from recall.itemcf import ItemCFRecall
from recall.popularity import PopularityRecall
from recall.fusion import FusionRecall
from ranking.feature_engineering import FeatureEngineering

from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd
STATE_KEYS = ['R', 'cat_df', 'cat_total', 'full_pop_count', 'items', 'neighbor_indices', 'neighbor_scores', 'popular_items', 'seen_cols_by_user', 'train_seen', 'user_snapshot', 'user_to_row']

class RecommendationPipeline(ItemCFRecall, PopularityRecall, FusionRecall, FeatureEngineering):

    def recommend(self, user_id, top_k=50, return_stages=False):
        if not 1 <= top_k <= 200:
            raise ValueError('top_k必须在1到200之间')
        recall = self.build_full_candidates(user_id)
        candidates = self.add_full_rank_attrs(recall)
        candidates['lgb_score'] = self.rank_model_v3.predict_proba(candidates[self.v3_features])[:, 1]
        ranked = candidates.sort_values('lgb_score', ascending=False, kind='stable').reset_index(drop=True)
        ranked['lgb_rank'] = np.arange(1, len(ranked) + 1)
        coarse = ranked.head(200).copy()
        final = coarse.head(top_k).copy()
        if not return_stages:
            return final[['user_id', 'video_id', 'lgb_score', 'lgb_rank']]
        return {'recall_top1000': recall, 'coarse_top200': coarse[['user_id', 'video_id', 'lgb_score', 'lgb_rank']], 'final_topk': final[['user_id', 'video_id', 'lgb_score', 'lgb_rank']]}

def load_pipeline(artifact_dir, stage='val'):
    """Load trusted local artifacts without executing their Python source."""
    artifact_dir = Path(artifact_dir).resolve()
    if stage not in {'val', 'test'}:
        raise ValueError('stage must be val or test')
    recall = joblib.load(artifact_dir / 'recall_state.joblib')
    feature = joblib.load(artifact_dir / 'feature_state.joblib')
    state = {key: recall[key] for key in STATE_KEYS}
    if stage == 'test':
        saved = joblib.load(artifact_dir / 'test_state_v1.joblib')
        mapping = {'R': 'test_R', 'items': 'test_items', 'user_to_row': 'test_user_to_row', 'train_seen': 'test_seen', 'popular_items': 'test_popular_items', 'full_pop_count': 'test_pop_count', 'neighbor_indices': 'test_neighbor_indices', 'neighbor_scores': 'test_neighbor_scores', 'user_snapshot': 'test_user_attrs', 'cat_total': 'test_cat_attrs'}
        state.update({key: saved[value] for key, value in mapping.items()})
        item_to_col = saved['test_item_to_col']
        state['seen_cols_by_user'] = {uid: [item_to_col[v] for v in seen if v in item_to_col] for uid, seen in saved['test_seen'].items()}
    pipe = RecommendationPipeline()
    pipe.state = SimpleNamespace(**state)
    pipe.v3_features = feature['v3_features']
    pipe.v3_cat_categories = feature['v3_cat_categories']
    pipe.rank_model_v3 = joblib.load(artifact_dir / 'lightgbm_v3.joblib')
    if len(pipe.v3_features) != 17:
        raise ValueError('Expected 17 V3 features')
    if pipe.rank_model_v3.feature_name_ != pipe.v3_features:
        raise ValueError('Model feature names do not match saved feature state')
    pipe.stage = stage
    return pipe

def train_v3(*args, **kwargs):
    from ranking.train_v3 import train_v3 as implementation
    return implementation(*args, **kwargs)
