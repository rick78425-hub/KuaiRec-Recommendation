import hashlib
from pathlib import Path
import numpy as np

def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
# ranked_ids：LightGBM 排序后的候选。
# fusion_ids：原始融合排序候选。
# targets：用户未来真实正向目标。
def ranking_metrics(ranked_ids, fusion_ids, targets):
    """Full-target denominator includes targets absent from the recall pool."""
    #Recall 的分母是用户全部未来正向目标，而不是召回出来的目标.没有被召回的视频也会计入分母，避免高估召回效果
    n = len(targets)
    if n == 0:
        raise ValueError('Evaluation requires at least one target')
    ranked_ids = list(ranked_ids)
    fusion_ids = list(fusion_ids)
    if len(set(ranked_ids)) != len(ranked_ids) or set(ranked_ids) != set(fusion_ids):
        raise ValueError('Ranked and fusion lists must contain the same unique candidates')
    hit = lambda ids, k: len(set(ids[:k]) & targets)
    ideal = float(np.sum(1 / np.log2(np.arange(2, min(n, 50) + 2))))

    def ndcg(ids):
        relevance = np.array([item in targets for item in ids[:50]], dtype=np.float64)
        return float(np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2))) / ideal)
    pool_hits = len(set(ranked_ids) & targets)
    return {'target_count': n, 'pool_hits': pool_hits, 'lgb50_hits': hit(ranked_ids, 50), 'lgb200_hits': hit(ranked_ids, 200), 'fusion50_hits': hit(fusion_ids, 50), 'fusion200_hits': hit(fusion_ids, 200), 'recall1000': pool_hits / n, 'lgb_recall200': hit(ranked_ids, 200) / n, 'fusion_recall200': hit(fusion_ids, 200) / n, 'fusion_recall50': hit(fusion_ids, 50) / n, 'lgb_recall50': hit(ranked_ids, 50) / n, 'fusion_ndcg50': ndcg(fusion_ids), 'lgb_ndcg50': ndcg(ranked_ids)}
