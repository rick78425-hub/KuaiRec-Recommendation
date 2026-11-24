import numpy as np
import pandas as pd

class FeatureEngineering:

    def add_full_rank_attrs(self, base):
        df = base.copy()
        df['_row_id'] = np.arange(len(df))
        #给候选补充用户整体历史
        df = df.merge(self.state.user_snapshot, on='user_id', how='left', validate='many_to_one')
        #给候选补充视频类别
        df = df.merge(self.state.cat_df[['video_id', 'cat_1', 'cat_2', 'cat_count']], on='video_id', how='left', validate='many_to_one')
        #给候选补充用户对该类别的历史偏好
        df = df.merge(self.state.cat_total, on=['user_id', 'cat_1'], how='left', validate='many_to_one')
        hist_cols = ['user_hist_count', 'user_hist_positive_count', 'user_hist_positive_rate', 'hist_cat_count', 'hist_cat_positive']
        df[hist_cols] = df[hist_cols].fillna(0)
        #平滑公式：避免用户对某个类别的历史交互太少，导致正向比例不可靠
        df['user_cat_rate'] = (df['hist_cat_positive'] + 10 * df['user_hist_positive_rate']) / (df['hist_cat_count'] + 10)
        for col in ['cat_1', 'cat_2']:
            categories = self.v3_cat_categories[col]
            values = df[col].fillna(-1).astype('int32')
            df[col] = pd.Categorical(values.where(values.isin(categories), -1), categories=categories)
        df = df.sort_values('_row_id').drop(columns='_row_id').reset_index(drop=True)
        assert np.array_equal(df[['user_id', 'video_id']].to_numpy(), base[['user_id', 'video_id']].to_numpy())
        return df
