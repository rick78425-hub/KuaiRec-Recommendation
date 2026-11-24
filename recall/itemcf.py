import numpy as np
import pandas as pd

class ItemCFRecall:
#ItemCF 不会在每次推荐时重新计算相似度。相似度矩阵和 Top-100 邻居已经预先计算并保存在 self.state 中,在线推荐只需要查表和累加。
    def get_itemcf(self, uid, k=1000):
        if uid not in self.state.user_to_row:
            return []
        history = self.state.R.getrow(self.state.user_to_row[uid]).indices
        nbrs = self.state.neighbor_indices[history].ravel()
        weights = self.state.neighbor_scores[history].ravel()
        valid = (nbrs >= 0) & (weights > 0)
        if not valid.any():
            return []
        scores = np.bincount(nbrs[valid], weights=weights[valid], minlength=len(self.state.items))
        scores[self.state.seen_cols_by_user.get(uid, [])] = 0
        cols = np.flatnonzero(scores > 0)
        cols = cols[np.argsort(-scores[cols], kind='stable')][:k]
        return [self.state.items[c] for c in cols]
