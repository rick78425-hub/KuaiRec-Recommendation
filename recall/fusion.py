import numpy as np
import pandas as pd

class FusionRecall:
#ItemCF 初始配额 700、popularity初始配额300。若去重或候选不足、会从两路候选中继续补齐，70/30不一定是最终贡献比例
    def fuse_recall(self, cf_list, pop_list, k=100, cf_ratio=0.7):
        result, used = ([], set())
        for source, quota in [(cf_list, int(k * cf_ratio)), (pop_list, k - int(k * cf_ratio))]:
            added = 0
            for item in source:
                if item in used:
                    continue
                result.append(item)
                used.add(item)
                added += 1
                if added == quota:
                    break
        for item in cf_list + pop_list:
            if len(result) >= k:
                break
            if item not in used:
                result.append(item)
                used.add(item)
        return result

    def build_full_candidates(self, uid):
        cf = self.get_itemcf(uid, 1000)
        pop = self.get_popularity(uid, 1000)
        pool = self.fuse_recall(cf, pop, k=1000, cf_ratio=0.7)
        scores = np.zeros(len(self.state.items), dtype=np.float64)
        if uid in self.state.user_to_row:
            history = self.state.R.getrow(self.state.user_to_row[uid]).indices
            nbrs = self.state.neighbor_indices[history].ravel()
            weights = self.state.neighbor_scores[history].ravel()
            valid = (nbrs >= 0) & (weights > 0)
            scores = np.bincount(nbrs[valid], weights=weights[valid], minlength=len(self.state.items))
        scores[self.state.seen_cols_by_user.get(uid, [])] = 0
        cols = np.flatnonzero(scores > 0)
        cols = cols[np.argsort(-scores[cols], kind='stable')]
        cf_rank_map = {self.state.items[c]: i + 1 for i, c in enumerate(cols)}
        seen = self.state.train_seen.get(uid, set())
        pop_all = [vid for vid in self.state.popular_items if vid not in seen]
        pop_rank_map = {vid: i + 1 for i, vid in enumerate(pop_all)}
        out = pd.DataFrame({'user_id': uid, 'video_id': pool})
        pool_cols = pd.Index(self.state.items).get_indexer(pool)
        assert (pool_cols >= 0).all(), '候选中出现不属于训练正向物品库的视频'
        out['cf_score'] = scores[pool_cols]
        out['cf_score_norm'] = out['cf_score'] / max(float(scores.max()), 1e-12)
        out['cf_rank'] = out['video_id'].map(cf_rank_map)
        out['pop_count'] = out['video_id'].map(self.state.full_pop_count)
        out['pop_rank'] = out['video_id'].map(pop_rank_map)
        out['in_cf_top1000'] = out['video_id'].isin(cf).astype('int8')
        out['in_pop_top1000'] = out['video_id'].isin(pop).astype('int8')
        out['fusion_order'] = np.arange(1, len(out) + 1)
        return out
