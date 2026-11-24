import numpy as np
import pandas as pd

class PopularityRecall:
#这里没有重新计算热门度，只是从已经计算好的热门榜中筛选
    def get_popularity(self, uid, k=1000):
        seen = self.state.train_seen.get(uid, set())
        return [item for item in self.state.popular_items if item not in seen][:k]
