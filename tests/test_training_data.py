import unittest
import pandas as pd

from data.training_data import split_history_future,choose_users,build_history_state,attach_history


class TrainingProtocolTests(unittest.TestCase):
    def test_boundary_ties_stay_in_future(self):
        data = pd.DataFrame({'timestamp':[1.,2.,2.,3.]})
        history,future,cutoff = split_history_future(data,0.5)
        self.assertEqual(cutoff,2.)
        self.assertEqual(history.timestamp.tolist(),[1.])
        self.assertEqual(future.timestamp.tolist(),[2.,2.,3.])

    def test_degenerate_time_split_rejected(self):
        with self.assertRaises(ValueError):
            split_history_future(pd.DataFrame({'timestamp':[2.,2.]}),0.8)

    def test_user_partition_reproducible_and_disjoint(self):
        config = {'preview_users':2,'extra_train_users':3,'valid_users':2,'preview_seed':42,'extra_seed':2026}
        first = choose_users(range(10),config)
        self.assertEqual(first,choose_users(range(10),config))
        self.assertFalse(first[1]&first[2])
        self.assertEqual((len(first[1]),len(first[2])),(5,2))

    def test_history_deduplicates_positive_but_filters_all_seen(self):
        history = pd.DataFrame({'user_id':[0,0,0,1], 'video_id':[10,10,20,20], 'watch_ratio':[2.,3.,0.,2.]})
        cats = pd.DataFrame({'video_id':[10,20], 'cat_1':[4,5], 'cat_2':[-1,-1], 'cat_count':[1,1]})
        state = build_history_state(history,cats,1.,2)
        self.assertEqual(state.R.nnz,2)
        self.assertEqual(state.train_seen[0],{10,20})
        row = state.user_snapshot.set_index('user_id').loc[0]
        self.assertEqual(row.user_hist_count,3)
        self.assertEqual(row.user_hist_positive_count,2)

    def test_history_join_preserves_labels_and_order(self):
        history = pd.DataFrame({'user_id':[0,0,1], 'video_id':[10,20,20], 'watch_ratio':[2.,0.,2.]})
        cats = pd.DataFrame({'video_id':[10,20], 'cat_1':[4,5], 'cat_2':[-1,-1], 'cat_count':[1,1]})
        state = build_history_state(history,cats,1.,2)
        base = pd.DataFrame({'user_id':[1,0], 'video_id':[10,20], 'label':[1,0]})
        enriched = attach_history(base,state,10)
        pd.testing.assert_frame_equal(enriched[['user_id','video_id','label']],base)
        self.assertAlmostEqual(enriched.iloc[1].user_cat_rate,5/11)


if __name__ == '__main__':
    unittest.main()
