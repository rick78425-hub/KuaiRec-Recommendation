from evaluation.metrics import file_hash, ranking_metrics
"""Reproduce full validation from frozen artifacts and raw validation targets."""
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd

def compare_saved(metrics, recommendations, artifact):
    expected = pd.read_csv(artifact / 'validation_metrics_v1.csv').set_index('user_id').sort_index()
    actual = metrics.set_index('user_id').sort_index()
    users_match = actual.index.equals(expected.index)
    deltas, mismatch_users = ({}, {})
    if users_match:
        for col in expected.columns:
            delta = np.abs(actual[col].to_numpy() - expected[col].to_numpy())
            deltas[col] = float(delta.max())
            mismatch_users[col] = int((delta > 1e-12).sum())
    old_rec = pd.read_csv(artifact / 'validation_recommendations_top50.csv')
    keys = ['user_id', 'rank']
    if old_rec.duplicated(keys).any() or recommendations.duplicated(keys).any():
        raise ValueError('Duplicate user/rank in saved or reproduced recommendations')
    joined = recommendations.merge(old_rec, on=keys, how='outer', suffixes=('_actual', '_saved'), indicator=True, validate='one_to_one')
    present = joined['_merge'].eq('both')
    ids_equal = joined.video_id_actual.eq(joined.video_id_saved)
    score_delta = np.abs(joined.lgb_score_actual - joined.lgb_score_saved)
    rec_checks = {'rows_actual': len(recommendations), 'rows_saved': len(old_rec), 'missing_or_extra_rows': int((~present).sum()), 'video_mismatch_rows': int((~ids_equal).sum()), 'score_mismatch_rows': int((~np.isclose(joined.lgb_score_actual, joined.lgb_score_saved, rtol=0, atol=1e-06)).sum()), 'maximum_score_difference': float(score_delta.max())}
    passed = users_match and bool(mismatch_users) and all((v == 0 for v in mismatch_users.values())) and all((rec_checks[k] == 0 for k in ['missing_or_extra_rows', 'video_mismatch_rows', 'score_mismatch_rows']))
    return {'passed': passed, 'metric_user_sets_match': users_match, 'metrics_max_absolute_difference': deltas, 'metrics_mismatching_users': mismatch_users, 'recommendations': rec_checks}

def evaluate_val(pipe, processed, artifact, output, batch_size=32):
    if pipe.stage != 'val':
        raise ValueError('Full validation requires the train-only history state')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    used_files = [processed / 'val.csv'] + [artifact / name for name in ['recall_state.joblib', 'feature_state.joblib', 'lightgbm_v3.joblib', 'validation_metrics_v1.csv', 'validation_recommendations_top50.csv']]
    before = {str(path): file_hash(path) for path in used_files}
    start = time.perf_counter()
    raw = pd.read_csv(processed / 'val.csv', usecols=['user_id', 'video_id', 'watch_ratio'])
    positive = raw.loc[raw.watch_ratio > 1, ['user_id', 'video_id']].drop_duplicates()
    targets = {}
    for uid, group in positive.groupby('user_id'):
        future = set(map(int, group.video_id)) - pipe.state.train_seen.get(int(uid), set())
        if future:
            targets[int(uid)] = future
    del raw, positive
    users = sorted(targets)
    if not users:
        raise ValueError('No validation users with new positive targets')
    print(f'Validation targets: {len(users)} users, {sum(map(len, targets.values()))} pairs', flush=True)
    rows, recommendation_parts = ([], [])
    timing = []
    candidate_total, zero_candidate_users = (0, 0)
    for offset in range(0, len(users), batch_size):
        batch_start = time.perf_counter()
        uids = users[offset:offset + batch_size]
        candidates = pd.concat([pipe.build_full_candidates(uid) for uid in uids], ignore_index=True)
        if candidates.duplicated(['user_id', 'video_id']).any():
            raise ValueError('Duplicate recall candidate')
        if len(candidates):
            features = pipe.add_full_rank_attrs(candidates)
            if not np.array_equal(features[['user_id', 'video_id']], candidates[['user_id', 'video_id']]):
                raise ValueError('Candidate-feature row order changed')
            scores = pipe.rank_model_v3.predict_proba(features[pipe.v3_features])[:, 1]
            if not np.isfinite(scores).all():
                raise ValueError('Nonfinite model scores')
            candidates['lgb_score'] = scores
        else:
            candidates['lgb_score'] = pd.Series(dtype='float64')
        groups = {int(uid): group for uid, group in candidates.groupby('user_id', sort=False)}
        for uid in uids:
            group = groups.get(uid, candidates.iloc[:0])
            pool = group.video_id.astype(int).tolist()
            if len(pool) > 1000 or set(pool) & pipe.state.train_seen.get(uid, set()):
                raise ValueError(f'Invalid candidate pool for user {uid}')
            candidate_total += len(pool)
            zero_candidate_users += int(not pool)
            ranked = group.sort_values('lgb_score', ascending=False, kind='stable')
            ranked_ids = ranked.video_id.astype(int).tolist()
            rows.append({'user_id': uid, **ranking_metrics(ranked_ids, pool, targets[uid])})
            top = ranked.head(50)
            recommendation_parts.append(pd.DataFrame({'user_id': uid, 'video_id': top.video_id.to_numpy(), 'lgb_score': top.lgb_score.to_numpy(), 'rank': np.arange(1, len(top) + 1)}))
        timing.append({'first_user': uids[0], 'users': len(uids), 'candidates': len(candidates), 'batch_seconds': time.perf_counter() - batch_start})
        if (offset // batch_size + 1) % 10 == 0 or offset + batch_size >= len(users):
            print(f'Validated {min(offset + batch_size, len(users))}/{len(users)} users; elapsed {time.perf_counter() - start:.1f}s', flush=True)
    metrics = pd.DataFrame(rows)
    recommendations = pd.concat(recommendation_parts, ignore_index=True)
    comparison = compare_saved(metrics, recommendations, artifact)
    summary = {'users': len(metrics), 'target_pairs': int(metrics.target_count.sum()), 'candidate_rows': candidate_total, 'zero_candidate_users': zero_candidate_users, 'macro_recall1000': float(metrics.recall1000.mean()), 'macro_lgb_recall200': float(metrics.lgb_recall200.mean()), 'macro_lgb_recall50': float(metrics.lgb_recall50.mean()), 'macro_fusion_recall50': float(metrics.fusion_recall50.mean()), 'macro_lgb_ndcg50': float(metrics.lgb_ndcg50.mean()), 'macro_fusion_ndcg50': float(metrics.fusion_ndcg50.mean()), 'micro_lgb_recall50': float(metrics.lgb50_hits.sum() / metrics.target_count.sum()), 'seconds': time.perf_counter() - start, 'batch_size': batch_size, 'protocol': {'history': 'train', 'positive_rule': 'watch_ratio > 1', 'target_rule': 'unique val positives minus all train seen; all nonempty target users retained', 'sorting': 'score descending, stable; preserve fusion order for ties', 'test_labels_read': False, 'model_retrained': False, 'latency_note': 'Batch elapsed time is not comparable to historical per-user latency.'}}
    metrics.to_csv(output / 'validation_metrics.csv', index=False)
    recommendations.to_csv(output / 'validation_recommendations_top50.csv', index=False)
    pd.DataFrame(timing).to_csv(output / 'batch_timing.csv', index=False)
    after = {str(path): file_hash(path) for path in used_files}
    manifest = {'python': platform.python_version(), 'input_sha256': before, 'inputs_unchanged': before == after, 'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'joblib']}, 'project_source_sha256': {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}}
    if not manifest['inputs_unchanged']:
        comparison['passed'] = False
    for filename, value in [('summary.json', summary), ('comparison.json', comparison), ('manifest.json', manifest)]:
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'summary': summary, 'comparison': comparison, 'inputs_unchanged': manifest['inputs_unchanged']}, ensure_ascii=False, indent=2), flush=True)
    return comparison['passed']
