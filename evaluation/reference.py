"""Check frozen inference against saved predictions, without reading test labels."""
import numpy as np
import pandas as pd

def verify_reference(pipe, artifact_dir):
    users = [0, 1, 2]
    if pipe.stage == "val":
        reference = pd.read_csv(artifact_dir / "recommend_reference_top50.csv")
    else:
        parts = []
        found = set()
        for chunk in pd.read_csv(artifact_dir / "test_recommendations_top50.csv.gz", chunksize=5000):
            selected = chunk.loc[chunk.user_id.isin(users)]
            parts.append(selected)
            found.update(selected.user_id.unique())
            if found == set(users):
                break
        reference = pd.concat(parts, ignore_index=True)
    checks = []
    for uid in users:
        actual = pipe.recommend(uid, top_k=50).reset_index(drop=True)
        expected = reference.loc[reference.user_id == uid].sort_values("lgb_rank").reset_index(drop=True)
        stages = pipe.recommend(uid, top_k=50, return_stages=True)
        pool = stages["recall_top1000"]
        checks.append({"user_id":uid, "reference_rows":len(expected),
            "ids_match": bool(np.array_equal(actual.video_id, expected.video_id)),
            "ranks_match": bool(np.array_equal(actual.lgb_rank, expected.lgb_rank)),
            "scores_match": bool(len(actual)==len(expected) and np.allclose(actual.lgb_score, expected.lgb_score, rtol=0, atol=1e-6)),
            "scores_finite": bool(np.isfinite(actual.lgb_score).all()),
            "scores_descending": bool(actual.lgb_score.is_monotonic_decreasing),
            "no_duplicate_candidates": bool(not pool.video_id.duplicated().any()),
            "no_seen_items": bool(not (set(pool.video_id) & pipe.state.train_seen.get(uid,set()))),
            "candidate_count_valid": bool(0 < len(pool) <= 1000)})
    passed = all(c["reference_rows"] == 50 and all(v for k,v in c.items() if k not in {"user_id","reference_rows"}) for c in checks)
    return {"stage":pipe.stage, "passed":passed, "checks":checks,
        "scope":"Three reference users only; no retraining or complete evaluation rerun."}
