"""Audit frozen Test outputs only; no model fitting or parameter selection."""
# 检查已经冻结的 Test 预测结果是否正确
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime
import hashlib
import json
import time
from io import StringIO
import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    with open(path,'rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def main(prediction_dir=None):
    config=json.loads((ROOT/'configs/local.json').read_text(encoding='utf-8'))
    art=Path(config['artifact_dir']);data=Path(config['processed_dir'])
    prediction_dir=Path(prediction_dir) if prediction_dir is not None else art
    output=ROOT/'results/runs'/('audit_test_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    state=joblib.load(art/'test_state_v1.joblib')
    raw=pd.read_csv(data/'test.csv',usecols=['user_id','video_id','watch_ratio'],dtype={'user_id':'int32','video_id':'int32','watch_ratio':'float32'})
    positives=raw.loc[raw.watch_ratio>1,['user_id','video_id']].drop_duplicates()
    mask=[v not in state['test_seen'].get(u,set()) for u,v in zip(positives.user_id,positives.video_id)]
    target=positives.loc[mask].copy()
    del raw,positives
    target['rel']=1
    target_counts=target.groupby('user_id').size()
    reference_top=pd.read_csv(art/'test_recommendations_top50.csv.gz')
    if reference_top.duplicated(['user_id','lgb_rank']).any(): raise ValueError('Duplicate Top50 rank')
    reference_top=reference_top.set_index(['user_id','lgb_rank']).sort_index()
    checks={name:0 for name in ['duplicate_pairs','invalid_rank_permutations','seen_recommendations',
        'nonfinite_scores','score_order_violations','top50_missing_rows','top50_video_mismatches','top50_score_mismatches']}
    parts=[]; counts=[]; carry=None;last_uid=None;row_count=0;top_rows=0
    def process(frame):
        nonlocal last_uid,row_count,top_rows
        if frame.empty:return
        if not frame.user_id.is_monotonic_increasing: raise ValueError('Candidate file is not grouped in sorted user order')
        if last_uid is not None and frame.user_id.iloc[0]<=last_uid: raise ValueError('Repeated user across candidate groups')
        for uid,g in frame.groupby('user_id',sort=False):
            n=len(g);counts.append({'user_id':int(uid),'candidate_count':n})
            checks['duplicate_pairs']+=int(g.video_id.duplicated().sum())
            valid=(n<=1000 and np.array_equal(g.lgb_rank.to_numpy(),np.arange(1,n+1))
                and np.array_equal(np.sort(g.fusion_order.to_numpy()),np.arange(1,n+1)))
            checks['invalid_rank_permutations']+=int(not valid)
            checks['seen_recommendations']+=len(set(g.video_id)&state['test_seen'].get(uid,set()))
            checks['nonfinite_scores']+=int((~np.isfinite(g.lgb_score)).sum())
            checks['score_order_violations']+=int(not g.lgb_score.is_monotonic_decreasing)
        last_uid=int(frame.user_id.iloc[-1]);row_count+=len(frame)
        top=frame.loc[frame.lgb_rank<=50].set_index(['user_id','lgb_rank']).sort_index()
        expected=reference_top.reindex(top.index)
        top_rows+=len(top)
        checks['top50_missing_rows']+=int(expected.video_id.isna().sum())
        checks['top50_video_mismatches']+=int((top.video_id!=expected.video_id).sum())
        checks['top50_score_mismatches']+=int((~np.isclose(top.lgb_score,expected.lgb_score,atol=1e-12,rtol=0)).sum())
        merged=frame.merge(target,on=['user_id','video_id'],how='left',validate='many_to_one')
        rel=merged.rel.fillna(0).to_numpy(dtype=np.int8)
        merged['hit_pool']=rel
        for name,rank,k in [('hit_lgb200','lgb_rank',200),('hit_lgb50','lgb_rank',50),('hit_fusion200','fusion_order',200),('hit_fusion50','fusion_order',50)]:
            merged[name]=rel*(merged[rank].to_numpy()<=k)
        for name,rank in [('dcg_lgb50','lgb_rank'),('dcg_fusion50','fusion_order')]:
            values=merged[rank].to_numpy()
            merged[name]=rel*np.where(values<=50,1/np.log2(values+1),0)
        parts.append(merged.groupby('user_id')[['hit_pool','hit_lgb200','hit_lgb50','hit_fusion200','hit_fusion50','dcg_lgb50','dcg_fusion50']].sum())
    for chunk in pd.read_csv(prediction_dir/'test_recommendations_top1000.csv.gz',usecols=['user_id','video_id','lgb_rank','fusion_order','lgb_score'],dtype={'lgb_rank':'int16','fusion_order':'int16'},chunksize=500000):
        if carry is not None:chunk=pd.concat([carry,chunk],ignore_index=True)
        final_user=chunk.user_id.iloc[-1]
        carry=chunk.loc[chunk.user_id==final_user].copy()
        process(chunk.loc[chunk.user_id!=final_user])
        print(f'Audited {row_count:,} candidate rows',flush=True)
    process(carry)
    actual_counts=pd.DataFrame(counts).set_index('user_id').sort_index()
    saved_counts=pd.read_csv(art/'test_candidate_counts.csv').set_index('user_id').sort_index()
    counts_match=actual_counts.equals(saved_counts)
    agg=pd.concat(parts).groupby(level=0).sum().reindex(target_counts.index,fill_value=0)
    agg['target_count']=target_counts
    agg['idcg50']=[sum(1/np.log2(np.arange(2,min(int(n),50)+2))) for n in target_counts]
    for result,hits in [('recall1000','hit_pool'),('recall200','hit_lgb200'),('recall50','hit_lgb50'),('fusion_recall200','hit_fusion200'),('fusion_recall50','hit_fusion50')]:agg[result]=agg[hits]/target_counts
    agg['ndcg50']=agg.dcg_lgb50/agg.idcg50;agg['fusion_ndcg50']=agg.dcg_fusion50/agg.idcg50
    saved_metrics=pd.read_csv(art/'test_metrics_by_user_v1.csv',index_col=0).sort_index()
    actual=pd.read_csv(StringIO(agg.to_csv()),index_col=0).sort_index()
    user_match=actual.index.equals(saved_metrics.index)
    metric_mismatches={col:int((~np.isclose(actual[col],saved_metrics[col],rtol=0,atol=1e-12)).sum()) for col in saved_metrics.columns} if user_match else {'user_set':1}
    summary=json.loads((art/'test_summary_v1.json').read_text(encoding='utf-8'))
    recomputed={'test_users_with_targets':len(agg),'test_target_pairs':int(target_counts.sum())}
    for name,col in [('macro_recall_1000','recall1000'),('macro_recall_200','recall200'),('macro_recall_50','recall50'),('macro_ndcg_50','ndcg50'),('fusion_macro_recall_50','fusion_recall50'),('fusion_macro_ndcg_50','fusion_ndcg50')]:recomputed[name]=float(agg[col].mean())
    for k,col in [(1000,'hit_pool'),(200,'hit_lgb200'),(50,'hit_lgb50')]:recomputed[f'micro_recall_{k}']=float(agg[col].sum()/target_counts.sum())
    recomputed['item_warm_coverage']=float(target.video_id.isin(set(state['test_items'])).mean())
    recomputed['item_cold_start_ratio']=1-recomputed['item_warm_coverage']
    summary_match=all(np.isclose(value,summary[key],rtol=0,atol=1e-12) for key,value in recomputed.items())
    baseline=json.loads((ROOT/'results/reports/audit_evidence.json').read_text(encoding='utf-8'))
    originals_unchanged=all(digest(path)==expected for path,expected in baseline['protected_file_sha256'].items())
    passed=all(n==0 for n in checks.values()) and counts_match and top_rows==len(reference_top) and all(n==0 for n in metric_mismatches.values()) and summary_match and originals_unchanged
    report={'passed':bool(passed),'candidate_rows':row_count,'candidate_users':len(counts),'top50_rows':top_rows,
        'integrity_errors':checks,'candidate_counts_match':counts_match,'metric_user_sets_match':user_match,
        'metric_mismatching_users':metric_mismatches,'summary_matches':bool(summary_match),'recomputed_summary':recomputed,
        'original_files_unchanged':originals_unchanged,'seconds':time.perf_counter()-start,'output':str(output),
        'precision_note':'Rank arrays retain original int16 dtype; NumPy log2 uses float32 here. The prior float64 diagnostic differed by about 1e-11 in macro NDCG. Per-user comparisons round-trip the new table through CSV, matching the original float32 DCG serialization; normalized metrics and summary retain strict 1e-12 checks.', 'prediction_directory':str(prediction_dir), 'scope':'Audit supplied frozen predictions against raw Test labels; no new predictions, training, or tuning. Bootstrap CIs not recomputed.'}
    agg.to_csv(output/'test_metrics_recomputed.csv')
    report_name='test_full_audit.json' if prediction_dir==art else 'test_reprediction_audit.json'
    for path in [output/'audit.json',ROOT/'results/reports'/report_name]:path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if not passed:raise SystemExit(1)
    return report

if __name__=='__main__':main()



