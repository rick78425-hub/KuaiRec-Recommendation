"""Frozen V3 full Test re-prediction followed by label-based evaluation."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime
import gzip
import hashlib
import json
import time
import numpy as np
import pandas as pd
from ranking.lightgbm_ranker import load_pipeline
from evaluation.test import main as audit_predictions

ROOT=Path(__file__).resolve().parent
def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    config_path=ROOT/'configs/reproduced_bundle.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    artifact=Path(config['artifact_dir'])
    inputs={str(p):sha(p) for p in artifact.glob('*.joblib')}
    inputs[str(config_path)]=sha(config_path)
    out=ROOT/'results/runs'/('test_end_to_end_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    pipe=load_pipeline(artifact,'test')
    users=sorted(pipe.state.train_seen)
    start=time.perf_counter();rows=0;counts=[]
    files={k:out/f'test_recommendations_top{k}.tmp.csv.gz' for k in [1000,50]}
    with gzip.open(files[1000],'wt',encoding='utf-8',newline='') as all_file,gzip.open(files[50],'wt',encoding='utf-8',newline='') as top_file:
        for offset in range(0,len(users),32):
            selected=users[offset:offset+32]
            candidates=pd.concat([pipe.build_full_candidates(uid) for uid in selected],ignore_index=True)
            features=pipe.add_full_rank_attrs(candidates)
            if not np.array_equal(candidates[['user_id','video_id']],features[['user_id','video_id']]):raise ValueError('Misaligned features')
            scores=pipe.rank_model_v3.predict_proba(features[pipe.v3_features])[:,1]
            if not np.isfinite(scores).all():raise ValueError('Nonfinite scores')
            candidates['lgb_score']=scores
            parts=[];tops=[]
            for uid,group in candidates.groupby('user_id',sort=True):
                ranked=group.sort_values('lgb_score',ascending=False,kind='stable').copy()
                ranked['lgb_rank']=np.arange(1,len(ranked)+1)
                parts.append(ranked);tops.append(ranked.head(50))
                counts.append({'user_id':int(uid),'candidate_count':len(ranked)})
                rows+=len(ranked)
            pd.concat(parts).to_csv(all_file,index=False,header=offset==0)
            pd.concat(tops).to_csv(top_file,index=False,header=offset==0)
            if (offset//32+1)%20==0 or offset+32>=len(users):
                print(f'Fresh Test predictions: {min(offset+32,len(users))}/{len(users)} users; {time.perf_counter()-start:.1f}s',flush=True)
    for k,path in files.items():path.replace(out/f'test_recommendations_top{k}.csv.gz')
    pd.DataFrame(counts).to_csv(out/'test_candidate_counts.csv',index=False)
    prediction_seconds=time.perf_counter()-start
    print('Predictions saved; now read Test labels for evaluation.',flush=True)
    audit=audit_predictions(prediction_dir=out)
    local=json.loads((ROOT/'configs/local.json').read_text(encoding='utf-8'))
    columns=['user_id','video_id','lgb_rank','lgb_score']
    new=pd.read_csv(out/'test_recommendations_top50.csv.gz',usecols=columns).sort_values(['user_id','lgb_rank']).reset_index(drop=True)
    old=pd.read_csv(Path(local['artifact_dir'])/'test_recommendations_top50.csv.gz',usecols=columns).sort_values(['user_id','lgb_rank']).reset_index(drop=True)
    top_match=np.array_equal(new[columns[:-1]],old[columns[:-1]]) and np.allclose(new.lgb_score,old.lgb_score,atol=1e-12,rtol=0)
    inputs_unchanged=all(sha(path)==value for path,value in inputs.items())
    report={'passed':bool(audit['passed'] and top_match and inputs_unchanged),'users_predicted':len(users),'candidate_rows':rows,
        'top50_rows':len(new),'top50_matches_original':bool(top_match),'prediction_seconds':prediction_seconds,
        'seconds':time.perf_counter()-start,'prediction_directory':str(out),'evaluation_directory':audit['output'],
        'metrics':audit['recomputed_summary'],'audit':audit,'input_sha256':inputs,'inputs_unchanged':inputs_unchanged,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for folder in ['recall','ranking','evaluation'] for p in (ROOT/folder).glob('*.py')},
        'protocol':'Frozen V3 from independently retrained model and rebuilt histories; predict all history users before reading Test labels; no tuning or retraining in this run.'}
    for path in [out/'end_to_end_report.json',ROOT/'results/reports/test_end_to_end_verification.json']:
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['audit','input_sha256','source_sha256']},ensure_ascii=False,indent=2),flush=True)
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
