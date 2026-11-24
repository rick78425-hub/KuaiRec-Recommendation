"""Full frozen DeepFM Val comparison; does not train or read Test labels."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import hashlib
import time
from datetime import datetime
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
from ranking.lightgbm_ranker import load_pipeline
from ranking.deepfm import DeepFMComparison

def main():
    config=json.loads((ROOT/'configs/local.json').read_text(encoding='utf-8'))
    art=Path(config['artifact_dir']).resolve()
    output=ROOT/'results/runs'/('deepfm_val_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    pipe=load_pipeline(art,'val')
    models={version:DeepFMComparison(art,version) for version in ['v1','v2']}
    raw=pd.read_csv(art.parent/'val.csv',usecols=['user_id','video_id','watch_ratio'])
    positive=raw.loc[raw.watch_ratio>1,['user_id','video_id']].drop_duplicates()
    targets={}
    for uid,group in positive.groupby('user_id'):
        values=set(map(int,group.video_id))-pipe.state.train_seen.get(int(uid),set())
        if values: targets[int(uid)]=values
    del raw,positive
    rows={v:[] for v in models}
    users=sorted(targets)
    start=time.perf_counter()
    for offset in range(0,len(users),32):
        uids=users[offset:offset+32]
        pool=pd.concat([pipe.build_full_candidates(uid) for uid in uids],ignore_index=True)
        features=pipe.add_full_rank_attrs(pool)
        features['coarse_score']=pipe.rank_model_v3.predict_proba(features[pipe.v3_features])[:,1]
        features['coarse_rank']=features.groupby('user_id').coarse_score.rank(ascending=False,method='first')
        coarse=features.loc[features.coarse_rank<=200].copy()
        scores={version:model.predict(coarse) for version,model in models.items()}
        for version,values in scores.items():
            if not np.isfinite(values).all(): raise ValueError('Nonfinite DeepFM score')
            coarse[version+'_score']=values
            coarse[version+'_rank']=coarse.groupby('user_id')[version+'_score'].rank(ascending=False,method='first')
        groups={uid:g for uid,g in coarse.groupby('user_id')}
        for uid,group in features.groupby('user_id'):
            target=targets[int(uid)]
            cg=groups[uid]
            common={'user_id':int(uid),'targets':len(target),'pool_hits':len(set(group.video_id)&target),
                'coarse_hits':len(set(cg.video_id)&target),
                'lgb50_hits':len(set(cg.loc[cg.coarse_rank<=50,'video_id'])&target)}
            for version in models:
                hits=len(set(cg.loc[cg[version+'_rank']<=50,'video_id'])&target)
                rows[version].append({**common,'dfm50_hits':hits})
        if (offset//32+1)%20==0 or offset+32>=len(users):
            print(f'DeepFM V1/V2: {min(offset+32,len(users))}/{len(users)} users; {time.perf_counter()-start:.1f}s',flush=True)
    report={'users':len(users),'seconds':time.perf_counter()-start,'models':{},'test_labels_read':False,'retrained':False}
    for version in models:
        frame=pd.DataFrame(rows[version]).set_index('user_id').sort_index()
        saved=pd.read_csv(art/f'deepfm_{version}_val_results.csv').set_index('user_id').sort_index()
        users_match=frame.index.equals(saved.index)
        mismatches={col:int((frame[col]!=saved[col]).sum()) for col in saved.columns} if users_match else {'users':1}
        report['models'][version]={'passed':users_match and all(n==0 for n in mismatches.values()),
            'mismatching_users_by_column':mismatches,
            'macro_deepfm_recall50':float((frame.dfm50_hits/frame.targets).mean()),
            'macro_lgb_recall50':float((frame.lgb50_hits/frame.targets).mean())}
        frame.to_csv(output/f'deepfm_{version}_val_results.csv')
    baseline=json.loads((ROOT/'results/reports/audit_evidence.json').read_text(encoding='utf-8'))
    for name,expected in baseline['protected_file_sha256'].items():
        with open(name,'rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=expected: raise ValueError('Original file changed: '+name)
    report['original_files_unchanged']=True
    report['passed']=all(m['passed'] for m in report['models'].values())
    report['output']=str(output)
    (output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'results/reports/deepfm_full_val_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    if not report['passed']: raise SystemExit(1)

if __name__=='__main__': main()
