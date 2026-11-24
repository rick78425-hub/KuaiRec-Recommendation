import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from ranking.lightgbm_ranker import load_pipeline
from ranking.deepfm import DeepFMComparison
art=Path(json.loads((root/'configs/local.json').read_text())['artifact_dir'])
pipe=load_pipeline(art,'val')
val=pd.read_csv(art.parent/'val.csv',usecols=['user_id','video_id','watch_ratio'])
checks=[]
for version in ['v1','v2']:
    model=DeepFMComparison(art,version)
    saved=pd.read_csv(art/f'deepfm_{version}_val_results.csv').set_index('user_id')
    for uid in [0,1,2]:
        candidates=pipe.build_full_candidates(uid)
        features=pipe.add_full_rank_attrs(candidates)
        features['coarse_score']=pipe.rank_model_v3.predict_proba(features[pipe.v3_features])[:,1]
        features['coarse_rank']=features.coarse_score.rank(ascending=False,method='first')
        coarse=features.loc[features.coarse_rank<=200].copy()
        scores=model.predict(coarse)
        coarse['dfm_rank']=pd.Series(scores,index=coarse.index).rank(ascending=False,method='first')
        targets=set(val.loc[(val.user_id==uid)&(val.watch_ratio>1),'video_id'])-pipe.state.train_seen.get(uid,set())
        hits=len(set(coarse.loc[coarse.dfm_rank<=50,'video_id'])&targets)
        checks.append({'version':version,'user_id':uid,'finite_scores':bool(np.isfinite(scores).all()),'dfm50_hits':hits,'saved_dfm50_hits':int(saved.loc[uid,'dfm50_hits']),'matches':hits==int(saved.loc[uid,'dfm50_hits'])})
baseline=json.loads((root/'results/reports/audit_evidence.json').read_text(encoding='utf-8'))
for name,expected in baseline['protected_file_sha256'].items():
    with open(name,'rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==expected
result={'passed':all(c['matches'] and c['finite_scores'] for c in checks),'checks':checks,'scope':'CPU inference on 3 Val users per version; no retraining or full DeepFM reevaluation','original_files_unchanged':True}
(root/'results/reports/deepfm_comparison_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
assert result['passed']
