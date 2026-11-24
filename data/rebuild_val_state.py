"""Rebuild the train-only validation inference state from original data."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import time
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix,diags
from data.training_data import read_categories

ROOT=Path(__file__).resolve().parents[1]

def main():
    local=json.loads((ROOT/'configs/local.json').read_text(encoding='utf-8'))
    config=json.loads((ROOT/'configs/train_v3.json').read_text(encoding='utf-8'))
    processed=Path(local['processed_dir'])
    artifact=Path(local['artifact_dir'])
    output=ROOT/'results/runs'/('rebuild_val_state_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    raw=pd.read_csv(processed/'train.csv',usecols=['user_id','video_id','watch_ratio'])
    positive=raw.loc[raw.watch_ratio>1,['user_id','video_id']].drop_duplicates()
    users=sorted(positive.user_id.unique())
    items=sorted(positive.video_id.unique())
    matrix=csr_matrix((np.ones(len(positive),dtype=np.float32),
        (pd.Index(users).get_indexer(positive.user_id),pd.Index(items).get_indexer(positive.video_id))),shape=(len(users),len(items)))
    counts=np.asarray(matrix.sum(axis=0)).ravel()
    diagonal=diags(1/np.sqrt(counts),format='csr')
    sim=(diagonal@(matrix.T@matrix).tocsr()@diagonal).tocsr()
    sim.setdiag(0)
    sim.eliminate_zeros()
    neighbors=np.full((len(items),100),-1,dtype=np.int32)
    weights=np.zeros((len(items),100),dtype=np.float32)
    for i in range(len(items)):
        row=sim.getrow(i)
        n=min(100,row.nnz)
        if n:
            chosen=np.argpartition(row.data,-n)[-n:]
            # Preserve the original full-Train cell 49 tie behavior.
            chosen=chosen[np.argsort(row.data[chosen])[::-1]]
            neighbors[i,:n],weights[i,:n]=row.indices[chosen],row.data[chosen]
    del sim,diagonal
    print('Train ItemCF rebuilt',flush=True)
    seen=raw.groupby('user_id').video_id.agg(set).to_dict()
    item_to_col={v:i for i,v in enumerate(items)}
    pop=positive.groupby('video_id').user_id.nunique()
    cats=read_categories(config['raw_zip_path'])
    raw['positive']=(raw.watch_ratio>1).astype('int8')
    attrs=raw.groupby('user_id',as_index=False).agg(user_hist_count=('positive','size'),user_hist_positive_count=('positive','sum'))
    attrs['user_hist_positive_rate']=attrs.user_hist_positive_count/attrs.user_hist_count
    raw['cat_1']=raw.video_id.map(cats.set_index('video_id').cat_1).fillna(-1).astype('int16')
    cat_attrs=raw.groupby(['user_id','cat_1'],as_index=False).agg(hist_cat_count=('positive','size'),hist_cat_positive=('positive','sum'))
    state={'R':matrix,'items':items,'user_to_row':{uid:i for i,uid in enumerate(users)},
        'seen_cols_by_user':{uid:np.array([item_to_col[v] for v in values if v in item_to_col],dtype=np.int32) for uid,values in seen.items()},
        'train_seen':seen,'popular_items':pop.sort_values(ascending=False).index.tolist(),'full_pop_count':pop,
        'neighbor_indices':neighbors,'neighbor_scores':weights,'user_snapshot':attrs,'cat_total':cat_attrs,'cat_df':cats}
    joblib.dump(state,output/'recall_state_rebuilt.joblib',compress=3)
    # Original artifacts are used only after reconstruction, for comparison.
    saved=joblib.load(artifact/'recall_state.joblib')
    checks={}
    for key,value in state.items():
        ref=saved[key]
        if key=='R': checks[key]=value.shape==ref.shape and (value!=ref).nnz==0
        elif isinstance(value,pd.DataFrame):
            columns=list(ref.columns)
            keys=['user_id','cat_1'] if key=='cat_total' else ['video_id'] if key=='cat_df' else ['user_id']
            try:
                pd.testing.assert_frame_equal(value[columns].sort_values(keys).reset_index(drop=True),ref.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=0)
                checks[key]=True
            except AssertionError: checks[key]=False
        elif isinstance(value,pd.Series):
            checks[key]=bool(value.sort_index().equals(ref.sort_index()))
        elif key=='seen_cols_by_user':
            checks[key]=value.keys()==ref.keys() and all(np.array_equal(np.sort(v),np.sort(ref[u])) for u,v in value.items())
        elif isinstance(value,dict): checks[key]=value==ref
        else: checks[key]=bool(np.array_equal(value,ref))
    report={'passed':all(checks.values()),'checks':{k:bool(v) for k,v in checks.items()},'seconds':time.perf_counter()-start,
        'output':str(output),'reconstruction_inputs':['train.csv','KuaiRec.zip/item_categories.csv'],
        'test_labels_read':False,'model_retrained':False,
        'note':'Feature dictionaries and model weights are still separately loaded; this rebuilds only recall/history state.'}
    baseline=json.loads((ROOT/'results/reports/audit_evidence.json').read_text(encoding='utf-8'))
    for name,expected in baseline['protected_file_sha256'].items():
        with open(name,'rb') as stream: assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
    report['original_files_unchanged']=True
    for path in [output/'comparison.json',ROOT/'results/reports/val_state_rebuild_verification.json']:
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    if not report['passed']: raise SystemExit(1)

if __name__=='__main__': main()
