"""Rebuild Test history exclusively from Train and Val, without Test labels."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
from data.training_data import build_history_state,read_categories

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    local=json.loads((ROOT/'configs/local.json').read_text(encoding='utf-8'))
    recipe=json.loads((ROOT/'configs/train_v3.json').read_text(encoding='utf-8'))
    data,art=Path(local['processed_dir']),Path(local['artifact_dir'])
    output=ROOT/'results/runs'/('rebuild_test_state_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    paths=[data/'train.csv',data/'val.csv',Path(recipe['raw_zip_path'])]
    before={str(p):digest(p) for p in paths}
    start=time.perf_counter()
    columns=['user_id','video_id','watch_ratio','timestamp']
    train=pd.read_csv(paths[0],usecols=columns)
    val=pd.read_csv(paths[1],usecols=columns,dtype={'user_id':'int32','video_id':'int32','watch_ratio':'float32'})
    if train.timestamp.max()>=val.timestamp.min(): raise ValueError('Train/Val time boundary overlap')
    seen_val=val.groupby('user_id').video_id.agg(lambda x:set(x.astype(int))).to_dict()
    metadata={'train_rows':len(train),'val_rows':len(val),'latest_history_timestamp':float(val.timestamp.max()),
        'positive_rule':'watch_ratio > 1','itemcf_top_n':100,'neighbor_sort':'stable descending',
        'val_watch_ratio_dtype':'float32, preserved from original Test history protocol'}
    history=pd.concat([train,val],ignore_index=True)
    del train,val
    state=build_history_state(history,read_categories(paths[2]),1.,100)
    del history
    print('Train+Val history and ItemCF rebuilt',flush=True)
    users=pd.Index(sorted(state.user_to_row))
    items=state.items
    rebuilt={'test_R':state.R,'test_users':users,'test_items':items,
        'test_user_to_row':state.user_to_row,'test_item_to_col':{v:i for i,v in enumerate(items)},
        'test_neighbor_indices':state.neighbor_indices,'test_neighbor_scores':state.neighbor_scores,
        'test_pop_count':state.full_pop_count,'test_popular_items':state.popular_items,
        'test_seen':state.train_seen,'test_user_attrs':state.user_snapshot,'test_cat_attrs':state.cat_total,'test_seen_val':seen_val}
    joblib.dump(rebuilt,output/'test_state_rebuilt.joblib',compress=3)
    saved=joblib.load(art/'test_state_v1.joblib')
    checks={}
    for key,value in rebuilt.items():
        expected=saved[key]
        if key=='test_R': checks[key]=value.shape==expected.shape and (value!=expected).nnz==0
        elif isinstance(value,pd.DataFrame):
            keys=['user_id','cat_1'] if key=='test_cat_attrs' else ['user_id']
            try:
                pd.testing.assert_frame_equal(value[list(expected.columns)].sort_values(keys).reset_index(drop=True),
                    expected.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=0)
                checks[key]=True
            except AssertionError: checks[key]=False
        elif isinstance(value,pd.Series):
            checks[key]=bool(value.sort_index().equals(expected.sort_index()))
        elif isinstance(value,dict): checks[key]=value==expected
        else: checks[key]=bool(np.array_equal(value,expected))
    original=json.loads((ROOT/'results/reports/audit_evidence.json').read_text(encoding='utf-8'))
    unchanged=all(digest(name)==sha for name,sha in original['protected_file_sha256'].items())
    inputs_unchanged=before=={str(p):digest(p) for p in paths}
    report={'passed':all(checks.values()) and unchanged and inputs_unchanged,'checks':checks,
        'seconds':time.perf_counter()-start,'output':str(output),'input_sha256':before,
        'inputs_unchanged':inputs_unchanged,'original_files_unchanged':unchanged,
        'test_labels_read':False,'reference_role':'Loaded original state only after rebuilding, for comparison',
        'protocol':metadata,'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'data/training_data.py',ROOT/'ranking/lightgbm_ranker.py']}}
    for target in [output/'comparison.json',ROOT/'results/reports/test_state_rebuild_verification.json']:
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if not report['passed']: raise SystemExit(1)

if __name__=='__main__': main()
