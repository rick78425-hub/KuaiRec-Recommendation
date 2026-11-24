"""Package verified rebuilt states and retrained V3; keep original inputs read only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime
import shutil
import json
import hashlib
import joblib
from ranking.lightgbm_ranker import load_pipeline
from evaluation.reference import verify_reference

ROOT=Path(__file__).resolve().parents[1]

def main():
    def report(name):
        value=json.loads((ROOT/'results/reports'/name).read_text(encoding='utf-8'))
        if not value['passed']: raise ValueError('Upstream verification failed: '+name)
        return value
    val=report('val_state_rebuild_verification.json')
    test=report('test_state_rebuild_verification.json')
    train=report('v3_training_verification.json')
    bundle=ROOT/'artifacts'/('v3_reproduced_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    bundle.mkdir(parents=True,exist_ok=False)
    sources={'recall_state.joblib':Path(val['output'])/'recall_state_rebuilt.joblib',
        'test_state_v1.joblib':Path(test['output'])/'test_state_rebuilt.joblib',
        'lightgbm_v3.joblib':Path(train['run'])/'lightgbm_v3_retrained.joblib',
        'feature_state.joblib':Path(train['run'])/'training_feature_state.joblib'}
    for name,path in sources.items(): shutil.copy2(path,bundle/name)
    local=json.loads((ROOT/'configs/local.json').read_text(encoding='utf-8'))
    original=Path(local['artifact_dir'])
    checks={stage:verify_reference(load_pipeline(bundle,stage),original) for stage in ['val','test']}
    passed=all(result['passed'] for result in checks.values())
    manifest={'passed':passed,'scope':'V3 inference bundle; no DeepFM, raw CSV or reference predictions required for recommendation',
        'bundle':str(bundle),'inputs':{name:str(path) for name,path in sources.items()},
        'sha256':{name:hashlib.sha256((bundle/name).read_bytes()).hexdigest() for name in sources},
        'sample_verification':checks,'runtime_reads_original_artifacts':False,
        'comparison_reads_original_reference_predictions':True}
    (bundle/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'reports/v3_bundle_verification.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if not passed: raise RuntimeError('Bundle verification failed')
    local['artifact_dir']=str(bundle)
    local['purpose']='rebuilt_v3_inference_only'
    local['supported_commands']=['recommend']
    local.pop('frozen_protocol',None)
    (ROOT/'configs/reproduced_bundle.json').write_text(json.dumps(local,ensure_ascii=False,indent=2),encoding='utf-8')
    note='''V3 rebuilt inference bundle
recall_state.joblib: rebuilt Train history for Val inference.
test_state_v1.joblib: rebuilt Train+Val history for Test inference.
lightgbm_v3.joblib: independently retrained and verified V3.
feature_state.joblib: training feature order and category dictionaries only.
manifest.json: source paths, hashes and six sample checks.
Use the project Python source and its dependencies with configs/reproduced_bundle.json.
Recommendation does not need the original artifacts or raw CSV files.
This is not a source-code distribution or a complete evaluation/DeepFM bundle.
'''
    (bundle/'README.txt').write_text(note,encoding='utf-8')
    print(json.dumps({'passed':passed,'bundle':str(bundle),'sample_checks':6},ensure_ascii=False))

if __name__=='__main__': main()
