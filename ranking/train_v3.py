"""Train and validate V3 without notebook state or original-artifact writes."""
import gc
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from data.training_data import V3_FEATURES, build_training_data
from evaluation.metrics import file_hash


def compare_training_table(train, artifact):
    original = joblib.load(artifact/'inference_extras.joblib')['rank_tr3']
    keys = ['user_id','video_id']
    actual = train.sort_values(keys).reset_index(drop=True)
    expected = original.sort_values(keys).reset_index(drop=True)
    ids_match = np.array_equal(actual[keys],expected[keys])
    mismatches,errors = {},{}
    if ids_match:
        for col in ['label']+V3_FEATURES:
            a,b = actual[col].astype('float64').to_numpy(),expected[col].astype('float64').to_numpy()
            same = np.isclose(a,b,atol=1e-12,rtol=0,equal_nan=True)
            mismatches[col] = int((~same).sum())
            finite = np.isfinite(a)&np.isfinite(b)
            errors[col] = float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.0
    vocab_match = all(actual[c].cat.categories.equals(expected[c].cat.categories) for c in ['cat_1','cat_2']) if ids_match else False
    result = {'passed':bool(ids_match and vocab_match and all(n==0 for n in mismatches.values())),
        'rows_actual':len(actual),'rows_saved':len(expected),'candidate_keys_match':ids_match,
        'category_vocabularies_match':vocab_match,'mismatching_rows_by_column':mismatches,'maximum_absolute_error':errors}
    del original,actual,expected
    gc.collect()
    return result


def train_v3(processed,artifact,output,project_root):
    config_path = project_root/'configs/train_v3.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['recall_k'] != 1000 or config['itemcf_ratio'] != 0.7:
        raise ValueError('Frozen training recipe requires recall_k=1000 and itemcf_ratio=0.7')
    zip_path = Path(config['raw_zip_path'])
    inputs = [processed/'train.csv',zip_path,config_path,artifact/'inference_extras.joblib',artifact/'lightgbm_v3.joblib']
    before = {str(p):file_hash(p) for p in inputs}
    (output/'resolved_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    start = time.perf_counter()
    train,valid,vocab,split = build_training_data(processed/'train.csv',zip_path,config)
    (output/'internal_split.json').write_text(json.dumps(split,ensure_ascii=False,indent=2),encoding='utf-8')
    table_comparison = compare_training_table(train,artifact)
    (output/'training_table_comparison.json').write_text(json.dumps(table_comparison,ensure_ascii=False,indent=2),encoding='utf-8')
    if not table_comparison['passed']:
        raise RuntimeError('Rebuilt training table differs from original; training stopped for review')
    print('All 800000 saved training rows match: keys, labels, 17 features and category vocabularies',flush=True)
    model = lgb.LGBMClassifier(**config['model_params'])
    model.fit(train[V3_FEATURES],train.label,eval_X=valid[V3_FEATURES],eval_y=valid.label,
        eval_metric='binary_logloss',callbacks=[lgb.early_stopping(config['early_stopping_rounds'],verbose=False)])
    prediction = model.predict_proba(valid[V3_FEATURES])[:,1]
    reference = joblib.load(artifact/'lightgbm_v3.joblib')
    reference_prediction = reference.predict_proba(valid[V3_FEATURES])[:,1]
    difference = np.abs(prediction-reference_prediction)
    model_comparison = {'internal_valid_rows':len(valid),'maximum_score_difference':float(difference.max()),
        'score_mismatch_rows':int((difference>1e-12).sum()),
        'best_iteration_actual':int(model.best_iteration_),'best_iteration_saved':int(reference.best_iteration_),
        'booster_text_identical':model.booster_.model_to_string()==reference.booster_.model_to_string()}
    model_comparison['passed'] = bool(model_comparison['score_mismatch_rows']==0 and model.best_iteration_==reference.best_iteration_)
    joblib.dump(model,output/'lightgbm_v3_retrained.joblib')
    joblib.dump({'v3_features':V3_FEATURES,'v3_cat_categories':vocab},output/'training_feature_state.joblib')
    # Save the development-validation predictions for repeatable checks; this is not external Val.
    predictions = valid[['user_id','video_id','label']].copy()
    predictions['score'] = prediction
    predictions.to_csv(output/'internal_validation_predictions.csv.gz',index=False)
    summary = {'train_rows':len(train),'internal_valid_rows':len(valid),
        'train_users':train.user_id.nunique(),'internal_valid_users':valid.user_id.nunique(),
        'best_iteration':int(model.best_iteration_),'internal_auc':float(roc_auc_score(valid.label,prediction)),
        'internal_logloss':float(log_loss(valid.label,prediction)),'seconds':time.perf_counter()-start,
        'table_comparison_passed':table_comparison['passed'],'model_comparison':model_comparison,
        'note':'Trained only from original Train with internal time split. Internal AUC is not full Val ranking quality.'}
    after = {str(p):file_hash(p) for p in inputs}
    manifest = {'python':platform.python_version(),'inputs_unchanged':before==after,'input_sha256':before,
        'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','lightgbm','joblib']},
        'source_sha256':{p.name:file_hash(p) for p in (project_root/'ranking').glob('*.py')}}
    for name,value in [('training_summary.json',summary),('manifest.json',manifest)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=int),encoding='utf-8')
    if not manifest['inputs_unchanged']:
        raise RuntimeError('An input changed during training')
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=int),flush=True)
    return bool(model_comparison['passed'])
