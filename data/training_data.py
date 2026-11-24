"""Rebuild V3 samples from train.csv and raw item categories only."""
import ast
import gc
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

from ranking.lightgbm_ranker import RecommendationPipeline


RECALL_FEATURES = ['cf_score','cf_score_norm','cf_rank','pop_count','pop_rank',
    'in_cf_top1000','in_pop_top1000','fusion_order']
HISTORY_FEATURES = ['user_hist_count','user_hist_positive_count','user_hist_positive_rate',
    'cat_1','cat_2','cat_count','hist_cat_count','hist_cat_positive','user_cat_rate']
V3_FEATURES = RECALL_FEATURES + HISTORY_FEATURES


def split_history_future(frame, quantile):
    if not 0 < quantile < 1:
        raise ValueError('quantile must be between 0 and 1')
    cutoff = float(frame.timestamp.quantile(quantile))
    history = frame.loc[frame.timestamp < cutoff].copy()
    future = frame.loc[frame.timestamp >= cutoff].copy()
    if history.empty or future.empty or history.timestamp.max() >= future.timestamp.min():
        raise ValueError('Internal split must be nonempty and strictly chronological')
    return history, future, cutoff


def choose_users(history_users, config):
    history_users = sorted(history_users)
    preview = pd.Series(history_users).sample(n=config['preview_users'],random_state=config['preview_seed']).tolist()
    remaining = sorted(set(history_users)-set(preview))
    count = config['extra_train_users'] + config['valid_users']
    extras = np.random.default_rng(config['extra_seed']).choice(remaining,size=count,replace=False).tolist()
    train = set(preview) | set(extras[:config['extra_train_users']])
    valid = set(extras[config['extra_train_users']:])
    assert not train & valid
    return preview+extras, train, valid


def read_categories(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name.endswith('item_categories.csv')]
        if len(names) != 1:
            raise ValueError('Expected exactly one item_categories.csv in archive')
        with archive.open(names[0]) as stream:
            cats = pd.read_csv(stream,usecols=['video_id','feat'])
    lists = cats.feat.apply(lambda x:ast.literal_eval(x) if pd.notna(x) else [])
    cats['cat_count'] = lists.apply(len)
    cats['cat_1'] = lists.apply(lambda x:x[0] if len(x)>0 else -1)
    cats['cat_2'] = lists.apply(lambda x:x[1] if len(x)>1 else -1)
    if cats.video_id.duplicated().any():
        raise ValueError('Duplicate video category metadata')
    return cats[['video_id','cat_1','cat_2','cat_count']]


def build_history_state(history, cats, threshold, top_n):
    positive = history.loc[history.watch_ratio > threshold,['user_id','video_id']].drop_duplicates()
    users = pd.Index(sorted(positive.user_id.unique()))
    items = pd.Index(sorted(positive.video_id.unique()))
    matrix = csr_matrix((np.ones(len(positive),dtype=np.float32),
        (users.get_indexer(positive.user_id), items.get_indexer(positive.video_id))),shape=(len(users),len(items)))
    cooccurrence = (matrix.T @ matrix).tocsr()
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    diagonal = diags(1/np.sqrt(counts),format='csr')
    similarity = (diagonal @ cooccurrence @ diagonal).tocsr()
    similarity.setdiag(0)
    similarity.eliminate_zeros()
    neighbors = np.full((len(items),top_n),-1,dtype=np.int32)
    weights = np.zeros((len(items),top_n),dtype=np.float32)
    for i in range(len(items)):
        row = similarity.getrow(i)
        n = min(top_n,row.nnz)
        if n:
            chosen = np.argpartition(row.data,-n)[-n:]
            chosen = chosen[np.argsort(-row.data[chosen],kind='stable')]
            neighbors[i,:n],weights[i,:n] = row.indices[chosen],row.data[chosen]
    del cooccurrence,similarity,diagonal
    gc.collect()
    seen = history.groupby('user_id').video_id.agg(set).to_dict()
    item_to_col = {v:i for i,v in enumerate(items)}
    popularity = positive.groupby('video_id').user_id.nunique().sort_values(ascending=False)
    h = history[['user_id','video_id','watch_ratio']].copy()
    h['positive'] = (h.watch_ratio > threshold).astype('int8')
    attrs = h.groupby('user_id',as_index=False).agg(user_hist_count=('positive','size'),user_hist_positive_count=('positive','sum'))
    attrs['user_hist_positive_rate'] = attrs.user_hist_positive_count / attrs.user_hist_count
    h['cat_1'] = h.video_id.map(cats.set_index('video_id').cat_1).fillna(-1).astype('int32')
    cat_attrs = h.groupby(['user_id','cat_1'],as_index=False).agg(hist_cat_count=('positive','size'),hist_cat_positive=('positive','sum'))
    return SimpleNamespace(R=matrix,items=items,user_to_row={v:i for i,v in enumerate(users)},
        seen_cols_by_user={uid:[item_to_col[v] for v in values if v in item_to_col] for uid,values in seen.items()},
        train_seen=seen,popular_items=popularity.index.tolist(),full_pop_count=popularity,
        neighbor_indices=neighbors,neighbor_scores=weights,user_snapshot=attrs,cat_total=cat_attrs,cat_df=cats)


def attach_history(base, state, smoothing):
    frame = base.copy()
    frame['_row_id'] = np.arange(len(frame))
    frame = frame.merge(state.user_snapshot,on='user_id',how='left',validate='many_to_one')
    frame = frame.merge(state.cat_df,on='video_id',how='left',validate='many_to_one')
    frame = frame.merge(state.cat_total,on=['user_id','cat_1'],how='left',validate='many_to_one')
    cols = ['user_hist_count','user_hist_positive_count','user_hist_positive_rate','hist_cat_count','hist_cat_positive']
    frame[cols] = frame[cols].fillna(0)
    frame['user_cat_rate'] = (frame.hist_cat_positive+smoothing*frame.user_hist_positive_rate)/(frame.hist_cat_count+smoothing)
    frame = frame.sort_values('_row_id').drop(columns='_row_id').reset_index(drop=True)
    assert np.array_equal(frame[['user_id','video_id']],base[['user_id','video_id']])
    return frame


def build_training_data(train_path, zip_path, config):
    raw = pd.read_csv(train_path,usecols=['user_id','video_id','timestamp','watch_ratio'])
    history,future,cutoff = split_history_future(raw,config['history_quantile'])
    del raw
    print(f'Internal history/future: {len(history)}/{len(future)} rows',flush=True)
    cats = read_categories(zip_path)
    state = build_history_state(history,cats,config['positive_threshold'],config['itemcf_top_n'])
    user_order,train_users,valid_users = choose_users(state.train_seen,config['sampling'])
    history_rows,future_rows = len(history),len(future)
    positives = future.loc[future.watch_ratio > config['positive_threshold'],['user_id','video_id']].drop_duplicates().assign(label=1)
    del history,future
    gc.collect()
    pipe = RecommendationPipeline()
    pipe.state = state
    parts = []
    for offset in range(0,len(user_order),100):
        ids = user_order[offset:offset+100]
        candidates = pd.concat([pipe.build_full_candidates(int(uid)) for uid in ids],ignore_index=True)
        candidates = candidates.merge(positives.loc[positives.user_id.isin(ids)],on=['user_id','video_id'],how='left',validate='one_to_one')
        candidates['label'] = candidates.label.fillna(0).astype('int8')
        parts.append(candidates[['user_id','video_id']+RECALL_FEATURES+['label']])
        print(f'Training candidates: {min(offset+100,len(user_order))}/{len(user_order)} users',flush=True)
    base = pd.concat(parts,ignore_index=True)
    if base.duplicated(['user_id','video_id']).any():
        raise ValueError('Duplicate training candidate pairs')
    train = attach_history(base.loc[base.user_id.isin(train_users)],state,config['category_smoothing'])
    valid = attach_history(base.loc[base.user_id.isin(valid_users)],state,config['category_smoothing'])
    vocab = {}
    for col in ['cat_1','cat_2']:
        tr,va = train[col].fillna(-1).astype('int32'),valid[col].fillna(-1).astype('int32')
        vocab[col] = sorted(set(tr.unique()) | {-1})
        train[col] = pd.Categorical(tr,categories=vocab[col])
        valid[col] = pd.Categorical(va.where(va.isin(vocab[col]),-1),categories=vocab[col])
    info = {'cutoff_timestamp':cutoff,'cutoff_shanghai':str(pd.to_datetime(cutoff,unit='s',utc=True).tz_convert('Asia/Shanghai')),
        'history_rows':history_rows,'future_rows':future_rows,'train_users':sorted(map(int,train_users)),
        'valid_users':sorted(map(int,valid_users)),'user_generation_order':list(map(int,user_order)),
        'train_rows':len(train),'valid_rows':len(valid),'train_positives':int(train.label.sum()),
        'valid_positives':int(valid.label.sum()),'uses_val':False,'uses_test':False,
        'snapshot_source':'train.csv rows with timestamp strictly before internal cutoff'}
    return train,valid,vocab,info
