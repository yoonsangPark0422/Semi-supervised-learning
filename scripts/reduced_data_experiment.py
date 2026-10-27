"""Shared exact 400-label/5000-unlabeled CIFAR-10 split and isolated trainer entry."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys
import numpy as np

BASE=Path('C:/Users/FORYOUCOM/Desktop')
MAIN=BASE/'Semi-supervised-learning-main/Semi-supervised-learning-main'
REPO=BASE/'Semi-supervised-learning-upload-worktree'
ROOT=MAIN/'results/reduced10pct_decon_real_last2_cifar10_seed1'
REAL=REPO/'dual real last/dual real last-2'

def labels():
    result=[]
    for i in range(1,6):
        with (MAIN/f'dual/data/cifar-10-batches-py/data_batch_{i}').open('rb') as f:
            result.extend(pickle.load(f,encoding='bytes')[b'labels'])
    return np.asarray(result)

def prepare():
    ROOT.mkdir(exist_ok=True)
    destination=ROOT/'shared_indices.npz'
    if destination.exists():raise RuntimeError('Split already exists; refusing overwrite')
    tree=ast.parse((REAL/'dataset/cifar_val_split.py').read_text(encoding='utf-8-sig'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_imb_data')
    env={'math':math}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<original_counts>','exec'),env)
    counts=env['make_imb_data'](400,10,100,'exp')
    y=labels();rng=np.random.RandomState(1)
    pool=[];labeled=[]
    for c,count in enumerate(counts):
        ids=rng.permutation(np.flatnonzero(y==c))[:500]
        pool.extend(ids);labeled.extend(ids[:count])
    pool=np.sort(np.asarray(pool,dtype=np.int64))
    labeled=np.asarray(labeled,dtype=np.int64)
    expanded=np.tile(labeled,math.ceil(32*512/400));rng.shuffle(expanded)
    assert len(np.unique(labeled))==400 and len(pool)==5000
    assert np.isin(labeled,pool).all()
    assert np.bincount(y[pool],minlength=10).tolist()==[500]*10
    assert np.bincount(y[labeled],minlength=10).tolist()==counts
    np.savez(destination,labeled=labeled,expanded=expanded,unlabeled=pool)
    meta=dict(seed=1,labeled_unique=400,unlabeled_unique=5000,labeled_counts=counts,
        unlabeled_counts=[500]*10,labeled_expanded=len(expanded),
        labeled_subset_of_unlabeled=True,requested_imbalance=100,
        realized_imbalance=max(counts)/min(counts),
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    (ROOT/'split_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))

def train(method,rest,audit_only=False):
    class AuditComplete(Exception):pass
    split=np.load(ROOT/'shared_indices.npz')
    expected=json.loads((ROOT/'split_manifest.json').read_text())
    assert hashlib.sha256((ROOT/'shared_indices.npz').read_bytes()).hexdigest()==expected['sha256']
    code=REAL if method=='real_last_dual_2' else BASE/'DeCon'
    filename='train_head_tail_exchange_sampler_alpha_val_split.py' if method=='real_last_dual_2' else 'train_dual_matched.py'
    sys.path.insert(0,str(code))
    spec=importlib.util.spec_from_file_location('reduced_experiment_trainer',code/filename)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    original=module.DATASET_GETTERS['cifar10']
    def fixed_split(args,all_labels,candidate_idxs=None):
        assert args.seed==1 and args.num_labeled==400 and args.num_classes==10
        assert args.val_samples_per_class==0 and args.expand_labels
        assert not args.resume and not getattr(args,'init_checkpoint','')
        assert args.batch_size==32 and args.eval_step==512
        y=np.asarray(all_labels)
        assert len(y)==50000
        assert np.bincount(y[split['unlabeled']],minlength=10).tolist()==[500]*10
        assert np.bincount(y[split['labeled']],minlength=10).tolist()==expected['labeled_counts']
        args.labeled_class_counts=expected['labeled_counts']
        args.num_labeled=400
        print('REDUCED DATA: labeled=400 unique, unlabeled=5000 balanced, shared exact IDs',flush=True)
        return split['expanded'].copy(),split['unlabeled'].copy()
    original.__globals__['x_u_split' if method=='real_last_dual_2' else 'dual_matched_x_u_split']=fixed_split
    def audited_getter(args,root):
        if method=='decon':assert args.match_dual_data
        ds=original(args,root)
        assert len(ds[0])==expected['labeled_expanded'] and len(ds[1])==5000 and len(ds[2])==10000
        audit=dict(expected,method=method,test_count=len(ds[2]),actual_labeled_length=len(ds[0]),actual_unlabeled_length=len(ds[1]),
            trainer=str(code/filename),trainer_sha256=hashlib.sha256((code/filename).read_bytes()).hexdigest())
        (Path(args.out)/'actual_data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
        print('ACTUAL DATA AUDIT PASSED: 400 unique labeled, 5000 unlabeled, 10000 official test',flush=True)
        if audit_only:
            if hasattr(args,'writer'):args.writer.close()
            raise AuditComplete()
        return ds
    module.DATASET_GETTERS['cifar10']=audited_getter
    sys.argv=[str(code/filename)]+rest
    try:module.main()
    except AuditComplete:pass

if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--method',choices=['real_last_dual_2','decon'])
    args,rest=parser.parse_known_args()
    if args.prepare:prepare()
    elif args.method:train(args.method,rest,args.audit_only)
    else:raise SystemExit('Specify --prepare or --method')
