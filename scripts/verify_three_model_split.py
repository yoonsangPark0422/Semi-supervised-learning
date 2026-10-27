"""Verify actual CIFAR-10 IDs selected by the three experiment loaders."""
import ast
import hashlib
import json
import logging
import math
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np

base = Path('C:/Users/FORYOUCOM/Desktop')
main = base / 'Semi-supervised-learning-main/Semi-supervised-learning-main'
repo = base / 'Semi-supervised-learning-upload-worktree'
paths = {
    'new_dual': main / 'dual/dataset/cifar_val_split.py',
    'real_last_2': repo / 'dual real last/dual real last-2/dataset/cifar_val_split.py',
    'decon': base / 'DeCon/dataset/data_dual_matched.py',
}
labels = []
for i in range(1, 6):
    with (main / f'dual/data/cifar-10-batches-py/data_batch_{i}').open('rb') as f:
        labels.extend(pickle.load(f, encoding='bytes')[b'labels'])
pool = np.arange(len(labels))
outputs = {}
for name, path in paths.items():
    names = {'dual_matched_x_u_split', 'make_dual_imb_data'} if name == 'decon' else {'x_u_split', 'make_imb_data'}
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(functions) == 2
    env = dict(np=np, math=math, logger=logging.getLogger(name))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), env)
    args = SimpleNamespace(num_labeled=4000, num_classes=10, imb_ratio=100,
                           imb_ratio_label=100, imb_type='exp', expand_labels=True,
                           batch_size=32, eval_step=512)
    np.random.seed(1)
    fn = env['dual_matched_x_u_split' if name == 'decon' else 'x_u_split']
    outputs[name] = fn(args, labels, pool)
    print(name, 'counts=', args.labeled_class_counts)
for pair in outputs.values():
    assert all(np.array_equal(a, b) for a, b in zip(pair, outputs['new_dual']))
assert len(outputs['new_dual'][1]) == 50000
assert np.bincount(np.asarray(labels)[outputs['new_dual'][1]]).tolist() == [5000]*10
manifest = {
    'seed': 1, 'all_three_exact_indices_equal': True,
    'labeled_unique_count': len(np.unique(outputs['new_dual'][0])),
    'labeled_expanded_count': len(outputs['new_dual'][0]),
    'unlabeled_count': 50000, 'unlabeled_per_class': [5000]*10,
    'labeled_expanded_sha256': hashlib.sha256(outputs['new_dual'][0].astype('<i8').tobytes()).hexdigest(),
    'unlabeled_sha256': hashlib.sha256(outputs['new_dual'][1].astype('<i8').tobytes()).hexdigest(),
    'loader_paths': {k: str(v) for k, v in paths.items()},
}
out = main / 'results/three_models_matched_fresh_cifar10_seed1/split_audit.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
