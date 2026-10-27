"""Frozen clean-validation mask audit, not historical augmented train selection."""
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

base = Path('C:/Users/FORYOUCOM/Desktop')
root = base / 'Semi-supervised-learning-main/Semi-supervised-learning-main/results'
trainer = base / 'Semi-supervised-learning-upload-worktree/dual real last/dual real last-2/train_head_tail_exchange_sampler_alpha_val_split.py'
tree = ast.parse(trainer.read_text(encoding='utf-8-sig'))
needed = {'classwise_topk_mask', 'checked_class_counts'}
env = dict(torch=torch, math=math, np=np)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in needed], type_ignores=[]), str(trainer), 'exec'), env)
args = SimpleNamespace(num_classes=10, classwise_quota='global', major_top_ratio=.2,
                       minority_threshold_gamma=1., min_threshold=.5)
counts = torch.tensor([1612,967,580,348,209,124,74,44,26,16])
out = root / 'validation_tail_selection_audit'
out.mkdir(exist_ok=True)
summaries = []
for seed in (1,2):
    path = root / f'dual_real_last_2_lla_hard_validation_cifar10_e50_s512_seed{seed}/validation_disagreement/epoch_050/samples.csv'
    rows = list(csv.DictReader(path.open(encoding='utf-8-sig')))
    conf = torch.tensor([float(r['minor_adjusted_confidence']) for r in rows])
    pred = torch.tensor([int(r['minor_adjusted_pred']) for r in rows])
    selected = torch.zeros(len(rows))
    # Five deterministic random batch arrangements measure quota sensitivity.
    for repeat in range(5):
        order = np.random.RandomState(1000+repeat).permutation(len(rows))
        for start in range(0,len(rows),224):
            ids = torch.tensor(order[start:start+224])
            selected[ids] += env['classwise_topk_mask'](conf[ids], pred[ids], counts, args, .5)
    detailed = []
    for i,r in enumerate(rows):
        y = int(r['target'])
        if y < 7:
            continue
        p = int(pred[i]); c = float(conf[i])
        gate = r['correction_gate_eligible'] == 'True'
        sent = r['minor_to_major_sent'] == 'True'
        reason = ('sent' if sent else 'predicted_non_tail' if p < 7 else
                  'confidence_below_0.95' if c < .95 else 'bidirectional_cross_conflict')
        rate = float(selected[i])/5
        self_reason = ('confidence_below_0.5' if c < .5 else
                       'correction_gate_suppressed' if gate else
                       'quota_excluded_all_partitions' if rate == 0 else
                       'quota_depends_on_batch' if rate < 1 else 'selected_all_partitions')
        detailed.append(dict(seed=seed, validation_index=r['validation_index'], true_class=y,
            major_pred=r['major_pred'], major_confidence=r['major_confidence'],
            minor_raw_pred=r['minor_raw_pred'], minor_adjusted_pred=p,
            minor_adjusted_confidence=c, correct=(y==p), cross_reason=reason,
            cross_correct=(sent and y==p), self_reason=self_reason,
            self_baseline_selection_rate=rate, self_final_selection_rate=0 if gate else rate,
            major_to_minor_sent=r['major_to_minor_sent']))
    with (out/f'seed{seed}_tail_samples.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(detailed[0])); writer.writeheader(); writer.writerows(detailed)
    groups = {'all_tail': detailed}
    groups.update({str(c):[r for r in detailed if r['true_class']==c] for c in (7,8,9)})
    for group,g in groups.items():
        s = dict(seed=seed, group=group,n=len(g),correct=sum(r['correct'] for r in g),
            cross_reasons=dict(Counter(r['cross_reason'] for r in g)),
            self_reasons=dict(Counter(r['self_reason'] for r in g)),
            correct_not_sent=sum(r['correct'] and r['cross_reason']!='sent' for r in g),
            wrong_sent=sum(not r['correct'] and r['cross_reason']=='sent' for r in g),
            wrong_self_always_selected=sum(not r['correct'] and r['self_final_selection_rate']==1 for r in g),
            wrong_prediction_destinations=dict(Counter(r['minor_adjusted_pred'] for r in g if not r['correct'])))
        summaries.append(s)
report = dict(method='Existing epoch50 EMA clean validation predictions; final thresholds; five batch permutations of 224; no model updates.',
    limitations='This is a counterfactual frozen validation audit, not actual historical weak-aug train masks. Self quota depends on batch; last batch may be smaller. Latest full-train models not used.',
    summaries=summaries)
(out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
