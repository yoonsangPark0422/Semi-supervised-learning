"""Held-out diagnostics only; never selects a training target from ground truth."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch


def prediction_rows(major_logits, minor_logits, targets, counts, args,
                    bias_scale, group_indices, start_index=0, cross_enabled=True):
    prior = counts.to(minor_logits.device).float()
    prior = prior / prior.sum()
    major_probs = torch.softmax(major_logits.detach() / args.T, dim=-1)
    raw_probs = torch.softmax(minor_logits.detach() / args.T, dim=-1)
    adjusted_probs = torch.softmax(
        minor_logits.detach() / args.T
        - bias_scale * args.minority_bias_strength * prior.log()[None, :], dim=-1)
    major_conf, major_pred = major_probs.max(-1)
    raw_conf, raw_pred = raw_probs.max(-1)
    adjusted_conf, adjusted_pred = adjusted_probs.max(-1)
    heads, _, tails = group_indices
    heads, tails = set(heads.tolist()), set(tails.tolist())

    def group(label):
        return 'head' if label in heads else 'tail' if label in tails else 'middle'

    for i in range(len(targets)):
        m, r, a, y = (int(x[i]) for x in (major_pred, raw_pred, adjusted_pred, targets))
        mc, rc, ac = (float(x[i]) for x in (major_conf, raw_conf, adjusted_conf))
        if m == r == a:
            origin = 'all_agree'
        elif m == r:
            origin = 'correction_created_disagreement'
        elif m == a:
            origin = 'correction_resolved_disagreement'
        elif r == a:
            origin = 'model_disagreement_unchanged_by_correction'
        else:
            origin = 'all_three_differ'
        major_candidate = cross_enabled and m in heads and mc >= args.tau_major_cross
        minor_candidate = cross_enabled and a in tails and ac >= args.tau_minor_cross
        conflict = major_candidate and minor_candidate
        skip = (major_candidate and not conflict
                and args.major_cross_skip_minor_conf_threshold >= 0
                and ac >= args.major_cross_skip_minor_conf_threshold)
        yield {
            'validation_index': start_index + i, 'target': y,
            'major_pred': m, 'major_confidence': mc,
            'minor_raw_pred': r, 'minor_raw_confidence': rc,
            'minor_adjusted_pred': a, 'minor_adjusted_confidence': ac,
            'raw_confidence_gap_major_minus_minor': mc - rc,
            'adjusted_confidence_gap_descriptive_only': mc - ac,
            'major_prob_of_minor_adjusted_label': float(major_probs[i, a]),
            'minor_raw_prob_of_major_label': float(raw_probs[i, m]),
            'minor_raw_prob_of_adjusted_label': float(raw_probs[i, a]),
            'major_group': group(m), 'minor_raw_group': group(r),
            'minor_adjusted_group': group(a), 'origin': origin,
            'disagree': m != a, 'correction_changed_label': r != a,
            'major_correct': m == y, 'minor_raw_correct': r == y,
            'minor_adjusted_correct': a == y,
            'major_only_correct': m == y and a != y,
            'minor_only_correct': a == y and m != y,
            'both_wrong': m != y and a != y,
            'correction_wrong_to_right': r != y and a == y,
            'correction_right_to_wrong': r == y and a != y,
            'major_cross_candidate': major_candidate,
            'minor_cross_candidate': minor_candidate,
            'cross_conflict': conflict,
            'major_to_minor_sent': major_candidate and not conflict and not skip,
            'minor_to_major_sent': minor_candidate and not conflict,
        }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups['all_validation'].append(row)
        if row['disagree']:
            groups['all_disagreements'].append(row)
        groups['origin/' + row['origin']].append(row)
        if row['disagree']:
            key = ('disagreement/' + row['origin'] + '/major_' + row['major_group']
                   + '/minor_raw_' + row['minor_raw_group']
                   + '/minor_adjusted_' + row['minor_adjusted_group'])
            groups[key].append(row)
        for mask in ('major_cross_candidate', 'minor_cross_candidate',
                     'cross_conflict', 'major_to_minor_sent', 'minor_to_major_sent'):
            if row[mask]:
                groups[mask].append(row)
                if row['disagree']:
                    groups[mask + '/disagreement'].append(row)
    metrics = ('major_only_correct', 'minor_only_correct', 'both_wrong',
               'major_correct', 'minor_raw_correct', 'minor_adjusted_correct',
               'correction_changed_label', 'correction_wrong_to_right',
               'correction_right_to_wrong')
    output = []
    for name, subset in sorted(groups.items()):
        entry = {'condition': name, 'n': len(subset)}
        for metric in metrics:
            entry[metric + '_count'] = sum(row[metric] for row in subset)
            entry[metric + '_rate'] = entry[metric + '_count'] / len(subset)
        output.append(entry)
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def diagnose_validation(args, loader, major, minor, class_counts,
                        group_indices, bias_scale, epoch, global_step):
    if args.val_samples_per_class <= 0:
        raise ValueError('Disagreement diagnostics require held-out validation, not official test.')
    if not isinstance(loader.sampler, torch.utils.data.SequentialSampler):
        raise ValueError('Sequential validation loader required for stable image indices.')
    modes = (major.training, minor.training)
    rows = []
    major.eval()
    minor.eval()
    try:
        for inputs, targets in loader:
            inputs = inputs.to(args.device)
            rows.extend(prediction_rows(
                major(inputs), minor(inputs), targets, class_counts, args,
                bias_scale, group_indices, len(rows), global_step >= args.pseudo_warmup))
    finally:
        major.train(modes[0])
        minor.train(modes[1])
    folder = Path(args.out) / 'validation_disagreement' / ('epoch_%03d' % epoch)
    folder.mkdir(parents=True, exist_ok=True)
    write_csv(folder / 'samples.csv', rows)
    write_csv(folder / 'conditions.csv', summarize(rows))
    metadata = {
        'dataset': args.dataset, 'seed': args.seed, 'epoch': epoch,
        'global_step': global_step, 'bias_scale': bias_scale,
        'minority_bias_strength': args.minority_bias_strength, 'temperature': args.T,
        'val_samples_per_class': args.val_samples_per_class,
        'model_source': 'ema' if args.use_ema else 'online',
        'tau_major_cross': args.tau_major_cross, 'tau_minor_cross': args.tau_minor_cross,
        'head_classes': group_indices[0].tolist(), 'tail_classes': group_indices[2].tolist(),
        'rate_denominator': 'n within each condition; rates are fractions, not percentages',
        'validation_index': 'position in deterministic validation split for this seed',
        'limitations': 'Clean validation images in eval mode; not actual augmented train batches. '
                      'Confidence gaps are descriptive, not calibrated reliability scores. '
                      'Do not fit a rule and report its accuracy on these same validation samples.',
    }
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return folder
