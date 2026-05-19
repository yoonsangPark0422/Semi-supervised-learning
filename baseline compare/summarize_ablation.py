import argparse
import csv
import math
import os


def read_final_metrics(path):
    with open(path, newline='') as csvfile:
        rows = list(csv.DictReader(csvfile))
    if not rows:
        return None
    row = rows[-1]
    acc = row.get('test_acc_final') or row.get('test_acc')
    macro_f1 = row.get('macro_f1_final') or row.get('macro_f1')
    return {
        'acc': float(acc) if acc not in ['', None] else math.nan,
        'macro_f1': float(macro_f1) if macro_f1 not in ['', None] else math.nan,
        'pseudo_pred_js': float(row['pseudo_pred_js'])
        if row.get('pseudo_pred_js') else math.nan,
        'pseudo_agreement_rate': float(row['pseudo_agreement_rate'])
        if row.get('pseudo_agreement_rate') else math.nan,
        'epoch_time_sec': float(row['epoch_time_sec'])
        if row.get('epoch_time_sec') else math.nan,
        'peak_allocated_mb': float(row['peak_allocated_mb'])
        if row.get('peak_allocated_mb') else math.nan,
    }


def mean_std(values):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/ablation_fast')
    args = parser.parse_args()

    grouped = {}
    for seed_name in os.listdir(args.root):
        seed_dir = os.path.join(args.root, seed_name)
        if not os.path.isdir(seed_dir):
            continue
        for exp_name in os.listdir(seed_dir):
            metrics_path = os.path.join(seed_dir, exp_name, 'train_metrics.csv')
            if not os.path.exists(metrics_path):
                continue
            grouped.setdefault(exp_name, []).append(read_final_metrics(metrics_path))

    fields = ['experiment', 'runs', 'acc_mean', 'acc_std', 'macro_f1_mean',
              'macro_f1_std', 'pseudo_pred_js_mean',
              'pseudo_agreement_rate_mean', 'epoch_time_sec_mean',
              'peak_allocated_mb_mean']
    out_path = os.path.join(args.root, 'summary.csv')
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for exp_name in sorted(grouped):
            rows = [row for row in grouped[exp_name] if row is not None]
            acc_mean, acc_std = mean_std([row['acc'] for row in rows])
            f1_mean, f1_std = mean_std([row['macro_f1'] for row in rows])
            js_mean, _ = mean_std([row['pseudo_pred_js'] for row in rows])
            agree_mean, _ = mean_std(
                [row['pseudo_agreement_rate'] for row in rows])
            time_mean, _ = mean_std([row['epoch_time_sec'] for row in rows])
            mem_mean, _ = mean_std([row['peak_allocated_mb'] for row in rows])
            writer.writerow({
                'experiment': exp_name,
                'runs': len(rows),
                'acc_mean': acc_mean,
                'acc_std': acc_std,
                'macro_f1_mean': f1_mean,
                'macro_f1_std': f1_std,
                'pseudo_pred_js_mean': js_mean,
                'pseudo_agreement_rate_mean': agree_mean,
                'epoch_time_sec_mean': time_mean,
                'peak_allocated_mb_mean': mem_mean,
            })
    print('wrote', out_path)


if __name__ == '__main__':
    main()
