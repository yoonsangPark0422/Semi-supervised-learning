import argparse
import csv
from pathlib import Path


METHODS = ["crest", "daso", "abc", "proposed"]


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def get_metric(row, *names):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def best_row(rows):
    if not rows:
        return None

    def score(row):
        value = get_metric(row, "test_acc_final", "test_acc")
        try:
            return float(value)
        except ValueError:
            return -1.0

    return max(rows, key=score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="result")
    parser.add_argument("--seed", default="5")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent / root
    seed_dir = root / f"seed_{args.seed}"

    rows = []
    for method in METHODS:
        method_dir = seed_dir / method
        metrics = read_csv(method_dir / "train_metrics.csv")
        latest = metrics[-1] if metrics else {}
        best = best_row(metrics) or {}
        rows.append({
            "method": method,
            "completed_epochs": latest.get("epoch", "0"),
            "latest_acc": get_metric(latest, "test_acc_final", "test_acc"),
            "latest_macro_f1": get_metric(latest, "macro_f1_final", "macro_f1"),
            "best_epoch": best.get("epoch", ""),
            "best_acc": get_metric(best, "test_acc_final", "test_acc"),
            "best_macro_f1": get_metric(best, "macro_f1_final", "macro_f1"),
            "peak_allocated_mb": latest.get("peak_allocated_mb", ""),
            "param_count_m": get_metric(
                latest, "param_count_m", "param_count_major_m"),
            "train_metrics": str(method_dir / "train_metrics.csv"),
            "per_class_eval": str(method_dir / "per_class_eval.csv"),
            "pseudo_stats": str(method_dir / "pseudo_stats.csv"),
        })

    out_csv = root / "comparison_report.csv"
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_md = root / "comparison_report.md"
    lines = [
        "# CIFAR-10 Baseline Comparison",
        "",
        "| Method | Epochs | Latest Acc | Latest Macro F1 | Best Epoch | Best Acc | Best Macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['completed_epochs']} | "
            f"{row['latest_acc']} | {row['latest_macro_f1']} | "
            f"{row['best_epoch']} | {row['best_acc']} | {row['best_macro_f1']} |"
        )
    lines.extend([
        "",
        "Generated from each method's train_metrics.csv under result/seed_5.",
        "Per-class accuracy/F1 and pseudo-label statistics are saved in each method folder.",
    ])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
