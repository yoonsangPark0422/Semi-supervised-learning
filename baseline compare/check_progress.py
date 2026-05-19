import argparse
import csv
from pathlib import Path


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="result")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", default="5")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent / root
    seed_dir = root / f"seed_{args.seed}"

    methods = ["crest", "daso", "abc", "proposed"]
    out_rows = []
    for method in methods:
        metrics_path = seed_dir / method / "train_metrics.csv"
        rows = read_rows(metrics_path)
        if rows:
            latest = rows[-1]
            completed = int(float(latest["epoch"]))
            avg_time = sum(as_float(row.get("epoch_time_sec")) for row in rows) / len(rows)
            remaining = max(0, args.epochs - completed) * avg_time
            acc = latest.get("test_acc_final") or latest.get("test_acc") or ""
            macro_f1 = latest.get("macro_f1_final") or latest.get("macro_f1") or ""
        else:
            completed = 0
            avg_time = 0.0
            remaining = 0.0
            acc = ""
            macro_f1 = ""
        out_rows.append({
            "method": method,
            "completed_epochs": completed,
            "target_epochs": args.epochs,
            "latest_acc": acc,
            "latest_macro_f1": macro_f1,
            "avg_epoch_time_sec": f"{avg_time:.2f}",
            "estimated_remaining_hours": f"{remaining / 3600.0:.2f}",
            "metrics_path": str(metrics_path),
        })

    out_path = root / "progress_summary.csv"
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {out_path}")
    for row in out_rows:
        print(
            f"{row['method']}: {row['completed_epochs']}/{row['target_epochs']} "
            f"acc={row['latest_acc']} macro_f1={row['latest_macro_f1']} "
            f"eta_h={row['estimated_remaining_hours']}"
        )


if __name__ == "__main__":
    main()
