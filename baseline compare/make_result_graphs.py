import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "result"
SEED_DIR = RESULT / "seed_5"
METHODS = ["crest", "daso", "abc", "proposed"]
LABELS = {
    "crest": "CReST",
    "daso": "DASO",
    "abc": "ABC",
    "proposed": "Proposed",
}
COLORS = {
    "crest": "#2f6bff",
    "daso": "#f28e2b",
    "abc": "#59a14f",
    "proposed": "#d62728",
}


def to_float(value):
    if value is None or value == "":
        return None
    return float(value)


def read_metrics(method):
    path = SEED_DIR / method / "train_metrics.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            acc = to_float(row.get("test_acc_final")) or to_float(row.get("test_acc_ensemble")) or to_float(row.get("test_acc"))
            f1 = to_float(row.get("macro_f1_final")) or to_float(row.get("macro_f1_ensemble")) or to_float(row.get("macro_f1"))
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "acc": acc,
                    "macro_f1": f1,
                    "train_loss": to_float(row.get("train_loss")),
                }
            )
    return rows


def scale_points(rows, key, x0, y0, width, height, ymin, ymax):
    points = []
    for row in rows:
        value = row[key]
        if value is None:
            continue
        x = x0 + (row["epoch"] - 1) / 99 * width
        y = y0 + height - (value - ymin) / (ymax - ymin) * height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def line_chart(all_metrics, key, title, y_label, filename, ymin=None, ymax=None):
    width, height = 980, 620
    x0, y0, plot_w, plot_h = 90, 70, 780, 430
    values = [row[key] for rows in all_metrics.values() for row in rows if row[key] is not None]
    if ymin is None:
        ymin = max(0.0, min(values) - 5.0)
    if ymax is None:
        ymax = min(100.0, max(values) + 3.0)
    if ymax <= ymin:
        ymax = ymin + 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial, sans-serif; fill:#222}.title{font-size:24px;font-weight:700}.axis{font-size:13px}.legend{font-size:14px}.grid{stroke:#e5e7eb;stroke-width:1}.axisline{stroke:#333;stroke-width:1.5}.line{fill:none;stroke-width:3}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{x0}" y="38">{html.escape(title)}</text>',
    ]

    for i in range(6):
        y = y0 + plot_h - i / 5 * plot_h
        value = ymin + i / 5 * (ymax - ymin)
        parts.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis" x="{x0 - 12}" y="{y + 4:.1f}" text-anchor="end">{value:.1f}</text>')

    for epoch in [1, 20, 40, 60, 80, 100]:
        x = x0 + (epoch - 1) / 99 * plot_w
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + plot_h}"/>')
        parts.append(f'<text class="axis" x="{x:.1f}" y="{y0 + plot_h + 24}" text-anchor="middle">{epoch}</text>')

    parts.append(f'<line class="axisline" x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}"/>')
    parts.append(f'<line class="axisline" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}"/>')
    parts.append(f'<text class="axis" x="{x0 + plot_w / 2}" y="{y0 + plot_h + 58}" text-anchor="middle">Epoch</text>')
    parts.append(f'<text class="axis" transform="translate(24 {y0 + plot_h / 2}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>')

    legend_x = x0 + plot_w + 35
    legend_y = y0 + 20
    for idx, method in enumerate(METHODS):
        rows = all_metrics[method]
        points = scale_points(rows, key, x0, y0, plot_w, plot_h, ymin, ymax)
        color = COLORS[method]
        parts.append(f'<polyline class="line" points="{points}" stroke="{color}"/>')
        ly = legend_y + idx * 32
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 28}" y2="{ly}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text class="legend" x="{legend_x + 38}" y="{ly + 5}">{LABELS[method]}</text>')

    parts.append("</svg>")
    (RESULT / filename).write_text("\n".join(parts), encoding="utf-8")


def bar_chart(all_metrics):
    width, height = 980, 620
    x0, y0, plot_w, plot_h = 95, 70, 760, 420
    latest = {m: all_metrics[m][-1]["acc"] for m in METHODS}
    best = {m: max(row["acc"] for row in all_metrics[m] if row["acc"] is not None) for m in METHODS}
    ymax = 100.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial, sans-serif; fill:#222}.title{font-size:24px;font-weight:700}.axis{font-size:13px}.value{font-size:13px;font-weight:700}.legend{font-size:14px}.grid{stroke:#e5e7eb;stroke-width:1}.axisline{stroke:#333;stroke-width:1.5}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{x0}" y="38">Final Accuracy Comparison</text>',
    ]
    for i in range(6):
        y = y0 + plot_h - i / 5 * plot_h
        value = i / 5 * ymax
        parts.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axis" x="{x0 - 12}" y="{y + 4:.1f}" text-anchor="end">{value:.0f}</text>')
    parts.append(f'<line class="axisline" x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}"/>')
    parts.append(f'<line class="axisline" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}"/>')
    parts.append(f'<text class="axis" transform="translate(24 {y0 + plot_h / 2}) rotate(-90)" text-anchor="middle">Accuracy (%)</text>')

    group_w = plot_w / len(METHODS)
    bar_w = 34
    for idx, method in enumerate(METHODS):
        cx = x0 + group_w * idx + group_w / 2
        for offset, kind, fill in [(-bar_w / 1.7, "Latest", "#9ca3af"), (bar_w / 1.7, "Best", COLORS[method])]:
            value = latest[method] if kind == "Latest" else best[method]
            h = value / ymax * plot_h
            x = cx + offset - bar_w / 2
            y = y0 + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{fill}"/>')
            parts.append(f'<text class="value" x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle">{value:.2f}</text>')
        parts.append(f'<text class="axis" x="{cx:.1f}" y="{y0 + plot_h + 28}" text-anchor="middle">{LABELS[method]}</text>')

    parts.append(f'<rect x="{x0 + plot_w + 35}" y="{y0 + 10}" width="16" height="16" fill="#9ca3af"/>')
    parts.append(f'<text class="legend" x="{x0 + plot_w + 60}" y="{y0 + 23}">Latest</text>')
    parts.append(f'<rect x="{x0 + plot_w + 35}" y="{y0 + 42}" width="16" height="16" fill="#d62728"/>')
    parts.append(f'<text class="legend" x="{x0 + plot_w + 60}" y="{y0 + 55}">Best</text>')
    parts.append("</svg>")
    (RESULT / "final_accuracy_comparison.svg").write_text("\n".join(parts), encoding="utf-8")


def write_dashboard():
    cards = [
        ("accuracy_curves.svg", "Accuracy Curves"),
        ("macro_f1_curves.svg", "Macro F1 Curves"),
        ("final_accuracy_comparison.svg", "Final Accuracy Comparison"),
    ]
    body = [
        "<!doctype html>",
        '<html lang="ko"><head><meta charset="utf-8"><title>CIFAR-10 Baseline Graphs</title>',
        "<style>body{font-family:Arial,sans-serif;margin:32px;background:#f6f7f9;color:#222}h1{margin:0 0 20px}.chart{background:#fff;border:1px solid #ddd;margin:0 0 28px;padding:16px}img{max-width:100%;height:auto}</style>",
        "</head><body><h1>CIFAR-10 Baseline Comparison Graphs</h1>",
    ]
    for filename, title in cards:
        body.append(f'<section class="chart"><h2>{title}</h2><img src="{filename}" alt="{title}"></section>')
    body.append("</body></html>")
    (RESULT / "comparison_graphs.html").write_text("\n".join(body), encoding="utf-8")


def main():
    all_metrics = {method: read_metrics(method) for method in METHODS}
    line_chart(all_metrics, "acc", "Accuracy by Epoch", "Accuracy (%)", "accuracy_curves.svg", ymin=0, ymax=100)
    line_chart(all_metrics, "macro_f1", "Macro F1 by Epoch", "Macro F1 (%)", "macro_f1_curves.svg", ymin=0, ymax=100)
    bar_chart(all_metrics)
    write_dashboard()
    print("wrote:")
    for name in [
        "accuracy_curves.svg",
        "macro_f1_curves.svg",
        "final_accuracy_comparison.svg",
        "comparison_graphs.html",
    ]:
        print(RESULT / name)


if __name__ == "__main__":
    main()
