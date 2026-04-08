"""Summarize and rank diffusion repair experiment metrics.

Usage:
    python experiments/diffusion_repair/summarize_metrics.py

Reads:
    experiments/diffusion_repair/results/metrics_*.json

Writes:
    experiments/diffusion_repair/RESULTS_SUMMARY.md
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
OUT_FILE = ROOT / "RESULTS_SUMMARY.md"


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(RESULTS_DIR.glob("metrics_*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        metrics = payload.get("metrics", {})
        config = payload.get("config", {})
        rows.append(
            {
                "tag": metrics.get("tag", path.stem.replace("metrics_", "")),
                "before": float(metrics.get("before_viol_rate", 0.0)),
                "after": float(metrics.get("after_viol_rate", 0.0)),
                "delta": float(metrics.get("viol_rate_delta", 0.0)),
                "efficacy": float(metrics.get("efficacy", 0.0)),
                "side_effect": float(metrics.get("side_effect_mse", 0.0)),
                "seed": config.get("seed", 0),
                "change_summary": metrics.get("change_summary", "n/a"),
            }
        )
    return rows


def build_summary(rows: list[dict]) -> str:
    if not rows:
        return "# Diffusion Repair Results Summary\n\nNo metrics files found.\n"

    ranked = sorted(rows, key=lambda r: (r["after"], r["side_effect"]))
    best = ranked[0]

    lines: list[str] = []
    lines.append("# Diffusion Repair Results Summary\n")
    lines.append(
        "Auto-generated from `results/metrics_*.json` using rank key: "
        "(after_viol_rate asc, side_effect_mse asc).\n"
    )

    lines.append("## Best Run\n")
    lines.append(f"- tag: {best['tag']}\n")
    lines.append(f"- seed: {best['seed']}\n")
    lines.append(f"- before_viol_rate: {best['before']:.4f}\n")
    lines.append(f"- after_viol_rate: {best['after']:.4f}\n")
    lines.append(f"- efficacy: {best['efficacy']:.4f}\n")
    lines.append(f"- side_effect_mse: {best['side_effect']:.4f}\n")
    lines.append(f"- change_summary: {best['change_summary']}\n")

    lines.append("\n## Leaderboard\n")
    lines.append(
        "| rank | tag | seed | before | after | delta | efficacy | side_effect_mse |\n"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|\n")

    for idx, r in enumerate(ranked, start=1):
        lines.append(
            f"| {idx} | {r['tag']} | {r['seed']} | "
            f"{r['before']:.4f} | {r['after']:.4f} | {r['delta']:.4f} | "
            f"{r['efficacy']:.4f} | {r['side_effect']:.4f} |\n"
        )

    return "".join(lines)


def main() -> None:
    rows = load_rows()
    content = build_summary(rows)
    OUT_FILE.write_text(content, encoding="utf-8")
    print(f"Wrote summary: {OUT_FILE}")


if __name__ == "__main__":
    main()
