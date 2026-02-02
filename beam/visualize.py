#!/usr/bin/env python3
"""
ビームサーチのログを可視化する。

使い方:
  ./a.out < input.txt 2>&1 >/dev/null | python visualize_beam_log.py
"""

import sys
import json
import os
import datetime
import matplotlib.pyplot as plt

# 出力先ディレクトリの設定
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"visualize/beam/{TIMESTAMP}"


def load_data():
    """標準入力からBEAM_LOG行のみを読み込む"""
    logs = []

    for line in sys.stdin:
        if "BEAM_LOG:" in line:
            try:
                json_str = line.split("BEAM_LOG:", 1)[1].strip()
                data = json.loads(json_str)
                logs.append(data)
            except json.JSONDecodeError:
                continue

    if not logs:
        print("No valid BEAM_LOG data found.")
        sys.exit(1)

    return logs


def plot_overview(logs):
    """全体サマリの描画"""
    turns = [l["turn"] for l in logs]
    has_hash_dedup = "hash_dedup" in logs[0]

    n_plots = 6 if has_hash_dedup else 5
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)

    ax_idx = 0

    # 1. コスト推移 (best / mean / worst)
    ax = axes[ax_idx]; ax_idx += 1
    ax.plot(turns, [l["best"] for l in logs], label="best", linewidth=2)
    ax.plot(turns, [l["mean"] for l in logs], label="mean", linestyle="--")
    ax.plot(turns, [l["worst"] for l in logs], label="worst", alpha=0.5)
    ax.fill_between(
        turns,
        [l["mean"] - l["std"] for l in logs],
        [l["mean"] + l["std"] for l in logs],
        alpha=0.15,
        label="mean +/- std",
    )
    ax.set_ylabel("Cost")
    ax.set_title("Cost (lower is better)")
    ax.legend(loc="upper right")
    ax.grid(True)

    # 2. コストのmeanからの差分 (best - mean, worst - mean, std)
    ax = axes[ax_idx]; ax_idx += 1
    diff_best = [l["best"] - l["mean"] for l in logs]
    diff_worst = [l["worst"] - l["mean"] for l in logs]
    std_vals = [l["std"] for l in logs]
    ax.plot(turns, diff_best, label="best - mean", linewidth=2, color="tab:blue")
    ax.plot(turns, diff_worst, label="worst - mean", linewidth=2, color="tab:red")
    ax.fill_between(
        turns,
        [-s for s in std_vals],
        std_vals,
        alpha=0.2,
        color="tab:orange",
        label="+/- std",
    )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("Cost diff from mean")
    ax.set_title("Cost Spread (difference from mean)")
    ax.legend(loc="upper right")
    ax.grid(True)

    # 3. ビーム幅 & ユニーク親数
    ax = axes[ax_idx]; ax_idx += 1
    ax.plot(
        turns,
        [l["beam_width_limit"] for l in logs],
        label="beam_width_limit",
        linewidth=2,
        color="tab:blue",
    )
    ax.plot(
        turns,
        [l["beam_width"] for l in logs],
        label="beam_width (actual)",
        linewidth=2,
        color="tab:orange",
    )
    ax.plot(
        turns,
        [l["unique_parents"] for l in logs],
        label="unique_parents",
        linestyle="--",
        color="tab:green",
    )
    ax.set_ylabel("Count")
    ax.set_title("Beam Width & Diversity")
    ax.legend(loc="upper right")
    ax.grid(True)

    # 3. 候補数 & 枝刈り数 (& ハッシュ重複数)
    ax = axes[ax_idx]; ax_idx += 1
    ax.plot(turns, [l["candidates"] for l in logs], label="candidates", linewidth=2)
    ax.plot(turns, [l["pruned"] for l in logs], label="pruned", linestyle="--")
    if has_hash_dedup:
        ax.plot(
            turns,
            [l["hash_dedup"] for l in logs],
            label="hash_dedup",
            linestyle=":",
        )
    ax.set_ylabel("Count")
    ax.set_title("Candidates & Pruning")
    ax.legend(loc="upper right")
    ax.grid(True)

    # 4. ハッシュ重複率 (hash版のみ)
    if has_hash_dedup:
        ax = axes[ax_idx]; ax_idx += 1
        dedup_rate = []
        for l in logs:
            total = l["candidates"]
            dedup_rate.append(l["hash_dedup"] / total * 100 if total > 0 else 0)
        ax.plot(turns, dedup_rate, color="tab:orange", linewidth=2)
        ax.set_ylabel("Rate (%)")
        ax.set_title("Hash Dedup Rate")
        ax.grid(True)

    # 5. 経過時間
    ax = axes[ax_idx]; ax_idx += 1
    elapsed = [l["elapsed"] for l in logs]
    dt = [elapsed[0]] + [elapsed[i] - elapsed[i - 1] for i in range(1, len(elapsed))]
    ax.bar(turns, dt, width=0.8, alpha=0.7)
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Turn")
    ax.set_title("Elapsed Time per Turn")
    ax.grid(True)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "overview.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Reading from stdin... Output to: {OUTPUT_DIR}")

    logs = load_data()

    print("Plotting...")
    plot_overview(logs)


if __name__ == "__main__":
    main()
