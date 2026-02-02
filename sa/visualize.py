import sys
import json
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 出力先ディレクトリの設定
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"visualize/sa/{TIMESTAMP}"

# 移動平均のウィンドウサイズ（大きいほど滑らかになる）
SMOOTHING_WINDOW = 20


def load_data():
    """標準入力からVISUALIZE行のみを読み込む"""
    global_stats = []
    move_stats_list = []

    # パイプまたはファイルリダイレクトからの入力を想定
    for line in sys.stdin:
        if "VISUALIZE:" in line:
            try:
                json_str = line.split("VISUALIZE:", 1)[1].strip()
                data = json.loads(json_str)

                # グローバルな統計
                g_stat = {k: v for k, v in data.items() if k != "moves"}
                global_stats.append(g_stat)

                # 近傍（Move）ごとの統計
                iter_count = data["iter"]
                time_val = data["time"]
                for m in data["moves"]:
                    m["iter"] = iter_count
                    m["time"] = time_val
                    move_stats_list.append(m)
            except json.JSONDecodeError:
                continue

    if not global_stats:
        print("No valid VISUALIZE data found.")
        sys.exit(1)

    return pd.DataFrame(global_stats), pd.DataFrame(move_stats_list)


def calculate_intervals(df_moves):
    """累積カウントから区間ごとの確率（受理率など）を計算する"""
    # name ごとにグループ化して差分をとる
    df_moves = df_moves.sort_values(by=["name", "iter"])

    # 差分計算（diff）
    cols_to_diff = ["tried", "evaluated", "accepted", "improved", "updated_best"]

    # グループごとにdiffをとる
    grouped = df_moves.groupby("name")[cols_to_diff]
    diffs = grouped.diff().fillna(0)

    # 元のデータフレームに結合
    df_intervals = df_moves.copy()
    for col in cols_to_diff:
        df_intervals[f"d_{col}"] = diffs[col]

    # 確率計算（0除算回避）
    # 区間試行回数が0の場合は確率0とする
    with np.errstate(divide="ignore", invalid="ignore"):
        df_intervals["rate_evaluated"] = np.where(
            df_intervals["d_tried"] > 0,
            df_intervals["d_evaluated"] / df_intervals["d_tried"],
            0.0,
        )
        df_intervals["rate_accept"] = np.where(
            df_intervals["d_tried"] > 0,
            df_intervals["d_accepted"] / df_intervals["d_tried"],
            0.0,
        )
        df_intervals["rate_improve"] = np.where(
            df_intervals["d_tried"] > 0,
            df_intervals["d_improved"] / df_intervals["d_tried"],
            0.0,
        )
        df_intervals["rate_update"] = np.where(
            df_intervals["d_tried"] > 0,
            df_intervals["d_updated_best"] / df_intervals["d_tried"],
            0.0,
        )

    return df_intervals


def plot_overview(df_global, df_moves_interval, suffix=""):
    """要件1: 全体サマリの描画"""
    fig, axes = plt.subplots(7, 1, figsize=(12, 28), sharex=True)

    move_names = df_moves_interval["name"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(move_names)))
    color_map = {name: colors[i] for i, name in enumerate(move_names)}

    # 1. Temperature
    ax = axes[0]
    ax.plot(df_global["time"], df_global["temp"], color="tab:orange")
    ax.set_ylabel("Temperature")
    ax.set_title("Overview Stats")
    ax.grid(True)

    # 2. Score & Best Score
    ax = axes[1]
    ax.plot(df_global["time"], df_global["score"], label="Current", alpha=0.6, lw=1)
    ax.plot(df_global["time"], df_global["best_score"], label="Best", color="red", lw=2)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True)

    # 3. Tried Count (Stacked Area Chart)
    ax = axes[2]
    # 各近傍のd_triedを時間ごとにピボット
    pivot_tried = df_moves_interval.pivot_table(
        index="time", columns="name", values="d_tried", fill_value=0
    )
    # 積み上げ面グラフ
    ax.stackplot(
        pivot_tried.index,
        [pivot_tried[name].values for name in pivot_tried.columns],
        labels=pivot_tried.columns,
        colors=[color_map.get(name, "gray") for name in pivot_tried.columns],
        alpha=0.8,
    )
    ax.set_ylabel("Tried Count (Stacked)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True)

    # 4. Evaluation Rate per Move (Interval)
    ax = axes[3]
    for name in move_names:
        subset = df_moves_interval[df_moves_interval["name"] == name]
        smoothed = (
            subset["rate_evaluated"]
            .rolling(window=SMOOTHING_WINDOW, min_periods=1)
            .mean()
        )
        ax.plot(subset["time"], smoothed, label=name, color=color_map[name])
    ax.set_ylabel("Evaluation Rate")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True)

    # 5. Accept Rate per Move (Interval)
    ax = axes[4]
    for name in move_names:
        subset = df_moves_interval[df_moves_interval["name"] == name]
        smoothed = (
            subset["rate_accept"].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
        )
        ax.plot(subset["time"], smoothed, label=name, color=color_map[name])
    ax.set_ylabel("Accept Rate")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True)

    # 6. Improve Rate per Move (Interval)
    ax = axes[5]
    for name in move_names:
        subset = df_moves_interval[df_moves_interval["name"] == name]
        smoothed = (
            subset["rate_improve"]
            .rolling(window=SMOOTHING_WINDOW, min_periods=1)
            .mean()
        )
        ax.plot(subset["time"], smoothed, label=name, color=color_map[name])
    ax.set_ylabel("Improve Rate")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True)

    # 7. Best Update Rate per Move (Interval)
    ax = axes[6]
    for name in move_names:
        subset = df_moves_interval[df_moves_interval["name"] == name]
        smoothed = (
            subset["rate_update"].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
        )
        ax.plot(subset["time"], smoothed, label=name, color=color_map[name])
    ax.set_ylabel("Best Update Rate")
    ax.set_xlabel("Time (s)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"00_overview{suffix}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_details(df_moves_interval, suffix=""):
    """要件2: 近傍ごとの詳細グラフ（1つの画像に縦に並べる）"""
    move_names = sorted(df_moves_interval["name"].unique())
    n_moves = len(move_names)

    if n_moves == 0:
        return

    # 全近傍のデータから y軸の最大値を計算
    all_rates = []
    for name in move_names:
        subset = df_moves_interval[df_moves_interval["name"] == name]
        r_accept = subset["rate_accept"].rolling(SMOOTHING_WINDOW, min_periods=1).mean()
        r_improve = (
            subset["rate_improve"].rolling(SMOOTHING_WINDOW, min_periods=1).mean()
        )
        r_update = subset["rate_update"].rolling(SMOOTHING_WINDOW, min_periods=1).mean()
        all_rates.extend(r_accept.values)
        all_rates.extend(r_improve.values)
        all_rates.extend(r_update.values)

    # y軸の範囲を決定（少し余裕を持たせる）
    y_max = max(all_rates) * 1.05 if all_rates else 1.0
    y_min = 0

    # 縦に並べるサブプロット
    fig, axes = plt.subplots(n_moves, 1, figsize=(10, 6 * n_moves), sharex=True)

    # move_nameが1つの場合はaxesをリストに変換
    if n_moves == 1:
        axes = [axes]

    for i, name in enumerate(move_names):
        subset = df_moves_interval[df_moves_interval["name"] == name]
        ax = axes[i]

        # 移動平均でスムージング
        r_accept = subset["rate_accept"].rolling(SMOOTHING_WINDOW, min_periods=1).mean()
        r_improve = (
            subset["rate_improve"].rolling(SMOOTHING_WINDOW, min_periods=1).mean()
        )
        r_update = subset["rate_update"].rolling(SMOOTHING_WINDOW, min_periods=1).mean()

        # 全て同じ軸にプロット
        ax.plot(subset["time"], r_accept, label="Accept Rate", color="tab:blue")
        ax.plot(subset["time"], r_improve, label="Improve Rate", color="tab:green")
        ax.plot(subset["time"], r_update, label="Best Update Rate", color="tab:red")
        ax.set_ylabel("Rate")
        ax.set_ylim(y_min, y_max)
        ax.grid(True)
        ax.set_title(f"{name} Details")
        ax.legend(loc="upper right")

    # 最後のサブプロットにのみx軸ラベルを設定
    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"01_move_details_all{suffix}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def filter_last_n_percent(df_global, df_moves, percent=90):
    """データの後半n%だけを抽出する"""
    if len(df_global) == 0:
        return df_global, df_moves

    # 時間の閾値を計算（後半n%の開始時刻）
    time_values = df_global["time"].values
    min_time = time_values.min()
    max_time = time_values.max()
    time_range = max_time - min_time
    threshold = min_time + time_range * (1 - percent / 100)

    # フィルタリング
    df_global_filtered = df_global[df_global["time"] >= threshold].copy()
    df_moves_filtered = df_moves[df_moves["time"] >= threshold].copy()

    return df_global_filtered, df_moves_filtered


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Reading from stdin... Output to: {OUTPUT_DIR}")

    df_global, df_moves = load_data()

    # 累積値から区間ごとの値を計算
    df_moves_interval = calculate_intervals(df_moves)

    # 全体データでプロット作成
    print("Plotting all data...")
    plot_overview(df_global, df_moves_interval, suffix="")
    plot_details(df_moves_interval, suffix="")

    # 後半9割のデータでプロット作成
    print("Plotting last 90% data...")
    df_global_90, df_moves_90 = filter_last_n_percent(df_global, df_moves, percent=90)
    df_moves_interval_90 = calculate_intervals(df_moves_90)
    plot_overview(df_global_90, df_moves_interval_90, suffix="_last90")
    plot_details(df_moves_interval_90, suffix="_last90")


if __name__ == "__main__":
    main()
