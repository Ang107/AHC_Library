# 焼きなまし法 (Simulated Annealing)

AtCoder Heuristic Contest (AHC) 用の焼きなましテンプレートと可視化ツールです。

## ファイル構成

| ファイル | 説明 |
|---|---|
| `sa.cpp` | 焼きなまし法のC++テンプレート |
| `visualize.py` | SA統計情報の可視化ツール |
| `tips.md` | 焼きなまし関連の最適化テクニック集 |
| `sample/` | 使用例（AHC058） |

## 特徴

- **高速な乱数生成器**: xoshiro256**ベース
- **テーブルルックアップ**: 温度減衰・log計算の高速化（線形補間付き）
- **複数の近傍操作**: `AliasWeightedSampler` による O(1) 重み付き選択（遅延テーブル再構築）
- **閾値ベースの採用判定**: `exp(-delta/T)` ではなく `T * log(rand)` との比較で高速化
- **統計情報の出力**: 可視化ツール連携用のJSON形式ログ
- **最大化/最小化の切り替え**: `MAXIMIZE` フラグで制御
- **山登り法への切り替え**: `ALLOW_WORSE_MOVES = false` で山登り法に

## 使い方

`TODO` となっている箇所を問題に合わせて実装する。

### 1. パラメータ設定（調整箇所）

```cpp
using score_t = int64_t;                           // スコアの型
constexpr bool MAXIMIZE = true;                    // true: 最大化, false: 最小化
constexpr double TIME_LIMIT = 1.95;                // 時間制限
constexpr double START_TEMP = 1000.0;              // 開始温度
constexpr double END_TEMP = 10.0;                  // 終了温度
constexpr bool USE_EXPONENTIAL_DECAY = true;       // true: 指数減衰, false: べき乗減衰
constexpr double POWER_DECAY_EXP = 1.0;           // べき乗減衰の指数 (1.0で線形, >1で序盤高温維持, <1で序盤急冷)
constexpr bool ALLOW_WORSE_MOVES = true;           // false: 山登り法
constexpr int TIME_CHECK_INTERVAL = 0x7F;          // 時間チェック頻度（ビットマスク）
constexpr int STATS_INTERVAL = 10 * (TIME_CHECK_INTERVAL + 1); // 統計出力頻度

// 近傍操作の定義
enum MoveType { SWAP, INSERT, REVERSE, NUM_MOVES };
vector<string> MOVE_NAMES = {"SWAP", "INSERT", "REVERSE"};
vector<double> MOVE_WEIGHTS = {1, 1, 1};
```

### 2. `Input` を定義

`input()` メソッドで入力を読み込む。

### 3. `State` を定義

| メソッド | 説明 |
|---|---|
| コンストラクタ | 初期解を生成 |
| `calc_score_full(in)` | フルスコアを計算 |

### 4. 近傍操作を実装 (`SA::run()` メソッド内)

- 各操作で差分スコアを計算
- ロールバック用変数を保存
- 採用時の更新処理とロールバック処理を実装

近傍操作は問題に応じて変更する。テンプレートのSWAP/INSERT/REVERSEは例。

```cpp
// 例: 巡回セールスマン問題の場合
enum MoveType { TWO_OPT, OR_OPT, NUM_MOVES };
vector<string> MOVE_NAMES = {"TWO_OPT", "OR_OPT"};
vector<double> MOVE_WEIGHTS = {3, 1};  // 2-opt を重視
```

### 5. `Solver` を実装

- `solve()` メソッドでSAを実行
- `print()` メソッドで解を出力

### デバッグモード

```cpp
constexpr bool DEBUG = true;  // falseにすると統計情報の出力を無効化
```

`DEBUG = true` の場合、標準エラー出力に `VISUALIZE:` プレフィックス付きの統計情報（JSON形式）が出力される。

## visualize.py - 可視化ツール

SA実行時の統計情報をグラフ化します。`VISUALIZE:` プレフィックス付きのJSON行をstdinから読み込みます。

### 生成されるグラフ

#### 1. 全体サマリ (`00_overview.png`, `00_overview_last90.png`)

7つのサブプロットで構成:

1. **Temperature**: 温度の推移
2. **Score**: 現在スコアとベストスコアの推移
3. **Tried Count (Stacked)**: 近傍ごとの試行回数（積み上げ面グラフ）
4. **Evaluation Rate**: 近傍ごとの評価率（evaluated / tried）
5. **Accept Rate**: 近傍ごとの受理率（accepted / tried）
6. **Improve Rate**: 近傍ごとの改善率（improved / tried）
7. **Best Update Rate**: 近傍ごとのベスト更新率（updated_best / tried）

#### 2. 近傍操作の詳細 (`01_move_details_all.png`, `01_move_details_all_last90.png`)

各近傍操作について、以下の3つの率を同一グラフにプロット（全近傍でy軸スケール統一）:
- Accept Rate（青）
- Improve Rate（緑）
- Best Update Rate（赤）

### 使い方

```bash
# 実行して可視化
./sa < input.txt 2>&1 | python visualize.py

# または、ログを保存してから可視化
./sa < input.txt 2> log.txt
cat log.txt | python visualize.py
```

グラフは `visualize/sa/YYYYMMDD_HHMMSS/` ディレクトリに保存されます。

### パラメータ

```python
SMOOTHING_WINDOW = 20  # 移動平均のウィンドウサイズ
```
