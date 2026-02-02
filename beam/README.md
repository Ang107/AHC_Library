# ビームサーチ (Beam Search)

AtCoder Heuristic Contest (AHC) 用のビームサーチライブラリです。Euler Tour ベースの木管理により、状態の forward/backward を効率的に行います。

参考: https://eijirou-kyopro.hatenablog.com/entry/2024/02/01/115639, https://atcoder.jp/contests/ahc032/submissions/52156462

## ファイル構成

| ファイル | 説明 |
|---|---|
| `beam.cpp` | ハッシュ重複除去あり版 |
| `beam_no_hash.cpp` | ハッシュ重複除去なし版 |
| `visualize.py` | ビームサーチ統計情報の可視化ツール |
| `tips.md` | ビームサーチ関連の最適化テクニック集 |
| `sample/` | 使用例（RCO Contest 2018 Qual A） |

## 特徴

- **Euler Tour ベースの木管理**: 状態を `move_forward` / `move_backward` で差分更新し、全状態のコピーを避ける
- **Segment Tree による候補選択**: ビーム幅分の上位候補を効率的に管理
- **ハッシュ重複除去** (`beam.cpp`): 同一ハッシュの状態は評価が良い方のみ残す
- **ベイズ推定による動的ビーム幅調整**: 正規-逆ガンマ分布で1ターンあたりの所要時間を推定し、+3σの余裕を持ってビーム幅を自動決定
- **統計情報の出力**: 可視化ツール連携用の `BEAM_LOG:` プレフィックス付きJSONログ

## 使い方

`TODO` となっている箇所を問題に合わせて実装する。

### 1. `Action` を定義

状態遷移に必要な最小限の情報を持つ構造体。メモリは小さく保つ。

### 2. `Evaluator` を定義

状態の評価値を計算する構造体。`evaluate()` はコスト（低いほど良い）を返す。

### 3. `State` を実装

| メソッド | 説明 |
|---|---|
| `make_initial_node()` | 初期状態の Evaluator (と Hash) を返す |
| `expand(evaluator, [hash,] parent, selector)` | 現在の状態から全ての次状態候補を `selector.push()` で追加 |
| `move_forward(action)` | action を適用して次の状態に遷移 |
| `move_backward(action)` | action を取り消して前の状態に戻る |

### 4. `Config` を設定

```cpp
beam_search::Config config;
config.max_turn = 100;           // 最大ターン数
config.expected_turn = 100;      // 終了目安ターン数
config.dynamic_beam = true;      // 動的ビーム幅調整
config.time_limit = 1.9;         // 制限時間 (秒)
config.max_beam_width = 1000;    // 最大ビーム幅
config.initial_beam_width = 500; // 初期ビーム幅 (動的調整の基準値)
config.min_beam_width = 10;      // 最小ビーム幅
config.warmup_turn = 3;          // 最初の X ターンの観測は捨てる
config.tour_capacity = 10000000; // Euler Tour の容量 (雑に大きく)
// beam.cpp のみ:
config.hash_map_capacity = 160000; // ビーム幅 * 派生先数 の 16 倍程度
```

### 5. 実行

```cpp
beam_search::State state;
auto actions = beam_search::beam_search(config, state, total_timer);
```

戻り値は根から最良の葉までの `Action` 列。

### デバッグモード

```cpp
constexpr bool DEBUG = true;  // falseにすると統計情報の出力を無効化
```

`DEBUG = true` の場合、標準エラー出力に `BEAM_LOG:` プレフィックス付きの統計情報（JSON形式）が出力される。

## 2つのバージョンの選び方

- **`beam.cpp`**: 状態のハッシュが定義でき、重複状態を除去したい場合に使う。`Candidate` に `Hash` フィールドがあり、`HashMap` で同一ハッシュをまとめる
- **`beam_no_hash.cpp`**: ハッシュ重複除去が不要、またはハッシュの定義が困難な場合に使う。Config に `hash_map_capacity` が不要でシンプル

## 問題タイプ別の使い分け

- **ターン数固定型**: `max_turn` まで探索し、最終ターンで最良の候補を返す
- **ターン数最小化型**: `selector.push(candidate, finished=true)` で実行可能解が見つかった時点で即座に返す

## visualize.py - 可視化ツール

ビームサーチ実行時の統計情報をグラフ化します。`BEAM_LOG:` プレフィックス付きのJSON行をstdinから読み込みます。

### 生成されるグラフ (`overview.png`)

5〜6つのサブプロットで構成（ハッシュ版は6つ）:

1. **Cost**: best / mean / worst コストの推移（mean ± std の帯付き）
2. **Cost Spread**: mean からの差分（best - mean, worst - mean, ± std）
3. **Beam Width & Diversity**: beam_width_limit, 実際のbeam_width, unique_parents の推移
4. **Candidates & Pruning**: 候補数、枝刈り数（ハッシュ版はhash_dedup数も）
5. **Hash Dedup Rate** (ハッシュ版のみ): ハッシュ重複除去率
6. **Elapsed Time per Turn**: ターンごとの経過時間

### 使い方

```bash
./a.out < input.txt 2>&1 >/dev/null | python visualize.py
```

グラフは `visualize/beam/YYYYMMDD_HHMMSS/` ディレクトリに保存されます。
