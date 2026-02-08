# perf によるタイムプロファイリング (WSL2)

## セットアップ

```bash
sudo apt install linux-tools-generic
```

WSL2ではカーネルバージョンの不一致で `perf` コマンドがエラーになる。
シンボリックリンクを作成して回避する:

```bash
sudo ln -sf /usr/lib/linux-tools/*/perf /usr/local/bin/perf
```

パスが不明な場合は `ls /usr/lib/linux-tools/*/perf` で確認する。

## 基本フロー

### 1. コンパイル

`-g` でデバッグ情報を付与する。`-pg` は不要。

```bash
g++ -O2 -g -o main main.cpp
```

### 2. プロファイル取得

```bash
perf record -g ./main < input.txt
```

`perf.data` が生成される。

主要オプション:
- `-g`: コールグラフを記録
- `-F 1000`: サンプリング周波数を指定（デフォルト4000）
- `--call-graph dwarf`: より正確なコールグラフ（`-g` で不十分な場合）

### 3. 結果の確認

用途に応じて以下のいずれかを使う。

#### CUI: perf report

```bash
# 関数ごとのフラット表示（おすすめ）
perf report --stdio --no-children | less

# 特定の関数をソースコード行レベルで確認
perf annotate <関数名>

# 簡易統計（record不要、実行時間やキャッシュミスの概要）
perf stat ./main < input.txt
```

#### GUI: FlameGraph（関数単位の可視化）

関数ごとのCPU時間割合をインタラクティブなSVGとして可視化する。

```bash
# 初回のみ
git clone https://github.com/brendangregg/FlameGraph.git ~/FlameGraph

# SVG生成
perf script | ~/FlameGraph/stackcollapse-perf.pl | ~/FlameGraph/flamegraph.pl > flamegraph.svg

# ブラウザで開く
explorer.exe flamegraph.svg
```

読み方:
- **横幅** = CPU時間の割合（広いほど重い）
- **縦方向** = コールスタック（下が呼び出し元、上が呼び出し先）
- マウスホバーで関数名と割合を確認、クリックでズームイン

#### GUI: hotspot（ソースコード行レベルの分析）

関数内のどの行が重いかまでGUIで確認できる。

```bash
# 初回のみ
sudo apt install hotspot

# 開く
hotspot perf.data
```

WSL2でGUIを表示するにはWSLgが有効になっている必要がある（Windows 11では標準で有効）。

## Tips

- **最適化は付けたまま計測する**: `-O2` を外すとボトルネックが変わる
- **入力は本番と同じサイズで**: 小さい入力だとボトルネックが変わることがある
- **権限エラーになる場合**: `sudo sysctl -w kernel.perf_event_paranoid=-1`
- **インライン展開された関数を見たい場合**: `perf report --inline`
