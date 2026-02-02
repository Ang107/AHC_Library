# AHC Library

AtCoder Heuristic Contest (AHC) 用のテンプレート・tipsをまとめたライブラリです。
参考: https://atcoder.jp/contests/ahc058/submissions/71725922

## ファイル構成

```
.
├── main.cpp              # 汎用テンプレート
├── html2md.py            # AtCoder問題HTMLからMarkdownへの変換スクリプト
├── sa/
│   ├── sa.cpp            # 焼きなまし法テンプレート
│   ├── tips.md           # 焼きなまし関連のTips
│   ├── visualize.py      # SA統計情報の可視化ツール
│   ├── sample/           # 使用例
│   └── README.md         # 焼きなましの詳細ドキュメント
└── beam/
    ├── beam.cpp           # ビームサーチテンプレート（ハッシュあり）
    ├── beam_no_hash.cpp   # ビームサーチテンプレート（ハッシュなし）
    ├── tips.md            # ビームサーチ関連のTips
    ├── visualize.py       # ビームサーチ統計情報の可視化ツール
    ├── sample/            # 使用例
    └── README.md          # ビームサーチの詳細ドキュメント
```

## main.cpp - 汎用テンプレート

AHC共通のベーステンプレートです。以下のユーティリティを含みます。

- **型エイリアス**: `i32`, `u64`, `f64` など
- **高速ハッシュマップ**: `hash_map`, `hash_set`（pb_dsベース）
- **デバッグ出力**: `debug()` で任意の型を標準エラーに出力
- **乱数生成器**: `RNG`（xoshiro256**ベース、Walker's Alias Method対応）
- **タイマー**: `timer` 構造体
- **コレクション出力**: vector, map, tuple等の `<<` オーバーロード
- **ビット操作**: `bit()`, `setbit()`, `getbit()`, `lsb()`, `msb()`

## sa/ - 焼きなまし法

テーブルルックアップによる高速な温度減衰・log計算、重み付き近傍選択、統計情報の可視化などを備えた焼きなましテンプレートです。詳細は [sa/README.md](sa/README.md) を参照してください。

## beam/ - ビームサーチ

ビームサーチのテンプレートです。ハッシュによる重複除去ありなしの2種類があります。詳細は [beam/README.md](beam/README.md) を参照してください。

## CLAUDE.SAMPLE.md - CLAUDE.mdのサンプル

AHCの各問題プロジェクトで使用する `CLAUDE.md` のサンプルです。新しい問題に取り組む際に、このファイルをコピーして `CLAUDE.md` として配置し、プロジェクトに合わせて内容を編集してください。

## html2md.py - 問題文変換

AtCoderの問題ページのHTMLから日本語の問題文を抽出し、Markdownファイル (`problem.md`) に変換するスクリプトです。
HTMLはページ上で `Ctrl` + `s` $\rarr$ `ウェブページ、HTML のみ` を選択して保存したものを用いてください。
```bash
python3 html2md.py <HTMLファイルのパス>
```

実行したディレクトリに `problem.md` が生成されます。
