# セグフォ時の最短デバッグ手順

## 1. まず ASan で実行（最短）
```bash
g++-12 -std=gnu++23 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -I/home/xxx/lib/ac-library-master -o main_asan main.cpp && ./main_asan < tools/in/0000.txt

```
- 意味: 壊した場所を高確率で直接出す。
- 見る場所: 最初の `ERROR` と最初に出る `file:line`（自分のコード）。

## 2. 位置が曖昧なら gdb
```bash
g++-12 -std=gnu++23 -O0 -g -I/home/xxx/lib/ac-library-master -o main main.cpp && gdb ./main
```
```gdb
run < tools/in/0000.txt
bt
frame 0
info locals
```
- 意味: 落ちた瞬間の関数と変数を確認。
- 見る場所: `bt` の先頭の自分の関数、`frame 0` の添字・ポインタ・サイズ。
