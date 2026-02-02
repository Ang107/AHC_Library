#!/usr/bin/env python3
"""AtCoderの問題HTMLから日本語の問題文をMarkdownに変換するスクリプト"""

import sys
import re
from html.parser import HTMLParser


class AtCoderHTML2MD(HTMLParser):
    def __init__(self):
        super().__init__()
        self.result = []
        self.in_lang_ja = False
        self.in_lang_en = False
        self.tag_stack = []
        self.ol_counter = 0
        self.skip = False
        self.pre_depth = 0
        self.title = ""
        self.in_title = False
        self.time_limit = ""
        self.memory_limit = ""
        self.in_time_memory = False
        self.time_memory_text = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "")

        # タイトル取得
        if tag == "span" and "h2" in cls:
            self.in_title = True
            return
        if self.in_title and tag in ("a", "span", "button"):
            self.in_title = False  # ボタンやリンクに入ったらタイトル終了
            return

        # 実行時間・メモリ制限
        if tag == "p" and not self.in_lang_ja:
            self.in_time_memory = True
            self.time_memory_text = ""
            return

        # lang-ja / lang-en の検出
        if tag == "span" and "lang-ja" in cls:
            self.in_lang_ja = True
            return
        if tag == "span" and "lang-en" in cls:
            self.in_lang_en = True
            self.skip = True
            return

        if not self.in_lang_ja or self.skip:
            return

        self.tag_stack.append(tag)

        if tag == "h3":
            self.result.append("\n## ")
        elif tag == "h4":
            self.result.append("\n### ")
        elif tag == "ul":
            self.result.append("\n")
        elif tag == "ol":
            self.ol_counter = 0
            self.result.append("\n")
        elif tag == "li":
            parent_list = None
            for t in reversed(self.tag_stack[:-1]):
                if t in ("ul", "ol"):
                    parent_list = t
                    break
            if parent_list == "ol":
                self.ol_counter += 1
                self.result.append(f"{self.ol_counter}. ")
            else:
                self.result.append("- ")
        elif tag == "pre":
            self.pre_depth += 1
            self.result.append("\n```\n")
        elif tag == "code" and self.pre_depth == 0:
            self.result.append("`")
        elif tag == "var":
            self.result.append("$")
        elif tag == "strong" or tag == "b":
            self.result.append("**")
        elif tag == "em" or tag == "i":
            self.result.append("*")
        elif tag == "a":
            href = attrs_dict.get("href", "")
            self.result.append("[")
            self.tag_stack[-1] = ("a", href)
        elif tag == "br":
            self.result.append("\n")
        elif tag == "hr":
            self.result.append("\n---\n")
        elif tag == "p":
            self.result.append("\n")
        elif tag == "sub":
            self.result.append("_{")
        elif tag == "sup":
            self.result.append("^{")
        elif tag == "img":
            alt = attrs_dict.get("alt", "")
            src = attrs_dict.get("src", "")
            self.result.append(f"![{alt}]({src})")

    def handle_endtag(self, tag):
        if tag == "span" and self.in_title:
            self.in_title = False
            return

        if tag == "p" and self.in_time_memory and not self.in_lang_ja:
            self.in_time_memory = False
            text = self.time_memory_text.strip()
            if "実行時間制限" in text or "メモリ制限" in text:
                self.time_limit = text
            return

        if tag == "span" and self.in_lang_en and self.skip:
            self.in_lang_en = False
            self.skip = False
            return
        if tag == "span" and self.in_lang_ja and not self.tag_stack:
            self.in_lang_ja = False
            return

        if not self.in_lang_ja or self.skip:
            return

        if not self.tag_stack:
            return

        current = self.tag_stack.pop()

        if tag == "h3" or tag == "h4":
            self.result.append("\n")
        elif tag == "li":
            self.result.append("\n")
        elif tag == "pre":
            self.pre_depth -= 1
            self.result.append("```\n")
        elif tag == "code" and self.pre_depth == 0:
            self.result.append("`")
        elif tag == "var":
            self.result.append("$")
        elif tag == "strong" or tag == "b":
            self.result.append("**")
        elif tag == "em" or tag == "i":
            self.result.append("*")
        elif tag == "a" or (isinstance(current, tuple) and current[0] == "a"):
            href = current[1] if isinstance(current, tuple) else ""
            self.result.append(f"]({href})")
        elif tag == "p":
            self.result.append("\n")
        elif tag == "sub":
            self.result.append("}")
        elif tag == "sup":
            self.result.append("}")
        elif tag == "section":
            self.result.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title += data
            return
        if self.in_time_memory and not self.in_lang_ja:
            self.time_memory_text += data
            return
        if not self.in_lang_ja or self.skip:
            return
        if self.pre_depth > 0:
            self.result.append(data)
        else:
            self.result.append(data)

    def handle_entityref(self, name):
        if not self.in_lang_ja or self.skip:
            if self.in_time_memory:
                import html
                self.time_memory_text += html.unescape(f"&{name};")
            return
        import html
        self.result.append(html.unescape(f"&{name};"))

    def handle_charref(self, name):
        if not self.in_lang_ja or self.skip:
            return
        import html
        self.result.append(html.unescape(f"&#{name};"))

    def get_markdown(self):
        md = "".join(self.result)
        # 連続空行を整理
        md = re.sub(r'\n{3,}', '\n\n', md)
        md = md.strip()

        header = f"# {self.title.strip()}\n"
        if self.time_limit:
            header += f"\n{self.time_limit}\n"
        header += "\n"

        return header + md + "\n"


def main():
    if len(sys.argv) < 2:
        print("Usage: python html2md.py <input.html>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]

    with open(input_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    parser = AtCoderHTML2MD()
    parser.feed(html_content)
    md = parser.get_markdown()

    with open("problem.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"problem.md を生成しました")


if __name__ == "__main__":
    main()
