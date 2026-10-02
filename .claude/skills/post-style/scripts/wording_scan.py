#!/usr/bin/env python3
"""扫描文章或图组件里的用词问题，逐条列出位置和建议。

用法：python3 wording_scan.py <文件> [<文件> ...]

它只找能机械识别的五类问题。每一条要么改掉，要么确认是误报。
自造术语（比如把「点积」写成别的叫法）脚本认不出来，要靠人读。
"""
import re
import sys

# 翻译腔 → 建议写法。左边是正则。
REPLACE = [
    (r"口径", "按……算 / 按……数"),
    (r"物化", "存一份 / 搬到 GPU 上"),
    (r"解耦", "分开 / 不再绑在一起"),
    (r"数据相关", "由输入决定"),
    (r"启发式", "近似算法 / 经验规则"),
    (r"感知负载", "按负载分配"),
    (r"训练动力学|学习动态", "训练效果 / 学得是否均匀"),
    (r"一致地(改善|提升)", "都有提升"),
    (r"消费自己|消费(?=.{0,6}输出)", "拿……当输入"),
    (r"主流的(?=.{0,8}(计算|all-to-all|GEMM))", "主 stream 上的"),
    (r"算术量", "计算量"),
    (r"墙钟", "实际耗时"),
    (r"(?<![一-鿿])形态", "形式 / 样子 / 做法"),
    (r"旋钮", "可调的量 / 参数"),
    (r"接线", "实现 / 连接方式"),
    (r"货币|付账|买来|买回", "代价是……"),
    (r"落盘", "存到磁盘上"),
    (r"满配", "完整的"),
    (r"远逊于|区别于|力量在于|名不副实", "换成口语说法"),
    (r"注脚|对应物|支点|支柱", "直接说它指什么"),
    (r"退化(成|回|为)", "就变成（推导里的严格用法可以保留）"),
]

CJK = r"[一-鿿]"


def strip_noise(line: str) -> str:
    """去掉行内公式和行内代码，避免把公式里的字当正文。"""
    line = re.sub(r"\$[^$]*\$", " § ", line)  # § 标记这里原来有公式
    line = re.sub(r"`[^`]*`", " ", line)
    return line


def cjk_len(s: str) -> int:
    return len(re.findall(CJK, s))


def width(s: str) -> int:
    """汉字数加英文词数，用来比较标题两半是否等长。"""
    return cjk_len(s) + len(re.findall(r"[A-Za-z0-9]+", s))


def scan(path: str):
    hits = []
    in_code = in_math = in_front = False
    is_astro = path.endswith(".astro")
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if i == 1 and s == "---":
            in_front = True
            continue
        if in_front:
            if s == "---":
                in_front = False
                continue
            if not s.startswith(("title:", "description:")):
                continue
        if s.startswith("```"):
            in_code = not in_code
            continue
        if s.startswith("$$"):
            in_math = not in_math if s == "$$" or s.count("$$") == 1 else in_math
            continue
        if in_code or in_math or s.startswith("import "):
            continue
        # 组件文件里的注释行不算正文；Markdown 里以 * 开头的是加粗或列表，要扫。
        if is_astro and (s.startswith(("//", "/*", "*/")) or s == "*" or s.startswith("* ")):
            continue
        line = strip_noise(raw)

        def add(kind, frag, tip):
            hits.append((i, kind, frag.strip()[:40], tip))

        # 1. 翻译腔
        for pat, tip in REPLACE:
            for m in re.finditer(pat, line):
                add("翻译腔", line[max(0, m.start() - 8): m.end() + 8], tip)

        # 2. 单字概念名：加粗、引号或表格首列里只有一个汉字
        for m in re.finditer(rf"\*\*({CJK})[。：:]?\*\*|「({CJK})」", line):
            add("单字概念名", m.group(0), "换成完整的词，例如 写入 / 擦除 / 衰减")
        m = re.match(rf"\s*\|\s*({CJK})\s*\|", line)
        is_header = i < len(lines) and re.match(r"\s*\|\s*:?-", lines[i])
        if m and not is_header:
            add("单字概念名", m.group(0), "表格首列用完整的词")

        # 3. 对仗式标题或加粗句：逗号两边一样长
        head, strict = None, False
        if s.startswith("#"):
            head = re.sub(r"^#+\s*", "", line.strip())
        else:
            m = re.match(r"\s*\*\*([^*]+)\*\*", line)
            if m:
                head, strict = m.group(1), True
        if head and "§" not in head:
            body = head.split("：", 1)[-1]
            parts = [p for p in re.split(r"[，、,]", body.rstrip("。")) if p.strip()]
            if len(parts) == 2:
                a, b = width(parts[0]), width(parts[1])
                pure = not re.search(r"[A-Za-z0-9]", body)
                # 加粗句只在两半都是纯中文时才报，减少误报
                if abs(a - b) <= 1 and 4 <= min(a, b) <= 9 and (pure or not strict):
                    add("对仗式标题", head, "写成一句说明内容的陈述句")

        # 4. key / value / query 写成了中文
        for m in re.finditer(r"(?<!关)键|查询", line):
            add("key/value/query", line[max(0, m.start() - 6): m.end() + 6], "全系列统一用 key / query")
        for m in re.finditer(r"键值|[旧新]值|值(通道|轴)", line):
            add("key/value/query", line[max(0, m.start() - 6): m.end() + 6], "如果指 value，用英文")

        # 5. 长定语：一个分句里有四个以上「的」
        for clause in re.split(r"[，。；：！？,;]", line):
            if clause.count("的") >= 4 and cjk_len(clause) >= 20:
                add("长定语", clause, "拆成两句")
    return hits


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    total = 0
    for path in sys.argv[1:]:
        hits = scan(path)
        total += len(hits)
        name = path.split("/")[-1]
        for ln, kind, frag, tip in hits:
            print(f"{name}:{ln} [{kind}] {frag}  →  {tip}")
    print(f"\n共 {total} 条。每一条要么改掉，要么确认是误报。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
