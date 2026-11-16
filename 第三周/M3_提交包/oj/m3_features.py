# -*- coding: utf-8 -*-
"""M3 传统方法：字符 n-gram 句对特征（仅依赖标准库）。

训练时由 scikit-learn 的 TfidfVectorizer 调用 ``char_ngrams`` 作为 analyzer；
推理时由 ``m3_model.TradModel`` 用同一函数复算 TF-IDF，因此提交包不需要
numpy/scipy/scikit-learn 也能离线运行。

设计要点：
- A、B 两句分别做字符 1-3 gram TF-IDF，再拼接 A、B、|A-B|、A*B 四个稀疏块；
- 另加顺序/重合/长度/数字/否定/标点等数值特征，避免只把两句相加丢失顺序信息；
- 所有词表只由 train 拟合（train_trad.py 负责），推理不读取任何标签。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher

# 数值特征顺序必须与训练/推理/错误分析完全一致；新增特征只能追加并升模型版本。
NUMERIC_FEATURE_NAMES = [
    "jaccard_bigram",
    "containment_a",
    "containment_b",
    "containment_min",
    "containment_diff",
    "len_a",
    "len_b",
    "len_min",
    "len_max",
    "len_ratio",
    "len_absdiff",
    "len_logratio",
    "prefix_ratio",
    "suffix_ratio",
    "seq_ratio",
    "unigram_jaccard",
    "same_char_bag",
    "exact_equal",
    "digit_jaccard",
    "digit_set_diff",
    "digit_count_a",
    "digit_count_b",
    "neg_count_a",
    "neg_count_b",
    "neg_count_diff",
    "neg_xor",
    "pronoun_diff",
    "punct_count_diff",
    "first_diff_pos_norm",
    "last_diff_pos_norm",
    "contains_a_in_b",
    "contains_b_in_a",
]

NEGATIONS = ("不", "没", "无", "未", "非", "别", "勿", "莫", "否", "缺乏", "没有")
PRONOUNS = ("他", "她", "它", "我", "你", "您", "我们", "他们", "她们", "咱们")
PUNCT = set("，。！？；：、,.!?;:（）()《》〈〉“”\"'‘’—…·")


def normalize_text(text: str) -> str:
    """去掉所有空白，保留标点；与 M2 词面基线的口径一致。"""
    return "".join((text or "").split())


def char_ngrams(text: str) -> list[str]:
    """字符 1-3 gram，作为 TfidfVectorizer 的 analyzer。"""
    s = normalize_text(text)
    out: list[str] = []
    if not s:
        return out
    out.extend(s)  # unigram
    for n in (2, 3):
        if len(s) >= n:
            out.extend(s[i:i + n] for i in range(len(s) - n + 1))
    return out


def _ngram_set(text: str, n: int) -> set[str]:
    s = normalize_text(text)
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _containment(a: set[str], b: set[str]) -> float:
    if not a:
        return 1.0 if not b else 0.0
    return len(a & b) / len(a)


def _num_tokens(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text or "")


def _neg_count(text: str) -> int:
    return sum((text or "").count(w) for w in NEGATIONS)


def _pronoun_count(text: str) -> int:
    return sum((text or "").count(w) for w in PRONOUNS)


def _punct_count(text: str) -> int:
    return sum(1 for ch in (text or "") if ch in PUNCT)


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _common_suffix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[-1 - i] == b[-1 - i]:
        i += 1
    return i


def pair_numeric_features(a: str, b: str) -> list[float]:
    """顺序敏感的句对数值特征，返回长度固定的 list[float]。"""
    a = normalize_text(a)
    b = normalize_text(b)
    la, lb = len(a), len(b)
    lmin, lmax = min(la, lb), max(la, lb)

    ba, bb = _ngram_set(a, 2), _ngram_set(b, 2)
    ua, ub = set(a), set(b)
    inter_b = len(ba & bb)
    union_b = len(ba | bb)
    cont_a = _containment(ba, bb)
    cont_b = _containment(bb, ba)

    da, db = _num_tokens(a), _num_tokens(b)
    sa, sb = set(da), set(db)
    digit_jac = _jaccard(sa, sb) if (sa or sb) else 1.0
    digit_diff = 0.0 if sa == sb else 1.0

    na, nb = _neg_count(a), _neg_count(b)
    pa, pb = _pronoun_count(a), _pronoun_count(b)
    pua, pub = _punct_count(a), _punct_count(b)

    # 第一个/最后一个不同字符的位置（归一化）；完全相同则为 1.0
    first_diff = 1.0
    last_diff = 1.0
    if a != b:
        n = min(la, lb)
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        first_diff = i / max(lmax, 1)
        j = 0
        while j < n and a[-1 - j] == b[-1 - j]:
            j += 1
        last_diff = (lmax - j) / max(lmax, 1)

    prefix = _common_prefix_len(a, b)
    suffix = _common_suffix_len(a, b)
    sm = SequenceMatcher(None, a, b, autojunk=False)
    seq_ratio = sm.ratio()

    same_bag = 1.0 if (a != b and sorted(a) == sorted(b)) else 0.0
    exact = 1.0 if a == b else 0.0

    return [
        inter_b / union_b if union_b else 1.0,          # jaccard_bigram
        cont_a,
        cont_b,
        min(cont_a, cont_b),
        cont_a - cont_b,
        float(la),
        float(lb),
        float(lmin),
        float(lmax),
        lmin / lmax if lmax else 1.0,
        float(abs(la - lb)),
        math.log((la + 1.0) / (lb + 1.0)),
        prefix / lmin if lmin else 1.0,
        suffix / lmin if lmin else 1.0,
        seq_ratio,
        _jaccard(ua, ub),
        same_bag,
        exact,
        digit_jac,
        digit_diff,
        float(len(da)),
        float(len(db)),
        float(na),
        float(nb),
        float(na - nb),
        1.0 if (na > 0) != (nb > 0) else 0.0,
        float(abs(pa - pb)),
        float(abs(pua - pub)),
        first_diff,
        last_diff,
        1.0 if (a and a in b) else 0.0,
        1.0 if (b and b in a) else 0.0,
    ]


def stats_dict(a: str, b: str) -> dict:
    vals = pair_numeric_features(a, b)
    return {k: v for k, v in zip(NUMERIC_FEATURE_NAMES, vals)}


def counts_for_text(text: str) -> Counter:
    return Counter(char_ngrams(text))
