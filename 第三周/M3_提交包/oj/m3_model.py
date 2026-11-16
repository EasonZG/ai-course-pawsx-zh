# -*- coding: utf-8 -*-
"""M3 可携带线性模型：只用 Python 标准库加载与推理。

模型目录由 ``oj/train_trad.py`` 导出，包含：
  meta.json   特征维度、块偏移、标准化参数、训练信息、文件哈希
  vocab.json  按索引排序的字符 n-gram 词表
  idf.bin     float64 IDF
  coef.bin    float64 线性权重（二分类，长度 = 4*V + n_numeric）

推理复算与 sklearn TfidfVectorizer 相同的 sublinear TF-IDF 和 L2 归一化，
不依赖 sklearn/numpy；这样提交包在只有 Python 标准库的环境也能跑。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from array import array
from collections import Counter

from m3_features import NUMERIC_FEATURE_NAMES, char_ngrams, pair_numeric_features


class ModelError(RuntimeError):
    pass


class ModelNotFoundError(ModelError):
    pass


class ModelIntegrityError(ModelError):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_doubles(path: str, expected: int | None = None) -> array:
    arr = array("d")
    size = os.path.getsize(path)
    if size % arr.itemsize:
        raise ModelIntegrityError(f"模型文件长度不是 float64 的整数倍：{path}")
    with open(path, "rb") as f:
        arr.fromfile(f, size // arr.itemsize)
    if expected is not None and len(arr) != expected:
        raise ModelIntegrityError(
            f"模型文件长度不符：{os.path.basename(path)} 期望 {expected}，实际 {len(arr)}"
        )
    return arr


class TradModel:
    """字符 n-gram 句对线性模型（纯标准库推理）。"""

    def __init__(self, model_dir: str, verify: bool = True):
        self.model_dir = os.path.abspath(model_dir)
        meta_path = os.path.join(self.model_dir, "meta.json")
        if not os.path.isfile(meta_path):
            raise ModelNotFoundError(
                f"找不到传统模型：{meta_path}\n"
                "请先在包根目录运行 python oj/train_trad.py 训练并导出模型；"
                "入口不会静默回退重训。"
            )
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)

        required = ["vocab.json", "idf.bin", "coef.bin"]
        missing = [n for n in required if not os.path.isfile(os.path.join(self.model_dir, n))]
        if missing:
            raise ModelNotFoundError(
                f"模型目录 {self.model_dir} 缺少文件：{', '.join(missing)}；"
                "请重新运行 python oj/train_trad.py。"
            )

        if verify:
            self._verify_files()

        with open(os.path.join(self.model_dir, "vocab.json"), encoding="utf-8") as f:
            tokens = json.load(f)
        if not isinstance(tokens, list) or not tokens:
            raise ModelIntegrityError("vocab.json 应为非空 token 列表")
        self.tokens = tokens
        self.vocab = {tok: i for i, tok in enumerate(tokens)}
        self.vocab_size = len(tokens)

        self.idf = _read_doubles(os.path.join(self.model_dir, "idf.bin"), self.vocab_size)
        self.n_features = int(self.meta["n_features"])
        self.coef = _read_doubles(os.path.join(self.model_dir, "coef.bin"), self.n_features)
        self.intercept = float(self.meta["intercept"])
        self.scaler_mean = [float(x) for x in self.meta["scaler_mean"]]
        self.scaler_scale = [float(x) for x in self.meta["scaler_scale"]]
        self.block_offsets = [int(x) for x in self.meta["block_offsets"]]
        self.sublinear_tf = bool(self.meta.get("sublinear_tf", True))
        self.numeric_names = list(self.meta.get("numeric_features", NUMERIC_FEATURE_NAMES))

        if len(self.numeric_names) != len(self.scaler_mean) or len(self.numeric_names) != len(self.scaler_scale):
            raise ModelIntegrityError("meta.json 中数值特征名与标准化参数长度不一致")
        expect = 4 * self.vocab_size + len(self.numeric_names)
        if self.n_features != expect:
            raise ModelIntegrityError(
                f"特征维度不一致：meta.n_features={self.n_features}，"
                f"4*V+numeric={expect}"
            )
        if self.meta.get("numeric_features") and self.meta["numeric_features"] != NUMERIC_FEATURE_NAMES:
            # 只允许完全一致，避免训练/推理特征错位
            raise ModelIntegrityError("模型数值特征顺序与当前 m3_features.py 不一致")

    def _verify_files(self) -> None:
        files = self.meta.get("files") or {}
        for name, want in files.items():
            path = os.path.join(self.model_dir, name)
            if not os.path.isfile(path):
                raise ModelIntegrityError(f"模型文件缺失：{name}")
            got = sha256_file(path)
            if got != want:
                raise ModelIntegrityError(
                    f"模型文件哈希不符：{name}（期望 {want[:12]}…，实际 {got[:12]}…）"
                )

    def transform_text(self, text: str) -> dict[int, float]:
        """返回 L2 归一化后的稀疏 TF-IDF {词表索引: 值}。"""
        counts = Counter(char_ngrams(text))
        if not counts:
            return {}
        vals: dict[int, float] = {}
        for tok, c in counts.items():
            idx = self.vocab.get(tok)
            if idx is None:
                continue
            tf = (1.0 + math.log(c)) if self.sublinear_tf else float(c)
            vals[idx] = tf * float(self.idf[idx])
        norm = math.sqrt(sum(v * v for v in vals.values()))
        if norm <= 0.0:
            return {}
        return {i: v / norm for i, v in vals.items()}

    def pair_features(self, a: str, b: str) -> tuple[dict[int, float], dict]:
        A = self.transform_text(a)
        B = self.transform_text(b)
        V = self.vocab_size
        off_a, off_b, off_abs, off_prod, off_num = self.block_offsets
        feats: dict[int, float] = {}

        for i, v in A.items():
            feats[off_a + i] = v
        for i, v in B.items():
            feats[off_b + i] = v

        # |A-B|：并集上的绝对差
        for i in set(A) | set(B):
            d = abs(A.get(i, 0.0) - B.get(i, 0.0))
            if d:
                feats[off_abs + i] = d
        # A*B：交集上的逐元素乘积
        for i, v in A.items():
            bv = B.get(i)
            if bv:
                feats[off_prod + i] = v * bv

        raw = pair_numeric_features(a, b)
        for j, x in enumerate(raw):
            scale = self.scaler_scale[j] or 1.0
            feats[off_num + j] = (x - self.scaler_mean[j]) / scale
        stats = {k: v for k, v in zip(self.numeric_names, raw)}
        return feats, stats

    def score_pair(self, a: str, b: str) -> tuple[float, dict]:
        feats, stats = self.pair_features(a, b)
        score = self.intercept
        for i, v in feats.items():
            score += float(self.coef[i]) * v
        return score, stats

    def predict_pair(self, a: str, b: str) -> tuple[int, float, dict]:
        score, stats = self.score_pair(a, b)
        return (1 if score > 0.0 else 0), score, stats

    def info(self) -> dict:
        return {
            "model_dir": self.model_dir,
            "model_version": self.meta.get("model_version"),
            "trained_at": self.meta.get("trained_at"),
            "n_features": self.n_features,
            "vocab_size": self.vocab_size,
            "n_numeric": len(self.numeric_names),
            "train_rows": self.meta.get("train_rows"),
            "train_seconds": self.meta.get("train_seconds"),
            "model_sha256": self.meta.get("model_sha256"),
            "config": self.meta.get("config"),
        }
