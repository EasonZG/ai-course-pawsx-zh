#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M3 传统方法训练：字符 n-gram 句对 TF-IDF + SGD(log_loss)。

只读 data/train.jsonl 训练，只读 data/splits/dev1200.jsonl 选模。
训练脚本不接受 holdout 路径；holdout 由独立的 eval_holdout_m3.py 在选模后评一次。

用法（包根目录）：
    python oj/train_trad.py
    python oj/train_trad.py --max-train 8000 --epochs 8 --runs-dir runs/m3_smoke --out-dir models/trad_smoke
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from m3_env import collect_environment  # noqa: E402
from m3_features import NUMERIC_FEATURE_NAMES, char_ngrams, pair_numeric_features  # noqa: E402

LABELS = ["different", "paraphrase"]
TZ = timezone(timedelta(hours=8))


def now_iso() -> str:
    return datetime.now(TZ).isoformat(timespec="milliseconds")


def utf8_stdio() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def label_of(row: dict) -> int:
    t = row.get("target")
    if isinstance(t, dict):
        t = t.get("label")
    if isinstance(t, bool) or not isinstance(t, int) or t not in (0, 1):
        raise ValueError(f"非法标签：{row.get('item_id')} target={row.get('target')!r}")
    return t


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def combined_hash(paths: list[str]) -> str:
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def build_pair_matrix(
    vectorizer: TfidfVectorizer,
    pairs: list[tuple[str, str]],
    scaler: StandardScaler | None = None,
) -> tuple[csr_matrix, StandardScaler, float]:
    t0 = time.time()
    A = [p[0] for p in pairs]
    B = [p[1] for p in pairs]
    XA = vectorizer.transform(A)
    XB = vectorizer.transform(B)
    absdiff = abs(XA - XB).tocsr()
    prod = XA.multiply(XB).tocsr()
    numeric = np.asarray([pair_numeric_features(a, b) for a, b in pairs], dtype=np.float64)
    if scaler is None:
        scaler = StandardScaler().fit(numeric)
    numeric_s = scaler.transform(numeric)
    X = hstack(
        [XA, XB, absdiff, prod, csr_matrix(numeric_s)],
        format="csr",
        dtype=np.float64,
    )
    X.sort_indices()
    del XA, XB, absdiff, prod, numeric, numeric_s
    gc.collect()
    return X, scaler, time.time() - t0


def metrics_dict(y_true: list[int], y_pred: list[int], y_prob: np.ndarray | None = None) -> dict:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
        "per_label": {
            LABELS[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i in range(2)
        },
        "confusion": {
            "rows_gold_cols_pred": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            "row_order": LABELS,
            "col_order": LABELS,
        },
        "n": len(y_true),
    }
    if y_prob is not None:
        out["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    return out


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_timeline(path: str, event: str, **kw) -> None:
    data = {"events": []}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"events": []}
    data.setdefault("events", []).append({"at": now_iso(), "event": event, **kw})
    write_json(path, data)


def export_model(
    out_dir: str,
    vectorizer: TfidfVectorizer,
    scaler: StandardScaler,
    clf: SGDClassifier,
    config: dict,
    dev_metrics: dict,
    args,
    data_hashes: dict,
    timings: dict,
    env: dict,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    tokens = [""] * len(vectorizer.vocabulary_)
    for tok, idx in vectorizer.vocabulary_.items():
        tokens[idx] = tok
    if any(t == "" for t in tokens):
        raise RuntimeError("词表存在空 token，导出中止")

    vocab_path = os.path.join(out_dir, "vocab.json")
    idf_path = os.path.join(out_dir, "idf.bin")
    coef_path = os.path.join(out_dir, "coef.bin")

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, separators=(",", ":"))
    vectorizer.idf_.astype(np.float64).tofile(idf_path)
    clf.coef_.ravel().astype(np.float64).tofile(coef_path)

    files = {
        "vocab.json": sha256_file(vocab_path),
        "idf.bin": sha256_file(idf_path),
        "coef.bin": sha256_file(coef_path),
    }
    model_sha = combined_hash([vocab_path, idf_path, coef_path])
    vocab_size = len(tokens)
    n_numeric = len(NUMERIC_FEATURE_NAMES)
    meta = {
        "schema": "m3.trad.model/1",
        "model_version": "trad-char-tfidf-sgd-v1",
        "trained_at": now_iso(),
        "config": config,
        "train_rows": int(args._train_rows),
        "dev_rows": int(args._dev_rows),
        "sampling_rule": args._sampling_rule,
        "n_features": int(4 * vocab_size + n_numeric),
        "vocab_size": int(vocab_size),
        "n_numeric": int(n_numeric),
        "block_offsets": [0, vocab_size, 2 * vocab_size, 3 * vocab_size, 4 * vocab_size],
        "numeric_features": NUMERIC_FEATURE_NAMES,
        "sublinear_tf": True,
        "scaler_mean": [float(x) for x in scaler.mean_],
        "scaler_scale": [float(x) for x in scaler.scale_],
        "intercept": float(clf.intercept_[0]),
        "dev_metrics": dev_metrics,
        "timings": timings,
        "data_sha256": data_hashes,
        "environment": {
            "python": env.get("python"),
            "platform": env.get("platform"),
            "cpu_name": env.get("cpu_name"),
            "cpu_count": env.get("cpu_count"),
            "memory_gb": env.get("memory_gb"),
            "gpu": env.get("gpu"),
            "packages": env.get("packages"),
            "torch_cuda_available": env.get("torch_cuda_available"),
        },
        "files": files,
        "model_sha256": model_sha,
    }
    meta_path = os.path.join(out_dir, "meta.json")
    write_json(meta_path, meta)
    return meta


def main() -> int:
    utf8_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.path.join(ROOT, "data", "train.jsonl"))
    ap.add_argument("--dev", default=os.path.join(ROOT, "data", "splits", "dev1200.jsonl"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "models", "trad"))
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "runs", "m3"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-features", type=int, default=200_000)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-train", type=int, default=0,
                    help="仅冒烟用：确定性抽取 N 条训练；0=全量 44,898")
    ap.add_argument("--expect-train", type=int, default=44898)
    ap.add_argument("--allow-any-dev", action="store_true",
                    help="仅测试用；正式训练必须 dev1200")
    args = ap.parse_args()

    if "holdout" in os.path.abspath(args.dev).lower() and not args.allow_any_dev:
        raise SystemExit("拒绝：训练/选模不得读取 holdout746 标签；请使用 data/splits/dev1200.jsonl")
    os.makedirs(args.runs_dir, exist_ok=True)
    os.makedirs(os.path.join(args.runs_dir, "logs"), exist_ok=True)
    log_path = os.path.join(args.runs_dir, "logs", "train_trad.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    log = logging.getLogger("train_trad")
    timeline_path = os.path.join(args.runs_dir, "timeline.json")
    t_start = time.time()
    append_timeline(timeline_path, "train_start", train=args.train, dev=args.dev, seed=args.seed)
    log.info("M3 传统方法训练开始")
    log.info("train=%s", args.train)
    log.info("dev=%s（只用于选模，不读 holdout）", args.dev)
    log.info("out_dir=%s runs_dir=%s", args.out_dir, args.runs_dir)

    train_rows = load_jsonl(args.train)
    dev_rows = load_jsonl(args.dev)
    if args.expect_train and args.max_train == 0 and len(train_rows) != args.expect_train:
        raise SystemExit(f"train 条数 {len(train_rows)} != 期望 {args.expect_train}；"
                         "如确认数据版本变化，用 --expect-train 0 关闭检查")
    if len(dev_rows) != 1200 and not args.allow_any_dev:
        raise SystemExit(f"dev 条数 {len(dev_rows)} != 1200；正式选模只允许 dev1200")

    if args.max_train and args.max_train < len(train_rows):
        rng = random.Random(args.seed)
        idx = sorted(rng.sample(range(len(train_rows)), args.max_train))
        train_rows = [train_rows[i] for i in idx]
        sampling_rule = f"冒烟：random.Random({args.seed}).sample 后按原顺序取 {args.max_train} 条"
    else:
        sampling_rule = "全量 train.jsonl，未采样"
    args._train_rows = len(train_rows)
    args._dev_rows = len(dev_rows)
    args._sampling_rule = sampling_rule

    y_train = [label_of(r) for r in train_rows]
    y_dev = [label_of(r) for r in dev_rows]
    pairs_train = [(r["text_pair"][0], r["text_pair"][1]) for r in train_rows]
    pairs_dev = [(r["text_pair"][0], r["text_pair"][1]) for r in dev_rows]
    if len(pairs_train) != len(y_train) or len(pairs_dev) != len(y_dev):
        raise SystemExit("text_pair 与 target 数量不一致")
    log.info("数据：train %d 条，dev %d 条；train 标签 %s", len(train_rows), len(dev_rows),
             {int(k): y_train.count(int(k)) for k in (0, 1)})
    log.info("采样规则：%s", sampling_rule)

    env = collect_environment()
    write_json(os.path.join(args.runs_dir, "environment.json"), env)
    log.info("环境：Python %s；CPU %s；GPU %s", env.get("python", "").split()[0],
             env.get("cpu_name"), env.get("gpu"))

    # 1) 只用 train 拟合词表
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        analyzer=char_ngrams,
        min_df=args.min_df,
        max_features=args.max_features,
        sublinear_tf=True,
        dtype=np.float32,
        lowercase=False,
    )
    all_train_texts = [t for p in pairs_train for t in p]
    vectorizer.fit(all_train_texts)
    vectorizer_seconds = time.time() - t0
    log.info("词表拟合完成：%d 个字符 n-gram，用时 %.2fs", len(vectorizer.vocabulary_), vectorizer_seconds)
    append_timeline(timeline_path, "vectorizer_fit", vocab_size=len(vectorizer.vocabulary_),
                    seconds=round(vectorizer_seconds, 3))

    # 2) 构造句对特征矩阵（train 拟合 scaler）
    X_train, scaler, build_seconds = build_pair_matrix(vectorizer, pairs_train, scaler=None)
    X_dev, _, dev_build_seconds = build_pair_matrix(vectorizer, pairs_dev, scaler=scaler)
    log.info("特征矩阵：train shape=%s nnz=%d；dev shape=%s nnz=%d；构造 %.2fs + %.2fs",
             X_train.shape, X_train.nnz, X_dev.shape, X_dev.nnz, build_seconds, dev_build_seconds)
    append_timeline(timeline_path, "matrix_built", train_shape=list(X_train.shape),
                    train_nnz=int(X_train.nnz), dev_shape=list(X_dev.shape),
                    dev_nnz=int(X_dev.nnz), seconds=round(build_seconds + dev_build_seconds, 3))

    candidates_spec = [
        {"order": 0, "name": "c1_alpha1e-5_none", "alpha": 1e-5, "class_weight": None},
        {"order": 1, "name": "c2_alpha1e-5_balanced", "alpha": 1e-5, "class_weight": "balanced"},
        {"order": 2, "name": "c3_alpha1e-4_none", "alpha": 1e-4, "class_weight": None},
        {"order": 3, "name": "c4_alpha1e-4_balanced", "alpha": 1e-4, "class_weight": "balanced"},
    ]
    candidates = []
    best_clf = None
    best_metrics = None
    best_spec = None
    cand_path = os.path.join(args.runs_dir, "candidates.jsonl")
    with open(cand_path, "w", encoding="utf-8") as cf:
        for spec in candidates_spec:
            log.info("候选 %s：alpha=%s class_weight=%s", spec["name"], spec["alpha"], spec["class_weight"])
            t1 = time.time()
            clf = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=float(spec["alpha"]),
                class_weight=spec["class_weight"],
                max_iter=int(args.epochs),
                tol=1e-4,
                random_state=args.seed,
                average=False,
                learning_rate="optimal",
                shuffle=True,
                verbose=0,
            )
            clf.fit(X_train, y_train)
            fit_seconds = time.time() - t1
            prob = clf.predict_proba(X_dev)
            pred = np.argmax(prob, axis=1)
            met = metrics_dict(y_dev, pred.tolist(), prob)
            met.update({
                "candidate": spec["name"],
                "order": spec["order"],
                "alpha": spec["alpha"],
                "class_weight": spec["class_weight"],
                "epochs": int(args.epochs),
                "fit_seconds": round(fit_seconds, 3),
                "n_iter": int(np.asarray(clf.n_iter_).ravel()[-1]) if np.asarray(clf.n_iter_).size else None,
                "coef_l2": float(np.linalg.norm(clf.coef_)),
                "intercept": float(clf.intercept_[0]),
                "train_rows": len(train_rows),
                "n_features": int(X_train.shape[1]),
                "train_nnz": int(X_train.nnz),
            })
            candidates.append(met)
            cf.write(json.dumps(met, ensure_ascii=False) + "\n")
            cf.flush()
            log.info("候选 %s 完成：dev acc=%.4f macro-F1=%.4f logloss=%.4f，用时 %.1fs",
                     spec["name"], met["accuracy"], met["macro_f1"], met.get("log_loss", float("nan")),
                     fit_seconds)
            append_timeline(timeline_path, "candidate_fit", candidate=spec["name"],
                            dev_accuracy=met["accuracy"], dev_macro_f1=met["macro_f1"],
                            seconds=round(fit_seconds, 3))
            if (best_metrics is None
                    or (met["accuracy"], met["macro_f1"]) > (best_metrics["accuracy"], best_metrics["macro_f1"])):
                best_clf, best_metrics, best_spec = clf, met, spec
            else:
                del clf
                gc.collect()

    log.info("选模：按 dev accuracy 降序、macro-F1 降序、配置顺序兜底；选中 %s", best_spec["name"])
    log.info("选中配置 dev：accuracy=%.4f macro-F1=%.4f", best_metrics["accuracy"], best_metrics["macro_f1"])

    # 保存 sklearn 侧 dev 预测，供导出模型做“重载/纯标准库推理”一致性核对
    dev_prob = best_clf.predict_proba(X_dev)
    dev_score = best_clf.decision_function(X_dev)
    dev_pred_path = os.path.join(args.runs_dir, "dev1200_sklearn_predictions.jsonl")
    with open(dev_pred_path, "w", encoding="utf-8") as f:
        for row, p, s, pr in zip(dev_rows, dev_prob, dev_score, np.argmax(dev_prob, axis=1)):
            f.write(json.dumps({
                "item_id": row["item_id"],
                "label": int(pr),
                "score": float(s),
                "prob_paraphrase": float(p[1]),
            }, ensure_ascii=False) + "\n")
    log.info("sklearn 侧 dev 预测已保存：%s", dev_pred_path)

    # 3) 导出可携带模型
    data_hashes = {
        "data/train.jsonl": sha256_file(args.train),
        "data/splits/dev1200.jsonl": sha256_file(args.dev),
    }
    timings = {
        "vectorizer_seconds": round(vectorizer_seconds, 3),
        "feature_build_seconds": round(build_seconds, 3),
        "dev_feature_build_seconds": round(dev_build_seconds, 3),
        "candidate_fit_seconds": {c["candidate"]: c["fit_seconds"] for c in candidates},
        "total_train_seconds": round(time.time() - t_start, 3),
    }
    meta = export_model(
        args.out_dir, vectorizer, scaler, best_clf, best_spec,
        best_metrics, args, data_hashes, timings, env,
    )
    log.info("模型导出：%s（%d 维，%d 词表，sha256=%s）",
             args.out_dir, meta["n_features"], meta["vocab_size"], meta["model_sha256"])
    append_timeline(timeline_path, "model_exported", model_dir=args.out_dir,
                    model_sha256=meta["model_sha256"], n_features=meta["n_features"])

    selection = {
        "schema": "m3.trad.selection/1",
        "selected_at": now_iso(),
        "selection_rule": "dev1200 accuracy 降序，macro-F1 降序，配置顺序（c1→c4）兜底",
        "selected_config": best_spec,
        "dev_metrics": best_metrics,
        "candidates": [
            {k: c[k] for k in ("candidate", "order", "alpha", "class_weight",
                               "accuracy", "macro_f1", "log_loss",
                               "fit_seconds", "n_iter", "coef_l2")}
            for c in candidates
        ],
        "model": {
            "dir": os.path.relpath(args.out_dir, ROOT).replace("\\", "/"),
            "n_features": meta["n_features"],
            "vocab_size": meta["vocab_size"],
            "n_numeric": meta["n_numeric"],
            "model_sha256": meta["model_sha256"],
            "files": meta["files"],
        },
        "data": {
            "train": {"path": "data/train.jsonl", "n": len(train_rows), "sha256": data_hashes["data/train.jsonl"]},
            "dev": {"path": "data/splits/dev1200.jsonl", "n": len(dev_rows), "sha256": data_hashes["data/splits/dev1200.jsonl"]},
            "sampling_rule": sampling_rule,
        },
        "timings": timings,
        "environment": meta["environment"],
        "holdout_evaluated": False,
        "note": "selection.json 写入后模型不再改动；holdout746 由 eval_holdout_m3.py 只评一次。",
    }
    write_json(os.path.join(args.runs_dir, "selection.json"), selection)
    append_timeline(timeline_path, "selection_written", selected=best_spec["name"],
                    model_sha256=meta["model_sha256"])
    with open(os.path.join(args.runs_dir, "train_command.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join([sys.executable, *sys.argv]) + "\n")

    log.info("selection.json 已写入；holdout746 未读取。总用时 %.1fs", time.time() - t_start)
    append_timeline(timeline_path, "train_end", total_seconds=round(time.time() - t_start, 3))
    print(json.dumps({
        "selected": best_spec["name"],
        "dev_accuracy": best_metrics["accuracy"],
        "dev_macro_f1": best_metrics["macro_f1"],
        "model_sha256": meta["model_sha256"],
        "n_features": meta["n_features"],
        "train_rows": len(train_rows),
        "train_seconds": timings["total_train_seconds"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
