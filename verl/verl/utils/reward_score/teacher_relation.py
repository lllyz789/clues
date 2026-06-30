from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any

import numpy as np


PAIR_LINE_RE = re.compile(
    r"^\s*\((?P<subject>[^,]+),\s*(?P<object>[^)]+)\)\s*:\s*"
    r"(?P<evidence>.*?)\s+Type\s*:\s*(?P<relation_type>.*?)\s*"
    r"\.\s*Final\s+Predicate\s*:\s*(?P<predicate>.*?)\s*$",
    re.IGNORECASE,
)

EMBED_MODEL_PATH = os.environ.get(
    "TEACHER_RELATION_EMBED_MODEL_PATH",
    "/root/autodl-tmp/lyz/model/all-MiniLM-L6-v2",
)
EMBED_DEVICE = os.environ.get("TEACHER_RELATION_EMBED_DEVICE")
EMBED_MAX_LENGTH = int(os.environ.get("TEACHER_RELATION_EMBED_MAX_LENGTH", "128"))

FORMAT_REWARD_WEIGHT = 1.0
RELATION_REWARD_WEIGHT = 4.0


def _norm_id(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _norm_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def _norm_predicate(value: Any) -> str:
    return _norm_text(value)


def _parse_lines(text: str) -> list[dict[str, str]]:
    records = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        match = PAIR_LINE_RE.match(line)
        if not match:
            continue
        item = {key: value.strip() for key, value in match.groupdict().items()}
        if item["subject"] and item["object"] and item["evidence"] and item["predicate"]:
            records.append(item)
    return records


def _format_valid(text: str) -> float:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return 0.0
    for line in lines:
        match = PAIR_LINE_RE.match(line)
        if not match:
            return 0.0
        item = {key: value.strip() for key, value in match.groupdict().items()}
        if not (item["subject"] and item["object"] and item["evidence"] and item["predicate"]):
            return 0.0
    return 1.0


def _pair_key(record: dict[str, Any]) -> tuple[str, str]:
    return _norm_id(record.get("subject", "")), _norm_id(record.get("object", ""))


def _pair_key_from_pair(pair: dict[str, Any]) -> tuple[str, str] | None:
    subject = pair.get("subject") if isinstance(pair, dict) else None
    obj = pair.get("object") if isinstance(pair, dict) else None
    if not isinstance(subject, dict) or not isinstance(obj, dict):
        return None
    if subject.get("id") is None or obj.get("id") is None:
        return None
    return _norm_id(subject["id"]), _norm_id(obj["id"])


def _build_references(ground_truth: Any) -> list[dict[str, str]]:
    if isinstance(ground_truth, dict):
        raw_refs = ground_truth.get("references")
        if isinstance(raw_refs, list):
            refs = []
            for ref in raw_refs:
                if not isinstance(ref, dict):
                    continue
                subject = ref.get("subject")
                obj = ref.get("object")
                evidence = ref.get("evidence")
                predicate = ref.get("predicate")
                if subject is None or obj is None or evidence is None or predicate is None:
                    continue
                refs.append(
                    {
                        "subject": str(subject).strip(),
                        "object": str(obj).strip(),
                        "evidence": str(evidence).strip(),
                        "predicate": str(predicate).strip(),
                    }
                )
            if refs:
                return refs

        answer = ground_truth.get("answer")
        if isinstance(answer, str):
            refs = _parse_lines(answer)
            if refs:
                return refs

    if isinstance(ground_truth, str):
        return _parse_lines(ground_truth)

    return []


@lru_cache(maxsize=1)
def _get_encoder() -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(EMBED_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(EMBED_MODEL_PATH, local_files_only=True)
    model.to(device)
    model.eval()
    return tokenizer, model, device


@lru_cache(maxsize=32768)
def _embedding(text: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    text = _norm_text(text)
    if not text:
        return np.empty((0,), dtype=np.float32)

    tokenizer, model, device = _get_encoder()
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=EMBED_MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * attention_mask, dim=1)
        counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        pooled = summed / counts
        pooled = F.normalize(pooled.float(), p=2, dim=-1)

    return pooled[0].cpu().numpy().astype(np.float32)


def _semantic_similarity(a: str, b: str) -> float:
    emb_a = _embedding(a)
    emb_b = _embedding(b)
    if emb_a.size == 0 or emb_b.size == 0:
        return 0.0
    value = float(np.dot(emb_a, emb_b))
    if not math.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _best_unused_pair_score(
    ref: dict[str, str],
    candidates: list[dict[str, str]],
    used: set[int],
) -> tuple[float, float, float, int | None]:
    best = (0.0, 0.0, 0.0, None)
    for idx, pred in enumerate(candidates):
        if idx in used:
            continue
        if _norm_predicate(ref["predicate"]) != _norm_predicate(pred["predicate"]):
            continue
        evidence_sim = _semantic_similarity(ref["evidence"], pred["evidence"])
        predicate_sim = 1.0
        score = evidence_sim
        if best[3] is None or score > best[0]:
            best = (score, evidence_sim, predicate_sim, idx)
    return best


def _zero_result(error_code: float = 0.0, num_pairs: float = 0.0, format_valid: float = 0.0) -> dict[str, float]:
    format_reward = format_valid * FORMAT_REWARD_WEIGHT
    return {
        "score": format_reward,
        "format_reward": format_reward,
        "relation_reward": 0.0,
        "acc_reward": 0.0,
        "evidence_similarity": 0.0,
        "predicate_similarity": 0.0,
        "matched_pairs": 0.0,
        "num_pairs": num_pairs,
        "format_valid": format_valid,
        "error_code": error_code,
    }


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, float]:
    refs: list[dict[str, str]] = []
    format_valid = 0.0
    try:
        refs = _build_references(ground_truth)
        if not refs and extra_info:
            refs = _build_references(extra_info.get("ground_truth") or extra_info.get("answer"))
        if not refs:
            return _zero_result(error_code=1.0)

        format_valid = _format_valid(solution_str or "")
        predictions = _parse_lines(solution_str or "")
        if not predictions:
            return _zero_result(error_code=2.0, num_pairs=float(len(refs)))
        format_reward = format_valid * FORMAT_REWARD_WEIGHT

        pred_by_pair: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for pred in predictions:
            pred_by_pair[_pair_key(pred)].append(pred)

        pairs = ground_truth.get("pairs") if isinstance(ground_truth, dict) else None
        expected_keys = []
        if isinstance(pairs, list):
            expected_keys = [_pair_key_from_pair(pair) for pair in pairs]

        used_by_pair: dict[tuple[str, str], set[int]] = defaultdict(set)
        pair_scores = []
        evidence_scores = []
        predicate_scores = []
        matched = 0

        for i, ref in enumerate(refs):
            key = expected_keys[i] if i < len(expected_keys) and expected_keys[i] is not None else _pair_key(ref)
            candidates = pred_by_pair.get(key, [])
            score, evidence_sim, predicate_sim, used_idx = _best_unused_pair_score(
                ref,
                candidates,
                used_by_pair[key],
            )
            if used_idx is not None:
                used_by_pair[key].add(used_idx)
                matched += 1
            pair_scores.append(score)
            evidence_scores.append(evidence_sim)
            predicate_scores.append(predicate_sim)

        num_pairs = len(refs)
        acc_reward = float(np.mean(pair_scores)) if pair_scores else 0.0
        relation_reward = acc_reward * RELATION_REWARD_WEIGHT
        return {
            "score": format_reward + relation_reward,
            "format_reward": format_reward,
            "relation_reward": relation_reward,
            "acc_reward": acc_reward,
            "evidence_similarity": float(np.mean(evidence_scores)) if evidence_scores else 0.0,
            "predicate_similarity": float(np.mean(predicate_scores)) if predicate_scores else 0.0,
            "matched_pairs": float(matched),
            "num_pairs": float(num_pairs),
            "format_valid": format_valid,
            "error_code": 0.0,
        }
    except Exception:
        return _zero_result(error_code=3.0, num_pairs=float(len(refs)), format_valid=format_valid)


def compute_score_batched(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    return [
        compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )
        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources,
            solution_strs,
            ground_truths,
            extra_infos,
            strict=True,
        )
    ]
