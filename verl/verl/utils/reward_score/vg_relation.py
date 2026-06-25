# Copyright 2026
#
# Rule-based VG scene-graph relation reward for verl custom_reward_function.

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
STRICT_RESPONSE_RE = re.compile(r"\s*<think>.*?</think>\s*<answer>\s*(.*?)\s*</answer>\s*", re.DOTALL)


def _norm_label(value: Any) -> str:
    return str(value).replace("_", " ").replace("-", " ").strip().lower()


def _category_from_obj(obj: dict[str, Any]) -> str:
    category = obj.get("category") or obj.get("name") or obj.get("label") or obj.get("class")
    if category is None:
        obj_id = obj.get("id", "")
        category = re.split(r"[#.]|\s+\d+$", str(obj_id), maxsplit=1)[0]
    return _norm_label(category)


def _get_box(obj: dict[str, Any]) -> list[float] | None:
    box = obj.get("box", obj.get("bbox"))
    box = _as_list(box)
    if box is None or len(box) != 4:
        return None
    try:
        box = [float(x) for x in box]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = box
    if x2 < x1 or y2 < y1:
        return None
    return box


def _convert_box_format(box: list[float], box_format: str) -> list[float]:
    if box_format == "yxyx":
        y1, x1, y2, x2 = box
        return [x1, y1, x2, y2]
    return box


def _as_list(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else None
    return None


def _iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def _extract_answer_json(text: str, require_think: bool) -> tuple[dict[str, Any] | None, bool]:
    pattern = STRICT_RESPONSE_RE if require_think else ANSWER_RE
    match = pattern.fullmatch(text)
    if not match:
        return None, False
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None, False
    if not isinstance(value, dict):
        return None, False
    return value, True


def _parse_ground_truth(ground_truth: Any, extra_info: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if ground_truth is None and extra_info:
        for key in ("ground_truth", "solution", "answer", "gt"):
            if key in extra_info:
                ground_truth = extra_info[key]
                break

    if isinstance(ground_truth, dict):
        if "objects" in ground_truth and "relationships" in ground_truth:
            return ground_truth
        if "answer" in ground_truth:
            return _parse_ground_truth(ground_truth["answer"], extra_info)

    if isinstance(ground_truth, str):
        text = ground_truth.strip()
        answer_match = ANSWER_RE.search(text)
        if answer_match:
            text = answer_match.group(1).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(value, dict) and "objects" in value and "relationships" in value:
            return value

    return None


def _normalize_graph(
    graph: dict[str, Any],
    *,
    box_format: str = "xyxy",
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]] | None:
    objects = graph.get("objects")
    relationships = graph.get("relationships")
    objects = _as_list(objects)
    relationships = _as_list(relationships)
    if objects is None or relationships is None:
        return None

    obj_by_id: dict[str, dict[str, Any]] = {}
    for obj in objects:
        if isinstance(obj, dict) and "id" in obj:
            box = _get_box(obj)
            if box is None:
                return None
            obj_id = _norm_label(obj["id"])
            obj_by_id[obj_id] = {
                "id": obj_id,
                "category": _category_from_obj(obj),
                "box": _convert_box_format(box, box_format),
            }
        elif isinstance(obj, (list, tuple)) and len(obj) == 2:
            obj_id_raw, box_raw = obj
            obj_id = _norm_label(obj_id_raw)
            box_list = _as_list(box_raw)
            if box_list is None or len(box_list) != 4:
                return None
            try:
                box = [float(x) for x in box_list]
            except (TypeError, ValueError):
                return None
            category = re.split(r"[#.]|\s+\d+$", str(obj_id_raw), maxsplit=1)[0]
            obj_by_id[obj_id] = {
                "id": obj_id,
                "category": _norm_label(category),
                "box": _convert_box_format(box, box_format),
            }
        else:
            return None

    normalized_rels: list[dict[str, str]] = []
    for rel in relationships:
        if isinstance(rel, dict) and all(key in rel for key in ("subject", "predicate", "object")):
            sub = _norm_label(rel["subject"])
            obj = _norm_label(rel["object"])
            pred = _norm_label(rel["predicate"])
        elif isinstance(rel, (list, tuple)) and len(rel) == 3:
            sub = _norm_label(rel[0])
            pred = _norm_label(rel[1])
            obj = _norm_label(rel[2])
        else:
            return None
        if sub not in obj_by_id or obj not in obj_by_id or not pred:
            return None
        normalized_rels.append({"subject": sub, "predicate": pred, "object": obj})

    return obj_by_id, normalized_rels


def _compute_object_reward(
    gt_objects: dict[str, dict[str, Any]],
    pred_objects: dict[str, dict[str, Any]],
    *,
    iou_threshold: float,
) -> dict[str, float]:
    if not gt_objects:
        return {"object_reward": 0.0, "matched_objects": 0.0, "num_gt_objects": 0.0, "num_pred_objects": 0.0}

    gt_list = list(gt_objects.values())
    pred_list = list(pred_objects.values())
    n_gt = len(gt_list)
    n_pred = len(pred_list)

    cost_matrix = np.zeros((n_gt, n_pred))
    for i, gt_obj in enumerate(gt_list):
        for j, pred_obj in enumerate(pred_list):
            if gt_obj["category"] == pred_obj["category"]:
                iou = _iou(gt_obj["box"], pred_obj["box"])
                if iou >= iou_threshold:
                    cost_matrix[i, j] = iou

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    total_score = 0.0
    matched_count = 0
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] >= iou_threshold:
            total_score += cost_matrix[r, c]
            matched_count += 1

    object_reward = total_score / n_gt
    return {
        "object_reward": object_reward,
        "matched_objects": float(matched_count),
        "num_gt_objects": float(n_gt),
        "num_pred_objects": float(n_pred),
    }


def _relation_matches(
    gt_rel: dict[str, str],
    pred_rel: dict[str, str],
    gt_objects: dict[str, dict[str, Any]],
    pred_objects: dict[str, dict[str, Any]],
    *,
    iou_threshold: float,
    require_object_category: bool,
) -> bool:
    if gt_rel["predicate"] != pred_rel["predicate"]:
        return False

    gt_sub = gt_objects[gt_rel["subject"]]
    gt_obj = gt_objects[gt_rel["object"]]
    pred_sub = pred_objects[pred_rel["subject"]]
    pred_obj = pred_objects[pred_rel["object"]]

    if require_object_category:
        if gt_sub["category"] != pred_sub["category"] or gt_obj["category"] != pred_obj["category"]:
            return False

    return _iou(gt_sub["box"], pred_sub["box"]) >= iou_threshold and _iou(gt_obj["box"], pred_obj["box"]) >= iou_threshold


def _compute_relation_metrics(
    gt_objects: dict[str, dict[str, Any]],
    gt_rels: list[dict[str, str]],
    pred_objects: dict[str, dict[str, Any]],
    pred_rels: list[dict[str, str]],
    *,
    iou_threshold: float,
    require_object_category: bool,
) -> dict[str, float]:
    if not gt_rels:
        return {"R": 0.0, "P": 0.0, "F1": 0.0, "mR": 0.0, "mP": 0.0, "mF1": 0.0,
                "matched": 0.0, "num_gt": 0.0}

    # Recall: for each GT rel, check if any pred matches
    gt_matched_by_pred: dict[str, list[bool]] = defaultdict(list)
    gt_matched_total = 0
    for gt_rel in gt_rels:
        matched = any(
            _relation_matches(
                gt_rel, pred_rel, gt_objects, pred_objects,
                iou_threshold=iou_threshold,
                require_object_category=require_object_category,
            )
            for pred_rel in pred_rels
        )
        gt_matched_total += int(matched)
        gt_matched_by_pred[gt_rel["predicate"]].append(matched)

    # Precision: for each pred rel, check if any GT matches
    pred_matched_by_pred: dict[str, list[bool]] = defaultdict(list)
    pred_matched_total = 0
    for pred_rel in pred_rels:
        matched = any(
            _relation_matches(
                gt_rel, pred_rel, gt_objects, pred_objects,
                iou_threshold=iou_threshold,
                require_object_category=require_object_category,
            )
            for gt_rel in gt_rels
        )
        pred_matched_total += int(matched)
        pred_matched_by_pred[pred_rel["predicate"]].append(matched)

    num_gt = len(gt_rels)
    num_pred = len(pred_rels)
    R = gt_matched_total / num_gt
    P = pred_matched_total / num_pred if num_pred > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

    # Per-predicate: collect all predicates from both GT and pred
    all_predicates = set(gt_matched_by_pred.keys()) | set(pred_matched_by_pred.keys())
    per_pred_f1s = []
    for pred_cat in all_predicates:
        gt_hits = gt_matched_by_pred.get(pred_cat, [])
        pred_hits = pred_matched_by_pred.get(pred_cat, [])
        r_c = (sum(gt_hits) / len(gt_hits)) if gt_hits else 0.0
        p_c = (sum(pred_hits) / len(pred_hits)) if pred_hits else 0.0
        f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
        per_pred_f1s.append(f1_c)

    mR = sum(sum(v) / len(v) for v in gt_matched_by_pred.values()) / len(gt_matched_by_pred)
    mP = (sum(sum(v) / len(v) for v in pred_matched_by_pred.values()) / len(pred_matched_by_pred)) if pred_matched_by_pred else 0.0
    mF1 = sum(per_pred_f1s) / len(per_pred_f1s) if per_pred_f1s else 0.0

    return {
        "R": R, "P": P, "F1": F1,
        "mR": mR, "mP": mP, "mF1": mF1,
        "matched": float(gt_matched_total), "num_gt": float(num_gt),
    }


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    *,
    iou_threshold: float = 0.5,
    f1_weight: float = 0.5,
    mf1_weight: float = 0.5,
    object_reward_weight: float = 1.0,
    relation_reward_weight: float = 4.0,
    require_think: bool = True,
    require_object_category: bool = True,
    box_format: str = "xyxy",
) -> dict[str, float]:
    """Compute combined object + relation reward.

    score = (object_reward_weight * object_reward + relation_reward_weight * relation_reward)
            / (object_reward_weight + relation_reward_weight)

    Object reward: for each GT object, find best matching pred (same category, IoU >= 0.5),
    score = IoU value (range 0.5~1.0). Average over all GT objects.

    Relation reward: 0.5 * F1 + 0.5 * mF1.

    The response must be exactly `<think>...</think><answer>{json}</answer>` when
    `require_think=True`. The answer JSON must contain `objects` and
    `relationships`; malformed outputs receive zero.
    """

    zero_result = {
        "score": 0.0,
        "format_reward": 0.0,
        "object_reward": 0.0,
        "matched_objects": 0.0,
        "num_gt_objects": 0.0,
        "num_pred_objects": 0.0,
        "relation_R": 0.0,
        "relation_P": 0.0,
        "relation_F1": 0.0,
        "relation_mR": 0.0,
        "relation_mP": 0.0,
        "relation_mF1": 0.0,
        "relation_reward": 0.0,
        "matched_relations": 0.0,
        "num_gt_relations": 0.0,
        "num_pred_relations": 0.0,
    }

    solution_str = solution_str or ""
    pred_graph, format_ok = _extract_answer_json(solution_str, require_think=require_think)
    if not format_ok or pred_graph is None:
        return zero_result

    gt_graph = _parse_ground_truth(ground_truth, extra_info)
    if gt_graph is None:
        result = dict(zero_result)
        result["format_reward"] = 1.0
        return result

    pred_norm = _normalize_graph(pred_graph, box_format=box_format)
    gt_norm = _normalize_graph(gt_graph, box_format=box_format)
    if pred_norm is None or gt_norm is None:
        return zero_result

    pred_objects, pred_rels = pred_norm
    gt_objects, gt_rels = gt_norm

    obj_metrics = _compute_object_reward(
        gt_objects, pred_objects, iou_threshold=iou_threshold,
    )

    rel_metrics = _compute_relation_metrics(
        gt_objects, gt_rels, pred_objects, pred_rels,
        iou_threshold=iou_threshold,
        require_object_category=require_object_category,
    )

    relation_reward = f1_weight * rel_metrics["F1"] + mf1_weight * rel_metrics["mF1"]
    object_reward = obj_metrics["object_reward"]
    total_weight = object_reward_weight + relation_reward_weight
    score = (object_reward_weight * object_reward + relation_reward_weight * relation_reward) / total_weight

    return {
        "score": score,
        "format_reward": 1.0,
        "object_reward": object_reward,
        "matched_objects": obj_metrics["matched_objects"],
        "num_gt_objects": obj_metrics["num_gt_objects"],
        "num_pred_objects": obj_metrics["num_pred_objects"],
        "relation_R": rel_metrics["R"],
        "relation_P": rel_metrics["P"],
        "relation_F1": rel_metrics["F1"],
        "relation_mR": rel_metrics["mR"],
        "relation_mP": rel_metrics["mP"],
        "relation_mF1": rel_metrics["mF1"],
        "relation_reward": relation_reward,
        "matched_relations": rel_metrics["matched"],
        "num_gt_relations": rel_metrics["num_gt"],
        "num_pred_relations": float(len(pred_rels)),
    }


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
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        )
    ]
