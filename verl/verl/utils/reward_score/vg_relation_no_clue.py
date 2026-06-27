"""
R1-SGG-style reward function for VG scene graph generation, adapted for verl.

Reward components (matching R1-SGG):
  1. format_reward: checks <CATEGORY>...</CATEGORY><OBJECT>...</OBJECT><RELATION>...</RELATION> structure
  2. node_acc_reward: category semantic similarity via bipartite matching
  3. node_box_reward: GIoU + exp(-L1) of matched object boxes
  4. edge_reward: triplet matching after object bipartite assignment

Uses the local LFM embedding model for semantic similarity.
"""

from __future__ import annotations

import json
import math
import os
import re
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


RELATION_GROUP_KEYS = (
    "spatial_relations",
    "contact_relations",
    "possession_relations",
    "action_relations",
    "motion_relations",
)

FORMAT_REWARD_WEIGHT = 1.0
NODE_REWARD_WEIGHT = 2.0
EDGE_REWARD_WEIGHT = 5.0

SEM_WEIGHT = 1.0
IOU_WEIGHT = 2.0
BOX_L1_WEIGHT = 5.0

LFM_MODEL_PATH = os.environ.get("VG_RELATION_LFM_MODEL_PATH", "/root/autodl-tmp/lyz/model/LFM")
LFM_MAX_LENGTH = int(os.environ.get("VG_RELATION_LFM_MAX_LENGTH", "512"))
LFM_DEVICE = os.environ.get("VG_RELATION_LFM_DEVICE")


def _norm_label(value: Any) -> str:
    s = str(value).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _category_from_id(obj_id: str) -> str:
    return _norm_label(re.split(r"[#.]|\s+\d+$", str(obj_id), maxsplit=1)[0])


def _safe_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        bbox = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in bbox):
        return None
    return bbox


def _as_list(value: Any) -> list | None:
    return value if isinstance(value, list) else None


def _get_bbox_value(obj: dict) -> Any:
    return obj.get("bbox")


@lru_cache(maxsize=1)
def _get_lfm_encoder() -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(LFM_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        LFM_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        LFM_MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch_dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


@lru_cache(maxsize=8192)
def _lfm_embedding(label: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    label = _norm_label(label)
    if not label:
        return np.empty((0,), dtype=np.float32)

    tokenizer, model, device = _get_lfm_encoder()
    inputs = tokenizer(
        label,
        padding=True,
        truncation=True,
        max_length=LFM_MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0]
        embedding = F.normalize(embedding.float(), p=2, dim=-1)

    return embedding[0].cpu().numpy().astype(np.float32)


@lru_cache(maxsize=8192)
def semantic_similarity(a: str, b: str) -> float:
    emb_a = _lfm_embedding(a)
    emb_b = _lfm_embedding(b)
    if emb_a.size == 0 or emb_b.size == 0:
        return 0.0
    return float(np.clip(np.dot(emb_a, emb_b), 0.0, 1.0))


def compute_iou(box_a: list, box_b: list) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def compute_giou(box_a: list, box_b: list) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    iou = inter / union
    c_x1 = min(box_a[0], box_b[0])
    c_y1 = min(box_a[1], box_b[1])
    c_x2 = max(box_a[2], box_b[2])
    c_y2 = max(box_a[3], box_b[3])
    area_c = (c_x2 - c_x1) * (c_y2 - c_y1)
    if area_c <= 0:
        return iou
    return iou - (area_c - union) / area_c


def box_l1(box_a: list, box_b: list) -> float:
    return sum(abs(a - b) for a, b in zip(box_a, box_b))


def _cost_function(pred: dict, gt: dict) -> float:
    iou = compute_giou(pred["bbox"], gt["bbox"])
    sem = semantic_similarity(_category_from_id(pred["id"]), _category_from_id(gt["id"]))
    return SEM_WEIGHT * (1.0 - sem) + IOU_WEIGHT * (1.0 - iou) + BOX_L1_WEIGHT * box_l1(pred["bbox"], gt["bbox"])


def bi_match(gt_objs: list[dict], pred_objs: list[dict]) -> list[dict]:
    num_gt = len(gt_objs)
    num_pred = len(pred_objs)
    if num_gt == 0 or num_pred == 0:
        return []
    pad = max(0, num_gt - num_pred)
    cost_matrix = np.zeros((num_pred + pad, num_gt))
    for i, pred in enumerate(pred_objs):
        for j, gt in enumerate(gt_objs):
            cost_matrix[i, j] = _cost_function(pred, gt)
    if pad:
        cost_matrix[num_pred:, :] = 1e5
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return [
        {"groundtruth": gt_objs[c], "prediction": pred_objs[r], "cost": cost_matrix[r, c]}
        for r, c in zip(row_ind, col_ind)
        if r < num_pred
    ]


def _loads_tag_json(payload: str) -> Any:
    payload = (payload or "").strip()
    if not payload:
        return None
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_tag_json(text: str, tag: str) -> Any:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return _loads_tag_json(match.group(1))


def _flatten_structured_relations(rel_payload: Any) -> list:
    if rel_payload is None:
        return []
    if not isinstance(rel_payload, dict):
        return []

    if "relations" in rel_payload:
        inner = rel_payload.get("relations")
        if isinstance(inner, dict):
            return _flatten_structured_relations(inner)
        return []

    flattened = []
    for key in RELATION_GROUP_KEYS:
        items = rel_payload.get(key, [])
        if isinstance(items, list):
            flattened.extend(items)
    return flattened


def _extract_structured_graph_data(text: str) -> dict | None:
    """Parse structured <OBJECT>/<RELATION> tags into the internal graph schema."""
    category_payload = _extract_tag_json(text, "CATEGORY")
    object_payload = _extract_tag_json(text, "OBJECT")
    relation_payload = _extract_tag_json(text, "RELATION")
    if category_payload is None or not isinstance(object_payload, dict) or relation_payload is None:
        return None
    return {
        "objects": object_payload.get("objects", []),
        "relations": _flatten_structured_relations(relation_payload),
    }


def _parse_pred_graph(text: str) -> dict | None:
    """Parse model output into normalized graph with id/bbox dicts."""
    data = _extract_structured_graph_data(text)
    if data is None:
        return None
    if not isinstance(data, dict):
        return None
    if "objects" not in data or "relations" not in data:
        return None
    raw_objects = _as_list(data["objects"])
    raw_relations = _as_list(data["relations"])
    if raw_objects is None or raw_relations is None:
        return None

    objects = []
    for obj in raw_objects:
        if not isinstance(obj, dict) or "id" not in obj or "bbox" not in obj:
            return None
        bbox = _safe_bbox(_get_bbox_value(obj))
        if bbox is None:
            return None
        objects.append({"id": _norm_label(obj["id"]), "bbox": bbox})

    relations = []
    for rel in raw_relations:
        if not isinstance(rel, dict) or not all(k in rel for k in ("subject", "predicate", "object")):
            return None
        relations.append({
            "subject": _norm_label(rel["subject"]),
            "predicate": _norm_label(rel["predicate"]),
            "object": _norm_label(rel["object"]),
        })

    if not objects:
        return None
    return {"objects": objects, "relations": relations}


def _parse_gt_graph(ground_truth: Any) -> dict | None:
    """Parse ground truth into normalized graph with id/bbox dicts."""
    if isinstance(ground_truth, str):
        structured = _extract_structured_graph_data(ground_truth)
        if structured is None:
            return None
        ground_truth = structured
    if not isinstance(ground_truth, dict):
        return None
    if "objects" not in ground_truth or "relations" not in ground_truth:
        return None
    raw_objects = _as_list(ground_truth["objects"])
    raw_relations = _as_list(ground_truth["relations"])
    if raw_objects is None or raw_relations is None:
        return None

    objects = []
    for obj in raw_objects:
        if not isinstance(obj, dict) or "id" not in obj or "bbox" not in obj:
            return None
        bbox = _safe_bbox(_get_bbox_value(obj))
        if bbox is None:
            return None
        objects.append({"id": _norm_label(obj["id"]), "bbox": bbox})

    relations = []
    for rel in raw_relations:
        if not isinstance(rel, dict) or not all(k in rel for k in ("subject", "predicate", "object")):
            return None
        relations.append({
            "subject": _norm_label(rel["subject"]),
            "predicate": _norm_label(rel["predicate"]),
            "object": _norm_label(rel["object"]),
        })

    if not objects:
        return None
    return {"objects": objects, "relations": relations}


def _format_reward(text: str) -> float:
    text = text.strip()
    category_payload = _extract_tag_json(text, "CATEGORY")
    object_payload = _extract_tag_json(text, "OBJECT")
    relation_payload = _extract_tag_json(text, "RELATION")
    if category_payload is None and object_payload is None and relation_payload is None:
        return 0.0
    if not isinstance(category_payload, dict) or "categories" not in category_payload:
        return 0.5
    if not isinstance(object_payload, dict) or "objects" not in object_payload:
        return 0.5
    if not isinstance(relation_payload, dict):
        return 0.5
    if not isinstance(relation_payload.get("relations"), dict):
        return 0.5

    strict = re.fullmatch(
        re.compile(
            r"\s*<CATEGORY>.*?</CATEGORY>\s*<OBJECT>.*?</OBJECT>\s*<RELATION>.*?</RELATION>\s*",
            re.DOTALL | re.IGNORECASE,
        ),
        text,
    )
    return 1.0 if strict else 0.5


def _node_acc_reward(gt_graph: dict, pred_graph: dict) -> float:
    gt_objs = gt_graph["objects"]
    pred_objs = pred_graph["objects"]
    if not gt_objs or not pred_objs:
        return 0.0
    assignments = bi_match(gt_objs, pred_objs)
    reward = 0.0
    for assign in assignments:
        gt_id = assign["groundtruth"]["id"]
        pred_id = assign["prediction"]["id"]
        reward += semantic_similarity(_category_from_id(gt_id), _category_from_id(pred_id))
    precision = reward / len(pred_objs)
    recall = reward / len(gt_objs)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _node_box_reward(gt_graph: dict, pred_graph: dict) -> float:
    gt_objs = gt_graph["objects"]
    pred_objs = pred_graph["objects"]
    if not gt_objs:
        return 0.0
    assignments = bi_match(gt_objs, pred_objs)
    reward = 0.0
    for assign in assignments:
        gt_box = assign["groundtruth"]["bbox"]
        pred_box = assign["prediction"]["bbox"]
        iou_val = compute_iou(gt_box, pred_box)
        l1_val = np.exp(-box_l1(gt_box, pred_box))
        reward += (iou_val * IOU_WEIGHT + l1_val * BOX_L1_WEIGHT) / (IOU_WEIGHT + BOX_L1_WEIGHT)
    return reward / len(gt_objs)


def _edge_reward(gt_graph: dict, pred_graph: dict) -> float:
    gt_objs = gt_graph["objects"]
    gt_rels = gt_graph["relations"]
    pred_objs = pred_graph["objects"]
    pred_rels = pred_graph["relations"]
    if not gt_rels:
        return 0.0

    assignments = bi_match(gt_objs, pred_objs)
    map_obj = {}
    for assign in assignments:
        gt_id = assign["groundtruth"]["id"]
        pred_id = assign["prediction"]["id"]
        map_obj[gt_id] = pred_id

    pred_pair_to_predicates = {}
    for rel in pred_rels:
        key = (rel["subject"], rel["object"])
        pred_pair_to_predicates.setdefault(key, []).append(rel["predicate"])

    reward = 0.0
    for gt_rel in gt_rels:
        sub, obj = gt_rel["subject"], gt_rel["object"]
        if sub not in map_obj or obj not in map_obj:
            continue
        sub_mapped = map_obj[sub]
        obj_mapped = map_obj[obj]
        pred_predicates = pred_pair_to_predicates.get((sub_mapped, obj_mapped))
        if pred_predicates:
            best_predicate_score = max(
                semantic_similarity(gt_rel["predicate"], pred_pred)
                for pred_pred in pred_predicates
            )
            reward += (
                semantic_similarity(_category_from_id(sub), _category_from_id(sub_mapped))
                * semantic_similarity(_category_from_id(obj), _category_from_id(obj_mapped))
                * best_predicate_score
            )

    return reward / max(1, len(gt_rels))


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, float]:
    """Compute R1-SGG-style reward for verl.

    Returns a dict with 'score' as the primary reward, plus component breakdowns.
    """
    zero = {
        "score": 0.0,
        "format_reward": 0.0,
        "node_acc_reward": 0.0,
        "node_box_reward": 0.0,
        "edge_reward": 0.0,
    }

    try:
        solution_str = solution_str or ""
        fmt = _format_reward(solution_str)
        if fmt == 0.0:
            return zero

        partial = dict(zero)
        partial["format_reward"] = fmt * FORMAT_REWARD_WEIGHT
        partial["score"] = fmt * FORMAT_REWARD_WEIGHT / (
            FORMAT_REWARD_WEIGHT + NODE_REWARD_WEIGHT * 2 + EDGE_REWARD_WEIGHT
        )

        # Parse ground truth
        gt_source = ground_truth
        if gt_source is None and extra_info:
            gt_source = extra_info.get("ground_truth") or extra_info.get("ground_truth_text")
        gt_graph = _parse_gt_graph(gt_source)
        if gt_graph is None:
            return partial

        # Parse prediction
        pred_graph = _parse_pred_graph(solution_str)
        if pred_graph is None:
            return partial

        # Normalize boxes to [0, 1] range (both gt and pred are in 0-1000 scale)
        for obj in gt_graph["objects"]:
            obj["bbox"] = [x / 1000.0 for x in obj["bbox"]]
        for obj in pred_graph["objects"]:
            obj["bbox"] = [x / 1000.0 for x in obj["bbox"]]

        # Compute component rewards
        node_acc = _node_acc_reward(gt_graph, pred_graph)
        node_box = _node_box_reward(gt_graph, pred_graph)
        edge = _edge_reward(gt_graph, pred_graph)

        # Weighted combination
        score = (
            fmt * FORMAT_REWARD_WEIGHT
            + node_acc * NODE_REWARD_WEIGHT
            + node_box * NODE_REWARD_WEIGHT
            + edge * EDGE_REWARD_WEIGHT
        ) / (FORMAT_REWARD_WEIGHT + NODE_REWARD_WEIGHT * 2 + EDGE_REWARD_WEIGHT)

        return {
            "score": score,
            "format_reward": fmt * FORMAT_REWARD_WEIGHT,
            "node_acc_reward": node_acc * NODE_REWARD_WEIGHT,
            "node_box_reward": node_box * NODE_REWARD_WEIGHT,
            "edge_reward": edge * EDGE_REWARD_WEIGHT,
        }
    except Exception:
        return zero
