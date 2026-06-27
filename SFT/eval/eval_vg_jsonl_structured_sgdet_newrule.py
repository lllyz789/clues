#!/usr/bin/env python3
"""Evaluate VG150 sgdet predictions from structured <CATEGORY>/<OBJECT>/<RELATION> JSONL.

This variant parses the new prompt format:
- <CATEGORY>...{"categories":[{"id":"..."}]}...</CATEGORY>
- <OBJECT>...{"objects":[{"id":"cat.1","bbox":[x1,y1,x2,y2]}, ...]}...</OBJECT>
- <RELATION>...{"relations":{"spatial_relations":[...], "possession_relations":[...], "interaction_relations":[...]}}...</RELATION>

The parser flattens all relation groups into legacy triplets
[(subject, predicate, object), ...] before running the existing SGDET evaluator.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

import eval_vg_jsonl_sgdet as base


DEFAULT_PRED_JSONL = (
    "/root/autodl-tmp/lyz/output/erejin_sft/new_acl/sft_v2-20260626-234146_ckpt1758_v2_temp_0.1_test500_pred.jsonl"
)

# Conservative canonical aliases used only for this evaluator. Both predicted
# labels and GT labels are mapped through these tables before SGDET matching.
OBJECT_CANONICAL_ALIASES = {
    "airplane": "plane",
    "boy": "person",
    "child": "person",
    "girl": "person",
    "guy": "person",
    "kid": "person",
    "lady": "person",
    "man": "person",
    "men": "person",
    "people": "person",
    "player": "person",
    "skier": "person",
    "woman": "person",
    "boot": "shoe",
    "sneaker": "shoe",
    "cap": "hat",
    "jean": "pant",
    "short": "pant",
}

PREDICATE_CANONICAL_ALIASES = {
    "wears": "wearing",
    "laying on": "lying on",
}


def _loads_jsonish(payload: str):
    payload = re.sub(r"^```(?:json)?|```$", "", (payload or "").strip(), flags=re.I).strip()
    if not payload:
        return None
    try:
        return json.loads(payload)
    except Exception:
        try:
            return ast.literal_eval(payload)
        except Exception:
            return None


def _extract_tag_json(text: str, tag: str):
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", flags=re.S | re.I)
    if not match:
        return None
    return _loads_jsonish(match.group(1))


def _flatten_structured_relations(rel_payload):
    """Flatten grouped relationships into a flat list of [subject, predicate, object] triples."""
    if rel_payload is None:
        return []
    if isinstance(rel_payload, list):
        return rel_payload
    if not isinstance(rel_payload, dict):
        return []

    flattened = []
    for key in (
        "spatial_relations",
        "possession_relations",
        "interaction_relations",
        "interactive_relations",
        "contact_relations",
        "action_relations",
        "motion_relations",
        "spatial",
        "possession",
        "interaction",
        "contact",
        "action",
        "motion",
    ):
        items = rel_payload.get(key, [])
        if isinstance(items, list):
            flattened.extend(items)

    if not flattened:
        inner = rel_payload.get("relations", None)
        if isinstance(inner, dict):
            return _flatten_structured_relations(inner)
        if isinstance(inner, list):
            return inner

    return flattened


def extract_structured_answer(text: str):
    """Parse <answer> or <OBJECT>/<RELATION> tags into the legacy answer schema."""
    answer_payload = _extract_tag_json(text, "answer")
    if isinstance(answer_payload, dict) and "objects" in answer_payload:
        objects = answer_payload.get("objects", [])
        relationships = _flatten_structured_relations(answer_payload.get("relationships"))
        return {
            "objects": objects,
            "relationships": relationships,
        }

    object_payload = _extract_tag_json(text, "OBJECT")
    relation_payload = _extract_tag_json(text, "RELATION")
    if object_payload is None and relation_payload is None:
        return None

    if not isinstance(object_payload, dict):
        return None

    objects = object_payload.get("objects", [])
    relationships = _flatten_structured_relations(relation_payload)

    return {
        "objects": objects,
        "relationships": relationships,
    }


def parse_object_image_id(record):
    images = record.get("images") or []
    if not images:
        return None
    path = images[0].get("path", "")
    matches = re.findall(r"(\d+)\.jpg", path)
    if matches:
        return int(matches[-1])
    return None


def build_index_aliases(ind_to_names, canonical_aliases):
    name_to_idx = {base.normalize_text(name): idx for idx, name in enumerate(ind_to_names)}
    idx_aliases = {idx: idx for idx in range(len(ind_to_names))}
    for source, target in canonical_aliases.items():
        source_idx = name_to_idx.get(base.normalize_text(source))
        target_idx = name_to_idx.get(base.normalize_text(target))
        if source_idx is not None and target_idx is not None:
            idx_aliases[source_idx] = target_idx
    return idx_aliases


def canonicalize_predicate_name(name, pred_to_idx):
    canonical = PREDICATE_CANONICAL_ALIASES.get(base.normalize_text(name), name)
    return canonical if canonical in pred_to_idx else name


def convert_pred_box(box, pred_box_scale, image_info, coord_mode):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0.0, pred_box_scale)
    if x2 <= x1 or y2 <= y1:
        return None

    if coord_mode == "square":
        scale_x = base.GT_BOX_SCALE / float(pred_box_scale)
        scale_y = scale_x
    elif coord_mode == "image":
        if image_info is None:
            return None
        width = float(image_info["width"])
        height = float(image_info["height"])
        max_dim = max(width, height)
        scale_x = (width / max_dim) * base.GT_BOX_SCALE / float(pred_box_scale)
        scale_y = (height / max_dim) * base.GT_BOX_SCALE / float(pred_box_scale)
    else:
        raise ValueError(f"unknown coord_mode: {coord_mode}")
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


class StanfordFilteredVGDataset:
    def __init__(self, vg_dir, image_ids, split_id=0, object_idx_aliases=None, predicate_idx_aliases=None):
        self.vg_dir = vg_dir
        self.ind_to_classes, self.ind_to_predicates = base.build_ind_to_names(vg_dir)
        self.object_idx_aliases = object_idx_aliases or {}
        self.predicate_idx_aliases = predicate_idx_aliases or {}
        self.ids = []
        self.annotations = []

        image_data = base.load_filtered_image_data(vg_dir)
        image_id_to_info = {int(item["image_id"]): item for item in image_data}
        h5_path = os.path.join(vg_dir, "VG-SGG.h5")
        with h5py.File(h5_path, "r") as h5:
            if len(image_data) != h5["split"].shape[0]:
                raise RuntimeError(
                    f"image_data/h5 length mismatch: {len(image_data)} vs {h5['split'].shape[0]}"
                )

            image_id_to_h5_index = {int(item["image_id"]): idx for idx, item in enumerate(image_data)}
            all_boxes = base.cxcywh_to_xyxy(h5["boxes_1024"][:])
            all_labels = h5["labels"][:, 0].astype(np.int64)
            all_relationships = h5["relationships"][:].astype(np.int64)
            all_predicates = h5["predicates"][:, 0].astype(np.int64)
            split = h5["split"][:]

            for image_id in image_ids:
                image_id = int(image_id)
                h5_index = image_id_to_h5_index.get(image_id)
                if h5_index is None or split[h5_index] != split_id:
                    continue

                first_box = int(h5["img_to_first_box"][h5_index])
                last_box = int(h5["img_to_last_box"][h5_index])
                if first_box < 0 or last_box < first_box:
                    continue

                image_info = image_id_to_info[image_id]
                boxes = all_boxes[first_box : last_box + 1]
                if boxes.shape[0] > 0:
                    # keep GT evaluation in the same 1024-max-dim space as base script
                    boxes = boxes.astype(np.float32, copy=True)

                labels = all_labels[first_box : last_box + 1]
                relation_labels = np.asarray(
                    [self.object_idx_aliases.get(int(label), int(label)) for label in labels],
                    dtype=np.int64,
                )

                first_rel = int(h5["img_to_first_rel"][h5_index])
                last_rel = int(h5["img_to_last_rel"][h5_index])
                if first_rel >= 0 and last_rel >= first_rel:
                    rel_obj_idx = all_relationships[first_rel : last_rel + 1] - first_box
                    predicates = all_predicates[first_rel : last_rel + 1]
                    predicates = np.asarray(
                        [self.predicate_idx_aliases.get(int(pred), int(pred)) for pred in predicates],
                        dtype=np.int64,
                    )
                    edges = np.column_stack((rel_obj_idx, predicates)).astype(np.int64)
                else:
                    edges = np.zeros((0, 3), dtype=np.int64)

                self.ids.append(image_id)
                self.annotations.append(
                    {
                        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                        "labels": torch.as_tensor(labels, dtype=torch.int64),
                        "relation_labels": torch.as_tensor(relation_labels, dtype=torch.int64),
                        "edges": torch.as_tensor(edges, dtype=torch.int64),
                    }
                )

    def get_groundtruth(self, index):
        ann = self.annotations[index]
        return ann["boxes"], ann.get("relation_labels", ann["labels"]), ann["edges"]

    @property
    def coco(self):
        if not hasattr(self, "_coco") or self._coco is None:
            from pycocotools.coco import COCO as _COCO

            _coco = _COCO()
            images = [{"id": img_id} for img_id in self.ids]
            categories = [
                {"id": idx, "name": name}
                for idx, name in enumerate(self.ind_to_classes)
                if name != "__background__"
            ]
            annotations = []
            ann_id = 0
            for img_id, ann in zip(self.ids, self.annotations):
                boxes = ann["boxes"].numpy()
                labels = ann["labels"].numpy()
                for box, cls in zip(boxes, labels):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": int(cls),
                            "bbox": [float(x1), float(y1), float(w), float(h)],
                            "area": float(w * h),
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1
            _coco.dataset = {"images": images, "annotations": annotations, "categories": categories}
            _coco.createIndex()
            self._coco = _coco
        return self._coco


def build_prediction(
    record,
    class_to_idx,
    pred_to_idx,
    synonym_map,
    pred_box_scale,
    image_info_map,
    coord_mode,
    object_idx_aliases,
):
    stats = base.Counter()
    image_id = parse_object_image_id(record)
    if image_id is None:
        stats["missing_image_id"] += 1
        return None, stats
    image_info = image_info_map.get(image_id)

    answer = extract_structured_answer(record.get("response", ""))
    if not isinstance(answer, dict):
        stats["bad_answer_json"] += 1
        return image_id, stats

    objects = answer.get("objects", [])
    relationships = answer.get("relationships", [])
    if not isinstance(objects, list) or not isinstance(relationships, list):
        stats["bad_answer_schema"] += 1
        return image_id, stats

    pred_boxes = []
    pred_labels = []
    pred_relation_labels = []
    id_to_index = {}
    for obj in objects:
        if isinstance(obj, (list, tuple)) and len(obj) == 2:
            obj_id = str(obj[0]).strip()
            category = base.strip_instance_id(obj_id)
            label = class_to_idx.get(category)
            box = convert_pred_box(obj[1], pred_box_scale, image_info, coord_mode)
        elif isinstance(obj, dict):
            obj_id = str(obj.get("id", "")).strip()
            category = base.strip_instance_id(obj.get("category") or obj_id)
            label = class_to_idx.get(category)
            box = convert_pred_box(obj.get("box") or obj.get("bbox"), pred_box_scale, image_info, coord_mode)
        else:
            stats["bad_object"] += 1
            continue
        if not obj_id:
            stats["missing_object_id"] += 1
            continue
        if label is None:
            stats["unknown_object"] += 1
            continue
        if box is None:
            stats["bad_box"] += 1
            continue
        if obj_id in id_to_index:
            stats["duplicate_object_id"] += 1
            continue

        id_to_index[obj_id] = len(pred_boxes)
        pred_boxes.append(box)
        pred_labels.append(label)
        pred_relation_labels.append(object_idx_aliases.get(int(label), int(label)))

    relation_rows = []
    for rel_order, rel in enumerate(relationships):
        if isinstance(rel, (list, tuple)) and len(rel) == 3:
            subject_id = str(rel[0]).strip()
            predicate_raw = str(rel[1]).strip()
            object_id = str(rel[2]).strip()
        elif isinstance(rel, dict):
            subject_id = str(rel.get("subject", "")).strip()
            predicate_raw = str(rel.get("predicate", "")).strip()
            object_id = str(rel.get("object", "")).strip()
        else:
            stats["bad_relationship"] += 1
            continue

        subj_idx = id_to_index.get(subject_id)
        obj_idx = id_to_index.get(object_id)
        if subj_idx is None or obj_idx is None:
            stats["relationship_missing_object"] += 1
            continue
        if subj_idx == obj_idx:
            stats["self_relationship"] += 1
            continue

        pred_name = base.normalize_predicate_name(predicate_raw, pred_to_idx, synonym_map)
        if pred_name is None:
            stats["unknown_predicate"] += 1
            continue
        pred_name = canonicalize_predicate_name(pred_name, pred_to_idx)
        relation_rows.append((rel_order, subj_idx, obj_idx, pred_to_idx[pred_name]))

    if not pred_boxes:
        stats["empty_objects_after_filter"] += 1
        return image_id, base.with_stats(base.dummy_prediction(len(pred_to_idx)), stats)

    if not relation_rows:
        stats["empty_relationships_after_filter"] += 1
        pred = {
            "boxes": torch.as_tensor(pred_boxes, dtype=torch.float32),
            "labels": torch.as_tensor(pred_labels, dtype=torch.int64),
            "scores": torch.ones((len(pred_boxes),), dtype=torch.float32),
            "graph": {
                "all_node_pairs": torch.zeros((0, 2), dtype=torch.int64),
                "all_relation": torch.zeros((0, len(pred_to_idx)), dtype=torch.float32),
                "pred_boxes": torch.as_tensor(pred_boxes, dtype=torch.float32),
                "pred_boxes_class": torch.as_tensor(pred_relation_labels, dtype=torch.int64),
                "pred_boxes_score": torch.ones((len(pred_boxes),), dtype=torch.float32),
            },
        }
        return image_id, base.with_stats(pred, stats)

    all_node_pairs = []
    all_relation = np.zeros((len(relation_rows), len(pred_to_idx)), dtype=np.float32)
    for row_idx, (_, subj_idx, obj_idx, pred_idx) in enumerate(relation_rows):
        all_node_pairs.append((subj_idx, obj_idx))
        all_relation[row_idx, pred_idx] = 1.0

    pred = {
        "boxes": torch.as_tensor(pred_boxes, dtype=torch.float32),
        "labels": torch.as_tensor(pred_labels, dtype=torch.int64),
        "scores": torch.ones((len(pred_boxes),), dtype=torch.float32),
        "graph": {
            "all_node_pairs": torch.as_tensor(all_node_pairs, dtype=torch.int64),
            "all_relation": torch.as_tensor(all_relation, dtype=torch.float32),
            "pred_boxes": torch.as_tensor(pred_boxes, dtype=torch.float32),
            "pred_boxes_class": torch.as_tensor(pred_relation_labels, dtype=torch.int64),
            "pred_boxes_score": torch.ones((len(pred_boxes),), dtype=torch.float32),
        },
    }
    stats["kept_objects"] += len(pred_boxes)
    stats["kept_relation_rows"] += len(relation_rows)
    return image_id, base.with_stats(pred, stats)


def load_predictions(args, class_to_idx, pred_to_idx, synonym_map, object_idx_aliases):
    predictions = OrderedDict()
    stats = base.Counter()
    image_info_map = base.build_image_info_map(args.vg_dir)
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(tqdm(f, desc="parse predictions")):
            if args.limit is not None and line_idx >= args.limit:
                break
            stats["lines"] += 1
            try:
                record = json.loads(line)
            except Exception:
                stats["bad_jsonl_line"] += 1
                continue
            image_id, pred_or_stats = build_prediction(
                record,
                class_to_idx,
                pred_to_idx,
                synonym_map,
                args.pred_box_scale,
                image_info_map,
                args.coord_mode,
                object_idx_aliases,
            )
            if image_id is None:
                stats.update(pred_or_stats)
                continue
            if isinstance(pred_or_stats, base.Counter):
                stats.update(pred_or_stats)
                predictions[image_id] = None
                continue
            if image_id in predictions:
                stats["duplicate_image_prediction"] += 1
                continue
            predictions[image_id] = pred_or_stats
            stats.update(pred_or_stats.get("_stats", base.Counter()))
            pred_or_stats.pop("_stats", None)
    return predictions, stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-jsonl", default=DEFAULT_PRED_JSONL)
    parser.add_argument("--vg-dir", default=base.DEFAULT_VG_DIR)
    parser.add_argument("--r1-root", default=base.DEFAULT_R1_ROOT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pred-box-scale", type=float, default=1000.0)
    parser.add_argument(
        "--coord-mode",
        choices=("image", "square"),
        default="image",
        help=(
            "image: x/y are normalized by image width/height, then mapped to VG 1024 max-dim; "
            "square: x/y are directly scaled from 1000 to 1024."
        ),
    )
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-synonyms", action="store_true")
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="VG split to evaluate: train=0, val=1, test=2.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.pred_jsonl = os.path.abspath(args.pred_jsonl)
    args.vg_dir = os.path.abspath(args.vg_dir)
    args.r1_root = os.path.abspath(args.r1_root)

    ind_to_classes, ind_to_predicates = base.build_ind_to_names(args.vg_dir)
    class_to_idx = {base.normalize_text(name): idx for idx, name in enumerate(ind_to_classes)}
    class_to_idx.pop("__background__", None)
    pred_to_idx = {base.normalize_text(name): idx for idx, name in enumerate(ind_to_predicates)}
    valid_predicates = set(pred_to_idx.keys()) - {"__background__"}
    synonym_map = {} if args.no_synonyms else base.load_predicate_synonyms(args.r1_root, valid_predicates)
    object_idx_aliases = build_index_aliases(ind_to_classes, OBJECT_CANONICAL_ALIASES)
    predicate_idx_aliases = build_index_aliases(ind_to_predicates, PREDICATE_CANONICAL_ALIASES)

    predictions, parse_stats = load_predictions(args, class_to_idx, pred_to_idx, synonym_map, object_idx_aliases)
    if not predictions:
        raise RuntimeError("No valid predictions were parsed.")

    split_to_id = {"train": 0, "val": 1, "test": 2}
    dataset = StanfordFilteredVGDataset(
        args.vg_dir,
        predictions.keys(),
        split_to_id[args.split],
        object_idx_aliases=object_idx_aliases,
        predicate_idx_aliases=predicate_idx_aliases,
    )
    predictions = OrderedDict((image_id, predictions.get(image_id)) for image_id in dataset.ids)
    patched_empty = base.patch_empty_graphs_for_zero_recall(predictions, dataset, len(ind_to_predicates))

    sys.path.insert(0, args.r1_root)
    os.chdir(args.r1_root)
    base.install_lightweight_import_fallbacks()
    from src.utils.sgg_eval import SggEvaluator

    evaluator = SggEvaluator(
        dataset,
        iou_types=("bbox", "relation"),
        mode="sgdet",
        num_rel_category=len(ind_to_predicates),
        iou_thres=args.iou_thres,
        num_workers=args.num_workers,
    )
    evaluator.update(predictions)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    result_dict = evaluator.sgg_result_dict
    metrics = base.collect_metrics(result_dict, "sgdet")

    if "bbox" in evaluator.coco_eval and len(evaluator.coco_eval["bbox"].stats) > 0:
        coco_stats = evaluator.coco_eval["bbox"].stats
        metrics["bbox_AP"] = float(coco_stats[0] * 100)
        metrics["bbox_AP50"] = float(coco_stats[1] * 100)
    else:
        metrics["bbox_AP"] = float("nan")
        metrics["bbox_AP50"] = float("nan")

    pred_triplet_counts = []
    for img_id in dataset.ids:
        pred = predictions.get(img_id)
        if pred is not None and "graph" in pred:
            pred_triplet_counts.append(int(pred["graph"]["all_node_pairs"].shape[0]))
        else:
            pred_triplet_counts.append(0)
    avg_pred_triplets = float(np.mean(pred_triplet_counts)) if pred_triplet_counts else 0.0

    gt_rel_images = sum(int(ann["edges"].shape[0] > 0) for ann in dataset.annotations)
    total_gt_rels = sum(int(ann["edges"].shape[0]) for ann in dataset.annotations)
    summary = {
        "prediction_file": args.pred_jsonl,
        "vg_dir": args.vg_dir,
        "box_format": "prediction [x1,y1,x2,y2]",
        "prediction_box_scale": args.pred_box_scale,
        "prediction_coord_mode": args.coord_mode,
        "response_parse_format": (
            "primary: <OBJECT>{objects:[[id,[x1,y1,x2,y2]],...]} plus "
            "<RELATION>{relations:{spatial_relations:[...], possession_relations:[...], interaction_relations:[...]}}</RELATION>"
        ),
        "gt_box_scale": base.GT_BOX_SCALE,
        "mode": "sgdet",
        "iou_thres": args.iou_thres,
        "num_prediction_lines": int(parse_stats["lines"]),
        "num_eval_images": len(dataset.ids),
        "num_eval_images_with_gt_rels": gt_rel_images,
        "num_gt_rels": total_gt_rels,
        "avg_pred_triplets_per_image": avg_pred_triplets,
        "num_empty_or_missing_predictions_patched_as_zero_recall": patched_empty,
        "parse_stats": dict(parse_stats),
        "object_canonical_aliases": OBJECT_CANONICAL_ALIASES,
        "predicate_canonical_aliases": PREDICATE_CANONICAL_ALIASES,
        "metrics": metrics,
    }

    output = args.output
    if output is None:
        output = str(Path(args.pred_jsonl).with_suffix("")) + "_structured_newrule_sgdet_eval.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSGDET R/mR using structured new-rule tags")
    print(
        f"eval images: {len(dataset.ids)}  images with gt rels: {gt_rel_images}  "
        f"gt rels: {total_gt_rels}  avg pred triplets/image: {avg_pred_triplets:.2f}"
    )
    print(f"prediction coord mode: {args.coord_mode}  box scale -> gt scale: {args.pred_box_scale:g} -> {base.GT_BOX_SCALE:g}")
    print("repeated subject-object pairs are kept as separate relation rows")
    for k in (20, 50, 100):
        print(
            f"@{k}: "
            f"R={metrics[f'R@{k}']*100:.2f}  "
            f"mR={metrics[f'mR@{k}']*100:.2f}  "
            f"ng-R={metrics[f'ng-R@{k}']*100:.2f}  "
            f"ng-mR={metrics[f'ng-mR@{k}']*100:.2f}"
        )
    print(f"bbox_AP={metrics['bbox_AP']:.2f}  bbox_AP50={metrics['bbox_AP50']:.2f}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
