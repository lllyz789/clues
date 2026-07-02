#!/usr/bin/env python3
"""Evaluate VG150 sgdet predictions stored as VLLM JSONL.

The prediction boxes are interpreted as [x1, y1, x2, y2] in a 0-1000 range.
By default x/y are normalized by the image width/height respectively, then
mapped to the Stanford VG h5 1024-max-dimension coordinate range.
"""

import argparse
import ast
import json
import os
import re
import sys
import types
from importlib.machinery import ModuleSpec
from collections import Counter, OrderedDict
from pathlib import Path
import h5py
import numpy as np
import torch
from tqdm import tqdm


DEFAULT_PRED_JSONL = (
#    "/root/autodl-tmp/lyz/output/erejin_sft/new_prompt-20260623-012658_ckpt3514_v2_test500_pred.jsonl"
    "/root/autodl-tmp/lyz/output/erejin_sft/no_clue_acl-20260622-160320_ckpt1757_v2_test500_pred.jsonl"
)
DEFAULT_VG_DIR = "/root/autodl-tmp/lyz/dataset/vg/vg_data/stanford_filtered"
DEFAULT_R1_ROOT = "/root/autodl-tmp/lyz/R1-SGG-master"
GT_BOX_SCALE = 1024.0
CORRUPTED_IMAGES = {"1592.jpg", "1722.jpg", "4616.jpg", "4617.jpg"}


def normalize_text(value):
    value = str(value).strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value


def strip_instance_id(value):
    return normalize_text(str(value).split("#", 1)[0].split(".", 1)[0])


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_ind_to_names(vg_dir):
    data = read_json(os.path.join(vg_dir, "VG-SGG-dicts.json"))
    ind_to_classes = ["__background__"]
    for i in range(1, len(data["idx_to_label"]) + 1):
        ind_to_classes.append(data["idx_to_label"][str(i)])

    ind_to_predicates = ["__background__"]
    for i in range(1, len(data["idx_to_predicate"]) + 1):
        ind_to_predicates.append(data["idx_to_predicate"][str(i)])

    return ind_to_classes, ind_to_predicates


def load_filtered_image_data(vg_dir):
    image_data = read_json(os.path.join(vg_dir, "image_data.json"))
    filtered = []
    for item in image_data:
        filename = f"{item['image_id']}.jpg"
        if filename in CORRUPTED_IMAGES:
            continue
        filtered.append(item)
    return filtered


def build_image_info_map(vg_dir):
    return {int(item["image_id"]): item for item in load_filtered_image_data(vg_dir)}


def cxcywh_to_xyxy(boxes):
    boxes = boxes.astype(np.float32, copy=True)
    boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2.0
    boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
    boxes = np.clip(boxes, 0.0, GT_BOX_SCALE)
    return boxes


class StanfordFilteredVGDataset:
    def __init__(self, vg_dir, image_ids, split_id=0):
        self.vg_dir = vg_dir
        self.ind_to_classes, self.ind_to_predicates = build_ind_to_names(vg_dir)
        self.ids = []
        self.annotations = []

        image_data = load_filtered_image_data(vg_dir)
        h5_path = os.path.join(vg_dir, "VG-SGG.h5")
        with h5py.File(h5_path, "r") as h5:
            if len(image_data) != h5["split"].shape[0]:
                raise RuntimeError(
                    f"image_data/h5 length mismatch: {len(image_data)} vs {h5['split'].shape[0]}"
                )

            image_id_to_h5_index = {
                int(item["image_id"]): idx for idx, item in enumerate(image_data)
            }
            all_boxes = cxcywh_to_xyxy(h5["boxes_1024"][:])
            all_labels = h5["labels"][:, 0].astype(np.int64)
            all_relationships = h5["relationships"][:].astype(np.int64)
            all_predicates = h5["predicates"][:, 0].astype(np.int64)
            split = h5["split"][:]

            for image_id in image_ids:
                h5_index = image_id_to_h5_index.get(int(image_id))
                if h5_index is None or split[h5_index] != split_id:
                    continue

                first_box = int(h5["img_to_first_box"][h5_index])
                last_box = int(h5["img_to_last_box"][h5_index])
                if first_box < 0 or last_box < first_box:
                    continue

                boxes = all_boxes[first_box : last_box + 1]
                labels = all_labels[first_box : last_box + 1]

                first_rel = int(h5["img_to_first_rel"][h5_index])
                last_rel = int(h5["img_to_last_rel"][h5_index])
                if first_rel >= 0 and last_rel >= first_rel:
                    rel_obj_idx = all_relationships[first_rel : last_rel + 1] - first_box
                    predicates = all_predicates[first_rel : last_rel + 1]
                    edges = np.column_stack((rel_obj_idx, predicates)).astype(np.int64)
                else:
                    edges = np.zeros((0, 3), dtype=np.int64)

                self.ids.append(int(image_id))
                self.annotations.append(
                    {
                        "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                        "labels": torch.as_tensor(labels, dtype=torch.int64),
                        "edges": torch.as_tensor(edges, dtype=torch.int64),
                    }
                )

    def get_groundtruth(self, index):
        ann = self.annotations[index]
        return ann["boxes"], ann["labels"], ann["edges"]

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
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(cls),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "area": float(w * h),
                        "iscrowd": 0,
                    })
                    ann_id += 1
            _coco.dataset = {"images": images, "annotations": annotations, "categories": categories}
            _coco.createIndex()
            self._coco = _coco
        return self._coco


def load_predicate_synonyms(r1_root, valid_predicates):
    synonym_path = os.path.join(r1_root, "src", "vg_synonyms.py")
    mapping = {}
    if not os.path.exists(synonym_path):
        return mapping

    text = Path(synonym_path).read_text(encoding="utf-8")
    match = re.search(r"rel2vg_list\s*=\s*(\[.*?\])\s*\n", text, flags=re.S)
    if not match:
        return mapping

    try:
        rel2vg_list = ast.literal_eval(match.group(1))
    except Exception:
        return mapping

    for item in rel2vg_list:
        source = normalize_text(item.get("source", ""))
        target = normalize_text(item.get("target", ""))
        if source and target in valid_predicates:
            mapping[source] = target
    return mapping


def normalize_predicate_name(name, pred_to_idx, synonym_map):
    name = normalize_text(name)
    if name in pred_to_idx:
        return name
    if name in synonym_map and synonym_map[name] in pred_to_idx:
        return synonym_map[name]

    candidates = []
    for prefix in ("is ", "are ", "a ", "an ", "the "):
        if name.startswith(prefix):
            candidates.append(name[len(prefix) :])
    if name.endswith(" on top of"):
        candidates.append("on")
    if name == "next to":
        candidates.append("near")

    for cand in candidates:
        cand = normalize_text(cand)
        if cand in pred_to_idx:
            return cand
        if cand in synonym_map and synonym_map[cand] in pred_to_idx:
            return synonym_map[cand]
    return None


def extract_answer_json(text):
    text = text or ""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S | re.I)
    payload = match.group(1) if match else text
    payload = re.sub(r"^```(?:json)?|```$", "", payload.strip(), flags=re.I).strip()

    for candidate in (payload,):
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return ast.literal_eval(candidate)
            except Exception:
                pass

    start = payload.find("{")
    end = payload.rfind("}")
    if start >= 0 and end > start:
        candidate = payload[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return ast.literal_eval(candidate)
            except Exception:
                pass
    return None


def extract_image_id(record):
    for key in ("image_id", "id"):
        if key in record:
            try:
                return int(record[key])
            except Exception:
                pass

    images = record.get("images") or []
    if images:
        path = images[0].get("path", "")
        matches = re.findall(r"(\d+)\.jpg", path)
        if matches:
            return int(matches[-1])
    return None


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
        scale_x = GT_BOX_SCALE / float(pred_box_scale)
        scale_y = scale_x
    elif coord_mode == "image":
        if image_info is None:
            return None
        width = float(image_info["width"])
        height = float(image_info["height"])
        max_dim = max(width, height)
        scale_x = (width / max_dim) * GT_BOX_SCALE / float(pred_box_scale)
        scale_y = (height / max_dim) * GT_BOX_SCALE / float(pred_box_scale)
    else:
        raise ValueError(f"unknown coord_mode: {coord_mode}")
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


def dummy_prediction(num_predicates):
    graph = {
        "all_node_pairs": torch.as_tensor([[0, 1]], dtype=torch.int64),
        "all_relation": torch.zeros((1, num_predicates), dtype=torch.float32),
        "pred_boxes": torch.as_tensor(
            [[-10000.0, -10000.0, -9999.0, -9999.0], [-9990.0, -9990.0, -9989.0, -9989.0]],
            dtype=torch.float32,
        ),
        "pred_boxes_class": torch.as_tensor([1, 1], dtype=torch.int64),
        "pred_boxes_score": torch.ones((2,), dtype=torch.float32),
    }
    graph["all_relation"][0, 1] = 1.0
    return {
        "boxes": graph["pred_boxes"],
        "labels": graph["pred_boxes_class"],
        "scores": graph["pred_boxes_score"],
        "graph": graph,
    }


def with_stats(prediction, stats):
    prediction["_stats"] = stats
    return prediction


def build_prediction(
    record,
    class_to_idx,
    pred_to_idx,
    synonym_map,
    pred_box_scale,
    image_info_map,
    coord_mode,
):
    stats = Counter()
    image_id = extract_image_id(record)
    if image_id is None:
        stats["missing_image_id"] += 1
        return None, stats
    image_info = image_info_map.get(image_id)

    answer = extract_answer_json(record.get("response", ""))
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
    id_to_index = {}
    for obj in objects:
        if isinstance(obj, (list, tuple)) and len(obj) == 2:
            obj_id = str(obj[0]).strip()
            category = strip_instance_id(obj_id)
            label = class_to_idx.get(category)
            box = convert_pred_box(obj[1], pred_box_scale, image_info, coord_mode)
        elif isinstance(obj, dict):
            obj_id = str(obj.get("id", "")).strip()
            category = strip_instance_id(obj.get("category") or obj_id)
            label = class_to_idx.get(category)
            box = convert_pred_box(obj.get("box"), pred_box_scale, image_info, coord_mode)
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

        pred_name = normalize_predicate_name(predicate_raw, pred_to_idx, synonym_map)
        if pred_name is None:
            stats["unknown_predicate"] += 1
            continue
        pred_idx = pred_to_idx[pred_name]
        relation_rows.append((rel_order, subj_idx, obj_idx, pred_idx))

    if not pred_boxes:
        stats["empty_objects_after_filter"] += 1
        return image_id, with_stats(dummy_prediction(len(pred_to_idx)), stats)

    if not relation_rows:
        stats["empty_relationships_after_filter"] += 1
        return image_id, with_stats({
            "boxes": torch.as_tensor(pred_boxes, dtype=torch.float32),
            "labels": torch.as_tensor(pred_labels, dtype=torch.int64),
            "scores": torch.ones((len(pred_boxes),), dtype=torch.float32),
            "graph": {
                "all_node_pairs": torch.zeros((0, 2), dtype=torch.int64),
                "all_relation": torch.zeros((0, len(pred_to_idx)), dtype=torch.float32),
                "pred_boxes": torch.as_tensor(pred_boxes, dtype=torch.float32),
                "pred_boxes_class": torch.as_tensor(pred_labels, dtype=torch.int64),
                "pred_boxes_score": torch.ones((len(pred_boxes),), dtype=torch.float32),
            },
        }, stats)

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
            "pred_boxes_class": torch.as_tensor(pred_labels, dtype=torch.int64),
            "pred_boxes_score": torch.ones((len(pred_boxes),), dtype=torch.float32),
        },
    }
    stats["kept_objects"] += len(pred_boxes)
    stats["kept_relation_rows"] += len(relation_rows)
    return image_id, with_stats(pred, stats)


def load_predictions(args, class_to_idx, pred_to_idx, synonym_map):
    predictions = OrderedDict()
    stats = Counter()
    image_info_map = build_image_info_map(args.vg_dir)
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
            )
            if image_id is None:
                stats.update(pred_or_stats)
                continue
            if isinstance(pred_or_stats, Counter):
                stats.update(pred_or_stats)
                predictions[image_id] = None
                continue
            if image_id in predictions:
                stats["duplicate_image_prediction"] += 1
                continue
            predictions[image_id] = pred_or_stats
            stats.update(pred_or_stats.get("_stats", Counter()))
            pred_or_stats.pop("_stats", None)
    return predictions, stats


def patch_empty_graphs_for_zero_recall(predictions, dataset, num_predicates):
    patched = 0
    for image_id, ann in zip(dataset.ids, dataset.annotations):
        if ann["edges"].shape[0] == 0:
            continue
        pred = predictions.get(image_id)
        if pred is None:
            predictions[image_id] = dummy_prediction(num_predicates)
            patched += 1
            continue
        if pred["graph"]["all_node_pairs"].shape[0] == 0:
            predictions[image_id] = dummy_prediction(num_predicates)
            patched += 1
    return patched


def install_lightweight_import_fallbacks():
    if "tabulate" not in sys.modules:
        tabulate_module = types.ModuleType("tabulate")
        tabulate_module.__spec__ = ModuleSpec("tabulate", loader=None)

        def tabulate(rows, headers=(), **_kwargs):
            rows = list(rows)
            if headers:
                rows = [headers] + rows
            return "\n".join(" | ".join(str(cell) for cell in row) for row in rows)

        tabulate_module.tabulate = tabulate
        sys.modules["tabulate"] = tabulate_module

    torchvision_module = types.ModuleType("torchvision")
    torchvision_module.__spec__ = ModuleSpec("torchvision", loader=None)
    torchvision_module.__version__ = "0.7.0"
    sys.modules["torchvision"] = torchvision_module

    try:
        import pycocotools  # noqa: F401
    except Exception:
        pycocotools_module = types.ModuleType("pycocotools")
        coco_module = types.ModuleType("pycocotools.coco")
        cocoeval_module = types.ModuleType("pycocotools.cocoeval")
        mask_module = types.ModuleType("pycocotools.mask")

        class COCO:
            def __init__(self, *args, **kwargs):
                self.dataset = {}

            def createIndex(self):
                return None

            @staticmethod
            def loadRes(*args, **kwargs):
                return COCO()

        class COCOeval:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("COCO bbox evaluation requires pycocotools.")

        coco_module.COCO = COCO
        cocoeval_module.COCOeval = COCOeval
        pycocotools_module.coco = coco_module
        pycocotools_module.cocoeval = cocoeval_module
        pycocotools_module.mask = mask_module
        sys.modules["pycocotools"] = pycocotools_module
        sys.modules["pycocotools.coco"] = coco_module
        sys.modules["pycocotools.cocoeval"] = cocoeval_module
        sys.modules["pycocotools.mask"] = mask_module


def scalar_mean(values):
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def collect_metrics(result_dict, mode):
    metrics = {}
    for k in (20, 50, 100):
        metrics[f"R@{k}"] = scalar_mean(result_dict[f"{mode}_recall"][k])
        metrics[f"mR@{k}"] = float(result_dict[f"{mode}_mean_recall"][k])
        metrics[f"ng-R@{k}"] = scalar_mean(result_dict[f"{mode}_recall_nogc"][k])
        metrics[f"ng-mR@{k}"] = float(result_dict[f"{mode}_ng_mean_recall"][k])
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-jsonl", default=DEFAULT_PRED_JSONL)
    parser.add_argument("--vg-dir", default=DEFAULT_VG_DIR)
    parser.add_argument("--r1-root", default=DEFAULT_R1_ROOT)
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

    ind_to_classes, ind_to_predicates = build_ind_to_names(args.vg_dir)
    class_to_idx = {normalize_text(name): idx for idx, name in enumerate(ind_to_classes)}
    class_to_idx.pop("__background__", None)
    pred_to_idx = {normalize_text(name): idx for idx, name in enumerate(ind_to_predicates)}
    valid_predicates = set(pred_to_idx.keys()) - {"__background__"}
    synonym_map = {} if args.no_synonyms else load_predicate_synonyms(args.r1_root, valid_predicates)

    predictions, parse_stats = load_predictions(args, class_to_idx, pred_to_idx, synonym_map)
    if not predictions:
        raise RuntimeError("No valid predictions were parsed.")

    split_to_id = {"train": 0, "val": 1, "test": 2}
    dataset = StanfordFilteredVGDataset(args.vg_dir, predictions.keys(), split_to_id[args.split])
    predictions = OrderedDict((image_id, predictions.get(image_id)) for image_id in dataset.ids)
    patched_empty = patch_empty_graphs_for_zero_recall(predictions, dataset, len(ind_to_predicates))

    sys.path.insert(0, args.r1_root)
    os.chdir(args.r1_root)
    install_lightweight_import_fallbacks()
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
    metrics = collect_metrics(result_dict, "sgdet")

    # bbox AP from COCO evaluator: stats order is [AP, AP50, AP75, APs, APm, APl]
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
        "relation_score_mode": "r1_onehot_per_output_relation",
        "relation_score_note": (
            "Each generated relation is kept as one one-hot row, matching R1-SGG "
            "sgg_gather_preds.py; repeated subject-object pairs are not merged."
        ),
        "gt_box_scale": GT_BOX_SCALE,
        "mode": "sgdet",
        "iou_thres": args.iou_thres,
        "num_prediction_lines": int(parse_stats["lines"]),
        "num_eval_images": len(dataset.ids),
        "num_eval_images_with_gt_rels": gt_rel_images,
        "num_gt_rels": total_gt_rels,
        "avg_pred_triplets_per_image": avg_pred_triplets,
        "num_empty_or_missing_predictions_patched_as_zero_recall": patched_empty,
        "parse_stats": dict(parse_stats),
        "metrics": metrics,
    }

    output = args.output
    if output is None:
        output = str(Path(args.pred_jsonl).with_suffix("")) + "_sgdet_eval.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSGDET R/mR using R1-SGG-style one-hot relation rows")
    print(f"eval images: {len(dataset.ids)}  images with gt rels: {gt_rel_images}  gt rels: {total_gt_rels}  avg pred triplets/image: {avg_pred_triplets:.2f}")
    print(f"prediction coord mode: {args.coord_mode}  box scale -> gt scale: {args.pred_box_scale:g} -> {GT_BOX_SCALE:g}")
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
