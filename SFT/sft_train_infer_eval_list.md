# SFT Train / Infer / Eval List

本文档整理当前仓库里和 VG SFT 流程相关的训练、推理、评测入口，以及它们默认使用的关键文件。

## 1. SFT Train

### 入口脚本
- `Sft/train/train_qwen3_vl_4b_vg_full_sft.sh`


### 直接依赖文件
- 基座模型：`model/Qwen3-VL-4B-Instruct`
- 训练数据：`jsondata/erejin_datasets/origin/bbox0-1000/vg_5cls_0-1000_swift_structured.jsonl`

### 训练输出
- 默认输出目录：`output/qwen3_vl_4b_vg_sft_full/`
- 默认日志：`output/qwen3_vl_4b_vg_sft_full/.../train.log`

## 2. Infer

### 测试集推理
- `SFT/infer/run_infer_first_sample_gt_pred.sh`

默认依赖：
- 图像列表：`jsondata/vg_test_14700_image_paths.json`
- 训练数据模板：`jsondata/erejin_datasets/origin/bbox0-1000/merged_5cls_0-1000_swift_structured.jsonl`
- Prompt：从训练数据第一条消息中读取
- checkpoint：默认从 `output/qwen3_vl_4b_vg_sft_full/.../checkpoint-*` 读取

## 3. Eval

### SGDET 评测
- `tools_eval/eval_vg_jsonl_structured_sgdet_newrule.py`

## 4. 常用中间文件

- `jsondata/vg_test_14700_image_paths.json`
- `jsondata/erejin_datasets/origin/bbox0-1000/vg_5cls_0-1000_swift_structured.jsonl`

## 5. 备注

- 训练脚本默认使用 `swift sft`
- 推理脚本默认使用 `swift infer`
- 评测脚本默认读取 `response` 字段里的结构化结果

