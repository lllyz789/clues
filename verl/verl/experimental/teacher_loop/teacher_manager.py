# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TEACHER_SYSTEM_PROMPT = (
    "You are given multiple localized object pairs from one image.\n"
    "Generate the complete visual evidence relation reasoning clues for all given pairs.\n"
    "Each output line must use the format:\n"
    "(subject_id, object_id): evidence sentence Type: relation_type. Final Predicate: predicate\n\n"
    "Example input:\nPairs:\n[\n  {\n"
    '    "subject": {"id": "person.1", "bbox": [167, 133, 392, 987]},\n'
    '    "object": {"id": "sidewalk.1", "bbox": [0, 540, 1000, 999]}\n'
    "  },\n  {\n"
    '    "subject": {"id": "umbrella.1", "bbox": [8, 21, 451, 321]},\n'
    '    "object": {"id": "person.1", "bbox": [167, 133, 392, 987]}\n'
    "  }\n]\n\n"
    "Example output:\n"
    "(person.1, sidewalk.1): person.1's feet contact sidewalk.1 directly beneath the body. "
    "Type: spatial_relations. Final Predicate: on\n"
    "(umbrella.1, person.1): umbrella.1 is held overhead, covering person.1 from above. "
    "Type: spatial_relations. Final Predicate: above\n\n"
    "Do not output extra text."
)


def _extract_clue_and_pairs_from_response(response_text: str) -> tuple[str | None, str | None]:
    """Extract CLUE text and build pairs JSON from student structured response.

    Returns (clue_text, pairs_json) or (None, None) if extraction fails.
    """
    # Extract OBJECT section
    obj_match = re.search(r"<OBJECT>(.*?)</OBJECT>", response_text, re.DOTALL)
    if not obj_match:
        return None, None

    # Extract CLUE section
    clue_match = re.search(r"<CLUE>(.*?)</CLUE>", response_text, re.DOTALL)
    if not clue_match:
        return None, None

    clue_text = clue_match.group(1).strip()
    if not clue_text:
        return None, None

    # Parse objects for bbox lookup
    try:
        obj_data = json.loads(obj_match.group(1).strip())
        objects = {o["id"]: o["bbox"] for o in obj_data.get("objects", [])}
    except (json.JSONDecodeError, KeyError, TypeError):
        return None, None

    # Parse clue lines to extract (subject, object) pairs
    pairs = []
    seen = set()
    for line in clue_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        pair_match = re.match(r"\(([^,]+),\s*([^)]+)\)", line)
        if pair_match:
            subj_id = pair_match.group(1).strip()
            obj_id = pair_match.group(2).strip()
            key = (subj_id, obj_id)
            if key in seen:
                continue
            seen.add(key)
            subj_bbox = objects.get(subj_id)
            obj_bbox = objects.get(obj_id)
            if subj_bbox is not None and obj_bbox is not None:
                pairs.append({
                    "subject": {"id": subj_id, "bbox": subj_bbox},
                    "object": {"id": obj_id, "bbox": obj_bbox},
                })

    if not pairs:
        return None, None

    pairs_json = json.dumps(pairs, ensure_ascii=False, separators=(",", ":"))
    return clue_text, pairs_json


def build_teacher_prefix_ids(
    tokenizer,
    pairs_json: str,
    multi_modal_data: Optional[dict[str, Any]] = None,
) -> list[int]:
    """Build teacher prefix token IDs (system + user + generation prompt).

    The caller appends the student's actual CLUE token IDs after this prefix.
    """
    user_content = (
        "Generate the complete relation reasoning clues for the following object pairs.\n\n"
        f"Pairs:\n{pairs_json}"
    )

    has_image = multi_modal_data and multi_modal_data.get("images")
    if has_image:
        user_msg_content = [{"type": "image"}, {"type": "text", "text": user_content}]
    else:
        user_msg_content = user_content

    prefix_msgs = [
        {"role": "system", "content": _TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg_content},
    ]

    prefix_text = tokenizer.apply_chat_template(
        prefix_msgs, tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(prefix_text, add_special_tokens=False)


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    if teacher_model_config.inference.temperature != 1.0:
        raise NotImplementedError("vLLM does not support temperature for prompt_logprobs.")

    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    return {
        "max_tokens": 1,
        "temperature": teacher_model_config.inference.temperature,
        "prompt_logprobs": num_logprobs,
    }


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client: dict[str, LLMServerClient] = teacher_client

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        # Shapes: # S, (1 or K+1), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings. For top-k teacher queries, the extra trailing slot stores
        # the sampled token if it is not present in the top-k set.
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
        return teacher_ids, teacher_logprobs

    async def compute_teacher_logprobs_reformatted(
        self,
        tokenizer,
        prompt_ids: list[int],
        response_ids: list[int],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher logprobs using reformatted input matching teacher training format.

        Instead of sending student prompt+response directly, reconstructs teacher input as:
          system(relation reasoning) + user(pairs JSON) + student_clue_token_ids

        The student's actual CLUE token IDs are appended directly (no re-encoding) so the
        teacher produces logprobs for the exact same tokens at each position.

        Returns teacher_ids and teacher_logprobs with the SAME shape as the original
        (prompt_ids + response_ids) sequence, but only the positions corresponding to
        <CLUE> tokens in the student response have meaningful values; all other positions
        are filled with zeros.
        """
        total_len = len(prompt_ids) + len(response_ids)
        topk = self.distillation_loss_config.topk or 1
        width = topk + 1 if self.distillation_loss_config.loss_settings.use_topk else 1

        default_ids = torch.zeros(total_len, width, dtype=torch.int32)
        default_logprobs = torch.zeros(total_len, width, dtype=torch.float32)

        response_text = tokenizer.decode(response_ids, skip_special_tokens=False)

        # Extract pairs from student <OBJECT> section
        _, pairs_json = _extract_clue_and_pairs_from_response(response_text)
        if pairs_json is None:
            logger.debug("Failed to extract pairs from student response for teacher reformatting")
            return default_ids, default_logprobs

        # Find CLUE span token indices in student response
        resp_text_lower = response_text.lower()
        clue_start_pos = resp_text_lower.find("<clue>")
        if clue_start_pos < 0:
            return default_ids, default_logprobs
        content_start = clue_start_pos + len("<clue>")
        clue_end_pos = resp_text_lower.find("</clue>", content_start)
        content_end = clue_end_pos if clue_end_pos >= 0 else len(response_text)

        # Map character offsets to token indices
        offsets = []
        cursor = 0
        for token_id in response_ids:
            token_text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
            offsets.append((cursor, cursor + len(token_text)))
            cursor += len(token_text)

        clue_token_indices = []
        for idx, (left, right) in enumerate(offsets):
            if right > content_start and left < content_end:
                clue_token_indices.append(idx)

        if not clue_token_indices:
            return default_ids, default_logprobs

        # Build teacher input: prefix + student's actual CLUE token IDs
        prefix_ids = build_teacher_prefix_ids(tokenizer, pairs_json, multi_modal_data)
        student_clue_ids = [response_ids[i] for i in clue_token_indices]

        # Call teacher with reformatted input
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]

        # vLLM needs one token of headroom because this request asks for
        # max_tokens=1. Keep the run alive for overlong OPD samples and only
        # distill the CLUE prefix that fits.
        max_model_len = teacher_model_config.inference.max_model_len
        if max_model_len is not None:
            max_prompt_len = max_model_len - 1
            available_clue_len = max_prompt_len - len(prefix_ids)
            if available_clue_len <= 0:
                logger.warning(
                    "Skipping teacher distillation for overlong reformatted prefix: "
                    "prefix_len=%s, max_model_len=%s, pairs_json_chars=%s",
                    len(prefix_ids),
                    max_model_len,
                    len(pairs_json),
                )
                return default_ids, default_logprobs
            if len(student_clue_ids) > available_clue_len:
                logger.warning(
                    "Truncating teacher CLUE tokens for overlong reformatted input: "
                    "prefix_len=%s, clue_len=%s, kept_clue_len=%s, max_model_len=%s, pairs_json_chars=%s",
                    len(prefix_ids),
                    len(student_clue_ids),
                    available_clue_len,
                    max_model_len,
                    len(pairs_json),
                )
                student_clue_ids = student_clue_ids[:available_clue_len]

        teacher_seq_ids = prefix_ids + student_clue_ids
        prefix_length = len(prefix_ids)

        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=teacher_seq_ids,
            sampling_params=_get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images") if multi_modal_data else None,
            video_data=multi_modal_data.get("videos") if multi_modal_data else None,
            audio_data=multi_modal_data.get("audios") if multi_modal_data else None,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        raw_teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        raw_teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])

        # vLLM prompt_logprobs are aligned to the position that predicts the
        # next token: raw_teacher_ids[j] scores teacher_seq_ids[j + 1].
        # Keep that same convention when mapping back to the full student
        # sequence, because response_from_nested() later left-shifts by one.
        teacher_clue_len = raw_teacher_ids.shape[0] - prefix_length
        n_to_map = min(teacher_clue_len, len(clue_token_indices))
        prompt_offset = len(prompt_ids)

        for i in range(n_to_map):
            teacher_pos = prefix_length + i - 1
            student_full_idx = prompt_offset + clue_token_indices[i] - 1
            if student_full_idx < total_len and teacher_pos < raw_teacher_ids.shape[0]:
                default_ids[student_full_idx] = raw_teacher_ids[teacher_pos]
                default_logprobs[student_full_idx] = raw_teacher_logprobs[teacher_pos]

        return default_ids, default_logprobs
