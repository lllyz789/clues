# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }


def compute_jsd_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Compute Jensen-Shannon Divergence (JSD) distillation loss using top-k log probabilities.

    JSD is a symmetric divergence measure:
    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q) is the mixture distribution.

    Properties:
    - Symmetric: JSD(P||Q) = JSD(Q||P)
    - Bounded: JSD ∈ [0, log(2)]
    - Smoother gradients than KL divergence
    - Encourages balanced exploration

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        config: DistillationConfig containing loss parameters.
        data_format: "thd" or "bshd".

    Returns:
        dict with:
        - distillation_losses: (bsz, seqlen/sp_size) - JSD values
        - student_mass: (bsz, seqlen/sp_size) - student prob mass in support
        - teacher_mass: (bsz, seqlen/sp_size) - teacher prob mass in support
        - overlap_count: (bsz, seqlen/sp_size) - number of overlapping top-k tokens
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. Split across sequence parallel groups if needed
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. Get student log probabilities
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    # 3. Gather student log probs at teacher's top-k positions
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)

    # 4. Convert to probabilities (for mixing)
    teacher_topk_probs = teacher_topk_log_probs.exp()  # (bsz, seqlen, topk)
    student_topk_probs = student_topk_log_probs.exp()  # (bsz, seqlen, topk)

    # 5. Apply optional log prob clamping
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        # Recompute probs after clamping
        teacher_topk_probs = teacher_topk_log_probs.exp()
        student_topk_probs = student_topk_log_probs.exp()

    # 6. Compute mixture distribution M = 0.5 * (P_teacher + P_student)
    mixture_probs = 0.5 * (teacher_topk_probs + student_topk_probs)  # (bsz, seqlen, topk)
    mixture_log_probs = mixture_probs.clamp_min(1e-10).log()  # Avoid log(0)

    # 7. Compute JSD = 0.5 * KL(teacher||M) + 0.5 * KL(student||M)
    # KL(teacher||M) = sum_i P_teacher(i) * [log P_teacher(i) - log M(i)]
    kl_teacher_m = teacher_topk_probs * (teacher_topk_log_probs - mixture_log_probs)
    kl_teacher_m = kl_teacher_m.sum(dim=-1)  # (bsz, seqlen)

    # KL(student||M) = sum_i P_student(i) * [log P_student(i) - log M(i)]
    kl_student_m = student_topk_probs * (student_topk_log_probs - mixture_log_probs)
    kl_student_m = kl_student_m.sum(dim=-1)  # (bsz, seqlen)

    # JSD = 0.5 * (KL_teacher + KL_student)
    distillation_losses = 0.5 * (kl_teacher_m + kl_student_m)

    # 8. Compute mass metrics (same as forward_kl for consistency)
    student_mass = student_topk_probs.sum(dim=-1)
    teacher_mass = teacher_topk_probs.sum(dim=-1)

    # 9. Compute overlap diagnostics (same as forward_kl)
    student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)

    # Compute overlap advantage (how much better on overlapping tokens)
    token_jsd_contrib = 0.5 * (
        teacher_topk_probs * (teacher_topk_log_probs - mixture_log_probs) +
        student_topk_probs * (student_topk_log_probs - mixture_log_probs)
    )
    overlap_token_advantage_sum = (-token_jsd_contrib * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }


def compute_forward_kl_topk_renorm(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Compute forward KL distillation loss with renormalization over local support set.

    The local support set = teacher top-K tokens + student sampled token (if not
    already in top-K). Both teacher and student distributions are renormalized over
    this support set before computing forward KL divergence.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, K+1) — last slot stores sampled token
            logprob if it is not in teacher top-K, otherwise 0.0.
        teacher_topk_ids: (bsz, seqlen, K+1) — last slot stores sampled token id
            if it is not in teacher top-K, otherwise 0.
        data_format: "thd" or "bshd".

    Returns:
        dict with distillation_losses, student_mass, teacher_mass (each: bsz, seqlen/sp_size).
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, K+1)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, K+1)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # Determine which positions have a valid extra sampled token in the last slot.
    # When sampled token is already in top-K, last slot is left as (id=0, logprob=0.0).
    has_extra = (teacher_topk_log_probs[..., -1] != 0.0)  # (bsz, seqlen)

    # Build validity mask: (bsz, seqlen, K+1)
    valid_mask = torch.ones_like(teacher_topk_ids, dtype=torch.bool)
    valid_mask[..., -1] = has_extra

    # Gather student logits at the support set positions
    student_logits_at_support = torch.gather(
        student_logits, dim=-1, index=teacher_topk_ids.long()
    )  # (bsz, seqlen, K+1)

    # Renormalize student: log_softmax over valid support positions only
    student_logits_masked = student_logits_at_support.float()
    student_logits_masked[~valid_mask] = -1e9
    student_log_probs_renorm = F.log_softmax(student_logits_masked, dim=-1)

    # Renormalize teacher: treat teacher log_probs as unnormalized scores in the
    # reduced support set, apply log_softmax to renormalize.
    teacher_logprobs_masked = teacher_topk_log_probs.float()
    teacher_logprobs_masked[~valid_mask] = -1e9
    teacher_log_probs_renorm = F.log_softmax(teacher_logprobs_masked, dim=-1)

    # Forward KL: sum_x p_teacher(x) * [log p_teacher(x) - log p_student(x)]
    teacher_probs_renorm = teacher_log_probs_renorm.exp()
    per_token_kl = teacher_probs_renorm * (teacher_log_probs_renorm - student_log_probs_renorm)
    per_token_kl[~valid_mask] = 0.0
    distillation_losses = per_token_kl.sum(dim=-1)  # (bsz, seqlen)

    # Mass metrics: original (non-renormalized) probability mass in the support set
    student_log_probs_full = F.log_softmax(student_logits, dim=-1)
    student_support_log_probs = torch.gather(
        student_log_probs_full, dim=-1, index=teacher_topk_ids.long()
    )
    student_support_probs = student_support_log_probs.exp()
    student_support_probs[~valid_mask] = 0.0
    student_mass = student_support_probs.sum(dim=-1)

    teacher_support_probs = teacher_topk_log_probs.exp()
    teacher_support_probs[~valid_mask] = 0.0
    teacher_mass = teacher_support_probs.sum(dim=-1)

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
    }
