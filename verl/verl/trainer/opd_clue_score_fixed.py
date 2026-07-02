def _opd_clue_score_fixed(
    tokenizer,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_ids: torch.Tensor | None,
    teacher_logprobs: torch.Tensor | None,
) -> list[dict[str, float]]:
    """Compute per-pair OPD scores from teacher logprobs over <CLUE> tokens.

    FIXED VERSION: Uses token-index based matching instead of character positions
    to avoid BPE tokenization alignment issues.

    Returns a list (one per sample) of dicts mapping "subject_id|object_id" -> opd_score.
    Each opd_score is exp(mean_teacher_logprob) over that pair's clue line tokens.
    Keys use "|" separator to match edge_per_pair from reward function.
    """
    import re

    batch_size = responses.shape[0]
    results: list[dict[str, float]] = [{} for _ in range(batch_size)]
    if teacher_logprobs is None or teacher_ids is None:
        return results

    if teacher_ids.dim() == 2:
        teacher_ids = teacher_ids.unsqueeze(-1)
    if teacher_logprobs.dim() == 2:
        teacher_logprobs = teacher_logprobs.unsqueeze(-1)
    if teacher_ids.shape != teacher_logprobs.shape:
        return results

    valid_mask = response_mask.bool()
    for i in range(batch_size):
        resp_len = int(valid_mask[i].sum().item())
        if resp_len <= 0:
            continue

        resp_ids = responses[i, :resp_len]

        # Find <CLUE> and </CLUE> token positions (not character positions)
        clue_start_tok = None
        clue_end_tok = None

        for idx in range(resp_len):
            token_text = tokenizer.decode([int(resp_ids[idx].item())], skip_special_tokens=False).lower()
            if clue_start_tok is None and '<clue>' in token_text:
                clue_start_tok = idx + 1  # Start after <CLUE> token
            elif clue_start_tok is not None and '</clue>' in token_text:
                clue_end_tok = idx  # End before </CLUE> token
                break

        if clue_start_tok is None or clue_end_tok is None or clue_start_tok >= clue_end_tok:
            continue

        # Decode the entire CLUE region once for parsing
        clue_token_ids = resp_ids[clue_start_tok:clue_end_tok]
        clue_text = tokenizer.decode(clue_token_ids.tolist(), skip_special_tokens=False)

        # Parse clue lines to find subject-object pairs
        clue_lines = []
        for line in clue_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # More flexible regex to handle format variations
            pair_match = re.search(r"\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)", stripped)
            if pair_match:
                subject = pair_match.group(1).strip().lower()
                obj = pair_match.group(2).strip().lower()
                pair_key = f"{subject}|{obj}"
                clue_lines.append((pair_key, stripped))

        if not clue_lines:
            continue

        # Find token ranges for each clue line using token-level search
        # Strategy: Look for line boundaries by detecting newline tokens and pair patterns
        line_token_ranges = []
        current_line_start = clue_start_tok
        current_line_idx = 0

        for tok_idx in range(clue_start_tok, clue_end_tok):
            token_text = tokenizer.decode([int(resp_ids[tok_idx].item())], skip_special_tokens=False)

            # Check if this token contains newline (marks end of previous line)
            if '\n' in token_text and tok_idx > current_line_start:
                # Check if the accumulated tokens match a clue line pattern
                line_tokens = resp_ids[current_line_start:tok_idx]
                line_text = tokenizer.decode(line_tokens.tolist(), skip_special_tokens=False).strip()

                # Try to match this line to one of our parsed clue lines
                for clue_idx, (pair_key, clue_line_text) in enumerate(clue_lines):
                    if clue_idx >= current_line_idx:  # Only match forward
                        # Check if this is the right line by looking for the pair pattern
                        pair_match = re.search(r"\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)", line_text)
                        if pair_match:
                            found_subject = pair_match.group(1).strip().lower()
                            found_obj = pair_match.group(2).strip().lower()
                            found_key = f"{found_subject}|{found_obj}"

                            if found_key == pair_key:
                                line_token_ranges.append((pair_key, current_line_start, tok_idx))
                                current_line_idx = clue_idx + 1
                                current_line_start = tok_idx + 1
                                break

        # Handle last line (no trailing newline)
        if current_line_start < clue_end_tok and current_line_idx < len(clue_lines):
            line_tokens = resp_ids[current_line_start:clue_end_tok]
            line_text = tokenizer.decode(line_tokens.tolist(), skip_special_tokens=False).strip()

            for clue_idx in range(current_line_idx, len(clue_lines)):
                pair_key, clue_line_text = clue_lines[clue_idx]
                pair_match = re.search(r"\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)", line_text)
                if pair_match:
                    found_subject = pair_match.group(1).strip().lower()
                    found_obj = pair_match.group(2).strip().lower()
                    found_key = f"{found_subject}|{found_obj}"

                    if found_key == pair_key:
                        line_token_ranges.append((pair_key, current_line_start, clue_end_tok))
                        break

        # Compute OPD score for each line using token indices directly
        for pair_key, tok_start, tok_end in line_token_ranges:
            teacher_lps = []

            for tok_idx in range(tok_start, tok_end):
                student_token_id = int(resp_ids[tok_idx].item())
                t_ids_pos = teacher_ids[i, tok_idx]
                t_lps_pos = teacher_logprobs[i, tok_idx].float()

                # Check if student token is in teacher's top-k
                match = t_ids_pos == student_token_id
                if match.any():
                    teacher_lp = t_lps_pos[match].max()
                else:
                    # IMPROVEMENT: Instead of skipping, use a penalty score
                    # This allows exploration while still providing gradient signal
                    teacher_lp = torch.tensor(-2.0, device=t_lps_pos.device)  # exp(-2) ≈ 0.135

                if not torch.isfinite(teacher_lp):
                    continue

                teacher_lps.append(teacher_lp)

            if teacher_lps:
                mean_teacher_lp = torch.stack(teacher_lps).mean()
                results[i][pair_key] = torch.exp(mean_teacher_lp).clamp(0.0, 1.0).item()
            else:
                results[i][pair_key] = 0.0

    return results
