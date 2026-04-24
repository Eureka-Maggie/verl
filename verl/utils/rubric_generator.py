# -*- coding: utf-8 -*-
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

"""Rubric generation and rubric-based scoring for RL training."""

import ast
import glob
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from verl.protocol import DataProto
from verl.utils.llm_client import LLMClient


class RubricGenerator:
    """Generates evaluation rubrics from rollout samples and scores rollouts using rubrics.

    Flow:
      1. generate_rubric(batch, step=1): sample rollouts + ground_truth → call LLM →
         save step_1.jsonl
      2. load_latest_rubrics(): find newest step_N.jsonl → return rubric list
      3. score_batch_to_token_level_tensor(batch, rubrics): for every rollout call judge
         LLM → sum(scores)/len(rubrics) → place score at last valid response token
      4. update_rubric(batch, prev_rubrics, step): on convergence, sample low-variance
         rollouts, call LLM with update_rubric.txt template → save step_N.jsonl
      5. cleanup_future_rubrics(resume_step): on restart, delete step_N.jsonl where
         N > resume_step to avoid analysis confusion.

    Args:
        rubric_template_path:       Path to init_rubric.txt template
        update_rubric_template_path: Path to update_rubric.txt template
        judge_template_path:        Path to judge.txt template
        output_dir: Base directory for saving rubric JSONL files
        exp_name: Experiment name (subdirectory under output_dir)
        llm_client: LLMClient instance
        tokenizer: HF tokenizer for decoding token ids
        num_prompts: Number of prompts to sample for rubric generation
        num_rollouts_per_prompt: Number of rollouts per prompt for rubric generation
        num_low_var_groups: Number of low-variance groups to sample for update
    """

    def __init__(
        self,
        rubric_template_path: str,
        judge_template_path: str,
        output_dir: str,
        exp_name: str,
        llm_client: LLMClient,
        tokenizer,
        num_prompts: int = 8,
        num_rollouts_per_prompt: int = 2,
        update_rubric_template_path: Optional[str] = None,
        num_low_var_groups: int = 8,
        judge_max_workers: int = 32,
    ):
        self.output_dir = Path(output_dir) / exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_client = llm_client
        self.tokenizer = tokenizer
        self.num_prompts = num_prompts
        self.num_rollouts_per_prompt = num_rollouts_per_prompt
        self.num_low_var_groups = num_low_var_groups
        self.judge_max_workers = judge_max_workers
        self._first_judge_lock = threading.Lock()
        self._first_judge_printed = False

        with open(rubric_template_path, "r", encoding="utf-8") as f:
            self.rubric_template = f.read()

        with open(judge_template_path, "r", encoding="utf-8") as f:
            self.judge_template = f.read()

        # update_rubric.txt template (optional; falls back to init template if not provided)
        if update_rubric_template_path is not None:
            with open(update_rubric_template_path, "r", encoding="utf-8") as f:
                self.update_rubric_template = f.read()
        else:
            self.update_rubric_template = self.rubric_template

    # ------------------------------------------------------------------
    # Rubric generation
    # ------------------------------------------------------------------

    def sample_rollouts(self, batch: DataProto) -> List[Dict]:
        """Sample rollouts with ground_truth for rubric generation.

        Returns a list of dicts with keys: prompt, response, ground_truth.
        """
        prompts_tensor = batch.batch["prompts"]      # (B, prompt_len)
        responses_tensor = batch.batch["responses"]  # (B, resp_len)
        reward_model_info = batch.non_tensor_batch.get("reward_model", None)

        # Group rollouts by decoded prompt text
        prompt_to_data: Dict[str, Dict] = {}
        for i in range(len(prompts_tensor)):
            prompt_text = self.tokenizer.decode(prompts_tensor[i], skip_special_tokens=True)
            response_text = self.tokenizer.decode(responses_tensor[i], skip_special_tokens=True)

            gt = None
            if reward_model_info is not None:
                info = reward_model_info[i]
                if isinstance(info, dict):
                    gt = info.get("ground_truth", None)

            if prompt_text not in prompt_to_data:
                prompt_to_data[prompt_text] = {"ground_truth": gt, "rollouts": []}

            prompt_to_data[prompt_text]["rollouts"].append({
                "prompt": prompt_text,
                "response": response_text,
                "ground_truth": gt,
            })

        # Sample prompts, then sample rollouts per prompt
        available = list(prompt_to_data.keys())
        num_to_sample = min(self.num_prompts, len(available))
        sampled_prompts = random.sample(available, num_to_sample)

        sampled_rollouts = []
        for prompt in sampled_prompts:
            rollouts = prompt_to_data[prompt]["rollouts"]
            n = min(self.num_rollouts_per_prompt, len(rollouts))
            sampled_rollouts.extend(random.sample(rollouts, n))

        print(f"[RubricGenerator] Sampled {len(sampled_rollouts)} rollouts "
              f"from {num_to_sample} prompts for rubric generation.")
        return sampled_rollouts

    def sample_low_variance_rollouts(
        self,
        batch: DataProto,
        seq_rewards: "np.ndarray",  # shape (B,), one value per trajectory
        rollout_n: int,
    ) -> List[Dict]:
        """Sample rollouts from low-variance groups (prompts where the model responds consistently).

        Groups trajectories by prompt index, computes per-group reward variance, then
        picks the ``num_low_var_groups`` groups with the smallest variance and samples
        ``num_rollouts_per_prompt`` trajectories from each.

        Args:
            batch:        DataProto after rollout
            seq_rewards:  numpy array of per-trajectory rewards, shape (B,)
            rollout_n:    number of rollouts per prompt (used to reconstruct group index)

        Returns:
            List of dicts with keys: prompt, response, ground_truth, group_var
        """
        prompts_tensor = batch.batch["prompts"]
        responses_tensor = batch.batch["responses"]
        reward_model_info = batch.non_tensor_batch.get("reward_model", None)

        B = len(seq_rewards)

        # Reconstruct group index from rollout position (same logic as ray_trainer)
        if "index" in batch.batch:
            index_cpu = batch.batch["index"].cpu().numpy()
        else:
            index_cpu = np.arange(B) // rollout_n

        # Build per-group data
        group_data: Dict[int, Dict] = {}
        for i in range(B):
            gid = int(index_cpu[i])
            if gid not in group_data:
                group_data[gid] = {"rewards": [], "indices": []}
            group_data[gid]["rewards"].append(seq_rewards[i])
            group_data[gid]["indices"].append(i)

        # Compute variance per group and sort ascending
        group_vars = {gid: float(np.var(d["rewards"])) for gid, d in group_data.items()}
        sorted_groups = sorted(group_vars.keys(), key=lambda g: group_vars[g])
        selected_groups = sorted_groups[: self.num_low_var_groups]

        sampled_rollouts = []
        for gid in selected_groups:
            indices = group_data[gid]["indices"]
            n = min(self.num_rollouts_per_prompt, len(indices))
            chosen = random.sample(indices, n)
            for idx in chosen:
                prompt_text = self.tokenizer.decode(prompts_tensor[idx], skip_special_tokens=True)
                response_text = self.tokenizer.decode(responses_tensor[idx], skip_special_tokens=True)
                gt = None
                if reward_model_info is not None:
                    info = reward_model_info[idx]
                    if isinstance(info, dict):
                        gt = info.get("ground_truth", None)
                sampled_rollouts.append({
                    "prompt": prompt_text,
                    "response": response_text,
                    "ground_truth": gt,
                    "group_var": group_vars[gid],
                })

        print(
            f"[RubricGenerator] Low-variance sampling: selected {len(selected_groups)} groups "
            f"(min_var={group_vars[sorted_groups[0]]:.4f}, "
            f"max_selected_var={group_vars[selected_groups[-1]]:.4f}), "
            f"total {len(sampled_rollouts)} rollouts."
        )
        return sampled_rollouts

    def _format_rollout_samples(self, rollouts: List[Dict]) -> str:
        """Format rollout samples for insertion into init_rubric.txt template."""
        parts = []
        for i, r in enumerate(rollouts, 1):
            gt_line = f"\n**Ground Truth:** {r['ground_truth']}" if r.get("ground_truth") else ""
            parts.append(
                f"### Sample {i} ###\n"
                f"**Prompt:**\n{r['prompt']}\n\n"
                f"**Response:**\n{r['response']}"
                f"{gt_line}\n"
            )
        return "\n".join(parts)

    def generate_rubric(self, batch: DataProto, step: int = 1) -> Optional[str]:
        """Generate rubric from rollout batch and save to step_{step}.jsonl.

        Args:
            batch: DataProto after rollout (before reward computation)
            step: Training step number

        Returns:
            Path to saved JSONL file, or None on failure.
        """
        print(f"\n{'='*60}\nGenerating rubric at step {step}\n{'='*60}")

        sampled = self.sample_rollouts(batch)
        formatted = self._format_rollout_samples(sampled)
        prompt = self.rubric_template.replace("[INSERT ROLLOUT SAMPLES HERE]", formatted)

        system_prompt = "You are an expert evaluator and reward-design assistant for RL training."
        print("\n" + "="*80)
        print("【RUBRIC GENERATION】完整送入LLM的prompt：")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")
        response_text, success = self.llm_client.simple_text_call(system_prompt, prompt)

        if not success:
            print(f"[RubricGenerator] LLM call failed: {response_text}")
            return None

        rubrics = self._extract_rubrics_from_response(response_text)
        if not rubrics:
            print(f"[RubricGenerator] Failed to extract rubrics.\nRaw response:\n{response_text}")
            return None

        output_path = self.output_dir / f"step_{step}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for rubric in rubrics:
                f.write(json.dumps(rubric, ensure_ascii=False) + "\n")

        print(f"[RubricGenerator] Saved {len(rubrics)} rubrics to {output_path}")
        return str(output_path)

    def update_rubric(
        self,
        batch: DataProto,
        prev_rubrics: List[Dict],
        step: int,
        seq_rewards: "np.ndarray",
        rollout_n: int,
    ) -> Optional[str]:
        """Generate an updated rubric at a convergence point.

        Samples low-variance rollouts from the current batch, formats them together
        with the previous rubric, and calls the LLM using update_rubric.txt template.
        Saves the result to step_{step}.jsonl.

        Args:
            batch:        DataProto after rollout (before reward computation)
            prev_rubrics: List of rubric dicts from the previous rubric file
            step:         Current training step (used for output filename)
            seq_rewards:  Per-trajectory reward numpy array, shape (B,)
            rollout_n:    Number of rollouts per prompt

        Returns:
            Path to saved JSONL file, or None on failure.
        """
        print(f"\n{'='*60}\nUpdating rubric at step {step} (convergence triggered)\n{'='*60}")

        sampled = self.sample_low_variance_rollouts(batch, seq_rewards, rollout_n)
        formatted_rollouts = self._format_rollout_samples(sampled)
        formatted_prev = self._format_prev_rubrics(prev_rubrics)

        prompt = (
            self.update_rubric_template
            .replace("[INSERT PREVIOUS RUBRICS HERE]", formatted_prev)
            .replace("[INSERT ROLLOUT SAMPLES HERE]", formatted_rollouts)
        )

        system_prompt = "You are an expert evaluator and reward-design assistant for RL training."
        print("\n" + "="*80)
        print("【RUBRIC UPDATE】完整送入LLM的prompt：")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")

        response_text, success = self.llm_client.simple_text_call(system_prompt, prompt)
        if not success:
            print(f"[RubricGenerator] LLM call failed during update: {response_text}")
            return None

        rubrics = self._extract_rubrics_from_response(response_text)
        if not rubrics:
            print(f"[RubricGenerator] Failed to extract rubrics from update response.\nRaw:\n{response_text}")
            return None

        output_path = self.output_dir / f"step_{step}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for rubric in rubrics:
                f.write(json.dumps(rubric, ensure_ascii=False) + "\n")

        print(f"[RubricGenerator] Saved {len(rubrics)} updated rubrics to {output_path}")
        return str(output_path)

    def _format_prev_rubrics(self, rubrics: List[Dict]) -> str:
        """Format previous rubric list for insertion into update_rubric.txt template."""
        lines = []
        for i, r in enumerate(rubrics, 1):
            lines.append(
                f"{i}. [{r.get('rubric_id', f'r{i}')}] {r['title']}\n"
                f"   Principle: {r['principle']}"
            )
        return "\n".join(lines)

    def cleanup_future_rubrics(self, resume_step: int) -> None:
        """Delete rubric files generated after resume_step.

        Called on trainer init when resuming from a checkpoint.  Any step_N.jsonl
        with N > resume_step was generated in a training run that did not complete,
        so it should be removed to avoid confusion about when rubrics were switched.

        Args:
            resume_step: The step we are resuming from (inclusive; this file is kept).
        """
        files = glob.glob(str(self.output_dir / "step_*.jsonl"))

        def _step_num(path: str) -> int:
            m = re.search(r"step_(\d+)\.jsonl", path)
            return int(m.group(1)) if m else -1

        deleted = []
        for f in files:
            n = _step_num(f)
            if n > resume_step:
                try:
                    Path(f).unlink()
                    deleted.append(f)
                except OSError as e:
                    print(f"[RubricGenerator] Warning: could not delete {f}: {e}")

        if deleted:
            print(
                f"[RubricGenerator] Cleanup: deleted {len(deleted)} future rubric file(s) "
                f"(resume_step={resume_step}):\n  " + "\n  ".join(deleted)
            )
        else:
            print(f"[RubricGenerator] Cleanup: no future rubric files to delete (resume_step={resume_step}).")

    def _extract_rubrics_from_response(self, response: str) -> List[Dict]:
        """Parse rubric dicts from LLM response text."""
        # Try full JSON array first
        try:
            parsed = json.loads(response)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

        # Extract individual JSON objects
        rubrics = []
        for match in re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL):
            try:
                obj = json.loads(match)
                if "rubric_id" in obj and "title" in obj and "principle" in obj:
                    rubrics.append(obj)
            except json.JSONDecodeError:
                continue
        return rubrics

    # ------------------------------------------------------------------
    # Rubric loading
    # ------------------------------------------------------------------

    @staticmethod
    def _step_num(path: str) -> int:
        m = re.search(r"step_(\d+)\.jsonl", path)
        return int(m.group(1)) if m else -1

    def load_latest_rubrics(self) -> List[Dict]:
        """Load rubrics from the newest step_N.jsonl in output_dir."""
        files = glob.glob(str(self.output_dir / "step_*.jsonl"))
        if not files:
            return []

        latest = max(files, key=self._step_num)
        rubrics = []
        with open(latest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rubrics.append(json.loads(line))

        print(f"[RubricGenerator] Loaded {len(rubrics)} rubrics from {latest}")
        return rubrics

    # ------------------------------------------------------------------
    # Rubric-based scoring
    # ------------------------------------------------------------------

    def _format_rubrics_for_judge(self, rubrics: List[Dict]) -> str:
        """Format rubric list for insertion into judge.txt template."""
        lines = []
        for i, r in enumerate(rubrics, 1):
            lines.append(f"{i}. {r['title']}: {r['principle']}")
        return "\n".join(lines)

    def _parse_judge_response(self, response, num_rubrics: int) -> List[int]:
        """Parse binary list from judge LLM response. Returns list of 0/1."""
        if isinstance(response, list):
            response = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in response
            )
        if not isinstance(response, str):
            response = str(response)
        # Try ast.literal_eval first (handles [0,1,1,0])
        try:
            scores = ast.literal_eval(response.strip())
            if isinstance(scores, list):
                scores = [int(s) for s in scores]
                if all(s in (0, 1) for s in scores):
                    return scores[:num_rubrics]
        except Exception:
            pass

        # Fallback: extract all 0/1 tokens
        matches = re.findall(r'\b[01]\b', response)
        if matches:
            return [int(m) for m in matches[:num_rubrics]]

        return []

    def score_rollout(self, prompt: str, response: str, rubrics: List[Dict],
                      max_parse_retries: int = 3) -> float:
        """Score a single rollout against rubrics using the judge LLM.

        HTTP-level retries with exponential backoff are handled by LLMClient.chat.
        This method adds up to max_parse_retries extra attempts when the LLM
        response is successfully received but cannot be parsed as a binary list.

        Returns normalized score in [-1, 1]:
          all satisfied → 1.0 / half → 0.0 / none → -1.0
        Returns -1.0 if all attempts fail.
        """
        rubric_text = self._format_rubrics_for_judge(rubrics)
        rollout_text = f"Prompt:\n{prompt}\n\nResponse:\n{response}"

        judge_prompt = self.judge_template \
            .replace("[INSERT RUBRICS HERE]", rubric_text) \
            .replace("[INSERT CANDIDATE ROLLOUT HERE]", rollout_text)

        system_prompt = "You are a strict binary judge for RL training."

        # 只打印第一条（线程安全）
        should_print_first = False
        with self._first_judge_lock:
            if not self._first_judge_printed:
                self._first_judge_printed = True
                should_print_first = True

        if should_print_first:
            print("\n" + "="*80)
            print("【JUDGE】第一条rollout送入judge LLM的完整prompt：")
            print("="*80)
            print(judge_prompt)
            print("="*80 + "\n")

        for attempt in range(1, max_parse_retries + 1):
            response_text, success = self.llm_client.simple_text_call(system_prompt, judge_prompt,
                                                                         max_tokens=2048)

            if should_print_first and attempt == 1:
                print("\n" + "="*80)
                print("【JUDGE】judge LLM的原始输出：")
                print("="*80)
                print(response_text)
                print("="*80 + "\n")

            if not success:
                # LLMClient already exhausted its own retries; no point retrying here.
                print(f"[RubricGenerator] Judge LLM call failed after all retries, returning -1.0.")
                return -1.0

            scores = self._parse_judge_response(response_text, len(rubrics))
            if not scores:
                print(
                    f"[RubricGenerator] Failed to parse judge response "
                    f"(parse attempt {attempt}/{max_parse_retries}): {response_text[:200]}"
                )
                if attempt < max_parse_retries:
                    print(f"[RubricGenerator] Retrying judge call for parse fix …")
                continue

            if len(scores) != len(rubrics):
                print(f"[RubricGenerator] Warning: expected {len(rubrics)} scores, got {len(scores)}")

            raw_score = sum(scores) / len(rubrics)  # [0, 1]
            return raw_score * 2 - 1                # [-1, 1]

        print(f"[RubricGenerator] All {max_parse_retries} parse attempts failed, returning -1.0.")
        return -1.0

    def score_batch_to_token_level_tensor(
        self, batch: DataProto, rubrics: List[Dict], max_workers: int = None
    ):
        """Score all rollouts in batch in parallel and return a token-level reward tensor.

        Uses a thread pool with max_workers concurrent judge LLM calls (defaults
        to self.judge_max_workers, configurable via rubric_generation.judge_max_workers).
        The rubric score for each sample is placed at its last valid response
        token. All other positions are 0.

        Args:
            batch: DataProto with prompts, responses, response_mask tensors
            rubrics: List of rubric dicts from load_latest_rubrics()
            max_workers: Max concurrent judge LLM calls (default 32)

        Returns:
            torch.Tensor of shape (batch_size, response_length)
        """
        prompts_tensor = batch.batch["prompts"]
        responses_tensor = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        batch_size = prompts_tensor.shape[0]
        response_length = responses_tensor.shape[1]
        device = prompts_tensor.device

        # 提前解码所有文本（主线程，避免tokenizer并发问题）
        prompt_texts = [
            self.tokenizer.decode(prompts_tensor[i], skip_special_tokens=True)
            for i in range(batch_size)
        ]
        response_texts = [
            self.tokenizer.decode(responses_tensor[i], skip_special_tokens=True)
            for i in range(batch_size)
        ]

        workers = max_workers if max_workers is not None else self.judge_max_workers
        scores = [0.0] * batch_size
        completed = [0]
        lock = threading.Lock()

        def _score_one(idx: int) -> Tuple[int, float]:
            s = self.score_rollout(prompt_texts[idx], response_texts[idx], rubrics)
            with lock:
                completed[0] += 1
                done = completed[0]
            if done % 32 == 0 or done == batch_size:
                print(f"[RubricGenerator] Scored {done}/{batch_size} rollouts")
            return idx, s

        print(f"[RubricGenerator] Scoring {batch_size} rollouts with {len(rubrics)} rubrics "
              f"(max_workers={workers})...")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_score_one, i): i for i in range(batch_size)}
            for future in as_completed(futures):
                idx, s = future.result()
                scores[idx] = s

        reward_tensor = torch.zeros(batch_size, response_length, device=device)
        for i in range(batch_size):
            valid_positions = response_mask[i].nonzero(as_tuple=False)
            if len(valid_positions) > 0:
                last_pos = valid_positions[-1].item()
                reward_tensor[i, last_pos] = scores[i]

        scores_summary = reward_tensor.sum(-1)
        print(f"[RubricGenerator] Batch scores — mean: {scores_summary.mean():.4f}, "
              f"min: {scores_summary.min():.4f}, max: {scores_summary.max():.4f}")
        return reward_tensor
