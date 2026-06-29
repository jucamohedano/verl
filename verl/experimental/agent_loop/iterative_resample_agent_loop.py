# Copyright 2026
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

import asyncio
import re
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer


@register("iterative_resample")
class IterativeResampleAgentLoop(AgentLoopBase):
    """Evaluation-only loop for grouped test-time resampling.

    Each round samples several independent attempts from the current context. If
    all attempts fail a simple label match, the failed outputs are fed back as a
    user message and the next round samples again.

    Use ``reward_extra_info`` / ``iter_final_prediction`` for downstream metrics;
    full decoded ``response_ids`` may include chat-template artifacts (e.g. role
    markers) when parsing validation dumps — strip or infer offline if needed.
    """

    def __init__(
        self,
        *args,
        attempts_per_round: int = 5,
        max_rounds: int = 3,
        per_attempt_max_tokens: Optional[int] = 768,
        max_feedback_chars: int = 6000,
        enable_feedback: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if attempts_per_round <= 0:
            raise ValueError("attempts_per_round must be positive")
        if max_rounds <= 0:
            raise ValueError("max_rounds must be positive")

        self.attempts_per_round = attempts_per_round
        self.max_rounds = max_rounds
        self.per_attempt_max_tokens = per_attempt_max_tokens
        self.max_feedback_chars = max_feedback_chars
        self.enable_feedback = enable_feedback
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        ground_truth = self._extract_ground_truth(kwargs)

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        prompt_ids = await self.apply_chat_template(messages, images=images, videos=videos)
        current_prompt_ids = list(prompt_ids)
        request_root = uuid4().hex

        response_ids: list[int] = []
        response_mask: list[int] = []
        attempts: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {}

        success = False
        success_round = 0
        total_attempts = 0
        best_response = ""
        best_response_ids: list[int] = []

        for round_idx in range(1, self.max_rounds + 1):
            outputs = await self._generate_round(
                current_prompt_ids=current_prompt_ids,
                sampling_params=sampling_params,
                request_root=request_root,
                round_idx=round_idx,
                image_data=images,
                video_data=videos,
                metrics=metrics,
            )

            failed_texts: list[str] = []
            for attempt_idx, output in enumerate(outputs, start=1):
                total_attempts += 1
                text = await self._decode(output.token_ids)
                matched = self._matches_label(text, ground_truth)
                attempts.append(
                    {
                        "round": round_idx,
                        "attempt": attempt_idx,
                        "matched": matched,
                        "text": text,
                    }
                )

                if matched:
                    if not success:
                        success = True
                        success_round = round_idx
                        best_response = text
                        best_response_ids = output.token_ids
                else:
                    failed_texts.append(text)

            if success:
                self._append_text(response_ids, response_mask, f"\n\n[round {success_round} successful attempt]\n", 0)
                self._append_tokens(response_ids, response_mask, best_response_ids, 1)
                break

            if self.enable_feedback:
                feedback = self._build_feedback(round_idx, failed_texts, final_round=round_idx == self.max_rounds)
                feedback_ids = await self.apply_chat_template(
                    [{"role": "user", "content": feedback}],
                    remove_system_prompt=True,
                )
                self._append_tokens(response_ids, response_mask, feedback_ids, 0)

                if round_idx == self.max_rounds:
                    break

                current_prompt_ids += feedback_ids
            else:
                if round_idx == self.max_rounds:
                    break

        final_prediction = best_response if success else (attempts[-1]["text"] if attempts else "")

        # Ensure response_ids is non-empty to satisfy downstream padding logic.
        if not response_ids:
            self._append_text(response_ids, response_mask, "\n", 0)

        extra_fields = {
            "extras": {
                "attempts": attempts,
                "ground_truth": ground_truth,
                "best_response": best_response,
                "final_prediction": final_prediction,
            },
            "reward_extra_info": {
                "iter_success": float(success),
                "iter_success_round": success_round,
                "iter_attempts_used": total_attempts,
                "iter_best_response": best_response,
                "iter_final_prediction": final_prediction,
            },
        }

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            reward_score=float(success),
            num_turns=1 + (success_round or self.max_rounds),
            metrics=metrics,
            extra_fields=extra_fields,
        )

    async def _generate_round(
        self,
        *,
        current_prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_root: str,
        round_idx: int,
        image_data,
        video_data,
        metrics: dict[str, Any],
    ):
        params = dict(sampling_params)
        if self.per_attempt_max_tokens is not None:
            params["max_tokens"] = self.per_attempt_max_tokens

        async def generate_one(attempt_idx: int):
            with simple_timer("generate_sequences", metrics):
                return await self.server_manager.generate(
                    request_id=f"{request_root}-r{round_idx}-a{attempt_idx}",
                    prompt_ids=current_prompt_ids,
                    sampling_params=params,
                    image_data=image_data,
                    video_data=video_data,
                )

        return await asyncio.gather(*(generate_one(i) for i in range(1, self.attempts_per_round + 1)))

    def _build_feedback(self, round_idx: int, failed_texts: list[str], *, final_round: bool) -> str:
        chunks = []
        for idx, text in enumerate(failed_texts, start=1):
            chunks.append(f"Attempt {idx}:\n{self._clip(text)}")
        failed_block = "\n\n".join(chunks)
        if final_round:
            return (
                f"All sampled answers in final round {round_idx} were incorrect.\n\n"
                f"Failed attempts:\n{failed_block}"
            )
        return (
            f"All sampled answers in round {round_idx} were incorrect.\n\n"
            f"Failed attempts:\n{failed_block}\n\n"
            "Use these failed attempts as evidence. Reflect on what they missed, then answer again. "
            "State the final class label clearly."
        )

    def _append_tokens(self, response_ids: list[int], response_mask: list[int], token_ids: list[int], mask_value: int):
        remaining = self.response_length - len(response_ids)
        if remaining <= 0:
            return
        token_ids = token_ids[:remaining]
        response_ids.extend(token_ids)
        response_mask.extend([mask_value] * len(token_ids))

    def _append_text(self, response_ids: list[int], response_mask: list[int], text: str, mask_value: int):
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        self._append_tokens(response_ids, response_mask, token_ids, mask_value)

    async def _decode(self, token_ids: list[int]) -> str:
        return await self.loop.run_in_executor(None, lambda: self.tokenizer.decode(token_ids, skip_special_tokens=True))

    def _clip(self, text: str) -> str:
        if len(text) <= self.max_feedback_chars:
            return text
        return text[: self.max_feedback_chars] + "\n...(truncated)"

    @staticmethod
    def _extract_ground_truth(kwargs: dict[str, Any]) -> str:
        reward_model = kwargs.get("reward_model", {})
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get("ground_truth", "")
        else:
            ground_truth = ""
        if isinstance(ground_truth, (list, tuple)):
            ground_truth = ground_truth[0] if ground_truth else ""
        return str(ground_truth)

    @classmethod
    def _matches_label(cls, text: str, ground_truth: str) -> bool:
        label = cls._normalise(ground_truth)
        if not label:
            return False
        answer_region = text.split("</think>")[-1]
        answer = cls._normalise(answer_region)
        return re.search(rf"(^| ){re.escape(label)}( |$)", answer) is not None

    @staticmethod
    def _normalise(text: str) -> str:
        text = text.lower().replace("_", " ").replace("-", " ")
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()
