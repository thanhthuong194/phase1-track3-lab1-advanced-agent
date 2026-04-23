from __future__ import annotations
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal
from .llm_runtime import OpenAICompatibleClient, llm_actor_answer, llm_evaluator, llm_reflector
from .mock_runtime import FAILURE_MODE_BY_QID, actor_answer, evaluator, reflector
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord
from .utils import count_tokens

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    runtime_mode: Literal["mock", "llm"] = "mock"
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    _client: OpenAICompatibleClient | None = field(default=None, init=False, repr=False)

    def _get_client(self) -> OpenAICompatibleClient:
        if self.runtime_mode != "llm":
            raise RuntimeError("LLM client is only available in llm mode.")
        if self._client is None:
            self._client = OpenAICompatibleClient.from_env(
                model=self.llm_model,
                base_url=self.llm_base_url,
                api_key=self.llm_api_key,
            )
        return self._client

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            attempt_started = perf_counter()
            if self.runtime_mode == "llm":
                client = self._get_client()
                answer, actor_tokens, actor_latency_ms = llm_actor_answer(client, example, reflection_memory)
                judge, evaluator_tokens, evaluator_latency_ms = llm_evaluator(client, example, answer)
                # TODO: Replace with actual token count from LLM response
                token_estimate = actor_tokens + evaluator_tokens
                if token_estimate <= 0:
                    token_estimate = count_tokens(
                        [
                            ACTOR_SYSTEM,
                            EVALUATOR_SYSTEM,
                            example.question,
                            answer,
                            judge.reason,
                            *judge.missing_evidence,
                            *judge.spurious_claims,
                            *reflection_memory,
                        ]
                    )
                # TODO: Replace with actual latency measurement
                latency_ms = actor_latency_ms + evaluator_latency_ms
            else:
                answer = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
                judge = evaluator(example, answer)
                # TODO: Replace with actual token count from LLM response
                token_estimate = count_tokens(
                    [
                        ACTOR_SYSTEM,
                        EVALUATOR_SYSTEM,
                        example.question,
                        answer,
                        judge.reason,
                        *judge.missing_evidence,
                        *judge.spurious_claims,
                        *reflection_memory,
                    ]
                )
                # TODO: Replace with actual latency measurement
                latency_ms = max(1, int((perf_counter() - attempt_started) * 1000))
            latency_ms = max(latency_ms, max(1, int((perf_counter() - attempt_started) * 1000)))
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)
            final_answer = answer
            final_score = judge.score
            if judge.score == 1:
                traces.append(trace)
                break
            
            # TODO: Học viên triển khai logic Reflexion tại đây
            # 1. Kiểm tra nếu agent_type là 'reflexion' và chưa hết số lần attempt
            # 2. Gọi hàm reflector để lấy nội dung reflection
            # 3. Cập nhật reflection_memory để Actor dùng cho lần sau
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                if self.runtime_mode == "llm":
                    reflection, reflection_tokens, reflection_latency_ms = llm_reflector(self._get_client(), example, attempt_id, judge, answer)
                    if reflection_tokens <= 0:
                        reflection_tokens = count_tokens(
                            [
                                REFLECTOR_SYSTEM,
                                reflection.failure_reason,
                                reflection.lesson,
                                reflection.next_strategy,
                            ]
                        )
                else:
                    reflection_started = perf_counter()
                    reflection = reflector(example, attempt_id, judge)
                    reflection_latency_ms = max(1, int((perf_counter() - reflection_started) * 1000))
                    reflection_tokens = count_tokens(
                        [
                            REFLECTOR_SYSTEM,
                            reflection.failure_reason,
                            reflection.lesson,
                            reflection.next_strategy,
                        ]
                    )
                reflections.append(reflection)
                reflection_memory.append(f"Attempt {reflection.attempt_id}: {reflection.lesson} | Strategy: {reflection.next_strategy}")
                if len(reflection_memory) > 3:
                    reflection_memory = reflection_memory[-3:]
                trace.reflection = reflection
                trace.token_estimate += reflection_tokens
                trace.latency_ms += reflection_latency_ms
            traces.append(trace)
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(
        self,
        runtime_mode: Literal["mock", "llm"] = "mock",
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
    ) -> None:
        super().__init__(
            agent_type="react",
            max_attempts=1,
            runtime_mode=runtime_mode,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
        )

class ReflexionAgent(BaseAgent):
    def __init__(
        self,
        max_attempts: int = 3,
        runtime_mode: Literal["mock", "llm"] = "mock",
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
    ) -> None:
        super().__init__(
            agent_type="reflexion",
            max_attempts=max_attempts,
            runtime_mode=runtime_mode,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
        )
