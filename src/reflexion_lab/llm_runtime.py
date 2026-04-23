from __future__ import annotations
import json
import os
from dataclasses import dataclass
from time import perf_counter
from urllib import error, request
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry

def _context_to_text(example: QAExample) -> str:
    lines: list[str] = []
    for idx, chunk in enumerate(example.context, start=1):
        lines.append(f"[{idx}] {chunk.title}: {chunk.text}")
    return "\n".join(lines)

def _safe_total_tokens(payload: dict) -> int:
    usage = payload.get("usage", {})
    total = usage.get("total_tokens")
    if isinstance(total, int) and total > 0:
        return total
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        return max(1, prompt_tokens + completion_tokens)
    return 0

def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM response does not contain valid JSON object.")
        return json.loads(text[start : end + 1])

@dataclass
class OpenAICompatibleClient:
    model: str
    base_url: str
    api_key: str
    timeout_sec: int = 60

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> "OpenAICompatibleClient":
        return cls(
            model=model or os.getenv("LLM_MODEL", "qwen2.5:0.5b"),
            base_url=(base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")).rstrip("/"),
            api_key=api_key or os.getenv("LLM_API_KEY", "dummy"),
        )

    def chat(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> tuple[str, int, int]:
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        started = perf_counter()
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(f"Failed to call LLM endpoint: {exc}") from exc
        latency_ms = max(1, int((perf_counter() - started) * 1000))
        payload = json.loads(raw)
        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("LLM endpoint returned no choices.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        tokens = _safe_total_tokens(payload)
        return content, tokens, latency_ms

def llm_actor_answer(client: OpenAICompatibleClient, example: QAExample, reflection_memory: list[str]) -> tuple[str, int, int]:
    memory_section = "\n".join(reflection_memory) if reflection_memory else "(empty)"
    user_prompt = f"""Question:
{example.question}

Context:
{_context_to_text(example)}

Reflection memory:
{memory_section}
"""
    answer, tokens, latency_ms = client.chat(ACTOR_SYSTEM, user_prompt, json_mode=False)
    return answer.strip(), tokens, latency_ms

def llm_evaluator(client: OpenAICompatibleClient, example: QAExample, answer: str) -> tuple[JudgeResult, int, int]:
    user_prompt = f"""Question:
{example.question}

Gold answer:
{example.gold_answer}

Predicted answer:
{answer}
"""
    raw_json, tokens, latency_ms = client.chat(EVALUATOR_SYSTEM, user_prompt, json_mode=True)
    parsed = _extract_json(raw_json)
    if parsed.get("score") not in (0, 1):
        raise ValueError(f"Evaluator returned invalid score: {parsed.get('score')}")
    result = JudgeResult(
        score=parsed["score"],
        reason=str(parsed.get("reason", "")),
        missing_evidence=[str(item) for item in parsed.get("missing_evidence", [])],
        spurious_claims=[str(item) for item in parsed.get("spurious_claims", [])],
    )
    return result, tokens, latency_ms

def llm_reflector(client: OpenAICompatibleClient, example: QAExample, attempt_id: int, judge: JudgeResult, answer: str) -> tuple[ReflectionEntry, int, int]:
    user_prompt = f"""Question:
{example.question}

Context:
{_context_to_text(example)}

Failed answer (attempt {attempt_id}):
{answer}

Evaluator feedback:
- reason: {judge.reason}
- missing_evidence: {judge.missing_evidence}
- spurious_claims: {judge.spurious_claims}

Return JSON with keys:
attempt_id, failure_reason, lesson, next_strategy
"""
    raw_json, tokens, latency_ms = client.chat(REFLECTOR_SYSTEM, user_prompt, json_mode=True)
    parsed = _extract_json(raw_json)
    reflection = ReflectionEntry(
        attempt_id=int(parsed.get("attempt_id", attempt_id)),
        failure_reason=str(parsed.get("failure_reason", judge.reason)),
        lesson=str(parsed.get("lesson", "Need tighter grounding to context.")),
        next_strategy=str(parsed.get("next_strategy", "Resolve each hop step-by-step and verify final entity.")),
    )
    return reflection, tokens, latency_ms
