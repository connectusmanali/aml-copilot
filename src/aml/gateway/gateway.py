"""The LLM gateway: the ONE path every model call takes.

In order, for each call:
  1. mask PII (account and customer IDs, emails, phones, cards)
  2. compute a call id = hash(task, provider, model, prompt version, masked prompt)
  3. replay mode: return the recorded response for that id (exact case replay, Step 9)
  4. cache: same call id seen before -> recorded response, no new cost
  5. budget: refuse if the case's token budget would be exceeded
  6. call the provider, retrying rate limits and server errors with exponential backoff
  7. record the call (masked text, tokens, latency, cost) through the recorder callback
  8. unmask the response for our own code

Banks route every model call through one service like this: one place to enforce data
handling, cost limits, logging and vendor choice.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from aml.gateway.pii import Masker
from aml.gateway.providers import Completion, Provider, ProviderError

# USD per million tokens (input, output). Approximate list prices; free tier costs nothing,
# but the estimate is what the same traffic would cost on a paid plan.
PRICES: dict[str, tuple[float, float]] = {
    "offline-v1": (0.0, 0.0),
    "gemini-flash-latest": (0.30, 2.50),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_tokens: int = 20_000
    used_tokens: int = 0

    def check(self, estimate: int) -> None:
        if self.used_tokens + estimate > self.max_tokens:
            raise BudgetExceeded(
                f"token budget exceeded: {self.used_tokens:,} used + ~{estimate:,} requested "
                f"> {self.max_tokens:,}"
            )

    def spend(self, tokens: int) -> None:
        self.used_tokens += tokens


@dataclass(frozen=True)
class CallRecord:
    call_id: str
    case_id: str | None
    task: str
    provider: str
    model: str
    prompt_version: str
    response: str  # MASKED
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    cached: bool = False


@dataclass
class LLMGateway:
    provider: Provider
    recorder: Callable[[CallRecord], None] | None = None
    replay: dict[str, str] | None = None  # call_id -> recorded masked response
    max_retries: int = 3
    backoff_s: float = 1.0
    sleep: Callable[[float], None] = time.sleep
    _cache: dict[str, str] = field(default_factory=dict)
    calls: list[CallRecord] = field(default_factory=list)

    def complete(
        self,
        task: str,
        system: str,
        user: str,
        *,
        prompt_version: str,
        budget: Budget,
        case_id: str | None = None,
        max_tokens: int = 1_500,
        json_mode: bool = False,
    ) -> str:
        """One LLM call, traced as a GenAI span (model, tokens, cost; never the prompt text)."""
        from aml.ops.tracing import span

        n = len(self.calls)
        with span(
            "llm.complete",
            **{
                "gen_ai.system": self.provider.name,
                "gen_ai.request.model": self.provider.model, 
                "aml.task": task,
                "aml.case_id": case_id,
                "aml.prompt_version": prompt_version,
            },
        ) as sp:
            out = self._complete(
                task,
                system,
                user,
                prompt_version=prompt_version,
                budget=budget,
                case_id=case_id,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
            if len(self.calls) > n:
                rec = self.calls[-1]
                sp.set_attribute("gen_ai.usage.input_tokens", rec.input_tokens)
                sp.set_attribute("gen_ai.usage.output_tokens", rec.output_tokens)
                sp.set_attribute("aml.cost_usd", rec.cost_usd)
                sp.set_attribute("aml.cached", rec.cached)
            return out

    def _complete(
        self,
        task: str,
        system: str,
        user: str,
        *,
        prompt_version: str,
        budget: Budget,
        case_id: str | None = None,
        max_tokens: int = 1_500,
        json_mode: bool = False,
    ) -> str:
        masker = Masker()
        m_system, m_user = masker.mask(system), masker.mask(user)
        call_id = hashlib.sha256(
            json.dumps(
                [task, self.provider.name, self.provider.model, prompt_version, m_system, m_user]
            ).encode()
        ).hexdigest()[:32]

        if self.replay is not None:
            if call_id not in self.replay:
                raise KeyError(f"replay: no recorded response for call {call_id} ({task})")
            return masker.unmask(self.replay[call_id])
        if call_id in self._cache:
            self._record(
                call_id, case_id, task, prompt_version, self._cache[call_id], 0, 0, 0, cached=True
            )
            return masker.unmask(self._cache[call_id])

        budget.check(len((m_system + m_user).split()) * 2 + max_tokens)
        started = time.perf_counter()
        completion = self._with_retries(m_system, m_user, max_tokens, json_mode)
        latency = int((time.perf_counter() - started) * 1000)
        budget.spend(completion.input_tokens + completion.output_tokens)
        self._cache[call_id] = completion.text
        self._record(
            call_id,
            case_id,
            task,
            prompt_version,
            completion.text,
            completion.input_tokens,
            completion.output_tokens,
            latency,
        )
        return masker.unmask(completion.text)

    def _with_retries(self, system: str, user: str, max_tokens: int, json_mode: bool) -> Completion:
        for attempt in range(self.max_retries + 1):
            try:
                return self.provider.complete(
                    system, user, max_tokens=max_tokens, json_mode=json_mode
                )
            except ProviderError as e:
                if not e.retryable or attempt == self.max_retries:
                    raise
                self.sleep(self.backoff_s * 2**attempt)
        raise AssertionError("unreachable")

    def _record(
        self,
        call_id: str,
        case_id: str | None,
        task: str,
        prompt_version: str,
        response: str,
        tin: int,
        tout: int,
        latency: int,
        cached: bool = False,
    ) -> None:
        pin, pout = PRICES.get(self.provider.model, (0.0, 0.0))
        rec = CallRecord(
            call_id,
            case_id,
            task,
            self.provider.name,
            self.provider.model,
            prompt_version,
            response,
            tin,
            tout,
            latency,
            round((tin * pin + tout * pout) / 1e6, 6),
            cached,
        )
        self.calls.append(rec)
        if self.recorder and not cached:
            self.recorder(rec)

    def total_cost(self) -> float:
        return round(sum(c.cost_usd for c in self.calls), 6)

    def summary(self) -> dict[str, Any]:
        return {
            "calls": len(self.calls),
            "cached": sum(c.cached for c in self.calls),
            "input_tokens": sum(c.input_tokens for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
            "cost_usd": self.total_cost(),
        }


def record_to_dict(rec: CallRecord) -> dict[str, Any]:
    return asdict(rec)


def build_gateway(recorder: Callable[[CallRecord], None] | None = None) -> LLMGateway:
    """Gateway for the configured provider (AML_LLM_PROVIDER: offline | gemini | anthropic)."""
    from aml.config import get_settings
    from aml.gateway.providers import AnthropicProvider, GeminiProvider, OfflineProvider

    s = get_settings()
    key = s.llm_api_key.get_secret_value() if s.llm_api_key else ""
    provider: Provider
    if s.llm_provider == "gemini" and key:
        provider = GeminiProvider(key, s.llm_model or "gemini-flash-latest", s.llm_embed_model)
    elif s.llm_provider == "anthropic" and key:
        provider = AnthropicProvider(key, s.llm_model or "claude-haiku-4-5-20251001")
    else:
        provider = OfflineProvider()
    return LLMGateway(provider, recorder=recorder)
