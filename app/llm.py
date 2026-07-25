"""Provider-agnostic LLM routing.

Roles (extract/score/escalate/write) each map to a provider+model via .env.
Providers:
  - "local":      any OpenAI-compatible endpoint (Ollama, llama.cpp, vLLM)
  - "openrouter": same wire format, remote — the "fuck it, burn credits" switch
  - "anthropic":  native Anthropic SDK for the quality tier

Paid calls check the monthly budget first; over budget they either fall back
to local (BUDGET_FALLBACK_TO_LOCAL=true) or raise BudgetExceeded.
"""
import json
import re

import anthropic
from openai import OpenAI

from . import db
from .config import PRICING, RoleTarget, settings

PAID_PROVIDERS = {"anthropic", "openrouter"}


class BudgetExceeded(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


TIMEOUT_S = 120.0  # a hung free-tier endpoint should fail visibly, not stall for 10min

_local = OpenAI(base_url=settings.local_base_url, api_key=settings.local_api_key,
                timeout=TIMEOUT_S, max_retries=1)
_openrouter = (
    OpenAI(base_url=settings.openrouter_base_url, api_key=settings.openrouter_api_key,
           timeout=TIMEOUT_S, max_retries=1)
    if settings.openrouter_api_key
    else None
)
_anthropic = (
    anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=TIMEOUT_S)
    if settings.anthropic_api_key
    else None
)


def _cost(model: str, in_tok: int, out_tok: int) -> float:
    if model in PRICING:
        pin, pout = PRICING[model]
        return in_tok / 1e6 * pin + out_tok / 1e6 * pout
    return 0.0


def _resolve(role: str) -> RoleTarget:
    target = settings.role(role)
    if target.provider in PAID_PROVIDERS:
        with db.connect() as conn:
            spent = db.monthly_spend(conn)
        if spent >= settings.monthly_budget_usd:
            if settings.budget_fallback_to_local:
                return RoleTarget("local", settings.local_model)
            raise BudgetExceeded(
                f"Monthly budget ${settings.monthly_budget_usd:.2f} reached (${spent:.2f} spent); "
                f"refusing paid call for role '{role}'."
            )
    if not target.model:
        default = {"local": settings.local_model, "openrouter": settings.openrouter_model}.get(target.provider)
        if not default:
            raise LLMError(f"Role '{role}' → provider '{target.provider}' needs an explicit model.")
        target = RoleTarget(target.provider, default)
    return target


def _call_target(role: str, target: RoleTarget, system: str, user: str, max_tokens: int) -> tuple[str, bool]:
    """One completion attempt against a specific provider+model. Raises on any
    transport/API failure or an empty completion — never falls back itself,
    that's _complete_raw's job so it can decide whether a fallback is allowed."""
    if target.provider == "anthropic":
        if _anthropic is None:
            raise LLMError("ANTHROPIC_API_KEY not set but a role routes to anthropic.")
        resp = _anthropic.messages.create(
            model=target.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
        text = "".join(b.text for b in resp.content if b.type == "text")
        truncated = resp.stop_reason == "max_tokens"
    else:
        client = _local if target.provider == "local" else _openrouter
        if client is None:
            raise LLMError(f"Provider '{target.provider}' not configured (missing API key?).")
        resp = client.chat.completions.create(
            model=target.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = resp.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        text = resp.choices[0].message.content or ""
        truncated = resp.choices[0].finish_reason == "length"

    db.log_usage(role, target.provider, target.model, in_tok, out_tok, _cost(target.model, in_tok, out_tok))
    if not text.strip():
        raise LLMError(f"Empty completion from {target.provider}:{target.model} for role '{role}'.")
    return text, truncated


def _complete_raw(role: str, system: str, user: str, max_tokens: int) -> tuple[str, bool]:
    """Returns (text, truncated) — truncated means the model hit max_tokens.

    A remote target (openrouter/anthropic) that fails outright — network
    error, 5xx, empty completion — falls back once to the local tier rather
    than failing the whole eval. This is separate from the budget-triggered
    fallback in _resolve: that one avoids spending money you don't have,
    this one covers "OpenRouter/Anthropic is just down right now" so an
    unattended overnight run degrades to local quality instead of erroring
    out entirely. Local itself has no further fallback — if it's also down,
    that's a real failure and should raise."""
    target = _resolve(role)
    try:
        return _call_target(role, target, system, user, max_tokens)
    except Exception as e:
        if target.provider == "local":
            raise
        from . import notify
        fallback = RoleTarget("local", settings.local_model)
        print(f"[llm] {target.provider}:{target.model} failed for role '{role}' "
              f"({e}); falling back to {fallback.provider}:{fallback.model}")
        notify.notify(
            "seeker: LLM fallback to local",
            f"role '{role}' — {target.provider}:{target.model} failed ({str(e)[:200]}), used local instead",
        )
        return _call_target(role, fallback, system, user, max_tokens)


def complete(role: str, system: str, user: str, max_tokens: int = 2000) -> str:
    """If the output hit max_tokens, retry once with double budget. Reasoning
    models (MiniMax, GLM) spend thinking tokens against the same cap, so the
    first budget guess can be far too small for the visible output."""
    text, truncated = _complete_raw(role, system, user, max_tokens)
    if truncated:
        text, truncated = _complete_raw(role, system, user, max_tokens * 2)
        if truncated:
            raise LLMError(
                f"{role} output hit max_tokens even after a doubled retry "
                f"({max_tokens * 2}). Tail: ...{text[-200:]}")
    return text


def _try_parse(text: str) -> dict | None:
    # strict=False: tolerate literal newlines/tabs inside strings — long
    # markdown-in-JSON outputs routinely have one, and it's not worth a retry.
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0), strict=False)
        except json.JSONDecodeError:
            pass
    return None


def complete_json(role: str, system: str, user: str, max_tokens: int = 2000) -> dict:
    """Complete and parse a JSON object, tolerating fenced/prefixed output.
    If the output was truncated at max_tokens, retry once with double budget."""
    sys_json = system + "\nRespond with a single JSON object and nothing else. Be concise."
    text, truncated = _complete_raw(role, sys_json, user, max_tokens)
    parsed = _try_parse(text)
    if parsed is None and truncated:
        text, truncated = _complete_raw(role, sys_json, user, max_tokens * 2)
        parsed = _try_parse(text)
    if parsed is None:
        why = ("output hit max_tokens even after a doubled retry — JSON truncated"
               if truncated else "output was not valid JSON")
        raise LLMError(f"Role '{role}': {why}. Tail: ...{text[-200:]}")
    return parsed
