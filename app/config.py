"""Settings, all from environment (.env). Role→provider routing lives here."""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


# Rough $/MTok pricing for cost logging. Unknown models log cost 0 with tokens
# still recorded. Update as pricing changes.
PRICING = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    # OpenRouter models (keyed exactly as passed in ROLE_*)
    "deepseek/deepseek-v4-flash": (0.09, 0.18),
    "minimax/minimax-m3": (0.30, 1.20),
    "z-ai/glm-5.2": (0.93, 3.00),
}


@dataclass
class RoleTarget:
    provider: str  # local | openrouter | anthropic
    model: str


def _parse_role(value: str, default: str) -> RoleTarget:
    raw = value or default
    provider, _, model = raw.partition(":")
    return RoleTarget(provider=provider, model=model)


@dataclass
class Settings:
    data_dir: str = _env("DATA_DIR", "data")
    db_path: str = field(init=False)

    # Volume tier (OpenAI-compatible: Ollama, llama.cpp, vLLM, OpenRouter...)
    local_base_url: str = _env("LOCAL_BASE_URL", "http://localhost:11434/v1")
    local_api_key: str = _env("LOCAL_API_KEY", "ollama")
    local_model: str = _env("LOCAL_MODEL", "qwen3.5:9b")

    openrouter_base_url: str = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_api_key: str = _env("OPENROUTER_API_KEY")
    openrouter_model: str = _env("OPENROUTER_MODEL", "")  # default model if role omits one

    anthropic_api_key: str = _env("ANTHROPIC_API_KEY")

    # Role routing: "provider" or "provider:model". Model falls back to the
    # provider's default (local_model / openrouter_model; anthropic requires one).
    role_extract: RoleTarget = field(init=False)
    role_score: RoleTarget = field(init=False)
    role_escalate: RoleTarget = field(init=False)

    # Pipeline knobs
    escalate_min_score: int = int(_env("ESCALATE_MIN_SCORE", "55"))
    review_min_score: int = int(_env("REVIEW_MIN_SCORE", "50"))  # below → auto-skip
    monthly_budget_usd: float = float(_env("MONTHLY_BUDGET_USD", "10"))
    budget_fallback_to_local: bool = _env("BUDGET_FALLBACK_TO_LOCAL", "true").lower() == "true"
    pipeline_hour: int = int(_env("PIPELINE_HOUR", "2"))  # local time, daily

    # Off-screen notifications (see app/notify.py). Both optional; unset means
    # that channel silently no-ops.
    discord_webhook_url: str = _env("DISCORD_WEBHOOK_URL")
    # Separate healthchecks.io checks — each is its own dead-man's-switch, so a
    # missed nightly pipeline run and a missed liveness probe alert distinctly
    # instead of one check masking the other's silence. Base URL each; /fail
    # appended on an explicit failure.
    healthchecks_liveness_url: str = _env("HEALTHCHECKS_LIVENESS_URL")
    healthchecks_pipeline_url: str = _env("HEALTHCHECKS_PIPELINE_URL")

    # Shared search geography — one household, one metro. Adzuna queries that
    # omit "where" use this center + radius instead of hand-picking
    # Seattle vs Bellevue per query. Not per-applicant on purpose.
    search_where: str = _env("SEARCH_WHERE", "Seattle, WA")
    search_distance_km: int = int(_env("SEARCH_DISTANCE_KM", "25"))

    # Adzuna credentials (free tier: https://developer.adzuna.com/)
    adzuna_app_id: str = _env("ADZUNA_APP_ID")
    adzuna_app_key: str = _env("ADZUNA_APP_KEY")
    adzuna_country: str = _env("ADZUNA_COUNTRY", "us")

    # USAJobs credentials (free: https://developer.usajobs.gov/apirequest/)
    usajobs_api_key: str = _env("USAJOBS_API_KEY")
    usajobs_email: str = _env("USAJOBS_EMAIL")

    def __post_init__(self):
        self.db_path = os.path.join(self.data_dir, "seeker.db")
        self.role_extract = _parse_role(_env("ROLE_EXTRACT"), "local")
        self.role_score = _parse_role(_env("ROLE_SCORE"), "local")
        self.role_escalate = _parse_role(_env("ROLE_ESCALATE"), "anthropic:claude-haiku-4-5")

    def role(self, name: str) -> RoleTarget:
        return getattr(self, f"role_{name}")


settings = Settings()
