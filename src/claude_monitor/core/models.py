"""Data models for Claude Monitor.
Core data structures for usage tracking, session management, and token calculations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class Provider(Enum):
    """Supported AI model providers."""

    ANTHROPIC = "anthropic"
    Z_AI = "z_ai"
    OPENAI = "openai"
    GOOGLE = "google"
    CUSTOM = "custom"


class CostMode(Enum):
    """Cost calculation modes for token usage analysis."""

    AUTO = "auto"
    CACHED = "cached"
    CALCULATED = "calculate"


@dataclass
class UsageEntry:
    """Individual usage record from AI model usage data."""

    timestamp: datetime
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    provider: Provider = Provider.ANTHROPIC
    message_id: str = ""
    request_id: str = ""


@dataclass
class TokenCounts:
    """Token aggregation structure with computed totals."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Get total tokens across all types."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_tokens
            + self.cache_read_tokens
        )


@dataclass
class BurnRate:
    """Token consumption rate metrics."""

    tokens_per_minute: float
    cost_per_hour: float


@dataclass
class UsageProjection:
    """Usage projection calculations for active blocks."""

    projected_total_tokens: int
    projected_total_cost: float
    remaining_minutes: float


@dataclass
class SessionBlock:
    """Aggregated session block representing a 5-hour period."""

    id: str
    start_time: datetime
    end_time: datetime
    entries: List[UsageEntry] = field(default_factory=list)
    token_counts: TokenCounts = field(default_factory=TokenCounts)
    is_active: bool = False
    is_gap: bool = False
    burn_rate: Optional[BurnRate] = None
    actual_end_time: Optional[datetime] = None
    per_model_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    models: List[str] = field(default_factory=list)
    sent_messages_count: int = 0
    cost_usd: float = 0.0
    limit_messages: List[Dict[str, Any]] = field(default_factory=list)
    projection_data: Optional[Dict[str, Any]] = None
    burn_rate_snapshot: Optional[BurnRate] = None

    @property
    def total_tokens(self) -> int:
        """Get total tokens from token_counts."""
        return self.token_counts.total_tokens

    @property
    def total_cost(self) -> float:
        """Get total cost - alias for cost_usd."""
        return self.cost_usd

    @property
    def duration_minutes(self) -> float:
        """Get duration in minutes."""
        if self.actual_end_time:
            duration = (self.actual_end_time - self.start_time).total_seconds() / 60
        else:
            duration = (self.end_time - self.start_time).total_seconds() / 60
        return max(duration, 1.0)


def normalize_model_name(model: str, provider: Provider = Provider.ANTHROPIC) -> str:
    """Normalize model name for consistent usage across the application.

    Handles various model name formats and maps them to standard keys.
    Now supports multiple providers and their model naming conventions.

    Args:
        model: Raw model name from usage data
        provider: Provider enum for provider-specific normalization

    Returns:
        Normalized model key

    Examples:
        >>> normalize_model_name("claude-3-opus-20240229")
        'claude-3-opus'
        >>> normalize_model_name("Claude 3.5 Sonnet")
        'claude-3-5-sonnet'
        >>> normalize_model_name("glm-4", Provider.Z_AI)
        'glm-4'
    """
    if not model:
        return ""

    model_lower = model.lower()

    # Handle provider-specific normalization
    if provider == Provider.Z_AI:
        return _normalize_z_ai_model(model_lower)
    elif provider == Provider.OPENAI:
        return _normalize_openai_model(model_lower)
    elif provider == Provider.GOOGLE:
        return _normalize_google_model(model_lower)
    elif provider == Provider.ANTHROPIC:
        return _normalize_anthropic_model(model_lower)
    else:
        # Default to basic cleaning for custom providers
        return model_lower.strip()


def _normalize_anthropic_model(model_lower: str) -> str:
    """Normalize Anthropic Claude model names."""
    if (
        "claude-opus-4-" in model_lower
        or "claude-sonnet-4-" in model_lower
        or "claude-haiku-4-" in model_lower
        or "sonnet-4-" in model_lower
        or "opus-4-" in model_lower
        or "haiku-4-" in model_lower
    ):
        return model_lower

    if "opus" in model_lower:
        if "4-" in model_lower:
            return model_lower
        return "claude-3-opus"
    if "sonnet" in model_lower:
        if "4-" in model_lower:
            return model_lower
        if "3.5" in model_lower or "3-5" in model_lower:
            return "claude-3-5-sonnet"
        return "claude-3-sonnet"
    if "haiku" in model_lower:
        if "3.5" in model_lower or "3-5" in model_lower:
            return "claude-3-5-haiku"
        return "claude-3-haiku"

    return model_lower


def _normalize_z_ai_model(model_lower: str) -> str:
    """Normalize Z.ai GLM model names."""
    if "glm-4.5-x" in model_lower:
        return "glm-4.5-x"
    elif "glm-4.5-airx" in model_lower:
        return "glm-4.5-airx"
    elif "glm-4.5-air" in model_lower:
        return "glm-4.5-air"
    elif "glm-4.5v" in model_lower or "glm-4.5-v" in model_lower:
        return "glm-4.5v"
    elif "glm-4.5-flash" in model_lower:
        return "glm-4.5-flash"
    elif "glm-4.5" in model_lower:
        return "glm-4.5"
    elif "glm-4-32b-0414-128k" in model_lower:
        return "glm-4-32b-0414-128k"
    elif "glm-4" in model_lower:
        return "glm-4"
    elif "glm-3" in model_lower:
        return "glm-3"
    elif "glm" in model_lower:
        return "glm"
    return model_lower


def _normalize_openai_model(model_lower: str) -> str:
    """Normalize OpenAI model names."""
    if "gpt-4" in model_lower:
        return "gpt-4"
    if "gpt-3.5" in model_lower or "gpt-3-5" in model_lower:
        return "gpt-3.5"
    if "gpt-3" in model_lower:
        return "gpt-3"
    return model_lower


def _normalize_google_model(model_lower: str) -> str:
    """Normalize Google model names."""
    if "gemini-pro" in model_lower:
        return "gemini-pro"
    if "gemini" in model_lower:
        return "gemini"
    return model_lower


def detect_provider_from_model(model: str) -> Provider:
    """Detect provider based on model name patterns.

    Args:
        model: Model name to analyze

    Returns:
        Detected Provider enum
    """
    if not model:
        return Provider.ANTHROPIC  # Default fallback

    model_lower = model.lower()
    
    if any(keyword in model_lower for keyword in ["claude", "anthropic"]):
        return Provider.ANTHROPIC
    elif any(keyword in model_lower for keyword in ["glm", "z_ai"]):
        return Provider.Z_AI
    elif any(keyword in model_lower for keyword in ["gpt", "openai"]):
        return Provider.OPENAI
    elif any(keyword in model_lower for keyword in ["gemini", "google", "palm"]):
        return Provider.GOOGLE
    
    return Provider.CUSTOM
