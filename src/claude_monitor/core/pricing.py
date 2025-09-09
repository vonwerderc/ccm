"""Pricing calculations for multiple AI model providers.

This module provides the PricingCalculator class for calculating costs
based on token usage and model pricing. It supports multiple providers
including Anthropic (Claude), Z.ai (GLM), OpenAI, and Google models,
with both simple and detailed cost calculations with caching.
"""

from typing import Any, Dict, Optional

from claude_monitor.core.models import CostMode, Provider, TokenCounts, normalize_model_name, detect_provider_from_model


class PricingCalculator:
    """Calculates costs based on model pricing with caching support.

    This class provides methods for calculating costs for individual models/tokens
    as well as detailed cost breakdowns for collections of usage entries.
    It supports custom pricing configurations and caches calculations for performance.

    Features:
    - Configurable pricing (from config or custom)
    - Fallback hardcoded pricing for robustness
    - Multi-provider support (Anthropic, Z.ai, OpenAI, Google)
    - Caching for performance
    - Support for all token types including cache
    - Backward compatible with both APIs
    """

    FALLBACK_PRICING: Dict[str, Dict[str, float]] = {
        # Anthropic Claude models (per 1M tokens)
        "claude-3-opus": {
            "input": 15.0,
            "output": 75.0,
            "cache_creation": 18.75,
            "cache_read": 1.5,
        },
        "claude-3-sonnet": {
            "input": 3.0,
            "output": 15.0,
            "cache_creation": 3.75,
            "cache_read": 0.3,
        },
        "claude-3-haiku": {
            "input": 0.25,
            "output": 1.25,
            "cache_creation": 0.3,
            "cache_read": 0.03,
        },
        "claude-3-5-sonnet": {
            "input": 3.0,
            "output": 15.0,
            "cache_creation": 3.75,
            "cache_read": 0.3,
        },
        "claude-3-5-haiku": {
            "input": 0.25,
            "output": 1.25,
            "cache_creation": 0.3,
            "cache_read": 0.03,
        },
        # Z.ai GLM models (per 1M tokens) - accurate pricing from Z.ai docs
        "glm-4.5": {
            "input": 0.6,
            "output": 2.2,
            "cache_creation": 0.75,
            "cache_read": 0.06,
        },
        "glm-4.5v": {
            "input": 0.6,
            "output": 1.8,
            "cache_creation": 0.75,
            "cache_read": 0.06,
        },
        "glm-4.5-x": {
            "input": 2.2,
            "output": 8.9,
            "cache_creation": 2.75,
            "cache_read": 0.22,
        },
        "glm-4.5-air": {
            "input": 0.2,
            "output": 1.1,
            "cache_creation": 0.25,
            "cache_read": 0.02,
        },
        "glm-4.5-airx": {
            "input": 1.1,
            "output": 4.5,
            "cache_creation": 1.375,
            "cache_read": 0.11,
        },
        "glm-4-32b-0414-128k": {
            "input": 0.1,
            "output": 0.1,
            "cache_creation": 0.125,
            "cache_read": 0.01,
        },
        "glm-4.5-flash": {
            "input": 0.0,
            "output": 0.0,
            "cache_creation": 0.0,
            "cache_read": 0.0,
        },
        # Legacy GLM-4 pricing (for backward compatibility)
        "glm-4": {
            "input": 0.6,
            "output": 2.2,
            "cache_creation": 0.75,
            "cache_read": 0.06,
        },
        # OpenAI models (per 1M tokens)
        "gpt-4": {
            "input": 30.0,
            "output": 60.0,
            "cache_creation": 37.5,
            "cache_read": 3.0,
        },
        "gpt-3.5": {
            "input": 0.5,
            "output": 1.5,
            "cache_creation": 0.625,
            "cache_read": 0.05,
        },
        # Google models (per 1M tokens) - estimated pricing
        "gemini-pro": {
            "input": 1.0,
            "output": 2.0,
            "cache_creation": 1.25,
            "cache_read": 0.1,
        },
    }

    def __init__(
        self, custom_pricing: Optional[Dict[str, Dict[str, float]]] = None
    ) -> None:
        """Initialize with optional custom pricing.

        Args:
            custom_pricing: Optional custom pricing dictionary to override defaults.
                          Should follow same structure as FALLBACK_PRICING.
        """
        # Use fallback pricing if no custom pricing provided
        self.pricing: Dict[str, Dict[str, float]] = custom_pricing or self.FALLBACK_PRICING.copy()
        self._cost_cache: Dict[str, float] = {}

    def calculate_cost(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        tokens: Optional[TokenCounts] = None,
        strict: bool = False,
    ) -> float:
        """Calculate cost with flexible API supporting both signatures.

        Args:
            model: Model name
            input_tokens: Number of input tokens (ignored if tokens provided)
            output_tokens: Number of output tokens (ignored if tokens provided)
            cache_creation_tokens: Number of cache creation tokens
            cache_read_tokens: Number of cache read tokens
            tokens: Optional TokenCounts object (takes precedence)

        Returns:
            Total cost in USD
        """
        # Handle synthetic model
        if model == "<synthetic>":
            return 0.0

        # Support TokenCounts object
        if tokens is not None:
            input_tokens = tokens.input_tokens
            output_tokens = tokens.output_tokens
            cache_creation_tokens = tokens.cache_creation_tokens
            cache_read_tokens = tokens.cache_read_tokens

        # Create cache key
        cache_key = (
            f"{model}:{input_tokens}:{output_tokens}:"
            f"{cache_creation_tokens}:{cache_read_tokens}"
        )

        # Check cache
        if cache_key in self._cost_cache:
            return self._cost_cache[cache_key]

        # Get pricing for model
        pricing = self._get_pricing_for_model(model, strict=strict)

        # Calculate costs (pricing is per million tokens)
        cost = (
            (input_tokens / 1_000_000) * pricing["input"]
            + (output_tokens / 1_000_000) * pricing["output"]
            + (cache_creation_tokens / 1_000_000)
            * pricing.get("cache_creation", pricing["input"] * 1.25)
            + (cache_read_tokens / 1_000_000)
            * pricing.get("cache_read", pricing["input"] * 0.1)
        )

        # Round to 6 decimal places
        cost = round(cost, 6)

        # Cache result
        self._cost_cache[cache_key] = cost
        return cost

    def _get_pricing_for_model(
        self, model: str, strict: bool = False
    ) -> Dict[str, float]:
        """Get pricing for a model with optional fallback logic.

        Args:
            model: Model name
            strict: If True, raise KeyError for unknown models

        Returns:
            Pricing dictionary with input/output/cache costs

        Raises:
            KeyError: If strict=True and model is unknown
        """
        # Try normalized model name first
        provider = detect_provider_from_model(model)
        normalized = normalize_model_name(model, provider)

        # Check configured pricing
        if normalized in self.pricing:
            pricing = self.pricing[normalized].copy()
            # Ensure cache pricing exists
            if "cache_creation" not in pricing:
                pricing["cache_creation"] = pricing["input"] * 1.25
            if "cache_read" not in pricing:
                pricing["cache_read"] = pricing["input"] * 0.1
            return pricing

        # Check original model name
        if model in self.pricing:
            pricing = self.pricing[model].copy()
            if "cache_creation" not in pricing:
                pricing["cache_creation"] = pricing["input"] * 1.25
            if "cache_read" not in pricing:
                pricing["cache_read"] = pricing["input"] * 0.1
            return pricing

        # If strict mode, raise KeyError for unknown models
        if strict:
            raise KeyError(f"Unknown model: {model}")

        # Fallback to hardcoded pricing based on model type and provider
        model_lower = model.lower()
        provider = detect_provider_from_model(model)
        
        if provider == Provider.Z_AI:
            if "glm-4.5-x" in model_lower:
                return self.FALLBACK_PRICING["glm-4.5-x"]
            elif "glm-4.5-airx" in model_lower:
                return self.FALLBACK_PRICING["glm-4.5-airx"]
            elif "glm-4.5-air" in model_lower:
                return self.FALLBACK_PRICING["glm-4.5-air"]
            elif "glm-4.5v" in model_lower or "glm-4.5-v" in model_lower:
                return self.FALLBACK_PRICING["glm-4.5v"]
            elif "glm-4.5-flash" in model_lower:
                return self.FALLBACK_PRICING["glm-4.5-flash"]
            elif "glm-4.5" in model_lower:
                return self.FALLBACK_PRICING["glm-4.5"]
            elif "glm-4-32b-0414-128k" in model_lower:
                return self.FALLBACK_PRICING["glm-4-32b-0414-128k"]
            elif "glm-4" in model_lower:
                return self.FALLBACK_PRICING["glm-4"]
            elif "glm-3" in model_lower:
                return self.FALLBACK_PRICING["glm-3"]
            else:
                # Default GLM pricing
                return self.FALLBACK_PRICING["glm-4.5"]
        elif provider == Provider.OPENAI:
            if "gpt-4" in model_lower:
                return self.FALLBACK_PRICING["gpt-4"]
            elif "gpt-3.5" in model_lower or "gpt-3-5" in model_lower:
                return self.FALLBACK_PRICING["gpt-3.5"]
            else:
                # Default GPT pricing
                return self.FALLBACK_PRICING["gpt-3.5"]
        elif provider == Provider.GOOGLE:
            if "gemini" in model_lower:
                return self.FALLBACK_PRICING["gemini-pro"]
            else:
                # Default to reasonable pricing
                return {"input": 1.0, "output": 2.0, "cache_creation": 1.25, "cache_read": 0.1}
        else:
            # Anthropic fallback
            if "opus" in model_lower:
                return self.FALLBACK_PRICING["claude-3-opus"]
            elif "haiku" in model_lower:
                return self.FALLBACK_PRICING["claude-3-haiku"]
            else:
                # Default to Sonnet pricing
                return self.FALLBACK_PRICING["claude-3-sonnet"]

    def calculate_cost_for_entry(
        self, entry_data: Dict[str, Any], mode: CostMode
    ) -> float:
        """Calculate cost for a single entry (backward compatibility).

        Args:
            entry_data: Entry data dictionary
            mode: Cost mode (for backward compatibility)

        Returns:
            Cost in USD
        """
        # If cost is present and mode is cached, use it
        if mode.value == "cached":
            cost_value = entry_data.get("costUSD") or entry_data.get("cost_usd")
            if cost_value is not None:
                return float(cost_value)

        # Otherwise calculate from tokens
        model = entry_data.get("model") or entry_data.get("Model")
        if not model:
            raise KeyError("Missing 'model' key in entry_data")

        # Extract token counts with different possible keys
        input_tokens = entry_data.get("inputTokens", 0) or entry_data.get(
            "input_tokens", 0
        )
        output_tokens = entry_data.get("outputTokens", 0) or entry_data.get(
            "output_tokens", 0
        )
        cache_creation = entry_data.get(
            "cacheCreationInputTokens", 0
        ) or entry_data.get("cache_creation_tokens", 0)
        cache_read = (
            entry_data.get("cacheReadInputTokens", 0)
            or entry_data.get("cache_read_input_tokens", 0)
            or entry_data.get("cache_read_tokens", 0)
        )

        return self.calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
        )
