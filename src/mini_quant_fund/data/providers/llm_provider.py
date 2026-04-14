"""
LLM Provider - Multi-provider LLM client for market analysis.

Supports:
- Google AI (Gemini) - Primary provider
- DeepSeek - Fallback provider

Features:
- Automatic failover between providers
- Response caching to minimize API costs
- Rate limiting to avoid quota issues
"""

import os
import logging
import hashlib
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    provider: str
    model: str
    tokens_used: int
    cached: bool
    timestamp: str


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate response from LLM."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass


class GoogleAIProvider(BaseLLMProvider):
    """Google AI (Gemini) provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_AI_API_KEY")
        self.model = "gemini-1.5-flash"
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models"
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate response using Google AI."""
        if not self.is_available():
            return None

        try:
            import requests

            url = f"{self.endpoint}/{self.model}:generateContent?key={self.api_key}"

            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": kwargs.get("temperature", 0.3),
                    "maxOutputTokens": kwargs.get("max_tokens", 1024),
                }
            }

            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                if "candidates" in data and len(data["candidates"]) > 0:
                    candidate = data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        return candidate["content"]["parts"][0].get("text", "")
            else:
                logger.warning(f"Google AI error: {response.status_code} - {response.text[:200]}")

        except Exception as e:
            logger.warning(f"Google AI generation failed: {e}")

        return None


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek provider."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = "deepseek-chat"
        self.endpoint = "https://api.deepseek.com/v1/chat/completions"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate response using DeepSeek."""
        if not self.is_available():
            return None

        try:
            import requests

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a financial analyst providing market insights."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": kwargs.get("temperature", 0.3),
                "max_tokens": kwargs.get("max_tokens", 1024),
            }

            response = requests.post(
                self.endpoint,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
            else:
                logger.warning(f"DeepSeek error: {response.status_code} - {response.text[:200]}")

        except Exception as e:
            logger.warning(f"DeepSeek generation failed: {e}")

        return None


class LLMProvider:
    """
    Multi-provider LLM client with failover and caching.

    Usage:
        provider = LLMProvider()
        response = provider.generate("What is the market sentiment for AAPL?")
    """

    def __init__(self):
        """Initialize with available providers."""
        self.providers: List[BaseLLMProvider] = []
        self._cache: Dict[str, LLMResponse] = {}
        self._cache_ttl = timedelta(minutes=15)  # Cache for 15 minutes

        # Initialize providers in order of preference
        google_provider = GoogleAIProvider()
        if google_provider.is_available():
            self.providers.append(google_provider)
            logger.info("Google AI provider initialized")

        deepseek_provider = DeepSeekProvider()
        if deepseek_provider.is_available():
            self.providers.append(deepseek_provider)
            logger.info("DeepSeek provider initialized")

        if not self.providers:
            logger.warning("No LLM providers available - check API keys in .env")

    def _cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached response is still valid."""
        if key not in self._cache:
            return False
        response = self._cache[key]
        cached_time = datetime.fromisoformat(response.timestamp)
        return datetime.utcnow() - cached_time < self._cache_ttl

    def generate(
        self,
        prompt: str,
        use_cache: bool = True,
        **kwargs
    ) -> Optional[LLMResponse]:
        """
        Generate LLM response with failover and caching.

        Args:
            prompt: The prompt to send to the LLM
            use_cache: Whether to use cached responses
            **kwargs: Additional parameters (temperature, max_tokens)

        Returns:
            LLMResponse with content and metadata, or None if all providers fail
        """
        # Check cache first
        cache_key = self._cache_key(prompt)
        if use_cache and self._is_cache_valid(cache_key):
            cached = self._cache[cache_key]
            cached.cached = True
            return cached

        # Try each provider in order
        for provider in self.providers:
            try:
                content = provider.generate(prompt, **kwargs)
                if content:
                    response = LLMResponse(
                        content=content,
                        provider=provider.__class__.__name__,
                        model=getattr(provider, 'model', 'unknown'),
                        tokens_used=len(content) // 4,  # Rough estimate
                        cached=False,
                        timestamp=datetime.utcnow().isoformat()
                    )

                    # Cache the response
                    self._cache[cache_key] = response

                    logger.debug(f"LLM response from {response.provider}: {len(content)} chars")
                    return response

            except Exception as e:
                logger.warning(f"Provider {provider.__class__.__name__} failed: {e}")
                continue

        logger.error("All LLM providers failed")
        return None

    def generate_json(
        self,
        prompt: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Generate and parse JSON response from LLM.

        Returns parsed JSON dict or None if generation/parsing fails.
        """
        # Add JSON instruction to prompt
        json_prompt = f"""{prompt}

Respond ONLY with valid JSON. No explanation or markdown."""

        response = self.generate(json_prompt, **kwargs)
        if not response:
            return None

        try:
            # Try to extract JSON from response
            content = response.content.strip()

            # Handle markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            return json.loads(content.strip())

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            return None

    @property
    def available_providers(self) -> List[str]:
        """List of available provider names."""
        return [p.__class__.__name__ for p in self.providers]

    def clear_cache(self):
        """Clear the response cache."""
        self._cache.clear()


# ---------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------
_llm_provider: Optional[LLMProvider] = None


def get_llm_provider() -> LLMProvider:
    """Get or create the global LLM provider instance."""
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = LLMProvider()
    return _llm_provider


# ---------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    provider = get_llm_provider()
    print(f"Available providers: {provider.available_providers}")

    if provider.providers:
        response = provider.generate("What is the current market sentiment for technology stocks? Keep it brief.")
        if response:
            print(f"\nProvider: {response.provider}")
            print(f"Response: {response.content[:500]}")
    else:
        print("No LLM providers available. Check your API keys.")
