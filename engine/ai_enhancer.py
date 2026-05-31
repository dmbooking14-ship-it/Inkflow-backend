import httpx
import hashlib
import time
from typing import Dict, Any, List, Optional
from config import (
    GEMINI_API_KEYS,
    DEEPSEEK_API_KEYS,
    OPENROUTER_API_KEY,
    GEMINI_MODEL_FLASH,
    GEMINI_MODEL_PRO,
    DEEPSEEK_MODEL,
    GEMINI_ENDPOINT,
    DEEPSEEK_ENDPOINT,
    OPENROUTER_ENDPOINT,
    BUSINESS_CONTEXT,
    GEMINI_CACHE_TTL_SECONDS,
    MAX_CACHE_ENTRIES,
    PRICING,
)


class AIEnhancer:
    """Multi-provider AI with caching, key rotation, and failover."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._gemini_key_index = 0
        self._deepseek_key_index = 0

    # ========== CACHE ==========
    def _cache_key(self, text: str, provider: str) -> str:
        """Generate a cache key from text + provider."""
        raw = f"{text.lower().strip()}_{provider}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        """Get cached response if not expired."""
        entry = self._cache.get(key)
        if entry:
            age = time.time() - entry["timestamp"]
            if age < GEMINI_CACHE_TTL_SECONDS:
                entry["reuseCount"] = entry.get("reuseCount", 0) + 1
                return entry["response"]
            else:
                del self._cache[key]
        return None

    def _cache_set(self, key: str, response: str) -> None:
        """Store response in cache."""
        # Limit cache size
        if len(self._cache) >= MAX_CACHE_ENTRIES:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k]["timestamp"],
            )
            del self._cache[oldest_key]
        self._cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "reuseCount": 0,
        }

    # ========== SYSTEM PROMPT ==========
    def _build_system_prompt(self, mode: str = "general", dashboard_data: Optional[Dict] = None) -> str:
        """Build optimized system prompt with business context and optional dashboard data."""
        bc = BUSINESS_CONTEXT
        prompt = (
            f"You are InkFlow Assistant v6.0, built by {bc['founder']} for {bc['product']} "
            f"— a SaaS booking platform for tattoo artists (pre-revenue, Stage 0). "
            f"Current: {bc['primary_challenge']}. "
            f"System 6 ready at {bc['system6_trigger']}. "
            f"Pricing: ${PRICING['standard']}/${PRICING['pro']}/${PRICING['premium']}/mo. "
        )

        # Mode-specific instructions
        mode_prompts = {
            "analytics": "MODE: Analytics. Focus on data, metrics, numbers. Be precise and quantitative. ",
            "outreach": "MODE: Outreach. Focus on drafting messages, tracking leads, follow-up strategy. ",
            "strategy": "MODE: Strategy. Focus on business decisions, competitor analysis, growth tactics. Think long-term. ",
            "coder": "MODE: Coder. Write production-ready code with error handling. Explain reasoning before code. No fluff. ",
            "analyst": "MODE: Analyst. Structure: 1) KEY METRIC 2) TREND 3) INSIGHT 4) RECOMMENDATION. Be quantitative. ",
            "general": "MODE: General. Be conversational and helpful across all topics. ",
        }
        prompt += mode_prompts.get(mode, mode_prompts["general"])

        # Inject dashboard data if available
        if dashboard_data:
            prompt += (
                f"LIVE DATA: {dashboard_data.get('totalArtists', 0)} artists, "
                f"${dashboard_data.get('mrr', 0)} MRR, "
                f"{dashboard_data.get('totalBookings', 0)} bookings, "
                f"{dashboard_data.get('bookings7d', 0)} this week, "
                f"{dashboard_data.get('dmsSent', 0)} DMs, "
                f"{dashboard_data.get('healthScore', 0)}% health. "
            )

        prompt += "Be thorough. Complete all sentences. "
        return prompt

    # ========== GEMINI ==========
    async def _call_gemini(
        self,
        user_text: str,
        model: str = GEMINI_MODEL_FLASH,
        mode: str = "general",
        dashboard_data: Optional[Dict] = None,
        image_base64: Optional[str] = None,
    ) -> Optional[str]:
        """Call Gemini API with key rotation."""
        if not GEMINI_API_KEYS:
            return None

        system_prompt = self._build_system_prompt(mode, dashboard_data)
        full_prompt = f"{system_prompt}\n\nUser: {user_text}"

        # Build request body
        parts = [{"text": full_prompt}]
        if image_base64:
            mime_type = "image/jpeg"
            if "data:image/png" in image_base64:
                mime_type = "image/png"
            elif "data:image/gif" in image_base64:
                mime_type = "image/gif"
            base64_data = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64_data,
                }
            })

        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 800,
            },
        }

        # Try each key
        num_keys = len(GEMINI_API_KEYS)
        for attempt in range(num_keys):
            idx = (self._gemini_key_index + attempt) % num_keys
            api_key = GEMINI_API_KEYS[idx]

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        f"{GEMINI_ENDPOINT}/{model}:generateContent?key={api_key}",
                        json=body,
                        headers={"Content-Type": "application/json"},
                    )

                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts_list = candidates[0].get("content", {}).get("parts", [])
                        if parts_list:
                            reply = parts_list[0].get("text", "")
                            if reply:
                                self._gemini_key_index = idx
                                return reply

                elif resp.status_code == 429:
                    # Rate limited — try next key
                    continue
                else:
                    # Other error — try next key
                    continue

            except (httpx.TimeoutException, httpx.RequestError):
                continue

        return None

    # ========== DEEPSEEK ==========
    async def _call_deepseek(
        self,
        user_text: str,
        mode: str = "general",
        dashboard_data: Optional[Dict] = None,
    ) -> Optional[str]:
        """Call DeepSeek API with key rotation."""
        if not DEEPSEEK_API_KEYS:
            return None

        system_prompt = self._build_system_prompt(mode, dashboard_data)
        full_prompt = f"{system_prompt}\n\nUser: {user_text}"

        body = {
            "model": DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": 1500,
        }

        num_keys = len(DEEPSEEK_API_KEYS)
        for attempt in range(num_keys):
            idx = (self._deepseek_key_index + attempt) % num_keys
            api_key = DEEPSEEK_API_KEYS[idx]

            try:
                async with httpx.AsyncClient(timeout=45.0) as client:
                    resp = await client.post(
                        DEEPSEEK_ENDPOINT,
                        json=body,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                    )

                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        reply = choices[0].get("message", {}).get("content", "")
                        if reply:
                            self._deepseek_key_index = idx
                            return reply

                elif resp.status_code in (429, 500, 502, 503):
                    continue
                elif resp.status_code in (401, 402, 403):
                    continue
                else:
                    continue

            except (httpx.TimeoutException, httpx.RequestError):
                continue

        return None

    # ========== OPENROUTER ==========
    async def _call_openrouter(
        self,
        user_text: str,
        mode: str = "general",
        dashboard_data: Optional[Dict] = None,
    ) -> Optional[str]:
        """Call OpenRouter API (fallback provider)."""
        if not OPENROUTER_API_KEY:
            return None

        system_prompt = self._build_system_prompt(mode, dashboard_data)
        full_prompt = f"{system_prompt}\n\nUser: {user_text}"

        body = {
            "model": "deepseek/deepseek-chat",
            "messages": [{"role": "user", "content": full_prompt}],
        }

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(
                    OPENROUTER_ENDPOINT,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "HTTP-Referer": "https://inkflow.app",
                        "Content-Type": "application/json",
                    },
                )

            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")

        except (httpx.TimeoutException, httpx.RequestError):
            pass

        return None

    # ========== FULL FAILOVER CHAIN ==========
    async def generate_insight(
        self,
        user_text: str,
        provider: str = "gemini",
        model: str = "flash",
        mode: str = "general",
        dashboard_data: Optional[Dict] = None,
        image_base64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point with full failover chain:
        Cache → Primary Provider → Secondary → Tertiary → None
        """
        # Check cache first
        cache_key = self._cache_key(user_text, provider)
        cached = self._cache_get(cache_key)
        if cached:
            return {
                "response": cached,
                "provider": provider,
                "source": "cache",
                "success": True,
            }

        gemini_model = GEMINI_MODEL_PRO if model == "pro" else GEMINI_MODEL_FLASH

        # Image requests must use Gemini
        if image_base64:
            result = await self._call_gemini(
                user_text, gemini_model, mode, dashboard_data, image_base64
            )
            if result:
                self._cache_set(cache_key, result)
                return {"response": result, "provider": "gemini", "source": "api", "success": True}
            return {"response": None, "provider": "gemini", "source": "none", "success": False}

        # Try primary provider
        if provider == "gemini":
            result = await self._call_gemini(user_text, gemini_model, mode, dashboard_data)
            if result:
                self._cache_set(cache_key, result)
                return {"response": result, "provider": "gemini", "source": "api", "success": True}
            # Failover to DeepSeek
            result = await self._call_deepseek(user_text, mode, dashboard_data)
            if result:
                self._cache_set(cache_key, result)
                return {"response": result, "provider": "deepseek", "source": "failover", "success": True}
            # Failover to OpenRouter
            result = await self._call_openrouter(user_text, mode, dashboard_data)
            if result:
                self._cache_set(cache_key, result)
                return {"response": result, "provider": "openrouter", "source": "failover", "success": True}

        elif provider == "deepseek":
            result = await self._call_deepseek(user_text, mode, dashboard_data)
            if result:
                self._cache_set(cache_key, result)
                return {"response": result, "provider": "deepseek", "source": "api", "success": True}
            # Failover to OpenRouter
            result = await self._call_openrouter(user_text, mode, dashboard_data)
            if result:
                self._cache_set(cache_key, result)
                return {"response": result, "provider": "openrouter", "source": "failover", "success": True}
            # Failover to Gemini
            result = await self._call_gemini(user_text, gemini_model, mode, dashboard_data)
            if result:
                self._cache_set(cache_key, result)
                return {"response": result, "provider": "gemini", "source": "failover", "success": True}

        elif provider == "openrouter":
            result = await self._call_openrouter(user_text, mode, dashboard_data)
            if result:
                self._cache_set(cache_key, result)
                return {"response": result, "provider": "openrouter", "source": "api", "success": True}
            # Failover to DeepSeek
            result = await self._call_deepseek(user_text, mode, dashboard_data)
            if result:
                self._cache_set(cache_key, result)
                return {"response": result, "provider": "deepseek", "source": "failover", "success": True}
            # Failover to Gemini
            result = await self._call_gemini(user_text, gemini_model, mode, dashboard_data)
            if result:
                self._cache_set(cache_key, result)
                return {"response": result, "provider": "gemini", "source": "failover", "success": True}

        return {"response": None, "provider": provider, "source": "none", "success": False}

    # ========== VISION-ONLY SHORTCUT ==========
    async def analyze_image(
        self,
        image_base64: str,
        prompt_text: str = "Describe this image in detail.",
    ) -> Optional[str]:
        """Analyze an image using Gemini Vision."""
        result = await self._call_gemini(
            user_text=prompt_text,
            model=GEMINI_MODEL_FLASH,
            mode="general",
            dashboard_data=None,
            image_base64=image_base64,
        )
        return result


# ========== MODULE-LEVEL SINGLETON ==========
_ai_enhancer: Optional[AIEnhancer] = None


def get_ai_enhancer() -> AIEnhancer:
    """Get or create the AI enhancer singleton."""
    global _ai_enhancer
    if _ai_enhancer is None:
        _ai_enhancer = AIEnhancer()
    return _ai_enhancer