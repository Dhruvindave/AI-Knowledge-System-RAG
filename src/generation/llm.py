import os
import time
import random
import traceback
from google import genai
from google.genai import errors


class GeminiServiceError(Exception):
    """
    Structured Gemini API error capturing HTTP status codes, error categories,
    user-friendly explanations, and actionable troubleshooting guidance.
    """

    def __init__(
        self,
        code: int = 0,
        category: str = "UNKNOWN",
        title: str = "Gemini Error",
        message: str = "An error occurred with Gemini.",
        advice: str = "Please try again.",
        retryable: bool = False,
        raw_error: Exception = None,
        technical_details: str = "",
    ):
        self.code = code
        self.category = category
        self.title = title
        self.message = message
        self.advice = advice
        self.retryable = retryable
        self.raw_error = raw_error
        self.technical_details = technical_details or (
            f"{type(raw_error).__name__}: {raw_error}\n\n{traceback.format_exc()}"
            if raw_error else ""
        )
        super().__init__(f"[{self.category}] {self.title}: {self.message}")


def parse_gemini_error(exc: Exception) -> GeminiServiceError:
    """
    Analyzes any exception from the Gemini API and returns a structured GeminiServiceError
    with specific categorization, status code, and clear user advice.
    """
    if isinstance(exc, GeminiServiceError):
        return exc

    err_str = str(exc)
    err_lower = err_str.lower()
    code = getattr(exc, "code", 0)

    # 1. 429 Rate Limit / Quota Exceeded (RPM or RPD)
    if code == 429 or "429" in err_str or "resource_exhausted" in err_lower or "rate limit" in err_lower or "quota" in err_lower:
        is_rate_limit = any(w in err_lower for w in ["rate limit", "per minute", "too many requests", "short interval", "reduce request frequency"])
        is_hard_quota = any(w in err_lower for w in ["quota exceeded", "daily", "per day", "free_tier", "exhausted quota", "quota metric"])

        if is_rate_limit and not is_hard_quota:
            return GeminiServiceError(
                code=429,
                category="RATE_LIMIT",
                title="Rate Limit Exceeded (Too Many Requests)",
                message="Too many requests were sent in a short period (requests-per-minute limit).",
                advice="Please wait 20–30 seconds before submitting another question.",
                retryable=True,
                raw_error=exc,
            )
        else:
            return GeminiServiceError(
                code=429,
                category="QUOTA_EXCEEDED",
                title="Gemini Model Quota Reached",
                message="The daily or overall request quota for your Gemini API key has been exhausted.",
                advice="Wait for your quota to reset (midnight PT), upgrade your Google AI Studio plan, or set a new key in GEMINI_API_KEY.",
                retryable=False,
                raw_error=exc,
            )

    # 2. 503 High Demand / Model Overloaded / Unavailable
    if code == 503 or "503" in err_str or "unavailable" in err_lower or "overloaded" in err_lower or "high demand" in err_lower:
        return GeminiServiceError(
            code=503,
            category="HIGH_DEMAND",
            title="Gemini Model Under High Demand",
            message="Google's Gemini servers are currently experiencing peak traffic and high demand. The model is temporarily overloaded.",
            advice="Automatic retries were attempted. Please wait 10–20 seconds and click 'Retry Question'.",
            retryable=True,
            raw_error=exc,
        )

    # 3. 500, 502, 504 Google Internal Server Errors & Timeouts
    if code in (500, 502, 504) or any(c in err_str for c in ["500", "502", "504"]) or "internal" in err_lower or "deadline_exceeded" in err_lower:
        return GeminiServiceError(
            code=code or 500,
            category="SERVER_ERROR",
            title="Google Internal Server Error",
            message="An unexpected server error or gateway timeout occurred on Google's Gemini infrastructure.",
            advice="This is a temporary service glitch on Google's end. Please retry in a few moments.",
            retryable=True,
            raw_error=exc,
        )

    # 4. 403 Permission Denied / Authentication / Regional Block
    if code == 403 or "403" in err_str or "permission_denied" in err_lower or "api_key_invalid" in err_lower:
        if any(w in err_lower for w in ["location", "region", "country", "unsupported"]):
            return GeminiServiceError(
                code=403,
                category="REGION_UNSUPPORTED",
                title="Gemini API Region Unsupported",
                message="Google Gemini API is currently unavailable in your geographical region or country.",
                advice="Access Gemini from a supported region or configure a proxy in a supported territory.",
                retryable=False,
                raw_error=exc,
            )
        return GeminiServiceError(
            code=403,
            category="PERMISSION_DENIED",
            title="Gemini API Key Permission Denied",
            message="The Gemini API key is invalid, revoked, expired, or lacks permission for this model.",
            advice="Verify that your GEMINI_API_KEY environment variable is set to a valid, active key from Google AI Studio.",
            retryable=False,
            raw_error=exc,
        )

    # 5. 400 Invalid Argument / Context Window Overflow
    if code == 400 or "400" in err_str or "invalid_argument" in err_lower:
        if any(w in err_lower for w in ["token", "payload", "context length", "too large"]):
            return GeminiServiceError(
                code=400,
                category="TOKEN_LIMIT",
                title="Context Exceeds Token Window Limit",
                message="The question and retrieved document context exceed Gemini's maximum input token limit.",
                advice="Try asking a more specific question, or reduce the number of retrieved chunks (TOP_K).",
                retryable=False,
                raw_error=exc,
            )
        return GeminiServiceError(
            code=400,
            category="INVALID_ARGUMENT",
            title="Invalid Request Parameters",
            message="The request to Gemini contained invalid parameters or malformed content.",
            advice="Please verify request inputs and prompt configuration.",
            retryable=False,
            raw_error=exc,
        )

    # 6. 404 Model Not Found / Retired
    if code == 404 or "404" in err_str or "not_found" in err_lower:
        return GeminiServiceError(
            code=404,
            category="MODEL_NOT_FOUND",
            title="Gemini Model Not Found",
            message="The requested Gemini model name does not exist or has been deprecated.",
            advice="The system will automatically try fallback models such as gemini-2.0-flash or gemini-1.5-flash.",
            retryable=True,
            raw_error=exc,
        )

    # 7. Network / Connection Errors
    if any(k in type(exc).__name__.lower() for k in ["connect", "timeout", "network", "socket", "urllib"]) or any(k in err_lower for k in ["name or service not known", "connection refused", "timed out", "connecterror"]):
        return GeminiServiceError(
            code=0,
            category="NETWORK_ERROR",
            title="Network Connection Failed",
            message="Unable to reach Google Gemini API servers. Network connection was lost or timed out.",
            advice="Check your internet connection, firewall, or DNS resolution, then retry.",
            retryable=True,
            raw_error=exc,
        )

    # 8. General fallback
    return GeminiServiceError(
        code=code,
        category="UNKNOWN",
        title="Gemini Service Error",
        message=f"Gemini encountered an error: {err_str}",
        advice="Please review the technical details below or try asking again.",
        retryable=True,
        raw_error=exc,
    )


def extract_response_text(response) -> str:
    """
    Safely inspects Gemini candidate finish reasons, safety ratings, and content parts.
    Prevents unhandled crashes on SAFETY, RECITATION, or empty responses.
    """
    candidates = getattr(response, "candidates", None)
    if not candidates:
        raise GeminiServiceError(
            code=0,
            category="EMPTY_RESPONSE",
            title="Empty Response from Gemini",
            message="Gemini returned no response candidates for this query.",
            advice="Try rephrasing your question.",
            retryable=True,
        )

    candidate = candidates[0]
    finish_reason = str(getattr(candidate, "finish_reason", "STOP")).upper()

    if "SAFETY" in finish_reason:
        ratings = getattr(candidate, "safety_ratings", [])
        flagged = []
        for r in ratings:
            prob = str(getattr(r, "probability", "")).upper()
            if getattr(r, "blocked", False) or "HIGH" in prob or "MEDIUM" in prob:
                cat = str(getattr(r, "category", "")).replace("HARM_CATEGORY_", "").title()
                flagged.append(f"{cat} ({prob})")
        details = f" Flagged: {', '.join(flagged)}" if flagged else ""
        raise GeminiServiceError(
            code=0,
            category="SAFETY_FILTER",
            title="Response Blocked by Safety Filters",
            message=f"Gemini's automated content filters blocked the generated response.{details}",
            advice="Rephrase your question or ensure the source document contains safe, appropriate language.",
            retryable=False,
        )

    if "RECITATION" in finish_reason:
        raise GeminiServiceError(
            code=0,
            category="RECITATION",
            title="Response Blocked by Recitation Policy",
            message="The response was blocked because it closely duplicated copyrighted training text.",
            advice="Ask the model to summarize or explain the concept in original words.",
            retryable=False,
        )

    # Safely extract text content
    text = ""
    try:
        text = response.text
    except Exception:
        # Fallback to candidate parts
        if getattr(candidate, "content", None) and getattr(candidate.content, "parts", None):
            parts_text = [p.text for p in candidate.content.parts if getattr(p, "text", None)]
            text = "".join(parts_text)

    if not text:
        raise GeminiServiceError(
            code=0,
            category="EMPTY_RESPONSE",
            title="No Text Generated",
            message=f"The model completed with status '{finish_reason}' but produced no readable text.",
            advice="Please try submitting your question again.",
            retryable=True,
        )

    if "MAX_TOKENS" in finish_reason:
        text += "\n\n*(Note: Output token limit reached; the response was truncated.)*"

    return text


class GeneratedResponse:
    """Wrapper that guarantees a safe .text attribute and model metadata."""
    def __init__(self, text: str, model: str = "", raw_response=None):
        self.text = text
        self.model = model
        self.raw_response = raw_response


class LLMResponseGenerator:

    def __init__(self):
        """
        Google Gemini GenAI configuration with environment variable support.
        """
        default_key = 'AQ.Ab8RN6LdZl8pxdRhnCTAMsjCvQuzFv1CRJt3JGnqdkmhUR-GhQ'
        api_key = os.getenv("GEMINI_API_KEY", default_key)
        self.client = genai.Client(api_key=api_key)

    def build_context(self, retrieved_chunks):
        """
        retrieved_chunks: list of dicts from store.search(), e.g.
            {"title": ..., "breadcrumb": ..., "content": ..., "distance": ...}
        """
        blocks = []

        for i, chunk in enumerate(retrieved_chunks, start=1):
            blocks.append(
                f"[Source {i}] {chunk['breadcrumb']}\n{chunk['content']}"
            )

        return "\n\n---\n\n".join(blocks)

    def build_prompt(self, user_question, context):
        return f"""You are answering a question using only the context provided below.
If the answer isn't contained in the context, say so clearly instead of guessing.
When relevant, mention which source(s) your answer is based on.

Context:
{context}

Question: {user_question}

Answer:"""

    def generate_response(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        fallback_models: tuple = ("gemini-2.0-flash", "gemini-1.5-flash"),
        max_retries: int = 3,
    ) -> GeneratedResponse:
        """
        Generates content from Gemini with:
        - Exponential backoff and jitter for transient errors (503 High Demand, 429 RPM, 500, timeouts)
        - Seamless model fallback if the primary model is overloaded or unavailable
        - Finish reason safety inspection
        - Actionable GeminiServiceError on unrecoverable errors
        """
        models_to_try = [model] + [m for m in fallback_models if m != model]
        last_error = None

        for current_model in models_to_try:
            for attempt in range(1, max_retries + 1):
                try:
                    raw_response = self.client.models.generate_content(
                        model=current_model,
                        contents=prompt,
                    )
                    safe_text = extract_response_text(raw_response)
                    return GeneratedResponse(
                        text=safe_text,
                        model=current_model,
                        raw_response=raw_response,
                    )

                except Exception as exc:
                    parsed_err = parse_gemini_error(exc)
                    last_error = parsed_err

                    # Non-retryable errors (e.g. invalid key, quota exhausted, context too long) should fail immediately
                    if not parsed_err.retryable:
                        raise parsed_err

                    # If this is a transient error (High Demand 503, Rate Limit 429 RPM, Server 500, Timeout)
                    # wait with exponential backoff if attempts remain on this model
                    if attempt < max_retries:
                        backoff = (1.5 ** attempt) + random.uniform(0.3, 0.9)
                        time.sleep(backoff)
                        continue
                    else:
                        # Retries exhausted for this model; will attempt fallback model if available
                        break

        # If all models and retries failed, raise the structured error
        if last_error:
            raise last_error
        raise GeminiServiceError(
            code=0,
            category="UNKNOWN",
            title="Generation Failed",
            message="Unable to generate a response from Gemini across all attempted models.",
            advice="Please try again in a few moments.",
            retryable=True,
        )
