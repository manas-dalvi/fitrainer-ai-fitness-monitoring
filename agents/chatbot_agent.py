"""
agents/chatbot_agent.py — Gemini-powered conversational chatbot for FitTrainer.

Uses the official google-genai SDK (google.genai).

Model: gemini-2.0-flash-lite
  - GA stable
  - Free tier: 30 RPM / 1500 RPD  (highest rate limit on free tier)
  - No artificial word/token limits — model answers freely
"""
from google import genai
from google.genai import types

_MODEL = "gemini-2.0-flash-lite"


class ChatbotAgent:
    MAX_REQUESTS = 50

    def __init__(self):
        self._api_key: str = ""
        self._request_count: int = 0
        self._client = None
        self._chat = None

    # ── Key management ────────────────────────────────────────────────────────

    def set_api_key(self, key: str):
        """Set or replace the Gemini API key and reset state."""
        self._api_key = key.strip()
        self._request_count = 0
        self._client = None
        self._chat = None
        if self._api_key:
            self._init_client()

    def _init_client(self):
        """(Re)create the genai client and chat session."""
        try:
            self._client = genai.Client(api_key=self._api_key)
            self._chat = self._client.chats.create(
                model=_MODEL,
            )
            print(f"[Chatbot] Initialized with {_MODEL}")
        except Exception as e:
            print(f"[Chatbot] Init error: {e}")
            self._client = None
            self._chat = None

    # ── Chat ──────────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> dict:
        """
        Send a message and return:
            {"reply": str, "requests_used": int, "limit_reached": bool}
        """
        if not self._api_key:
            return {
                "reply": "No Gemini API key set. Click ⚙ Settings and paste your key.",
                "requests_used": self._request_count,
                "limit_reached": False,
            }

        if self._request_count >= self.MAX_REQUESTS:
            return {
                "reply": f"Request limit reached ({self.MAX_REQUESTS}/{self.MAX_REQUESTS}). Restart session to reset.",
                "requests_used": self._request_count,
                "limit_reached": True,
            }

        if not self._client or not self._chat:
            # Attempt re-init once
            self._init_client()
            if not self._chat:
                return {
                    "reply": "Chatbot failed to initialize. Try saving your API key again via ⚙ Settings.",
                    "requests_used": self._request_count,
                    "limit_reached": False,
                }

        try:
            self._request_count += 1
            response = self._chat.send_message(user_message)
            reply = response.text.strip()
        except Exception as e:
            self._request_count -= 1  # don't count failed requests
            # Try fallback answers before showing an error
            fallback = self._get_fallback(user_message)
            if fallback:
                reply = fallback
            else:
                err = str(e)
                if "API_KEY_INVALID" in err or "INVALID_ARGUMENT" in err:
                    reply = "Invalid API key. Please check your key in ⚙ Settings."
                elif "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                    reply = "Gemini rate limit hit. Here's a tip: " + (self._get_fallback(user_message) or "Wait a moment and try again.")
                else:
                    reply = f"Error: {err[:80]}"

        return {
            "reply": reply,
            "requests_used": self._request_count,
            "limit_reached": self._request_count >= self.MAX_REQUESTS,
        }

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        """Reset request counter and start a fresh chat session."""
        self._request_count = 0
        if self._client:
            self._chat = self._client.chats.create(model=_MODEL)

    # ── Fallback answers ──────────────────────────────────────────────────────

    _FALLBACKS = {
        "curl": (
            "Focus on controlled tempo — 2 seconds up, 3 seconds down. Pin elbows to your sides, "
            "fully extend at the bottom, and squeeze hard at the top. Avoid swinging your body. "
            "Progressive overload with strict form beats heavy cheating."
        ),
        "squat": (
            "Drive through your heels with chest up and core braced. Hit parallel or below — "
            "hip crease at knee level. Keep knees tracking over toes, never caving inward. "
            "Warm up with bodyweight sets and add weight gradually each week."
        ),
        "push": (
            "Keep your body in a straight plank line — no sagging hips or piking up. "
            "Hands shoulder-width, elbows at 45 degrees. Lower until chest nearly touches floor. "
            "Controlled descent, explosive push. Build volume before adding difficulty."
        ),
        "plank": (
            "Engage your core by pulling belly button to spine. Keep a straight line from head to heels — "
            "don't let hips sag or pike. Squeeze glutes, breathe steadily. "
            "Start with 30-second holds and build to 60 seconds with perfect form."
        ),
        "deadlift": (
            "Hinge at the hips, not the lower back. Keep the bar close to your body, shoulders over the bar. "
            "Brace your core hard before each pull. Drive through heels and lock out with glutes. "
            "Never round your lower back under load."
        ),
        "press": (
            "Start with dumbbells at shoulder height, palms forward. Press straight overhead to full lockout. "
            "Keep core tight to avoid arching your lower back. Control the descent slowly. "
            "Don't bounce at the bottom — pause briefly and press again."
        ),
    }

    def _get_fallback(self, message: str):
        """Return a canned answer if the message matches a known topic, else None."""
        msg = message.lower()
        for keyword, answer in self._FALLBACKS.items():
            if keyword in msg:
                return answer
        return None
