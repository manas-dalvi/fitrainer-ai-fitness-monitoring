"""
agents/coach_agent.py — LLM-powered coaching with rich fallback bank.
Uses Anthropic claude-haiku-4-5 for speed. Falls back to keyed messages.
Called from ThreadPoolExecutor — never blocks camera thread.
"""
import os
import json
import urllib.request
import urllib.error
import random


# ── Fallback message bank ─────────────────────────────────────────────────────
# Keyed by (exercise, primary_error_id) or (exercise, 'good') or ('any', context)
FALLBACKS: dict[tuple, list[str]] = {
    # SQUAT
    ("squat", "knee_cave_left"):   [
        "Drive that left knee outward — imagine pushing the floor apart with your feet.",
        "Left knee caving inward. Think: knees track over pinky toes.",
        "Push your left knee out to match your foot angle. Strong stance.",
    ],
    ("squat", "knee_cave_right"):  [
        "Right knee caving — drive it out over your right toes.",
        "Keep the right knee in line with your foot throughout the squat.",
    ],
    ("squat", "forward_lean"):     [
        "Chest stays up! Brace your core and keep your torso more vertical.",
        "You're leaning forward too much. Try elevating your heels slightly or work on ankle mobility.",
    ],
    ("squat", "insufficient_depth"): [
        "Go a little deeper — aim for thighs parallel to the floor.",
        "Depth needs work. Control the descent and hit that parallel position.",
    ],
    ("squat", "good"):             [
        "Solid squat! Maintain that depth and controlled tempo.",
        "Great form. Keep that chest up and drive through your heels.",
        "Looking strong. That depth is perfect — own every rep.",
    ],

    # PUSHUP
    ("pushup", "elbow_flare"):     [
        "Rotate your hands outward slightly and tuck elbows to 45 degrees.",
        "Elbows are flaring wide — protect your shoulders by tucking them in.",
    ],
    ("pushup", "hip_sag"):         [
        "Squeeze your glutes and brace your core — keep a straight line body.",
        "Hips are dropping. Imagine someone placed a glass of water on your back.",
    ],
    ("pushup", "hip_pike"):        [
        "Lower your hips down to neutral — straight line from head to heels.",
        "Hips too high. Push them down into a proper plank position.",
    ],
    ("pushup", "good"):            [
        "Clean push-up! Full range of motion and solid body alignment.",
        "Great form. Control the eccentric — slow down on the way down.",
        "Textbook push-up. Keep that core locked in.",
    ],

    # CURL
    ("curl", "elbow_forward"):     [
        "Pin your elbow to your side — the upper arm stays completely still.",
        "Elbow is drifting forward. Isolate that bicep — upper arm stays fixed.",
    ],
    ("curl", "body_swing"):        [
        "No swinging — if you need momentum the weight is too heavy.",
        "Keep your back straight and control the movement with just your arm.",
    ],
    ("curl", "good"):              [
        "Clean curl! Squeeze at the top and control the negative.",
        "Great isolation. Full extension at the bottom for maximum range.",
        "Solid curl form. Slow the eccentric phase for more gains.",
    ],

    # PRESS
    ("press", "back_arch"):        [
        "Brace your core hard — do not let your lower back arch under load.",
        "Lower back is arching. Tighten your abs and glutes before every press.",
    ],
    ("press", "incomplete_lockout"): [
        "Fully lock out overhead — squeeze the shoulder at the top of every rep.",
        "Get that full extension overhead. Partial reps mean partial gains.",
    ],
    ("press", "good"):             [
        "Strong press! Full lockout and tight core throughout.",
        "Great shoulder press. Keep that core engaged on every rep.",
        "Solid overhead work. Maintain that upright torso.",
    ],

    # LUNGE
    ("lunge", "knee_cave_left"):   [
        "Drive your left knee outward — it should track directly over your toes.",
    ],
    ("lunge", "forward_lean"):     [
        "Keep your torso upright in the lunge — chest up, eyes forward.",
        "You're leaning forward. Stand tall and step out further if needed.",
    ],
    ("lunge", "good"):             [
        "Controlled lunge! Both knees at 90 degrees is the target.",
        "Great lunge form. Equal weight distribution front and back.",
    ],

    # PLANK
    ("plank", "hip_sag"):          [
        "Squeeze your glutes and brace your core — hips are dropping.",
        "Engage your core properly. If you can't hold position, rest and reset.",
    ],
    ("plank", "hip_pike"):         [
        "Push your hips down — you want a straight line, not a triangle.",
    ],
    ("plank", "good"):             [
        "Perfect plank alignment. Breathe steadily and hold that position.",
        "Strong plank. Keep squeezing glutes and bracing abs.",
    ],

    # GENERIC MILESTONES
    ("any", "rep_5"):   ["5 reps in — you're warmed up. Focus on quality now."],
    ("any", "rep_10"):  ["10 reps done. Strong session — keep that standard."],
    ("any", "rep_20"):  ["20 reps! Excellent stamina. Form is your top priority now."],
    ("any", "fatigue"): ["Fatigue showing — slow down, each rep with full control."],
    ("any", "improving"): ["Your scores are climbing this session. The work is paying off."],
    ("any", "set_done"): [
        "Good set! Rest up, then come back focused on your form cues.",
        "Set complete. Use the rest to shake out and reset your technique.",
        "Solid set. Next one — apply what you've learned.",
    ],
    ("any", "session_start"): [
        "Let's go! Focus on your breathing and controlled movement.",
        "Session started. Prioritise quality over quantity on every rep.",
    ],
}


def _pick(exercise: str, error_ids: list, context: str = None) -> str:
    """Pick best fallback message given exercise + active errors."""
    # 1. Specific error match
    for eid in error_ids:
        msgs = FALLBACKS.get((exercise, eid))
        if msgs:
            return random.choice(msgs)
    # 2. Good form for this exercise
    if not error_ids:
        msgs = FALLBACKS.get((exercise, "good"))
        if msgs:
            return random.choice(msgs)
    # 3. Generic context
    if context:
        msgs = FALLBACKS.get(("any", context))
        if msgs:
            return random.choice(msgs)
    # 4. Ultimate fallback
    return "Stay controlled — quality over quantity on every rep."


# ── CoachAgent ────────────────────────────────────────────────────────────────

class CoachAgent:
    def __init__(self):
        self._api_key = ""
        self._last_messages: list[str] = []  # avoid repetition

    def set_api_key(self, key: str):
        self._api_key = key.strip()

    def generate(self, cv_data: dict, progress: dict, fatigue: dict,
                 profile, exercise: str, rep_count: int, context: str = None) -> str:
        """
        Generate a coaching message.
        Tries Anthropic first, falls back to curated bank.
        Always returns a non-empty string.
        """
        error_ids = [e["id"] for e in cv_data.get("errors", [])]
        fallback  = _pick(exercise, error_ids, context)

        if not self._api_key:
            return self._dedupe(fallback)

        # Build a tight prompt — haiku responds in <1s for 80 token limit
        tone_desc = {
            "encouraging": "warm and encouraging",
            "strict":      "direct and no-nonsense",
            "scientific":  "precise and data-driven",
        }.get(getattr(profile, "coaching_tone", "encouraging"), "encouraging")

        errors_str = ", ".join(error_ids) if error_ids else "none"
        depth      = cv_data.get("depth", "—")
        stability  = cv_data.get("stability", "—")
        trend      = progress.get("trend", "steady")
        avg        = progress.get("avg_score", 0)
        fatigue_l  = fatigue.get("level", "low")

        system = (
            f"You are a {tone_desc} fitness coach giving real-time feedback. "
            "Reply with ONE sentence only (max 18 words). Be specific and actionable. "
            "No greetings, no filler words."
        )
        user = (
            f"Exercise: {exercise}. Rep {rep_count}. "
            f"Depth: {depth}. Stability: {stability}. "
            f"Errors: {errors_str}. Trend: {trend}. Avg score: {avg}. "
            f"Fatigue: {fatigue_l}. Give a coaching cue."
        )

        try:
            payload = json.dumps({
                "model": "claude-haiku-4-5",
                "max_tokens": 80,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }).encode()

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "x-api-key":         self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read())
                msg  = data["content"][0]["text"].strip()
                return self._dedupe(msg)

        except Exception:
            return self._dedupe(fallback)

    def _dedupe(self, msg: str) -> str:
        """Avoid repeating the same message consecutively."""
        if msg in self._last_messages[-2:]:
            return msg + " Stay consistent."
        self._last_messages.append(msg)
        if len(self._last_messages) > 10:
            self._last_messages.pop(0)
        return msg

    def reset(self):
        self._last_messages = []
