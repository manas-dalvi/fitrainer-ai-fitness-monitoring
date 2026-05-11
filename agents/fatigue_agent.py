"""
agents/fatigue_agent.py — Fatigue detection from score trend and rep timing.
Pure Python, no LLM.
"""
import time


class FatigueAgent:
    def __init__(self):
        self._rep_times: list[float] = []   # timestamps of each rep

    def on_rep(self):
        """Call on every rep completion to track pacing."""
        self._rep_times.append(time.time())
        if len(self._rep_times) > 20:
            self._rep_times.pop(0)

    def analyze(self, scores: list[float], session_secs: float) -> dict:
        """
        Returns {level, message, velocity_drop}
        level: low | medium | high
        """
        if len(scores) < 4:
            return {"level": "low", "message": "", "velocity_drop": False}

        recent   = scores[-5:]
        baseline = scores[:max(1, len(scores) - 5)]
        avg_recent   = sum(recent)   / len(recent)
        avg_baseline = sum(baseline) / len(baseline)
        drop = avg_baseline - avg_recent

        # Rep velocity: are reps slowing down?
        velocity_drop = False
        if len(self._rep_times) >= 4:
            early_gap = self._rep_times[-4] - self._rep_times[-5] if len(self._rep_times) >= 5 else None
            late_gap  = self._rep_times[-1] - self._rep_times[-2]
            if early_gap and late_gap > early_gap * 1.4:
                velocity_drop = True

        # Fatigue level
        if drop > 20 or (velocity_drop and drop > 10):
            level = "high"
            message = "⚠ Fatigue detected — slow down and focus on form."
        elif drop > 10 or velocity_drop:
            level = "medium"
            message = "Getting tired. Maintain control on each rep."
        else:
            level = "low"
            message = ""

        return {"level": level, "message": message, "velocity_drop": velocity_drop}

    def reset(self):
        self._rep_times = []
