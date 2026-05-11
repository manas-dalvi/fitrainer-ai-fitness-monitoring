"""
agents/progress_agent.py — Session analytics, trend detection, adaptive rep target.
Pure Python, no LLM. Uses profile history for cross-session comparison.
"""


class ProgressAgent:
    def evaluate(self, scores: list[float], rep_count: int, profile) -> dict:
        """
        Returns {message, trend, avg_score, adaptive_reps}
        """
        result = {
            "message":       "Keep going — build consistency!",
            "trend":         "steady",
            "avg_score":     0,
            "adaptive_reps": 10,
        }

        if len(scores) < 3:
            return result

        recent   = scores[-5:]
        overall  = scores
        avg_recent  = sum(recent)  / len(recent)
        avg_overall = sum(overall) / len(overall)

        # Trend
        if avg_recent > avg_overall + 5:
            trend = "improving"
        elif avg_recent < avg_overall - 5:
            trend = "declining"
        else:
            trend = "steady"

        # Message
        if avg_recent >= 88:
            message = "🔥 Excellent form — you're in the zone!"
        elif avg_recent >= 75:
            message = "👍 Good quality reps. Push for full depth."
        elif avg_recent >= 60:
            message = "⚡ Form needs work — slow down and nail each rep."
        else:
            message = "Focus on technique over speed right now."

        if trend == "improving":
            message += " Trending up!"

        # Adaptive rep target
        # Cross-session comparison
        exercise = getattr(profile, "_current_exercise", "squat")
        prev_best = profile.best_form_scores.get(exercise, 0)
        if avg_recent >= 85 and rep_count >= 8:
            adaptive_reps = min(15, 10 + int((avg_recent - 85) / 5))
        elif avg_recent < 65:
            adaptive_reps = max(5, 8 - 1)
        else:
            adaptive_reps = 10

        return {
            "message":       message,
            "trend":         trend,
            "avg_score":     round(avg_recent, 1),
            "adaptive_reps": adaptive_reps,
        }
