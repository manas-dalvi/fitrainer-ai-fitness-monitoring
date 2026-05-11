"""
agents/safety_agent.py — Hard-limit safety checks. Deterministic, no LLM.
"""


class SafetyAgent:
    HARD_LIMITS = {
        "knee_cave_left":  ("critical", "⛔ Left knee caving — stop and reset your stance"),
        "knee_cave_right": ("critical", "⛔ Right knee caving — stop and reset your stance"),
        "hip_sag":         ("critical", "⛔ Hips dropping — reset plank/pushup position"),
        "back_arch":       ("critical", "⛔ Lower back arching — brace core immediately"),
    }

    def check(self, cv_data: dict, exercise: str) -> dict:
        """
        Returns {critical: [str], warnings: [str]}
        Critical = must stop / immediate correction.
        """
        critical = []
        warnings = []
        errors = cv_data.get("errors", [])

        for err in errors:
            eid = err["id"]
            msg = err["message"]
            sev = err["severity"]
            if sev == "critical":
                critical.append(msg)
            else:
                warnings.append(msg)

        # Aggregate: if multiple critical, keep highest priority
        if len(critical) > 2:
            critical = critical[:2]

        return {"critical": critical, "warnings": warnings}
