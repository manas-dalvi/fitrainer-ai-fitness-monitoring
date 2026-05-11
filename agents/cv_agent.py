"""
agents/cv_agent.py — Computer vision form analysis.
Pure Python, no LLM. Runs every processed frame.
"""
from exercises import EXERCISE_CONFIG
from profile import adaptive_thresh


class CVAgent:
    """Analyzes joint angles and produces structured error list per frame."""

    # Error definitions per exercise
    # Each: (id, severity, check_fn, message)
    ERROR_DEFS = {
        "squat": [
            ("knee_cave_left",  "critical",
             lambda a: a.get("left_knee_dev", 0) < -0.05,
             "LEFT_KNEE",  "Push left knee outward over your toes"),
            ("knee_cave_right", "critical",
             lambda a: a.get("right_knee_dev", 0) > 0.05,
             "RIGHT_KNEE", "Push right knee outward over your toes"),
            ("forward_lean",    "warning",
             lambda a: a.get("spine", 180) < 130 and a.get("left_knee", 180) < 150,
             "LEFT_SHOULDER", "Chest up — keep torso more upright"),
            ("insufficient_depth", "warning",
             lambda a: a.get("left_knee", 180) > 130 and a.get("left_knee", 180) < 155,
             "LEFT_HIP",   "Go deeper — aim for thighs parallel to floor"),
        ],
        "pushup": [
            ("elbow_flare",  "warning",
             lambda a: (a.get("left_shoulder", 90) or 0) > 70 and a.get("left_elbow", 180) < 130,
             "LEFT_ELBOW",  "Tuck elbows — keep them at 45° from body"),
            ("hip_sag",      "critical",
             lambda a: (a.get("body_line", 180) or 180) < 155 and a.get("body_line", 180) > 0,
             "LEFT_HIP",    "Engage core — hips are dropping"),
            ("hip_pike",     "warning",
             lambda a: (a.get("body_line", 180) or 180) > 175,
             "LEFT_HIP",    "Lower hips — straight line head to heels"),
        ],
        "curl": [
            ("elbow_forward", "warning",
             lambda a: (a.get("elbow_drift", 0) or 0) > 0.07,
             "LEFT_ELBOW",   "Pin elbow to your side — isolate the bicep"),
            ("body_swing",    "warning",
             lambda a: (a.get("spine", 180) or 180) < 155,
             "LEFT_SHOULDER","Keep back straight — no body swing"),
        ],
        "press": [
            ("back_arch",          "critical",
             lambda a: (a.get("spine", 180) or 180) < 152,
             "LEFT_HIP",    "Brace core — do not arch lower back"),
            ("incomplete_lockout", "warning",
             lambda a: (a.get("left_elbow", 180) or 180) < 150 and
                       (a.get("left_elbow", 180) or 180) > 100,
             "LEFT_ELBOW",  "Fully extend arms overhead at lockout"),
        ],
        "lunge": [
            ("knee_cave_left", "critical",
             lambda a: a.get("left_knee_dev", 0) < -0.04,
             "LEFT_KNEE",   "Push left knee out — track over toes"),
            ("forward_lean",  "warning",
             lambda a: (a.get("spine", 180) or 180) < 125,
             "LEFT_SHOULDER","Keep torso upright in the lunge"),
        ],
        "plank": [
            ("hip_sag",  "critical",
             lambda a: (a.get("body_line", 180) or 180) < 155 and (a.get("body_line", 180) or 180) > 0,
             "LEFT_HIP", "Squeeze glutes and core — hips are dropping"),
            ("hip_pike", "warning",
             lambda a: (a.get("body_line", 180) or 180) > 175,
             "LEFT_HIP", "Lower hips to neutral plank position"),
        ],
    }

    def __init__(self):
        self._error_frame_count: dict[str, int] = {}
        self._DEBOUNCE = 2  # frames before error is confirmed

    def analyze(self, angles: dict, exercise: str, fitness_level: str = "intermediate") -> dict:
        """
        Returns:
          depth    : excellent | good | average | poor
          stability: stable | unstable | poor
          errors   : list of {id, severity, joint, message}
          form_score: 0-100 (frame-level, not rep-level)
        """
        errors = []
        defs = self.ERROR_DEFS.get(exercise, [])

        for err_id, severity, check_fn, joint, msg in defs:
            try:
                triggered = check_fn(angles)
            except Exception:
                triggered = False

            if triggered:
                self._error_frame_count[err_id] = self._error_frame_count.get(err_id, 0) + 1
            else:
                self._error_frame_count[err_id] = 0

            if self._error_frame_count.get(err_id, 0) >= self._DEBOUNCE:
                errors.append({
                    "id":       err_id,
                    "severity": severity,
                    "joint":    joint,
                    "message":  msg,
                })

        # Depth classification
        depth = "—"
        if exercise in ("squat", "lunge"):
            angle = angles.get("left_knee")
            if angle is not None:
                down_t = EXERCISE_CONFIG[exercise]["down_thresh"]
                adj = adaptive_thresh(down_t, fitness_level, "lenient")
                if angle <= adj - 5:
                    depth = "excellent"
                elif angle <= adj + 5:
                    depth = "good"
                elif angle <= adj + 20:
                    depth = "average"
                else:
                    depth = "poor"
        elif exercise in ("pushup", "curl", "press"):
            angle = angles.get("left_elbow")
            if angle is not None:
                if exercise == "curl":
                    depth = "excellent" if angle <= 55 else "good" if angle <= 70 else "average" if angle <= 90 else "poor"
                else:
                    depth = "excellent" if angle <= 95 else "good" if angle <= 110 else "average" if angle <= 125 else "poor"

        # Stability
        stability = "stable"
        crit_ids = [e["id"] for e in errors if e["severity"] == "critical"]
        warn_ids = [e["id"] for e in errors if e["severity"] == "warning"]
        if any("knee_cave" in eid or "hip_sag" in eid or "back_arch" in eid for eid in crit_ids):
            stability = "poor"
        elif warn_ids:
            stability = "unstable"

        # Frame-level form score
        frame_score = 100 - len([e for e in errors if e["severity"] == "critical"]) * 25 \
                          - len([e for e in errors if e["severity"] == "warning"])  * 10
        frame_score = max(0, frame_score)

        return {
            "depth":      depth,
            "stability":  stability,
            "errors":     errors,
            "form_score": frame_score,
            "angle":      angles.get("left_knee") or angles.get("left_elbow") or 0,
        }

    def reset(self):
        self._error_frame_count = {}
