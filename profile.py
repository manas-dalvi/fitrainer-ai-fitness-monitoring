"""
profile.py — User profile: load/save from profile.json, BMR, adaptive thresholds.
"""
import json
import pathlib
import time
from dataclasses import dataclass, asdict, field

PROFILE_PATH = pathlib.Path(__file__).parent / "profile.json"


@dataclass
class UserProfile:
    # Basic
    name:             str   = "Athlete"
    age:              int   = 25
    sex:              str   = "male"       # male | female
    height_cm:        float = 170.0
    weight_kg:        float = 70.0
    # Fitness
    fitness_level:    str   = "intermediate"  # beginner | intermediate | advanced
    goal:             str   = "build_muscle"  # build_muscle | lose_weight | endurance | rehab
    injuries:         str   = ""
    # Coaching
    coaching_tone:    str   = "encouraging"   # encouraging | strict | scientific
    # API
    anthropic_key:    str   = ""
    gemini_key:       str   = ""
    # Meta
    is_onboarded:     bool  = False
    sessions_done:    int   = 0
    total_reps_ever:  int   = 0
    best_form_scores: dict  = field(default_factory=dict)  # exercise -> best score
    last_session:     float = 0.0

    # ── Mifflin-St Jeor BMR ──────────────────────────────
    @property
    def bmr(self) -> float:
        s = 5 if self.sex == "male" else -161
        return 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age + s

    @property
    def tdee(self) -> float:
        mult = {"beginner": 1.375, "intermediate": 1.55, "advanced": 1.725}
        return self.bmr * mult.get(self.fitness_level, 1.55)


def load_profile() -> UserProfile:
    if PROFILE_PATH.exists():
        try:
            data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
            # Only set fields that exist in dataclass
            valid_fields = {f.name for f in UserProfile.__dataclass_fields__.values()}
            filtered = {k: v for k, v in data.items() if k in valid_fields}
            return UserProfile(**filtered)
        except Exception:
            pass
    return UserProfile()


def save_profile(profile: UserProfile) -> None:
    PROFILE_PATH.write_text(
        json.dumps(asdict(profile), indent=2),
        encoding="utf-8"
    )


def update_profile_after_session(profile: UserProfile,
                                  exercise: str,
                                  scores: list[float],
                                  rep_count: int) -> UserProfile:
    """Called at session end to persist progress."""
    profile.sessions_done    += 1
    profile.total_reps_ever  += rep_count
    profile.last_session      = time.time()

    if scores:
        avg = sum(scores) / len(scores)
        prev_best = profile.best_form_scores.get(exercise, 0)
        profile.best_form_scores[exercise] = round(max(prev_best, avg), 1)

    save_profile(profile)
    return profile


# ── Adaptive threshold ────────────────────────────────────────────────────────

def adaptive_thresh(base: float, fitness_level: str, direction: str = "lenient") -> float:
    """
    Adjust a threshold based on fitness level.
    direction='lenient': beginners get more room (higher down threshold = easier).
    direction='strict': beginners get more room (lower up threshold = easier).
    """
    adj = {"beginner": 10, "intermediate": 0, "advanced": -8}
    delta = adj.get(fitness_level, 0)
    if direction == "lenient":
        return base + delta   # down threshold: higher = easier to reach
    else:
        return base - delta   # up threshold: lower = easier to return from
