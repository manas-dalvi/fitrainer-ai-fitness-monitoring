"""
gamification.py — XP, levels, badges, streak logic.
Pure Python, no external deps.
"""

XP_THRESHOLDS = [0, 150, 400, 800, 1400, 2200, 3200, 4500]
LEVEL_NAMES   = ["Rookie", "Trainee", "Active", "Committed",
                 "Intermediate", "Advanced", "Expert", "Elite"]

BADGE_DEFS = {
    "first_rep":    ("🎯", "First Rep",       "Complete your first rep"),
    "rep_10":       ("💪", "10 Reps",          "Complete 10 total reps"),
    "rep_50":       ("⚡", "50 Reps",           "Complete 50 total reps"),
    "rep_100":      ("🔱", "Century Club",     "Complete 100 total reps"),
    "perfect_form": ("⭐", "Perfect Form",     "Score 95+ on a single rep"),
    "streak_5":     ("🔥", "On Fire",           "5-rep streak of quality reps"),
    "streak_10":    ("🏆", "Unstoppable",       "10-rep streak"),
    "consistent":   ("📈", "Consistent",        "Last 5 reps all scored above 70"),
    "comeback":     ("💫", "Comeback",          "Improve form after a poor rep"),
    "ironwill":     ("🦾", "Iron Will",         "Complete 3 sets in one session"),
}


def level_for_xp(xp: int) -> tuple[int, str]:
    """Return (level_number, level_name) for given XP total."""
    for i in range(len(XP_THRESHOLDS) - 1, -1, -1):
        if xp >= XP_THRESHOLDS[i]:
            return i + 1, LEVEL_NAMES[min(i, len(LEVEL_NAMES) - 1)]
    return 1, LEVEL_NAMES[0]


def xp_next_level(xp: int) -> int:
    """XP required for next level (0 if max level)."""
    level, _ = level_for_xp(xp)
    if level >= len(XP_THRESHOLDS):
        return XP_THRESHOLDS[-1]
    return XP_THRESHOLDS[min(level, len(XP_THRESHOLDS) - 1)]


class Gamification:
    def __init__(self):
        self.xp              = 0
        self.level           = 1
        self.level_name      = LEVEL_NAMES[0]
        self.streak          = 0
        self.earned_badges   : set  = set()
        self._prev_score     = None
        self._sets_completed = 0

    # ── Core events ──────────────────────────────────────────────────────────

    def on_rep(self, score: float, all_scores: list) -> dict:
        """
        Called on every completed rep.
        Returns: {xp_gained, new_level, new_badges, xp_total}
        """
        # Base XP + form bonus
        base = 2
        form_bonus = 5 if score >= 88 else 3 if score >= 75 else 0

        # Streak multiplier (caps at 2.0× at streak 10)
        mult = min(1.0 + self.streak * 0.1, 2.0)
        gained = max(1, int((base + form_bonus) * mult))

        self.xp += gained

        # Streak update
        if score >= 72:
            self.streak += 1
        else:
            self.streak = 0

        # Level check
        new_level, new_name = level_for_xp(self.xp)
        level_up = new_level > self.level
        self.level      = new_level
        self.level_name = new_name

        # Badges
        new_badges = self._check_badges(score, all_scores)

        self._prev_score = score
        return {
            "xp_gained":  gained,
            "xp_total":   self.xp,
            "level":      self.level,
            "level_name": self.level_name,
            "level_up":   level_up,
            "streak":     self.streak,
            "new_badges": new_badges,
        }

    def on_set_complete(self) -> int:
        """Returns XP gained for completing a set."""
        gained = 25
        self.xp += gained
        self._sets_completed += 1
        new_badges = []
        if self._sets_completed >= 3 and "ironwill" not in self.earned_badges:
            self.earned_badges.add("ironwill")
            new_badges.append("ironwill")
        return gained

    def _check_badges(self, score: float, all_scores: list) -> list:
        new = []
        total = len(all_scores)

        checks = [
            ("first_rep",    total >= 1),
            ("rep_10",       total >= 10),
            ("rep_50",       total >= 50),
            ("rep_100",      total >= 100),
            ("perfect_form", score >= 95),
            ("streak_5",     self.streak >= 5),
            ("streak_10",    self.streak >= 10),
            ("consistent",   len(all_scores) >= 5 and min(all_scores[-5:]) > 70),
            ("comeback",     (self._prev_score is not None
                              and self._prev_score < 60 and score >= 75)),
        ]
        for badge_id, condition in checks:
            if condition and badge_id not in self.earned_badges:
                self.earned_badges.add(badge_id)
                new.append(badge_id)
        return new

    def reset_session(self):
        """Reset streak and set counter at start of new session."""
        self.streak          = 0
        self._sets_completed = 0
        self._prev_score     = None

    def get_badge_info(self, badge_id: str) -> dict:
        icon, name, desc = BADGE_DEFS.get(badge_id, ("🏅", badge_id, ""))
        return {"id": badge_id, "icon": icon, "name": name, "desc": desc}

    def get_all_earned(self) -> list:
        return [self.get_badge_info(b) for b in self.earned_badges]

    def xp_progress_pct(self) -> float:
        cur_thresh = XP_THRESHOLDS[min(self.level - 1, len(XP_THRESHOLDS) - 1)]
        next_thresh = xp_next_level(self.xp)
        if next_thresh <= cur_thresh:
            return 100.0
        return round(100.0 * (self.xp - cur_thresh) / (next_thresh - cur_thresh), 1)
