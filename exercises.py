"""
exercises.py — Exercise definitions, rep counting state machine, angle extraction.
Supports: squat, pushup, curl, press, lunge, plank
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── Landmark indices (MediaPipe 33-point model) ───────────────────────────────
LM = {
    "NOSE":            0,
    "LEFT_SHOULDER":  11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW":     13, "RIGHT_ELBOW":    14,
    "LEFT_WRIST":     15, "RIGHT_WRIST":    16,
    "LEFT_HIP":       23, "RIGHT_HIP":      24,
    "LEFT_KNEE":      25, "RIGHT_KNEE":     26,
    "LEFT_ANKLE":     27, "RIGHT_ANKLE":    28,
    "LEFT_FOOT":      31, "RIGHT_FOOT":     32,
}

# Skeleton connections for drawing overlay
SKELETON = [
    (11,12),(11,13),(13,15),(12,14),(14,16),   # arms
    (11,23),(12,24),(23,24),                    # torso
    (23,25),(25,27),(24,26),(26,28),            # legs
    (27,31),(28,32),                            # feet
]

# Upper-body only — for seated exercises (curl, press)
SKELETON_UPPER_BODY = [
    (11,12),(11,13),(13,15),(12,14),(14,16),   # shoulders + arms
    (11,23),(12,24),                            # shoulder → hip (visible torso partial)
]

# Which exercises use upper-body-only skeleton / confidence
UPPER_BODY_EXERCISES = {"curl", "press"}

# Per-exercise skeleton override (falls back to full SKELETON if not listed)
SKELETON_BY_EXERCISE: dict = {
    "curl":  SKELETON_UPPER_BODY,
    "press": SKELETON_UPPER_BODY,
}


# ── Exercise configuration ────────────────────────────────────────────────────

EXERCISE_CONFIG = {
    "squat": {
        "display":       "Barbell Squat",
        "primary_angle": "left_knee",
        "up_thresh":     155,    # extended (standing)
        "down_thresh":   100,    # flexed (bottom)
        "ideal_bottom":  90,
        "description":   "Hip-width stance. Chest up, knees track toes. Go to parallel.",
        "camera_tip":    "Stand side-on or slightly front-facing. Full body visible.",
        "met":           5.0,
    },
    "pushup": {
        "display":       "Push-Up",
        "primary_angle": "left_elbow",
        "up_thresh":     150,
        "down_thresh":   100,
        "ideal_bottom":  90,
        "description":   "Hands shoulder-width. Body plank-straight. Elbows 45° from body.",
        "camera_tip":    "Camera side-on at floor level. Full body in frame.",
        "met":           3.8,
    },
    "curl": {
        "display":       "Bicep Curl",
        "primary_angle": "curl_angle",   # best available arm (left or right)
        "up_thresh":     150,    # arm fully extended (high angle = start/end position)
        "down_thresh":    50,    # arm fully curled (low angle = top of curl)
        "ideal_bottom":   45,    # ideal curl depth
        "description":   "Sit or stand. Elbow pinned to side. Curl fully up, lower fully down.",
        "camera_tip":    "Front-facing camera. One arm and elbow clearly visible.",
        "met":           3.0,
    },
    "press": {
        "display":       "Shoulder Press",
        "primary_angle": "left_elbow",
        "up_thresh":     155,
        "down_thresh":    90,
        "ideal_bottom":   85,
        "description":   "Start at shoulder height. Press directly overhead to lockout.",
        "camera_tip":    "Camera front-facing. Full body including extended arms visible.",
        "met":           3.5,
    },
    "lunge": {
        "display":       "Lunge",
        "primary_angle": "left_knee",
        "up_thresh":     155,
        "down_thresh":    95,
        "ideal_bottom":   90,
        "description":   "Step forward, both knees at 90°. Front knee over ankle.",
        "camera_tip":    "Camera side-on. Full body visible including both legs.",
        "met":           4.5,
    },
    "plank": {
        "display":       "Plank Hold",
        "primary_angle": None,   # static exercise — no rep counting
        "up_thresh":     None,
        "down_thresh":   None,
        "ideal_bottom":  None,
        "description":   "Forearms/hands on floor. Straight line head to heel. Breathe.",
        "camera_tip":    "Camera side-on at floor level. Full body in frame.",
        "met":           3.0,
    },
}


# ── Angle calculation ─────────────────────────────────────────────────────────

def angle_between(a, b, c) -> float:
    """
    Angle at joint B between segments A→B and C→B.
    Uses arctan2 (same as uploaded code but with normalisation safeguard).
    Returns degrees 0-180.
    """
    v1 = np.array([a[0]-b[0], a[1]-b[1]])
    v2 = np.array([c[0]-b[0], c[1]-b[1]])
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-7 or n2 < 1e-7:
        return 180.0
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def get_xy(landmarks: list, name: str) -> Optional[tuple]:
    """Extract (x, y) from landmark list by name."""
    idx = LM.get(name)
    if idx is None or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    vis = lm.get("vis", lm.get("visibility", 0))
    if vis < 0.25:
        return None
    return (lm["x"], lm["y"])


def get_exercise_angles(landmarks: list, exercise: str) -> dict:
    """
    Compute all relevant joint angles for the given exercise.
    Returns dict of angle_name -> degrees (or None if landmarks invisible).
    """
    angles = {}

    def a(name_a, name_b, name_c, key):
        pa = get_xy(landmarks, name_a)
        pb = get_xy(landmarks, name_b)
        pc = get_xy(landmarks, name_c)
        if pa and pb and pc:
            angles[key] = angle_between(pa, pb, pc)

    if exercise in ("squat", "lunge"):
        a("LEFT_HIP",  "LEFT_KNEE",  "LEFT_ANKLE",  "left_knee")
        a("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", "right_knee")
        a("LEFT_SHOULDER",  "LEFT_HIP",  "LEFT_KNEE",  "left_hip")
        a("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE", "right_hip")
        # Knee-cave detection: lateral diff
        lk = get_xy(landmarks, "LEFT_KNEE")
        rk = get_xy(landmarks, "RIGHT_KNEE")
        lh = get_xy(landmarks, "LEFT_HIP")
        rh = get_xy(landmarks, "RIGHT_HIP")
        la = get_xy(landmarks, "LEFT_ANKLE")
        ra = get_xy(landmarks, "RIGHT_ANKLE")
        if lk and la:
            angles["left_knee_dev"] = lk[0] - la[0]
        if rk and ra:
            angles["right_knee_dev"] = rk[0] - ra[0]
        # Spine forward lean proxy
        ls = get_xy(landmarks, "LEFT_SHOULDER")
        rs = get_xy(landmarks, "RIGHT_SHOULDER")
        if ls and rs and lh and rh and lk and rk:
            sh_mid = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
            hp_mid = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
            kn_mid = ((lk[0]+rk[0])/2, (lk[1]+rk[1])/2)
            angles["spine"] = angle_between(sh_mid, hp_mid, kn_mid)

    elif exercise in ("pushup", "plank"):
        a("LEFT_SHOULDER",  "LEFT_ELBOW",  "LEFT_WRIST",  "left_elbow")
        a("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "right_elbow")
        # Hip alignment (sag / pike)
        ls = get_xy(landmarks, "LEFT_SHOULDER")
        lh = get_xy(landmarks, "LEFT_HIP")
        la = get_xy(landmarks, "LEFT_ANKLE")
        if ls and lh and la:
            angles["body_line"] = angle_between(ls, lh, la)
        # Elbow flare: elbow-shoulder-hip angle
        le = get_xy(landmarks, "LEFT_ELBOW")
        if ls and le and lh:
            angles["left_shoulder"] = angle_between(le, ls, lh)

    elif exercise == "curl":
        # ── Single-arm seated friendly ─────────────────────────────────────────
        # Use a lower visibility threshold so partially-occluded seated shoulders
        # still contribute. Pick whichever arm is more visible / available.

        def get_xy_loose(name: str):
            """Like get_xy but accepts vis >= 0.12 for seated curl."""
            idx = LM.get(name)
            if idx is None or idx >= len(landmarks):
                return None
            lm = landmarks[idx]
            vis = lm.get("vis", lm.get("visibility", 0))
            if vis < 0.12:
                return None
            return (lm["x"], lm["y"])

        def arm_angle(sh, el, wr):
            """Compute angle if all three points available, else None."""
            ps = get_xy_loose(sh)
            pe = get_xy_loose(el)
            pw = get_xy_loose(wr)
            if ps and pe and pw:
                return angle_between(ps, pe, pw)
            return None

        l_ang = arm_angle("LEFT_SHOULDER",  "LEFT_ELBOW",  "LEFT_WRIST")
        r_ang = arm_angle("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST")

        if l_ang is not None:
            angles["left_elbow"] = l_ang
        if r_ang is not None:
            angles["right_elbow"] = r_ang

        # Determine best single arm to use as primary rep-counting angle
        if l_ang is not None and r_ang is not None:
            # Both visible — pick the one with higher wrist visibility (active arm)
            l_vis = landmarks[LM["LEFT_WRIST"]].get("vis", 0)  if LM["LEFT_WRIST"]  < len(landmarks) else 0
            r_vis = landmarks[LM["RIGHT_WRIST"]].get("vis", 0) if LM["RIGHT_WRIST"] < len(landmarks) else 0
            angles["curl_angle"] = l_ang if l_vis >= r_vis else r_ang
        elif l_ang is not None:
            angles["curl_angle"] = l_ang
        elif r_ang is not None:
            angles["curl_angle"] = r_ang
        # (if neither available, curl_angle absent — rep counter holds state)

        # Elbow drift: shoulder-elbow horizontal offset (whichever side is active)
        l_wv = landmarks[LM["LEFT_WRIST"]].get("vis", 0)  if LM["LEFT_WRIST"]  < len(landmarks) else 0
        r_wv = landmarks[LM["RIGHT_WRIST"]].get("vis", 0) if LM["RIGHT_WRIST"] < len(landmarks) else 0
        if l_ang is not None and (r_ang is None or l_wv >= r_wv):
            active_sh, active_el = "LEFT_SHOULDER",  "LEFT_ELBOW"
        else:
            active_sh, active_el = "RIGHT_SHOULDER", "RIGHT_ELBOW"
        ls_pt = get_xy_loose(active_sh)
        le_pt = get_xy_loose(active_el)
        if ls_pt and le_pt:
            angles["elbow_drift"] = abs(le_pt[0] - ls_pt[0])


        # Body swing proxy (spine) — optional; skipped if hip/ankle invisible when seated
        ls2 = get_xy(landmarks, "LEFT_SHOULDER") or get_xy(landmarks, "RIGHT_SHOULDER")
        lh  = get_xy(landmarks, "LEFT_HIP")      or get_xy(landmarks, "RIGHT_HIP")
        la  = get_xy(landmarks, "LEFT_ANKLE")    or get_xy(landmarks, "RIGHT_ANKLE")
        if ls2 and lh and la:
            angles["spine"] = angle_between(ls2, lh, la)

    elif exercise == "press":
        a("LEFT_SHOULDER",  "LEFT_ELBOW",  "LEFT_WRIST",  "left_elbow")
        a("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "right_elbow")
        # Back arch
        ls = get_xy(landmarks, "LEFT_SHOULDER")
        lh = get_xy(landmarks, "LEFT_HIP")
        la = get_xy(landmarks, "LEFT_ANKLE")
        if ls and lh and la:
            angles["spine"] = angle_between(ls, lh, la)

    return angles


# ── Rep Counter ───────────────────────────────────────────────────────────────

class RepCounter:
    """
    Hysteretic state machine for rep counting.
    States: NEUTRAL(0) → AT_TOP(1) → GOING_DOWN(2) → AT_BOTTOM(3) → GOING_UP(4) → AT_TOP
    Requires CONFIRM_FRAMES consecutive frames to change state.
    """
    NEUTRAL = 0; AT_TOP = 1; GOING_DOWN = 2; AT_BOTTOM = 3; GOING_UP = 4
    CONFIRM_FRAMES = 3

    def __init__(self, up_thresh: float, down_thresh: float):
        self.up_thresh   = up_thresh
        self.down_thresh = down_thresh
        self.state       = self.NEUTRAL
        self.count       = 0
        self._pending    = None
        self._confirm    = 0
        self._rep_frames = 0  # frames since rep start (for min-rep filter)

    def reset(self, up_thresh: float = None, down_thresh: float = None):
        self.state    = self.NEUTRAL
        self.count    = 0
        self._pending = None
        self._confirm = 0
        if up_thresh is not None:
            self.up_thresh   = up_thresh
        if down_thresh is not None:
            self.down_thresh = down_thresh

    def update(self, angle: float) -> tuple[bool, str]:
        """
        Feed angle; returns (rep_completed: bool, phase: str).
        """
        self._rep_frames += 1

        # Determine raw target state
        if angle >= self.up_thresh:
            raw = self.AT_TOP
        elif angle <= self.down_thresh:
            raw = self.AT_BOTTOM
        elif self.state in (self.AT_TOP, self.GOING_DOWN, self.NEUTRAL):
            raw = self.GOING_DOWN
        else:
            raw = self.GOING_UP

        rep_completed = False

        # Hysteresis
        if raw != self.state:
            if raw == self._pending:
                self._confirm += 1
            else:
                self._pending = raw
                self._confirm = 1

            if self._confirm >= self.CONFIRM_FRAMES:
                prev_state   = self.state
                self.state   = raw
                self._confirm = 0

                # Rep complete: was going up, now at top
                if raw == self.AT_TOP and prev_state == self.GOING_UP:
                    if self._rep_frames >= 8:  # at least 8 frames = not a glitch
                        self.count      += 1
                        rep_completed    = True
                        self._rep_frames = 0

                # Start tracking on first descent from top
                if raw == self.GOING_DOWN and prev_state == self.AT_TOP:
                    self._rep_frames = 0

        # Phase string for UI
        phase_map = {
            self.NEUTRAL:    "waiting",
            self.AT_TOP:     "top",
            self.GOING_DOWN: "down",
            self.AT_BOTTOM:  "bottom",
            self.GOING_UP:   "up",
        }
        return rep_completed, phase_map.get(self.state, "waiting")


# ── Form scorer ───────────────────────────────────────────────────────────────

def score_rep(exercise: str, bottom_angle: float, errors: list) -> float:
    """
    Score a completed rep 0–100.
    Based on depth achievement and error penalties.
    """
    cfg = EXERCISE_CONFIG.get(exercise, {})
    ideal = cfg.get("ideal_bottom", 90)
    down_t = cfg.get("down_thresh", 100)

    # Depth score: 100 if exactly at ideal, decreases away from ideal
    depth_score = max(0.0, 100.0 - abs(bottom_angle - ideal) * 1.5)

    # Error penalties
    penalty = 0
    for err in errors:
        if err.get("severity") == "critical":
            penalty += 25
        elif err.get("severity") == "warning":
            penalty += 12

    score = max(0.0, min(100.0, depth_score - penalty))
    return round(score, 1)
