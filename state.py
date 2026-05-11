"""
state.py — Thread-safe shared state for Fitrainer v2.
All threads read/write through this object.
"""
import threading
import time
from collections import deque


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()

        # ── Camera / Pose ──────────────────────────────
        self.pose_detected   = False
        self.confidence      = 0.0
        self.current_angle   = 0.0
        self.raw_landmarks   = []       # list of {x,y,z,vis} for overlay JS
        self.phase           = "waiting"  # waiting | down | up | hold

        # ── Session ────────────────────────────────────
        self.exercise        = "squat"
        self.session_active  = False
        self.session_start   = 0.0
        self.set_num         = 1

        # ── Reps / Scoring ─────────────────────────────
        self.rep_count       = 0
        self.form_score      = 0.0      # last rep
        self.scores          = []       # all rep scores this session
        self.active_errors   = []       # current frame error ids
        self.angles          = {}       # current computed angles dict

        # ── Gamification ───────────────────────────────
        self.xp              = 0
        self.level           = 1
        self.level_name      = "Rookie"
        self.streak          = 0
        self.earned_badges   = []       # persisted badge ids
        self.new_badge       = None     # most recently earned badge (for toast)

        # ── Agent outputs ──────────────────────────────
        self.coach_msg       = "Press Start Session to begin."
        self.safety_critical = []
        self.safety_warnings = []
        self.fatigue_level   = "low"
        self.fatigue_msg     = ""
        self.progress_msg    = ""
        self.progress_trend  = "steady"
        self.adaptive_reps   = 10       # suggested reps for next set

        # ── LLM state ──────────────────────────────────
        self.llm_pending     = False
        self.last_llm_rep    = -5       # trigger every N reps

        # ── Frame buffer (NOT in snapshot) ─────────────
        self._frame_buf      = deque(maxlen=2)  # JPEG bytes

    # ── Mutations ──────────────────────────────────────
    def update(self, **kwargs):
        """Thread-safe bulk update."""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def push_frame(self, jpeg_bytes: bytes):
        """Push encoded JPEG frame (called from camera thread only)."""
        self._frame_buf.append(jpeg_bytes)

    def get_frame(self) -> bytes | None:
        """Get latest frame (called from Flask /video route)."""
        if not self._frame_buf:
            return None
        return self._frame_buf[-1]

    # ── Read ───────────────────────────────────────────
    def snapshot(self) -> dict:
        """
        Return a serialisable snapshot for SSE / JSON.
        Excludes frame buffer and lock.
        """
        with self._lock:
            s = self.__dict__.copy()

        # Remove non-serialisable / private fields
        for key in ("_lock", "_frame_buf"):
            s.pop(key, None)

        # Compute derived fields
        avg_score = 0
        if s["scores"]:
            avg_score = int(sum(s["scores"]) / len(s["scores"]))

        best_score = int(max(s["scores"])) if s["scores"] else 0
        consistency = 0
        if len(s["scores"]) > 1:
            import statistics
            consistency = max(0, int(100 - statistics.stdev(s["scores"])))
        elif s["scores"]:
            consistency = 100

        s["avg_score"]   = avg_score
        s["best_score"]  = best_score
        s["consistency"] = consistency
        s["session_secs"] = int(time.time() - s["session_start"]) if s["session_active"] else 0

        # Round floats
        s["form_score"]   = round(s["form_score"], 1)
        s["confidence"]   = round(s["confidence"], 2)
        s["current_angle"] = round(s["current_angle"], 1)

        return s

    def reset_session(self, exercise: str):
        """Reset all session-specific fields for a new session."""
        with self._lock:
            self.exercise        = exercise
            self.session_active  = True
            self.session_start   = time.time()
            self.set_num         = 1
            self.rep_count       = 0
            self.form_score      = 0.0
            self.scores          = []
            self.active_errors   = []
            self.streak          = 0
            self.phase           = "waiting"
            self.coach_msg       = f"Session started — {exercise}. Get into position."
            self.safety_critical = []
            self.safety_warnings = []
            self.fatigue_level   = "low"
            self.fatigue_msg     = ""
            self.progress_msg    = ""
            self.progress_trend  = "steady"
            self.adaptive_reps   = 10
            self.llm_pending     = False
            self.last_llm_rep    = -5
            self.new_badge       = None

    def end_session(self):
        with self._lock:
            self.session_active = False
            self.phase          = "waiting"
