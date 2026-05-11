"""
agents/orchestrator.py — Coordinates all agents.
Frame-sync agents (cv, safety, fatigue) run on every processed frame.
LLM coach dispatched asynchronously via ThreadPoolExecutor.
"""
import concurrent.futures
import threading

from agents.cv_agent       import CVAgent
from agents.safety_agent   import SafetyAgent
from agents.fatigue_agent  import FatigueAgent
from agents.progress_agent import ProgressAgent
from agents.coach_agent    import CoachAgent


class Orchestrator:
    LLM_REP_INTERVAL = 3   # call LLM every N reps at minimum

    def __init__(self):
        self.cv       = CVAgent()
        self.safety   = SafetyAgent()
        self.fatigue  = FatigueAgent()
        self.progress = ProgressAgent()
        self.coach    = CoachAgent()

        self._executor    = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._llm_future  : concurrent.futures.Future | None = None
        self._llm_lock    = threading.Lock()
        self._last_llm_rep = -self.LLM_REP_INTERVAL

    # ── Frame-level (runs every processed frame, synchronous) ─────────────────

    def process_frame(self, angles: dict, exercise: str,
                      fitness_level: str = "intermediate") -> dict:
        """
        Run CV + Safety synchronously. Returns partial state dict.
        Does NOT touch coach or trigger LLM.
        """
        cv_data = self.cv.analyze(angles, exercise, fitness_level)
        safety  = self.safety.check(cv_data, exercise)

        return {
            "active_errors":   cv_data["errors"],
            "form_score":      cv_data["form_score"],
            "safety_critical": safety["critical"],
            "safety_warnings": safety["warnings"],
            "cv_depth":        cv_data["depth"],
            "cv_stability":    cv_data["stability"],
        }

    # ── Rep-level (runs on rep completion) ────────────────────────────────────

    def process_rep(self, state_snapshot: dict, profile,
                    scores: list, rep_count: int) -> dict:
        """
        Run fatigue + progress synchronously.
        Dispatch coach LLM asynchronously.
        Returns sync results immediately; LLM result arrives later via poll.
        """
        exercise      = state_snapshot.get("exercise", "squat")
        session_secs  = state_snapshot.get("session_secs", 0)
        fitness_level = getattr(profile, "fitness_level", "intermediate")

        self.fatigue.on_rep()
        fatigue  = self.fatigue.analyze(scores, session_secs)
        progress = self.progress.evaluate(scores, rep_count, profile)

        # Dispatch LLM if enough reps have passed
        should_call_llm = (rep_count - self._last_llm_rep) >= self.LLM_REP_INTERVAL

        if should_call_llm:
            self._last_llm_rep = rep_count
            cv_data = {
                "errors":    state_snapshot.get("active_errors", []),
                "depth":     state_snapshot.get("cv_depth", "—"),
                "stability": state_snapshot.get("cv_stability", "—"),
            }
            self._dispatch_llm(cv_data, progress, fatigue, profile, exercise, rep_count)

        return {
            "fatigue_level": fatigue["level"],
            "fatigue_msg":   fatigue["message"],
            "progress_msg":  progress["message"],
            "progress_trend":progress["trend"],
            "adaptive_reps": progress["adaptive_reps"],
        }

    # ── Set / session events ──────────────────────────────────────────────────

    def on_set_complete(self, scores: list, rep_count: int,
                        profile, exercise: str) -> str:
        """Called when user ends a set. Returns coach message for the rest period."""
        progress = self.progress.evaluate(scores, rep_count, profile)
        fatigue  = self.fatigue.analyze(scores, 0)
        cv_data  = {"errors": [], "depth": "—", "stability": "—"}
        # Always call LLM for set completion if API available
        self._last_llm_rep = rep_count  # reset so next call fires after N reps
        msg = self.coach.generate(cv_data, progress, fatigue, profile,
                                  exercise, rep_count, context="set_done")
        return msg

    def on_session_end(self, scores: list, rep_count: int, profile,
                       exercise: str) -> str:
        """Called at session end for progress summary."""
        progress = self.progress.evaluate(scores, rep_count, profile)
        fatigue  = self.fatigue.analyze(scores, 0)
        cv_data  = {"errors": [], "depth": "—", "stability": "—"}
        summary  = self.coach.generate(cv_data, progress, fatigue, profile,
                                       exercise, rep_count, context="improving")
        return summary

    # ── LLM dispatch ──────────────────────────────────────────────────────────

    def _dispatch_llm(self, cv_data, progress, fatigue, profile,
                      exercise, rep_count):
        """Submit LLM call to thread pool. Non-blocking."""
        with self._llm_lock:
            # Cancel old future if still pending
            if self._llm_future and not self._llm_future.done():
                self._llm_future.cancel()
            self._llm_future = self._executor.submit(
                self.coach.generate,
                cv_data, progress, fatigue, profile, exercise, rep_count
            )

    def poll_llm(self) -> str | None:
        """
        Check if LLM future is done. Returns message or None.
        Called from the SSE loop / camera thread every few frames.
        """
        with self._llm_lock:
            if self._llm_future and self._llm_future.done():
                try:
                    result = self._llm_future.result(timeout=0)
                    self._llm_future = None
                    return result
                except Exception:
                    self._llm_future = None
        return None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def set_api_key(self, key: str):
        self.coach.set_api_key(key)

    def reset_session(self):
        self.cv.reset()
        self.fatigue.reset()
        self.coach.reset()
        self._last_llm_rep = -self.LLM_REP_INTERVAL
        with self._llm_lock:
            self._llm_future = None

    def shutdown(self):
        self._executor.shutdown(wait=False)
