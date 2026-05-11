"""
pose_engine.py — MediaPipe PoseLandmarker (Tasks API) wrapper.
Uses VIDEO mode with monotonic timestamps for live camera stream.
Works with mediapipe 0.10.x.
"""
import time
import pathlib
import threading

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from exercises import SKELETON, LM, SKELETON_BY_EXERCISE, UPPER_BODY_EXERCISES

MODEL_PATH = pathlib.Path(__file__).parent / "models" / "pose_landmarker_lite.task"

# Overlay colours
COLOR_GOOD     = (0, 220, 100)   # green
COLOR_WARNING  = (0, 165, 255)   # orange
COLOR_CRITICAL = (0, 60, 255)    # red
COLOR_NEUTRAL  = (180, 180, 180) # grey


class PoseEngine:
    """
    Wraps MediaPipe PoseLandmarker in VIDEO mode.
    Thread-safe: single lock around detect_for_video.
    """
    def __init__(self):
        self._lock        = threading.Lock()
        self._landmarker  = None
        self._ts_offset   = time.monotonic()
        self._initialized = False
        self._init()

    def _init(self):
        if not MODEL_PATH.exists():
            print(f"[PoseEngine] WARNING: model not found at {MODEL_PATH}")
            print("[PoseEngine] Run setup.py to download it.")
            return
        try:
            opts = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.45,
                min_pose_presence_confidence=0.45,
                min_tracking_confidence=0.45,
                output_segmentation_masks=False,
            )
            self._landmarker  = vision.PoseLandmarker.create_from_options(opts)
            self._initialized = True
            print("[PoseEngine] Loaded OK")
        except Exception as e:
            print(f"[PoseEngine] Init error: {e}")

    # ── Core ──────────────────────────────────────────────────────────────────

    def process_frame(self, bgr: np.ndarray, exercise: str = "") -> tuple[list, float]:
        """
        Run pose detection on a BGR frame.
        Returns (landmarks_list, confidence).
        landmarks_list: list of 33 dicts {x, y, z, vis} in 0–1 normalized coords.
        """
        if not self._initialized or self._landmarker is None:
            return [], 0.0

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Monotonic timestamp in ms (required by VIDEO mode)
        ts_ms = int((time.monotonic() - self._ts_offset) * 1000)

        with self._lock:
            try:
                result = self._landmarker.detect_for_video(mp_img, ts_ms)
            except Exception as e:
                return [], 0.0

        if not result.pose_landmarks or not result.pose_landmarks[0]:
            return [], 0.0

        raw = result.pose_landmarks[0]
        landmarks = []
        for lm in raw:
            landmarks.append({
                "x":   float(lm.x),
                "y":   float(lm.y),
                "z":   float(lm.z),
                "vis": float(lm.visibility) if hasattr(lm, "visibility") else 0.5,
            })

        # Confidence: for upper-body exercises (curl, press) use arm joints only
        # so seated users with invisible legs still get a valid confidence score
        if exercise in UPPER_BODY_EXERCISES:
            arm_idxs = [
                LM["LEFT_SHOULDER"],  LM["LEFT_ELBOW"],  LM["LEFT_WRIST"],
                LM["RIGHT_SHOULDER"], LM["RIGHT_ELBOW"], LM["RIGHT_WRIST"],
            ]
            vis_vals = [landmarks[i]["vis"] for i in arm_idxs if i < len(landmarks)]
            # Use mean of top-4 (best-visible joints) so one occluded joint doesn’t kill it
            vis_sorted = sorted(vis_vals, reverse=True)[:4]
            confidence = (sum(vis_sorted) / len(vis_sorted)) if vis_sorted else 0.0
        else:
            key_idxs = [LM["LEFT_HIP"], LM["LEFT_KNEE"], LM["LEFT_SHOULDER"], LM["LEFT_ANKLE"]]
            vis_vals = [landmarks[i]["vis"] for i in key_idxs if i < len(landmarks)]
            confidence = min(vis_vals) if vis_vals else 0.0

        return landmarks, confidence

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_overlay(self, frame: np.ndarray, landmarks: list,
                     active_errors: list, angles: dict,
                     exercise: str = "") -> np.ndarray:
        """
        Draw skeleton, joint circles, and angle labels on frame.
        For upper-body exercises (curl, press) only arm/shoulder connections are drawn.
        """
        if not landmarks:
            return frame

        h, w = frame.shape[:2]

        # Build set of error joints
        error_joints  = set()
        critical_joints = set()
        for err in active_errors:
            joint = err.get("joint")
            if joint and joint in LM:
                if err.get("severity") == "critical":
                    critical_joints.add(LM[joint])
                else:
                    error_joints.add(LM[joint])

        # Choose skeleton: upper-body only for seated exercises
        skel = SKELETON_BY_EXERCISE.get(exercise, SKELETON)

        # ── Draw connections ───────────────────────────────────────────────
        for a_idx, b_idx in skel:
            if a_idx >= len(landmarks) or b_idx >= len(landmarks):
                continue
            la = landmarks[a_idx]
            lb = landmarks[b_idx]
            if la["vis"] < 0.3 or lb["vis"] < 0.3:
                continue
            x1, y1 = int(la["x"] * w), int(la["y"] * h)
            x2, y2 = int(lb["x"] * w), int(lb["y"] * h)
            is_err = a_idx in error_joints or b_idx in error_joints
            is_crit = a_idx in critical_joints or b_idx in critical_joints
            color = COLOR_CRITICAL if is_crit else COLOR_WARNING if is_err else COLOR_GOOD
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

        # ── Draw joints ────────────────────────────────────────────────────
        for i, lm in enumerate(landmarks):
            if lm["vis"] < 0.3:
                continue
            px, py = int(lm["x"] * w), int(lm["y"] * h)
            if i in critical_joints:
                color, radius = COLOR_CRITICAL, 8
            elif i in error_joints:
                color, radius = COLOR_WARNING, 7
            else:
                color, radius = COLOR_GOOD, 4
            cv2.circle(frame, (px, py), radius, color, -1)
            # Pulse ring for critical joints
            if i in critical_joints:
                cv2.circle(frame, (px, py), 14, COLOR_CRITICAL, 1)

        # ── Angle labels ───────────────────────────────────────────────────
        ANGLE_LABELS = {
            "left_knee":  LM["LEFT_KNEE"],
            "right_knee": LM["RIGHT_KNEE"],
            "left_elbow": LM["LEFT_ELBOW"],
            "right_elbow":LM["RIGHT_ELBOW"],
            "left_hip":   LM["LEFT_HIP"],
        }
        for key, idx in ANGLE_LABELS.items():
            val = angles.get(key)
            if val is None or idx >= len(landmarks):
                continue
            if landmarks[idx]["vis"] < 0.35:
                continue
            px = int(landmarks[idx]["x"] * w) + 10
            py = int(landmarks[idx]["y"] * h) - 5
            txt = f"{int(val)}"
            cv2.putText(frame, txt, (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)

        return frame

    def is_ready(self) -> bool:
        return self._initialized

    def close(self):
        if self._landmarker:
            try:
                self._landmarker.close()
            except Exception:
                pass
