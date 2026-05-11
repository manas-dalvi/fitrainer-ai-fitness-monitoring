"""
app.py — Fitrainer v2 main Flask application.

Architecture:
  Thread 1 (camera_thread): cap.read() → MediaPipe → form check → rep count → SharedState
  Thread 2 (LLM, pool):     Coach.generate() → SharedState.coach_msg
  Thread 3 (Flask):         /video (MJPEG), /events (SSE), /api/* routes

Critical invariants:
  - camera_thread NEVER calls Flask routes or blocks on LLM
  - /video ONLY reads SharedState._frame_buf
  - SSE /events ONLY reads SharedState.snapshot()
  - All mutations go through state.update() or state.reset_session()
"""

import os
import sys
import time
import json
import pathlib
import threading

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, render_template

# ── Local imports ─────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from state        import SharedState
from profile      import UserProfile, load_profile, save_profile, update_profile_after_session
from exercises    import EXERCISE_CONFIG, RepCounter, get_exercise_angles, score_rep, LM, UPPER_BODY_EXERCISES
from gamification import Gamification, BADGE_DEFS
from pose_engine  import PoseEngine
from agents.orchestrator import Orchestrator
from agents.chatbot_agent import ChatbotAgent

# ── Singletons ────────────────────────────────────────────────────────────────
state    = SharedState()
gam      = Gamification()
orch     = Orchestrator()
pose_eng = PoseEngine()
profile  = load_profile()
chatbot  = ChatbotAgent()

# Load API keys from profile
if profile.anthropic_key:
    orch.set_api_key(profile.anthropic_key)
if profile.gemini_key:
    chatbot.set_api_key(profile.gemini_key)

# Flask app
app = Flask(__name__, template_folder="templates")

# ── Camera thread ─────────────────────────────────────────────────────────────
_camera_thread  : threading.Thread | None = None
_camera_stop    = threading.Event()
_rep_counter    : RepCounter | None       = None
_bottom_angle   : float                   = 999.0  # track lowest angle this rep
PROCESS_EVERY   = 1   # process every frame — immediate overlay response


def _camera_loop():
    global _rep_counter, _bottom_angle

    # Try multiple camera indices with DirectShow backend (avoids WMF conflicts on Windows)
    cap = None
    for cam_idx in [0, 1, 2]:
        _cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if _cap.isOpened():
            cap = _cap
            print(f"[Camera] Opened camera index {cam_idx} (DSHOW)")
            break
        _cap.release()
        time.sleep(0.1)

    if cap is None:
        # Fallback: plain VideoCapture without backend
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Camera] ERROR: cannot open any camera. Check device permissions.")
            return
        print("[Camera] Opened camera index 0 (fallback)")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_n    = 0
    fail_count = 0
    MAX_FAILS  = 30
    print("[Camera] Started")

    while not _camera_stop.is_set():
        ok, frame = cap.read()
        if not ok:
            fail_count += 1
            if fail_count >= MAX_FAILS:
                print("[Camera] Stream lost — attempting reopen...")
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("[Camera] Reopen failed. Stopping.")
                    break
                fail_count = 0
            time.sleep(0.05)
            continue

        fail_count = 0
        frame = cv2.flip(frame, 1)  # mirror for natural feel
        frame_n += 1

        # Process every N frames
        if frame_n % PROCESS_EVERY == 0 and state.session_active:
            _process_frame(frame)

        # Always draw reps / angle overlay on raw frame
        _draw_hud(frame)

        # Encode and push to frame buffer
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok2:
            state.push_frame(buf.tobytes())

    cap.release()
    print("[Camera] Stopped")


def _process_frame(frame: np.ndarray):
    """Run pose + form check + rep counting. Called from camera thread only."""
    global _rep_counter, _bottom_angle

    exercise = state.exercise
    cfg = EXERCISE_CONFIG.get(exercise, {})

    # ── Pose detection ──────────────────────────────────────────────────────
    # Pass exercise so the engine can use appropriate joints for confidence
    landmarks, confidence = pose_eng.process_frame(frame, exercise)

    # Upper-body exercises (curl, press) only need arm joints visible — lower bar
    conf_threshold = 0.15 if exercise in UPPER_BODY_EXERCISES else 0.35

    state.update(
        pose_detected=confidence > conf_threshold,
        confidence=confidence,
        raw_landmarks=landmarks,
    )

    if confidence < conf_threshold:
        return

    # ── Angle computation ───────────────────────────────────────────────────
    fitness_level = getattr(profile, "fitness_level", "intermediate")
    angles = get_exercise_angles(landmarks, exercise)
    state.update(angles=angles)

    # Primary angle for rep counter
    primary_key = cfg.get("primary_angle")
    primary_angle = angles.get(primary_key) if primary_key else None
    if primary_angle is not None:
        state.update(current_angle=primary_angle)

    # ── Frame-level agent eval (sync: cv + safety) ──────────────────────────
    frame_result = orch.process_frame(angles, exercise, fitness_level)
    state.update(
        active_errors   = frame_result["active_errors"],
        form_score      = frame_result["form_score"],
        safety_critical = frame_result["safety_critical"],
        safety_warnings = frame_result["safety_warnings"],
    )

    # ── Draw overlay on frame ───────────────────────────────────────────────
    pose_eng.draw_overlay(frame, landmarks,
                          frame_result["active_errors"], angles, exercise)

    # ── Rep counting (plank is time-based — skip) ───────────────────────────
    if exercise == "plank" or primary_angle is None:
        return

    if _rep_counter is None:
        _init_rep_counter(exercise, fitness_level)

    rep_done, phase = _rep_counter.update(primary_angle)
    state.update(phase=phase)

    # Track bottom angle this rep
    if phase == "bottom":
        _bottom_angle = min(_bottom_angle, primary_angle)

    # ── Rep completion ──────────────────────────────────────────────────────
    if rep_done:
        _on_rep_complete(exercise, angles, fitness_level)
        _bottom_angle = 999.0  # reset for next rep

    # ── Poll LLM result ─────────────────────────────────────────────────────
    llm_msg = orch.poll_llm()
    if llm_msg:
        state.update(coach_msg=llm_msg, llm_pending=False)


def _init_rep_counter(exercise: str, fitness_level: str):
    global _rep_counter
    from profile import adaptive_thresh
    cfg = EXERCISE_CONFIG.get(exercise, {})
    up   = adaptive_thresh(cfg.get("up_thresh",   155), fitness_level, "strict")
    down = adaptive_thresh(cfg.get("down_thresh",  100), fitness_level, "lenient")
    _rep_counter = RepCounter(up_thresh=up, down_thresh=down)
    # Establish top position
    for _ in range(4):
        _rep_counter.update(cfg.get("up_thresh", 155) + 10)


def _on_rep_complete(exercise: str, angles: dict, fitness_level: str):
    """Handle a completed rep: score, XP, badges, dispatch agents."""
    global profile

    new_rep_count = state.rep_count + 1
    scores_so_far = list(state.scores)

    # Score this rep
    errors = state.active_errors
    rep_score = score_rep(exercise, _bottom_angle, errors)
    scores_so_far.append(rep_score)

    # Gamification
    gam_result = gam.on_rep(rep_score, scores_so_far)

    # New badges → toast
    new_badge_data = None
    if gam_result["new_badges"]:
        bid = gam_result["new_badges"][0]
        icon, name, desc = BADGE_DEFS.get(bid, ("🏅", bid, ""))
        new_badge_data = {"id": bid, "icon": icon, "name": name}

    # Update state
    state.update(
        rep_count    = new_rep_count,
        form_score   = rep_score,
        scores       = scores_so_far,
        xp           = gam_result["xp_total"],
        level        = gam_result["level"],
        level_name   = gam_result["level_name"],
        streak       = gam_result["streak"],
        new_badge    = new_badge_data,
        earned_badges= list(gam.earned_badges),
    )

    # Rep-level agents (fatigue + progress) — sync & fast
    snap = state.snapshot()
    rep_result = orch.process_rep(snap, profile, scores_so_far, new_rep_count)
    state.update(
        fatigue_level = rep_result["fatigue_level"],
        fatigue_msg   = rep_result["fatigue_msg"],
        progress_msg  = rep_result["progress_msg"],
        progress_trend= rep_result["progress_trend"],
        adaptive_reps = rep_result["adaptive_reps"],
    )


def _draw_hud(frame: np.ndarray):
    """Draw rep count, form score, and safety alerts directly on frame."""
    snap = state.snapshot()
    h, w = frame.shape[:2]

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    rep   = snap.get("rep_count", 0)
    score = snap.get("form_score", 0)
    phase = snap.get("phase", "waiting")
    ex    = snap.get("exercise", "squat").upper()

    cv2.putText(frame, f"REPS: {rep}",
                (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, f"SCORE: {int(score)}",
                (160, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{ex}  {phase.upper()}",
                (350, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 1, cv2.LINE_AA)

    # Critical error banner
    crits = snap.get("safety_critical", [])
    if crits:
        cv2.rectangle(frame, (0, h-52), (w, h), (0, 0, 200), -1)
        cv2.putText(frame, crits[0][:60],
                    (10, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Pose confidence bar (bottom-right mini)
    conf = snap.get("confidence", 0)
    bar_w = int(conf * 120)
    cv2.rectangle(frame, (w-130, h-20), (w-130+bar_w, h-10),
                  (0, 200, 100), -1)
    cv2.putText(frame, "CV", (w-150, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    """MJPEG stream — reads from frame buffer only, no processing."""
    def gen():
        while True:
            frame_bytes = state.get_frame()
            if frame_bytes:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + frame_bytes + b"\r\n")
            time.sleep(0.033)  # ~30fps cap on stream
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/events")
def events():
    """
    SSE endpoint — pushes state JSON every 200ms.
    Browser uses EventSource('/events') — no polling needed.
    """
    def gen():
        while True:
            snap = state.snapshot()
            # Clear one-shot new_badge after sending
            if snap.get("new_badge"):
                state.update(new_badge=None)
            data = json.dumps(snap)
            yield f"data: {data}\n\n"
            time.sleep(0.2)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/status")
def api_status():
    return jsonify({
        "cv_ready":    pose_eng.is_ready(),
        "api_key_set": bool(profile.anthropic_key),
        "is_onboarded":profile.is_onboarded,
        "session":     state.session_active,
    })


@app.route("/api/profile", methods=["GET"])
def get_profile():
    import dataclasses
    d = dataclasses.asdict(profile)
    d.pop("anthropic_key", None)          # never send key to browser
    d["bmr"]  = round(profile.bmr)
    d["tdee"] = round(profile.tdee)
    return jsonify(d)


@app.route("/api/profile", methods=["POST"])
def set_profile():
    global profile
    data = request.get_json() or {}
    import dataclasses

    fields = {f.name for f in dataclasses.fields(UserProfile)}
    for k, v in data.items():
        if k in fields and k != "anthropic_key":
            try:
                setattr(profile, k, type(getattr(profile, k))(v))
            except Exception:
                pass

    # API key handled separately
    key = data.get("anthropic_key", "").strip()
    if key:
        profile.anthropic_key = key
        orch.set_api_key(key)

    profile.is_onboarded = True
    save_profile(profile)
    return jsonify({"ok": True, "bmr": round(profile.bmr), "tdee": round(profile.tdee)})


@app.route("/api/session/start", methods=["POST"])
def session_start():
    global _rep_counter, _bottom_angle
    data     = request.get_json() or {}
    exercise = data.get("exercise", "squat")
    if exercise not in EXERCISE_CONFIG:
        return jsonify({"error": f"Unknown exercise: {exercise}"}), 400

    state.reset_session(exercise)
    gam.reset_session()
    orch.reset_session()

    # Re-init rep counter for new exercise
    _rep_counter  = None
    _bottom_angle = 999.0

    # Store on profile for progress agent cross-reference
    profile._current_exercise = exercise

    return jsonify({
        "ok":       True,
        "exercise": exercise,
        "display":  EXERCISE_CONFIG[exercise]["display"],
        "camera_tip": EXERCISE_CONFIG[exercise]["camera_tip"],
    })


@app.route("/api/session/end_set", methods=["POST"])
def end_set():
    global _rep_counter, _bottom_angle
    scores    = list(state.scores)
    rep_count = state.rep_count
    exercise  = state.exercise

    xp_gained = gam.on_set_complete()
    msg       = orch.on_set_complete(scores, rep_count, profile, exercise)
    state.update(
        set_num   = state.set_num + 1,
        coach_msg = msg,
        xp        = gam.xp,
    )
    # Reset rep counter for new set (keep session totals)
    _rep_counter  = None
    _bottom_angle = 999.0

    return jsonify({
        "ok":       True,
        "set_num":  state.set_num,
        "xp_gained":xp_gained,
        "message":  msg,
    })


@app.route("/api/session/end", methods=["POST"])
def session_end():
    global profile
    scores    = list(state.scores)
    rep_count = state.rep_count
    exercise  = state.exercise

    summary = orch.on_session_end(scores, rep_count, profile, exercise)
    profile = update_profile_after_session(profile, exercise, scores, rep_count)

    # Nutrition estimate
    cal = (EXERCISE_CONFIG[exercise]["met"] *
           profile.weight_kg *
           (state.snapshot().get("session_secs", 0) / 3600.0))

    state.end_session()

    return jsonify({
        "ok":            True,
        "total_reps":    rep_count,
        "avg_score":     round(sum(scores)/len(scores), 1) if scores else 0,
        "best_score":    round(max(scores), 1) if scores else 0,
        "calories":      round(cal, 1),
        "xp_total":      gam.xp,
        "level":         gam.level,
        "level_name":    gam.level_name,
        "summary":       summary,
        "badges":        gam.get_all_earned(),
        "personal_best": profile.best_form_scores.get(exercise, 0),
    })


@app.route("/api/exercise", methods=["POST"])
def switch_exercise():
    global _rep_counter, _bottom_angle
    data     = request.get_json() or {}
    exercise = data.get("exercise", "squat")
    if exercise not in EXERCISE_CONFIG:
        return jsonify({"error": "Unknown exercise"}), 400
    if state.session_active:
        state.update(exercise=exercise)
        orch.cv.reset()
        _rep_counter  = None
        _bottom_angle = 999.0
        profile._current_exercise = exercise
    return jsonify({
        "ok":        True,
        "exercise":  exercise,
        "display":   EXERCISE_CONFIG[exercise]["display"],
        "camera_tip":EXERCISE_CONFIG[exercise]["camera_tip"],
        "description":EXERCISE_CONFIG[exercise]["description"],
    })


@app.route("/api/exercises")
def list_exercises():
    return jsonify([
        {"id": k, "display": v["display"], "description": v["description"],
         "camera_tip": v["camera_tip"]}
        for k, v in EXERCISE_CONFIG.items()
    ])


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400
    result = chatbot.chat(message)
    return jsonify(result)


@app.route("/api/settings", methods=["POST"])
def save_settings():
    global profile
    data = request.get_json() or {}
    gemini_key = data.get("gemini_key", "").strip()
    if gemini_key:
        profile.gemini_key = gemini_key
        chatbot.set_api_key(gemini_key)
        save_profile(profile)
    return jsonify({"ok": True, "message": "API key successfully updated"})


@app.route("/api/badges")
def badges():
    all_defs = [
        {"id": bid, "icon": icon, "name": name, "desc": desc,
         "earned": bid in gam.earned_badges}
        for bid, (icon, name, desc) in BADGE_DEFS.items()
    ]
    return jsonify(all_defs)


# ── App startup ───────────────────────────────────────────────────────────────

def start_camera():
    global _camera_thread
    _camera_stop.clear()
    _camera_thread = threading.Thread(target=_camera_loop, daemon=True, name="CameraThread")
    _camera_thread.start()


def stop_camera():
    _camera_stop.set()


# Start camera automatically when app loads
start_camera()
