"""
setup.py — First-time setup: download pose model, validate imports.
Run once before starting: python setup.py
"""
import os, sys, pathlib, urllib.request

BASE    = pathlib.Path(__file__).parent
MODELS  = BASE / "models"
MODEL   = MODELS / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

def run():
    print("\n=== Fitrainer Setup ===\n")
    MODELS.mkdir(exist_ok=True)
    (BASE / "templates").mkdir(exist_ok=True)

    # 1. Check Python
    if sys.version_info < (3, 9):
        print("ERROR: Python 3.9+ required"); sys.exit(1)
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}: OK")

    # 2. Check deps
    for pkg, imp in [("flask","flask"),("mediapipe","mediapipe"),
                      ("cv2","cv2"),("numpy","numpy")]:
        try:
            __import__(imp); print(f"  {pkg}: OK")
        except ImportError:
            print(f"  {pkg}: MISSING — installing...")
            os.system(f"{sys.executable} -m pip install {pkg} -q")

    # 3. Download model
    if MODEL.exists() and MODEL.stat().st_size > 1_000_000:
        print(f"Model: OK ({MODEL.stat().st_size//1024}KB)")
    else:
        print("Downloading pose model (~12MB)...")
        try:
            def prog(c,bs,tot):
                pct = int(c*bs*100/tot) if tot>0 else 0
                print(f"\r  {pct}%", end="", flush=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL, reporthook=prog)
            print(f"\nModel downloaded: {MODEL.stat().st_size//1024}KB")
        except Exception as e:
            print(f"\nDownload failed: {e}")
            print(f"Manual download URL:\n  {MODEL_URL}")
            print(f"Save to: {MODEL}")

    # 4. Create profile if missing
    pf = BASE / "profile.json"
    if not pf.exists():
        import json
        pf.write_text(json.dumps({"is_onboarded": False}, indent=2))
        print("profile.json created")

    print("\n=== Setup Complete. Run: python run.py ===\n")

if __name__ == "__main__":
    run()
