"""
run.py — Fitrainer v2 entry point.
Finds Python, downloads model if missing, starts Flask, opens browser.
"""
import os, sys, time, socket, pathlib, threading, subprocess, webbrowser

BASE = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE))
PORT = 5000
HOST = "127.0.0.1"

def load_env():
    env = BASE / ".env"
    if env.exists():
        for ln in env.read_text().splitlines():
            if "=" in ln and not ln.startswith("#"):
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

def check_model():
    m = BASE / "models" / "pose_landmarker_lite.task"
    if not m.exists():
        print("\nPose model not found — running setup...")
        import setup; setup.run()

def wait_ready(host, port, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5): return True
        except OSError: time.sleep(0.25)
    return False

def open_browser(url):
    time.sleep(0.2)
    ready = wait_ready(HOST, PORT, timeout=18)
    if not ready:
        print(f"\nServer took too long. Open manually: {url}"); return
    print(f"Opening {url}")
    opened = False
    if sys.platform == "win32":
        try: os.startfile(url); opened = True
        except: pass
        if not opened:
            try: subprocess.Popen(["cmd","/c","start","",url], shell=False,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); opened = True
            except: pass
    elif sys.platform == "darwin":
        try: subprocess.Popen(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); opened = True
        except: pass
    else:
        for cmd in ["xdg-open","gnome-open","sensible-browser"]:
            try: subprocess.Popen([cmd, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); opened = True; break
            except: continue
    if not opened:
        try: webbrowser.open(url)
        except: print(f"Please open manually: {url}")

def main():
    print("\n========================================")
    print("  FITRAINER v2 — AI Fitness Coach")
    print("========================================\n")
    load_env()
    check_model()

    from app import app
    url = f"http://{HOST}:{PORT}"
    print(f"Server: {url}")
    print("Stop:   Ctrl+C\n")
    print(f"If browser does not open: {url}\n")

    t = threading.Thread(target=open_browser, args=(url,), daemon=True)
    t.start()

    try:
        app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nFitrainer stopped.")
    except OSError as e:
        if "10048" in str(e) or "Address already in use" in str(e):
            print(f"\nPort {PORT} in use. Try: python run.py (or kill other process)")
        else: raise

if __name__ == "__main__":
    main()
