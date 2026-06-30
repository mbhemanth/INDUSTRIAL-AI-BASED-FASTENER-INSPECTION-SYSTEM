# =========================================================
# INDUSTRIAL INSPECTION SYSTEM — FLASK + DUAL CAMERA
# SINGLE YOLO MODEL SHARED
# KEEPING ORIGINAL UI/VISUALIZATION
# JETSON / DOCKER STABLE VERSION
#
# CHANGE FROM ORIGINAL:
#   - Spatial dedup grid added to prevent double-count
#     when car stops/starts and ByteTrack re-assigns ID
#   - Everything else is IDENTICAL to original
#
# OTHER FEATURES (unchanged):
#   - PLC connection (pymodbus) — reads model from registers
#   - OK / NG signal written to PLC register 29 per vehicle
#   - Manual OK signal via browser R key / button
#   - /signal_log endpoint to check last signals
# =========================================================

from flask import Flask, Response, render_template_string, jsonify, request as freq
import subprocess
import cv2
import threading
import time
import torch
import numpy as np
from ultralytics import YOLO
import shutil
from pathlib import Path
from pymodbus.client import ModbusTcpClient

# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

# =========================================================
# RTSP CONFIG
# =========================================================

USER = "Hemanth"
PASS = "Tansam%40123"

RTSP_URL_1 = f"rtsp://{USER}:{PASS}@192.168.0.11/MediaInput/h264/stream_2"
RTSP_URL_2 = f"rtsp://{USER}:{PASS}@192.168.0.10/MediaInput/h264/stream_2"

# =========================================================
# PLC CONFIG
# =========================================================

PLC_IP   = "192.168.251.1"
PLC_PORT = 502

plc_client    = None
plc_connected = False
plc_lock      = threading.Lock()

# =========================================================
# LOWER RESOLUTION FOR JETSON STABILITY
# =========================================================

WIDTH    = 800
HEIGHT   = 600
IMG_SIZE = 640

# =========================================================
# BYTETRACK YAMLs
# =========================================================

def get_bytetrack_yaml():
    try:
        import ultralytics
        pkg_dir = Path(ultralytics.__file__).parent
        yaml_path = pkg_dir / "cfg" / "trackers" / "bytetrack.yaml"
        if yaml_path.exists():
            return str(yaml_path)
    except Exception:
        pass
    return "bytetrack.yaml"


_bt_yaml = get_bytetrack_yaml()

BYTETRACK_YAML_RH = "bytetrack_rh.yaml"
BYTETRACK_YAML_LH = "bytetrack_lh.yaml"

shutil.copy(_bt_yaml, BYTETRACK_YAML_RH)
shutil.copy(_bt_yaml, BYTETRACK_YAML_LH)

print(f"✅ ByteTrack YAMLs: {BYTETRACK_YAML_RH}, {BYTETRACK_YAML_LH}")

# =========================================================
# YOLO MODELS
# =========================================================

MODEL_PATH = "clipv8m_fp32.engine"

model_rh = YOLO(MODEL_PATH, task="detect")
model_lh = YOLO(MODEL_PATH, task="detect")

USE_GPU = torch.cuda.is_available()

if USE_GPU:
    print("✅ CUDA GPU ENABLED")
else:
    print("⚠ CPU MODE")

# =========================================================
# JETSON STABILITY SETTINGS
# =========================================================

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled   = True

running            = True
current_model_name = "WAITING MODEL"

# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = {
    0: "clip_ok",
    1: "clip_ng",
    2: "bolt_ok",
    3: "bolt_ng",
    4: "dummy_ok",
    5: "dummy_ng",
}

# =========================================================
# REQUIRED COUNTS
# =========================================================

REQUIRED = {
    "clip":  7,
    "bolt":  2,
    "dummy": 2,
}

# =========================================================
# COUNT LINE
# =========================================================

COUNT_LINE_Y = 350

# =========================================================
# SPATIAL DEDUP SETTINGS  ← ONLY NEW THING ADDED
# After any object is counted, that 80×80px grid cell on
# the count line is blocked for 3 seconds.
# So if the car stops and ByteTrack drops+reassigns the ID,
# the new ID cannot count again in the same spot.
# Tune these two values if needed:
#   DEDUP_CELL_SIZE — shrink to 60 if clips are very close
#   DEDUP_SECONDS   — increase if car stops longer than 3s
# =========================================================

DEDUP_CELL_SIZE = 80
DEDUP_SECONDS   = 3.0

# =========================================================
# GLOBAL VEHICLE RESULT
# =========================================================

vehicle_results = {
    "RH SIDE": None,
    "LH SIDE": None,
}

vehicle_lock = threading.Lock()

# =========================================================
# COOLDOWN / HOLD
# =========================================================

cooldown_active     = False
cooldown_start_time = 0
COOLDOWN_SECONDS    = 20

# =========================================================
# SIGNAL LOG — stores last 50 signals for /signal_log
# =========================================================

signal_log  = []
signal_lock = threading.Lock()

# =========================================================
# PLC CONNECT
# =========================================================

def connect_plc():
    global plc_client, plc_connected

    try:
        if plc_client is not None:
            try:
                plc_client.close()
            except Exception:
                pass

        plc_client = ModbusTcpClient(
            PLC_IP,
            port=PLC_PORT,
            timeout=5,
            retries=0
        )

        plc_connected = plc_client.connect()

        if plc_connected:
            print(f"✅ PLC Connected → {PLC_IP}:{PLC_PORT}")
        else:
            print(f"❌ PLC Connection Failed → {PLC_IP}:{PLC_PORT}")

    except Exception as e:
        print(f"PLC CONNECTION ERROR: {e}")
        plc_connected = False

# =========================================================
# PLC WRITE SIGNAL
# value=1 → NG   value=2 → OK
# Writes register 29, waits 200ms, resets to 0
# =========================================================

def write_plc_signal(value):
    global plc_connected

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    label     = "OK" if value == 2 else "NG"

    if value == 2:
        print(f"✅ [{timestamp}] PLC SIGNAL → OK (reg29={value})")
    else:
        print(f"❌ [{timestamp}] PLC SIGNAL → NG (reg29={value})")

    with signal_lock:
        signal_log.append({
            "time":  timestamp,
            "value": value,
            "label": label
        })
        if len(signal_log) > 50:
            signal_log.pop(0)

    if not plc_connected:
        print("⚠ PLC not connected — signal logged only")
        return

    try:
        with plc_lock:
            wr1 = plc_client.write_register(29, value)
            time.sleep(0.2)
            wr2 = plc_client.write_register(29, 0)

        if wr1 is None or wr2 is None:
            print("⚠ PLC WRITE FAILED — no response")
            plc_connected = False
            return

        if wr1.isError() or wr2.isError():
            print("⚠ PLC WRITE ERROR RESPONSE")
            plc_connected = False
            return

        print(f"✅ PLC WRITE SUCCESS → reg29={value} then reset 0")

    except Exception as e:
        print(f"PLC WRITE ERROR: {e}")
        plc_connected = False

# =========================================================
# PLC THREAD — connects PLC + reads model name from registers
# reg0==1  → DUSTER
# reg1==2  → TEKTON
# =========================================================

def plc_thread():
    global plc_connected, current_model_name

    connect_plc()

    while running:
        try:
            if not plc_connected:
                print("⏳ PLC disconnected. Retry after 10 sec...")
                time.sleep(10)
                connect_plc()
                continue

            with plc_lock:
                r = plc_client.read_holding_registers(address=0, count=2)

            if r is None or r.isError():
                print("⚠ PLC read failed. Keeping connection, not reconnecting immediately.")
                time.sleep(2)
                continue

            reg0 = r.registers[0]
            reg1 = r.registers[1]

            if reg0 == 1:
                current_model_name = "DUSTER"
            elif reg1 == 2:
                current_model_name = "TEKTON"
            else:
                current_model_name = "UNKNOWN MODEL"

        except Exception as e:
            print(f"PLC THREAD ERROR: {e}")
            plc_connected = False

        time.sleep(1)

# =========================================================
# FINAL RESULT CHECK — sends OK/NG to PLC + starts hold
# =========================================================

def check_and_send_final_result():
    global vehicle_results, cooldown_active, cooldown_start_time

    with vehicle_lock:
        rh = vehicle_results["RH SIDE"]
        lh = vehicle_results["LH SIDE"]

        if rh is None or lh is None:
            return

        print(f"FINAL RESULT => RH:{rh} | LH:{lh}")

        if rh == "OK" and lh == "OK":
            write_plc_signal(2)
            print("✅ FINAL VEHICLE RESULT = OK")
        else:
            write_plc_signal(1)
            print("❌ FINAL VEHICLE RESULT = NG")

        cooldown_active     = True
        cooldown_start_time = time.time()
        print("⏳ HOLD STARTED")

        vehicle_results["RH SIDE"] = None
        vehicle_results["LH SIDE"] = None

# =========================================================
# DRAW NUMBER CIRCLES
# =========================================================

def draw_number_circles(frame, start_x, y, total, history):
    for i in range(total):
        number     = str(i + 1)
        color      = (255, 255, 255)
        thickness  = 2
        text_color = (255, 255, 255)

        if i < len(history):
            status = history[i]
            if status == "OK":
                color     = (0, 255, 0)
                thickness = -1
            elif status == "NG":
                color     = (0, 0, 255)
                thickness = -1

        cv2.circle(frame, (start_x + i * 52, y), 18, color, thickness)
        cv2.putText(frame, number, (start_x - 8 + i * 52, y + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

# =========================================================
# DRAW STATUS PANEL
# =========================================================

def draw_status_panel(frame, cam):
    counts = cam.counts

    cv2.putText(frame, cam.side_name, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    cv2.putText(frame, f"MODEL : {current_model_name}", (250, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    # PLC status indicator top-right
    plc_status_color = (0, 255, 0) if plc_connected else (0, 0, 255)
    plc_status_text  = "PLC:OK" if plc_connected else "PLC:FAIL"
    cv2.putText(frame, plc_status_text, (630, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, plc_status_color, 2)

    # CLIP
    cv2.putText(frame, "CLIP", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    draw_number_circles(frame, 220, 110, REQUIRED["clip"], cam.clip_history)

    # BOLT
    cv2.putText(frame, "BOLT", (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    draw_number_circles(frame, 220, 220, REQUIRED["bolt"], cam.bolt_history)

    # DUMMY
    cv2.putText(frame, "DUMMY", (20, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    draw_number_circles(frame, 220, 330, REQUIRED["dummy"], cam.dummy_history)

    any_ng = (
        counts["clip_ng"]  > 0 or
        counts["bolt_ng"]  > 0 or
        counts["dummy_ng"] > 0
    )

    all_ok = (
        counts["clip_ok"]  == 7 and
        counts["bolt_ok"]  == 2 and
        counts["dummy_ok"] == 2 and
        counts["clip_ng"]  == 0 and
        counts["bolt_ng"]  == 0 and
        counts["dummy_ng"] == 0
    )

    if any_ng:
        cv2.putText(frame, "NG", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 7)
    elif all_ok:
        cv2.putText(frame, "OK", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 7)

    return frame

# =========================================================
# FFMPEG PIPE
# =========================================================

def open_ffmpeg_pipe(rtsp_url):
    command = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-fflags",         "nobuffer",
        "-flags",          "low_delay",
        "-i",              rtsp_url,
        "-vf",             f"scale={WIDTH}:{HEIGHT}",
        "-f",              "rawvideo",
        "-pix_fmt",        "bgr24",
        "-"
    ]
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8
    )

# =========================================================
# CAMERA CLASS
# =========================================================

class CameraStream:

    def __init__(self, url, side_name, model, tracker_yaml):
        self.url          = url
        self.side_name    = side_name
        self.model        = model
        self.tracker_yaml = tracker_yaml

        self.latest_frame    = None
        self.processed_frame = None
        self.lock            = threading.Lock()

        # BYTETRACK STATE
        self.track_prev_cy     = {}
        self.counted_ids       = set()
        self.track_class_votes = {}

        self.vehicle_active      = False
        self.result_sent         = False
        self.final_bolt_detected = False

        self.clip_history  = []
        self.bolt_history  = []
        self.dummy_history = []

        self.counts = {
            "clip_ok":  0,
            "clip_ng":  0,
            "bolt_ok":  0,
            "bolt_ng":  0,
            "dummy_ok": 0,
            "dummy_ng": 0,
        }

        # SPATIAL DEDUP  ← only addition vs original __init__
        self.spatial_dedup = {}

        self.pipe = open_ffmpeg_pipe(self.url)

        threading.Thread(target=self.grab_frames, daemon=True).start()

    # =====================================================
    # RESET
    # =====================================================

    def reset_inspection(self):
        print(f"🔄 RESET : {self.side_name}")
        for key in self.counts:
            self.counts[key] = 0
        self.track_prev_cy.clear()
        self.counted_ids.clear()
        self.track_class_votes.clear()
        self.spatial_dedup.clear()          # ← clear dedup on reset
        self.vehicle_active      = False
        self.result_sent         = False
        self.final_bolt_detected = False
        self.clip_history.clear()
        self.bolt_history.clear()
        self.dummy_history.clear()
        print(f"✅ RESET COMPLETE : {self.side_name}")

    # =====================================================
    # FRAME CAPTURE
    # =====================================================

    def grab_frames(self):
        frame_size = WIDTH * HEIGHT * 3
        print(f"✅ {self.side_name} ffmpeg started")

        while running:
            raw = self.pipe.stdout.read(frame_size)
            if len(raw) != frame_size:
                print(f"⚠ {self.side_name} reconnecting...")
                try:
                    self.pipe.kill()
                except Exception:
                    pass
                time.sleep(2)
                self.pipe = open_ffmpeg_pipe(self.url)
                continue

            frame = np.frombuffer(raw, np.uint8).reshape((HEIGHT, WIDTH, 3))
            with self.lock:
                self.latest_frame = frame

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def set_processed_frame(self, frame):
        with self.lock:
            self.processed_frame = frame

    def get_frame(self):
        with self.lock:
            return self.processed_frame

# =========================================================
# CAMERA OBJECTS  ← identical to original, no direction arg
# =========================================================

cam_rh = CameraStream(RTSP_URL_1, "RH SIDE", model_rh, BYTETRACK_YAML_RH)
cam_lh = CameraStream(RTSP_URL_2, "LH SIDE", model_lh, BYTETRACK_YAML_LH)

# =========================================================
# PROCESS CAMERA
# =========================================================

def process_camera(cam):
    global cooldown_active, cooldown_start_time

    frame = cam.get_latest_frame()
    if frame is None:
        return

    # =====================================================
    # HOLD MODE
    # =====================================================

    if cooldown_active:
        remaining = COOLDOWN_SECONDS - int(time.time() - cooldown_start_time)
        if remaining > 0:
            frame = draw_status_panel(frame, cam)
            cv2.putText(frame, f"HOLD : {remaining}s", (380, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 4)
            cam.set_processed_frame(frame)
            return
        else:
            cooldown_active = False
            cam_rh.reset_inspection()
            cam_lh.reset_inspection()
            print("✅ HOLD COMPLETE")

    # =====================================================
    # YOLO TRACK
    # =====================================================

    with torch.no_grad():
        results = cam.model.track(
            frame,
            imgsz=IMG_SIZE,
            conf=0.15,
            iou=0.45,
            tracker=cam.tracker_yaml,
            persist=True,
            device=0 if USE_GPU else "cpu",
            half=False,
            verbose=False,
            stream=False
        )

    result = results[0]

    # =====================================================
    # DRAW LINE
    # =====================================================

    cv2.line(frame, (0, COUNT_LINE_Y), (WIDTH, COUNT_LINE_Y), (0, 255, 255), 3)

    # =====================================================
    # CLEAN EXPIRED SPATIAL DEDUP CELLS
    # =====================================================

    now = time.time()
    cam.spatial_dedup = {
        k: t for k, t in cam.spatial_dedup.items()
        if now - t < DEDUP_SECONDS
    }

    # =====================================================
    # DETECTIONS
    # =====================================================

    if result.boxes is not None and result.boxes.id is not None:
        boxes     = result.boxes
        track_ids = boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            cls        = int(box.cls[0])
            class_name = CLASS_NAMES.get(cls)
            if class_name is None:
                continue

            conf = float(box.conf[0])
            if conf < 0.3:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            crossed = False

            # TRACK VOTES
            if track_id not in cam.track_class_votes:
                cam.track_class_votes[track_id] = {}
            votes = cam.track_class_votes[track_id]
            votes[class_name] = votes.get(class_name, 0) + 1

            prev_cy = cam.track_prev_cy.get(track_id)
            cam.track_prev_cy[track_id] = cy

            # =====================================================
            # CROSSING — IDENTICAL TO ORIGINAL
            # Counts both directions so detection is unchanged
            # =====================================================
            crossed_line = (
                prev_cy is not None and (
                    (prev_cy < COUNT_LINE_Y <= cy) or
                    (prev_cy > COUNT_LINE_Y >= cy)
                )
            )

            # =====================================================
            # SPATIAL DEDUP  ← only new check added here
            # Grid cell at crossing point blocked for DEDUP_SECONDS
            # after any count — stops re-count on ID reassignment
            # =====================================================
            cell_key     = (cx // DEDUP_CELL_SIZE, cy // DEDUP_CELL_SIZE)
            cell_blocked = cell_key in cam.spatial_dedup

            if cell_blocked and crossed_line and track_id not in cam.counted_ids:
                print(f"⚠ {cam.side_name} | DEDUP BLOCKED | cell={cell_key} | {class_name}")

            if crossed_line and track_id not in cam.counted_ids and not cell_blocked:
                best_class = max(votes, key=votes.get)

                cam.counted_ids.add(track_id)
                cam.spatial_dedup[cell_key] = now   # lock cell
                cam.counts[best_class] += 1
                crossed = True

                print(f"{cam.side_name} | {best_class} | total={cam.counts[best_class]}")

                if   best_class == "clip_ok":  cam.clip_history.append("OK")
                elif best_class == "clip_ng":  cam.clip_history.append("NG")
                elif best_class == "bolt_ok":  cam.bolt_history.append("OK")
                elif best_class == "bolt_ng":  cam.bolt_history.append("NG")
                elif best_class == "dummy_ok": cam.dummy_history.append("OK")
                elif best_class == "dummy_ng": cam.dummy_history.append("NG")

                cam.vehicle_active = True

                if "bolt" in best_class:
                    total_bolt = cam.counts["bolt_ok"] + cam.counts["bolt_ng"]
                    if total_bolt >= 2:
                        cam.final_bolt_detected = True

            # DRAW BOX
            color = (0, 0, 255) if "ng" in class_name else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
            label = class_name + (" COUNTED" if crossed else "")
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # =====================================================
    # FINAL RESULT — write OK/NG to PLC register 29
    # =====================================================

    if cam.vehicle_active and cam.final_bolt_detected and not cam.result_sent:
        all_ok = (
            cam.counts["clip_ok"]  == 7 and
            cam.counts["dummy_ok"] == 2 and
            cam.counts["bolt_ok"]  == 2 and
            cam.counts["clip_ng"]  == 0 and
            cam.counts["dummy_ng"] == 0 and
            cam.counts["bolt_ng"]  == 0
        )

        with vehicle_lock:
            vehicle_results[cam.side_name] = "OK" if all_ok else "NG"

        print(f"{cam.side_name} FINAL RESULT = {vehicle_results[cam.side_name]}")
        cam.result_sent = True
        check_and_send_final_result()

    # UI
    frame = draw_status_panel(frame, cam)
    cam.set_processed_frame(frame)

# =========================================================
# CAMERA + PLC THREADS
# =========================================================

def rh_thread():
    while running:
        try:
            process_camera(cam_rh)
        except Exception as e:
            print(f"RH ERROR: {e}")
        time.sleep(0.03)


def lh_thread():
    while running:
        try:
            process_camera(cam_lh)
        except Exception as e:
            print(f"LH ERROR: {e}")
        time.sleep(0.03)


threading.Thread(target=rh_thread,  daemon=True).start()
threading.Thread(target=lh_thread,  daemon=True).start()
threading.Thread(target=plc_thread, daemon=True).start()

# =========================================================
# COMBINED STREAM
# =========================================================

def generate_combined():
    while True:
        frame_rh = cam_rh.get_frame()
        frame_lh = cam_lh.get_frame()

        if frame_rh is None or frame_lh is None:
            black = np.zeros((HEIGHT, WIDTH * 2, 3), dtype=np.uint8)
            cv2.putText(black, "WAITING FOR CAMERAS...", (300, HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            _, buf = cv2.imencode(".jpg", black)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
            continue

        rh_disp  = cv2.resize(frame_rh, (960, 1080))
        lh_disp  = cv2.resize(frame_lh, (960, 1080))
        combined = cv2.hconcat([rh_disp, lh_disp])

        # FINAL BANNER
        with vehicle_lock:
            rh_res = vehicle_results.get("RH SIDE")
            lh_res = vehicle_results.get("LH SIDE")

        if rh_res is not None and lh_res is not None:
            if rh_res == "OK" and lh_res == "OK":
                banner_text  = "VEHICLE OK"
                banner_color = (0, 255, 0)
            else:
                banner_text  = "VEHICLE NG"
                banner_color = (0, 0, 255)
            cv2.putText(combined, banner_text, (700, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, banner_color, 6)

        # HOLD DISPLAY
        if cooldown_active:
            remaining = COOLDOWN_SECONDS - int(time.time() - cooldown_start_time)
            cv2.putText(combined, f"HOLD : {remaining}s", (760, 1030),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

        # PLC STATUS on combined frame
        plc_col  = (0, 255, 0) if plc_connected else (0, 0, 255)
        plc_text = f"PLC {PLC_IP} : {'CONNECTED' if plc_connected else 'DISCONNECTED'}"
        cv2.putText(combined, plc_text, (700, 1060),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, plc_col, 2)

        _, buf = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")

        time.sleep(0.03)

# =========================================================
# HTML
# =========================================================

HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Inspection System</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #0a0a0a;
    font-family: Courier New;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
}
header {
    width: 100%;
    background: #111;
    border-bottom: 2px solid #00ff88;
    padding: 14px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
header h1 { color: #00ff88; font-size: 1.4rem; letter-spacing: 4px; }
#plc-badge {
    font-size: 0.9rem;
    padding: 4px 14px;
    border-radius: 4px;
    font-weight: bold;
    letter-spacing: 2px;
}
.btn-row { display: flex; gap: 16px; margin: 16px; }
.btn {
    border: none;
    padding: 10px 28px;
    font-size: 16px;
    font-family: Courier New;
    font-weight: bold;
    cursor: pointer;
    letter-spacing: 2px;
    border-radius: 4px;
    transition: opacity 0.15s;
}
.btn:active { opacity: 0.7; }
.btn-reset { background: #ff9900; color: #000; }
.btn-ok    { background: #00cc44; color: #000; }
.btn-ng    { background: #cc0000; color: #fff; }
.signal-toast {
    display: none;
    position: fixed;
    top: 24px;
    right: 24px;
    padding: 14px 28px;
    font-size: 1.3rem;
    font-family: Courier New;
    font-weight: bold;
    border-radius: 6px;
    z-index: 999;
    letter-spacing: 2px;
}
.stream-wrap { width: 100%; max-width: 1920px; padding: 20px; }
.stream-wrap img { width: 100%; border: 2px solid #222; }
</style>
</head>
<body>

<header>
    <h1>INDUSTRIAL INSPECTION SYSTEM</h1>
    <span id="plc-badge" style="background:#333;color:#aaa;">PLC CHECKING...</span>
</header>

<div class="btn-row">
    <button class="btn btn-reset" onclick="doReset()">⟳ MANUAL RESET</button>
    <button class="btn btn-ok"    onclick="doManualOK()">✔ MANUAL OK SIGNAL &nbsp;[R]</button>
    <button class="btn btn-ng"    onclick="doManualNG()">✘ MANUAL NG SIGNAL</button>
</div>

<div id="toast" class="signal-toast"></div>

<div class="stream-wrap">
    <img src="/video_feed">
</div>

<script>
function showToast(msg, color) {
    var t = document.getElementById('toast');
    t.textContent  = msg;
    t.style.background = color;
    t.style.color  = '#fff';
    t.style.display = 'block';
    setTimeout(function() { t.style.display = 'none'; }, 2500);
}

function doReset() {
    fetch('/reset').then(function() { showToast('RESET DONE', '#ff9900'); });
}

function doManualOK() {
    fetch('/manual_signal?value=2').then(r => r.json()).then(function(d) {
        showToast('✔ OK SIGNAL SENT', '#00cc44');
    });
}

function doManualNG() {
    fetch('/manual_signal?value=1').then(r => r.json()).then(function(d) {
        showToast('✘ NG SIGNAL SENT', '#cc0000');
    });
}

/* Keyboard: R = Manual OK (same as original cv2 code) */
document.addEventListener('keydown', function(e) {
    if (e.key === 'r' || e.key === 'R') doManualOK();
});

/* Poll PLC status every 3 seconds */
function updatePLCBadge() {
    fetch('/plc_status').then(r => r.json()).then(function(d) {
        var badge = document.getElementById('plc-badge');
        if (d.connected) {
            badge.style.background = '#004400';
            badge.style.color      = '#00ff88';
            badge.textContent      = 'PLC ' + d.ip + ' : CONNECTED | MODEL: ' + d.model;
        } else {
            badge.style.background = '#440000';
            badge.style.color      = '#ff4444';
            badge.textContent      = 'PLC ' + d.ip + ' : DISCONNECTED';
        }
    }).catch(function() {});
}

updatePLCBadge();
setInterval(updatePLCBadge, 3000);
</script>
</body>
</html>
"""

# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def index():
    return render_template_string(HOME_HTML)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_combined(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/reset")
def manual_reset():
    cam_rh.reset_inspection()
    cam_lh.reset_inspection()
    print("🔄 MANUAL RESET")
    return "RESET OK", 200


@app.route("/manual_signal")
def manual_signal():
    try:
        value = int(freq.args.get("value", 2))
    except ValueError:
        value = 2
    write_plc_signal(value)
    return jsonify({"status": "sent", "value": value,
                    "label": "OK" if value == 2 else "NG"})


@app.route("/plc_status")
def plc_status():
    return jsonify({
        "connected": plc_connected,
        "ip":        PLC_IP,
        "port":      PLC_PORT,
        "model":     current_model_name,
    })


@app.route("/signal_log")
def get_signal_log():
    with signal_lock:
        return jsonify(list(signal_log))


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("🚀 Starting Flask Inspection Server...")
    print(f"   Dedup cell size : {DEDUP_CELL_SIZE}px")
    print(f"   Dedup cooldown  : {DEDUP_SECONDS}s")
    app.run(host="0.0.0.0", port=5001, threaded=True)
