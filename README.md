# Industrial Inspection System using YOLOv8, Flask, and PLC

## Overview

The Industrial Inspection System is a real-time computer vision application developed using Flask, YOLOv8, OpenCV, and Modbus PLC communication. The system inspects automotive components using two industrial cameras and automatically determines whether each vehicle passes or fails inspection.

The application performs AI-based object detection, tracks objects using ByteTrack, communicates inspection results to a PLC, and provides a live monitoring dashboard through a web interface.

---

## Features

- Real-time dual camera inspection
- YOLOv8 object detection
- ByteTrack object tracking
- Automatic OK/NG decision
- PLC communication using Modbus TCP
- Live web dashboard using Flask
- Manual OK, NG, and Reset controls
- Real-time PLC connection monitoring
- Vehicle inspection history
- Spatial deduplication to prevent duplicate counting
- GPU acceleration support (NVIDIA Jetson)

---

## Technologies Used

- Python
- Flask
- OpenCV
- YOLOv8 (Ultralytics)
- PyTorch
- NumPy
- Modbus TCP (pymodbus)
- ByteTrack
- FFmpeg
- HTML
- JavaScript

---

## Project Structure

```
Industrial-Inspection-System/
│
├── app.py
├── clipv8m_fp32.engine
├── bytetrack_rh.yaml
├── bytetrack_lh.yaml
├── requirements.txt
├── README.md
└── static/
```

---

## System Workflow

1. Capture video from two RTSP cameras.
2. Detect components using YOLOv8.
3. Track detected objects using ByteTrack.
4. Count clips, bolts, and dummy parts.
5. Prevent duplicate counting using spatial deduplication.
6. Verify required component counts.
7. Send OK or NG signal to the PLC.
8. Display inspection results on the Flask dashboard.

---

## Required Components

### Hardware

- NVIDIA Jetson Device (Recommended)
- Two Industrial IP Cameras
- PLC supporting Modbus TCP
- Local Network

### Software

- Python 3.10+
- CUDA (Optional)
- FFmpeg
- Git

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/industrial-inspection-system.git
cd industrial-inspection-system
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
python app.py
```

The application starts on

```
http://localhost:5001
```

---

## PLC Communication

The system communicates with the PLC using Modbus TCP.

Default Configuration

```
PLC IP : 192.168.251.1
Port   : 502
```

Signals

| Result | Register | Value |
|---------|----------|-------|
| OK | 29 | 2 |
| NG | 29 | 1 |
| Reset | 29 | 0 |

---

## Inspection Criteria

The vehicle is considered **OK** when:

- 7 Clip OK
- 2 Bolt OK
- 2 Dummy OK
- No NG detections

Otherwise, the vehicle is classified as **NG**.

---

## Dashboard Features

- Live video streaming
- PLC connection status
- Current production model
- Object count visualization
- Manual Reset button
- Manual OK signal
- Manual NG signal
- Vehicle inspection status

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard |
| `/video_feed` | GET | Live video stream |
| `/reset` | GET | Reset inspection |
| `/manual_signal` | GET | Send manual OK/NG signal |
| `/plc_status` | GET | PLC connection status |
| `/signal_log` | GET | Inspection signal history |

---

## Future Improvements

- Database integration
- Inspection report generation
- User authentication
- Production analytics dashboard
- Cloud monitoring
- Email alerts
- Multi-camera support

---

## Author

**Hemanth M B**

B.Tech Computer Science and Engineering (AI & ML)

SRM Institute of Science and Technology

---

## License

This project is intended for educational and industrial research purposes.
