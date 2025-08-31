from flask import Flask, Response, jsonify
import cv2
import threading
import time
import pandas as pd
import json
import os

app = Flask(__name__)


class CameraStreamer:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.detector = cv2.QRCodeDetector()
        self.last_qr = ""

    def classify_qr(self, qr_data: str) -> str:
        qr_lower = qr_data.lower()
        if qr_data.startswith("MB-") or "mien bac" in qr_lower:
            return "Miền Bắc"
        if qr_data.startswith("MT-") or "mien trung" in qr_lower:
            return "Miền Trung"
        if qr_data.startswith("MN-") or "mien nam" in qr_lower:
            return "Miền Nam"
        return "Miền khác"

    def save_qr_data(self, qr_entry):
        data_file = "qr_data.json"
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []

        data.append(qr_entry)

        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def generate_frames(self):
        while True:
            success, frame = self.camera.read()
            if not success:
                break

                # QR detection
            data, points, _ = self.detector.detectAndDecode(frame)

            if points is not None and data and data != self.last_qr:
                # Vẽ khung QR
                points = points.astype(int).reshape(-1, 2)
                for j in range(len(points)):
                    pt1 = tuple(points[j])
                    pt2 = tuple(points[(j + 1) % len(points)])
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 3)

                    # Save QR data
                qr_region = self.classify_qr(data)
                qr_entry = {
                    "data": data,
                    "type": "QRCODE",
                    "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "region": qr_region,
                }
                self.save_qr_data(qr_entry)
                self.last_qr = data

                # Hiển thị text
                cv2.rectangle(frame, (points[0][0], points[0][1] - 35),
                              (points[0][0] + len(data) * 12, points[0][1] - 5),
                              (0, 255, 0), -1)
                cv2.putText(frame, data, (points[0][0], points[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\\r\\n'
                   b'Content-Type: image/jpeg\\r\\n\\r\\n' + frame + b'\\r\\n')


camera_streamer = CameraStreamer()


@app.get('/video_feed')
def video_feed():
    return Response(camera_streamer.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.get('/api/qr_data')
def get_qr_data():
    data_file = "qr_data.json"
    if os.path.exists(data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify([])


@app.route('/api/last_qr')
def get_last_qr():
    data_file = "qr_data.json"
    if os.path.exists(data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if data:
                return jsonify({"data": data[-1]['data']})
    return jsonify({"data": ""})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)