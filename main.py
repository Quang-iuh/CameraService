from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import threading
import time
import base64
import json

app = Flask(__name__)
CORS(app)  # Cho phép Streamlit app truy cập


class CameraService:
    def __init__(self):
        self.camera = None
        self.is_streaming = False
        self.lock = threading.Lock()
        self.detector = cv2.QRCodeDetector()

    def initialize_camera(self, camera_index=0):
        """Khởi tạo camera USB"""
        try:
            self.camera = cv2.VideoCapture(camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 15)
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False

    def generate_frames(self):
        """Generator để stream video frames"""
        while self.is_streaming:
            with self.lock:
                if self.camera is None:
                    break

                success, frame = self.camera.read()
                if not success:
                    break

                    # Detect QR code
                data, points, _ = self.detector.detectAndDecode(frame)

                if points is not None and data:
                    # Vẽ khung QR như trong VideoProcessor
                    points = points.astype(int).reshape(-1, 2)
                    for j in range(len(points)):
                        pt1 = tuple(points[j])
                        pt2 = tuple(points[(j + 1) % len(points)])
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 3)

                        # Hiển thị text
                    cv2.rectangle(frame, (points[0][0], points[0][1] - 35),
                                  (points[0][0] + len(data) * 12, points[0][1] - 5),
                                  (0, 255, 0), -1)
                    cv2.putText(frame, data, (points[0][0], points[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    # Encode frame
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.033)  # ~30 FPS

    def get_frame_with_qr(self):
        """Lấy single frame với QR detection để Streamlit xử lý"""
        with self.lock:
            if self.camera is None:
                return None, None

            success, frame = self.camera.read()
            if not success:
                return None, None

                # Detect QR
            data, points, _ = self.detector.detectAndDecode(frame)

            # Encode frame to base64
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                return frame_b64, data

            return None, None

    def start_streaming(self):
        self.is_streaming = True

    def stop_streaming(self):
        self.is_streaming = False
        if self.camera:
            self.camera.release()

        # Global camera service instance


camera_service = CameraService()
@app.route('/camera/status')
def camera_status():
    return jsonify({
        "camera_initialized": camera_service.camera is not None,
        "is_streaming": camera_service.is_streaming,
        "camera_opened": camera_service.camera.isOpened() if camera_service.camera else False
    })

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.time()})


@app.route('/camera/start')
def start_camera():
    """Khởi động camera"""
    if camera_service.initialize_camera():
        camera_service.start_streaming()
        return jsonify({"status": "started", "message": "Camera initialized successfully"})
    else:
        return jsonify({"status": "error", "message": "Failed to initialize camera"}), 500


@app.route('/camera/stop')
def stop_camera():
    """Dừng camera"""
    camera_service.stop_streaming()
    return jsonify({"status": "stopped", "message": "Camera stopped"})


@app.route('/camera/stream')
def video_stream():
    """Stream video endpoint"""
    return Response(camera_service.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera/frame')
def get_frame():
    """Lấy single frame với QR detection"""
    frame_b64, qr_data = camera_service.get_frame_with_qr()

    if frame_b64:
        return jsonify({
            "frame": frame_b64,
            "qr_data": qr_data,
            "timestamp": time.time()
        })
    else:
        return jsonify({"error": "Failed to capture frame"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)