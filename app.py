import cv2
import mediapipe as mp
from ultralytics import YOLO
import datetime
import time
import smtplib
import os
import math

from flask import Flask, Response, jsonify, request, render_template
from flask_cors import CORS

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SCREENSHOT_DIR = "logs/screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

# KONFIGURASI EMAIL - GANTI DENGAN EMAIL ANDA
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "hafizsuryaw@gmail.com",
    "sender_password": "gkehhwwlzwvjlzdk",
    "supervisor_email": "hafizsuryawijaya10@gmail.com"
}

# VARIABEL GLOBAL - TIDAK ADA DATA DEFAULT
violations_list = []
tab_violations = []
current_student = {"nim": "", "name": ""}  # KOSONG, tidak ada data demo
is_exam_active = False
exam_start_time = None
camera = None

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(0.5)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("⏳ Loading YOLO...")
yolo_model = YOLO("yolov8n.pt")
print("✅ YOLO ready")

PHONE_CLASSES = ["cell phone"]
BOOK_CLASSES = ["book"]

class SmartViolationDetector:
    def __init__(self):
        self.looking_away_start = None
        self.no_face_start = None
        self.multiple_faces_start = None
        self.phone_detected_start = None
        self.book_detected_start = None
        
        self.NO_FACE_TOLERANCE = 3
        self.LOOKING_AWAY_DELAY = 2
        self.MULTIPLE_FACES_DELAY = 2
        self.OBJECT_DETECTION_DELAY = 2

    def add_violation(self, vtype, desc, frame=None):
        screenshot = None

        if frame is not None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{current_student['nim']}_{vtype.replace(' ', '_')}_{ts}.jpg"
            path = os.path.join(SCREENSHOT_DIR, filename)
            cv2.imwrite(path, frame)
            screenshot = filename

        violation = {
            "timestamp": datetime.datetime.now().isoformat(),
            "violation_type": vtype,
            "description": desc,
            "screenshot": screenshot
        }
        violations_list.append(violation)
        print(f"⚠️ {vtype}: {desc}")

    def calculate_head_rotation(self, landmarks, w, h):
        nose_tip = landmarks[1]
        chin = landmarks[152]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        
        nose_2d = (int(nose_tip.x * w), int(nose_tip.y * h))
        chin_2d = (int(chin.x * w), int(chin.y * h))
        left_eye_2d = (int(left_eye.x * w), int(left_eye.y * h))
        right_eye_2d = (int(right_eye.x * w), int(right_eye.y * h))
        
        eye_center_x = (left_eye_2d[0] + right_eye_2d[0]) / 2
        face_center_x = w / 2
        
        yaw_normalized = (eye_center_x - face_center_x) / (w / 2)
        yaw_angle = yaw_normalized * 45
        
        if yaw_angle < -15:
            direction = "LEFT"
        elif yaw_angle > 15:
            direction = "RIGHT"
        else:
            direction = "CENTER"
        
        return direction, abs(yaw_angle)

    def detect_face_and_pose(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_count = 0
        direction = "CENTER"
        angle = 0

        results = face_detection.process(rgb)
        if results.detections:
            face_count = len(results.detections)
            for d in results.detections:
                mp_drawing.draw_detection(frame, d)

        mesh = face_mesh.process(rgb)
        if mesh.multi_face_landmarks and face_count == 1:
            landmarks = mesh.multi_face_landmarks[0].landmark
            direction, angle = self.calculate_head_rotation(landmarks, w, h)

        return frame, face_count, direction, angle

    def detect_objects(self, frame):
        objs = []
        results = yolo_model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = r.names[cls]
                if conf > 0.6:
                    if name in PHONE_CLASSES:
                        objs.append("HP")
                    if name in BOOK_CLASSES:
                        objs.append("Buku")
        return objs

    def check_violations(self, face_count, direction, objects, frame):
        now = time.time()

        if face_count == 0:
            if self.no_face_start is None:
                self.no_face_start = now
            elif now - self.no_face_start >= self.NO_FACE_TOLERANCE:
                self.add_violation("Tidak Ada Wajah", f"Wajah hilang lebih dari {self.NO_FACE_TOLERANCE} detik", frame)
                self.no_face_start = None
        else:
            self.no_face_start = None

        if face_count > 1:
            if self.multiple_faces_start is None:
                self.multiple_faces_start = now
            elif now - self.multiple_faces_start >= self.MULTIPLE_FACES_DELAY:
                self.add_violation(
                    "Terdeteksi Lebih dari 1 Wajah",
                    f"{face_count} wajah terdeteksi - kemungkinan ada orang lain",
                    frame
                )
                self.multiple_faces_start = None
        else:
            self.multiple_faces_start = None

        if face_count == 1 and direction != "CENTER":
            if self.looking_away_start is None:
                self.looking_away_start = now
            elif now - self.looking_away_start >= self.LOOKING_AWAY_DELAY:
                self.add_violation(
                    f"Menoleh {direction}",
                    f"Kepala menoleh {direction} lebih dari {self.LOOKING_AWAY_DELAY} detik",
                    frame
                )
                self.looking_away_start = None
        else:
            self.looking_away_start = None

        if "HP" in objects:
            if self.phone_detected_start is None:
                self.phone_detected_start = now
            elif now - self.phone_detected_start >= self.OBJECT_DETECTION_DELAY:
                self.add_violation("Deteksi HP", f"HP terdeteksi selama lebih dari {self.OBJECT_DETECTION_DELAY} detik", frame)
                self.phone_detected_start = None
        else:
            self.phone_detected_start = None

        if "Buku" in objects:
            if self.book_detected_start is None:
                self.book_detected_start = now
            elif now - self.book_detected_start >= self.OBJECT_DETECTION_DELAY:
                self.add_violation("Deteksi Buku", f"Buku terdeteksi selama lebih dari {self.OBJECT_DETECTION_DELAY} detik", frame)
                self.book_detected_start = None
        else:
            self.book_detected_start = None

detector = SmartViolationDetector()

def send_email():
    """Kirim email laporan setelah ujian selesai"""
    
    # VALIDASI: Harus ada data mahasiswa
    if not current_student.get('nim') or not current_student.get('name'):
        print("❌ Data mahasiswa tidak lengkap, email tidak dikirim")
        return False
    
    msg = MIMEMultipart()
    msg["From"] = EMAIL_CONFIG["sender_email"]
    msg["To"] = EMAIL_CONFIG["supervisor_email"]
    msg["Subject"] = f"Laporan Ujian - {current_student['name']} ({current_student['nim']})"

    total_camera = len(violations_list)
    total_tab = len(tab_violations)
    total_all = total_camera + total_tab
    
    exam_duration = "Tidak tercatat"
    if exam_start_time is not None:
        try:
            duration = datetime.datetime.now() - exam_start_time
            minutes = int(duration.total_seconds() / 60)
            seconds = int(duration.total_seconds() % 60)
            exam_duration = f"{minutes} menit {seconds} detik"
        except Exception as e:
            print(f"⚠️ Error menghitung durasi: {e}")
            exam_duration = "Tidak dapat dihitung"

    body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            .header {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 30px; 
                text-align: center; 
                border-radius: 10px;
            }}
            .content {{ padding: 20px; }}
            .safe {{ color: #10b981; font-weight: bold; font-size: 18px; }}
            .warning {{ color: #f59e0b; font-weight: bold; font-size: 18px; }}
            .danger {{ color: #ef4444; font-weight: bold; font-size: 18px; }}
            .info-box {{
                background: #f9fafb;
                border-left: 4px solid #667eea;
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            .violation-item {{
                background: #fef2f2;
                border-left: 4px solid #ef4444;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
            }}
            .no-violation-box {{
                background: #ecfdf5;
                border-left: 4px solid #10b981;
                padding: 20px;
                margin: 20px 0;
                border-radius: 5px;
                text-align: center;
            }}
            .violation-number {{
                background: #ef4444;
                color: white;
                padding: 5px 10px;
                border-radius: 50%;
                font-weight: bold;
                margin-right: 10px;
            }}
            .violation-type {{
                color: #dc2626;
                font-weight: bold;
                font-size: 16px;
            }}
            .violation-time {{
                color: #6b7280;
                font-size: 14px;
                margin-top: 5px;
            }}
            .violation-desc {{
                color: #374151;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Laporan Hasil Ujian Online</h1>
            <p>Sistem AI Proctoring - Laporan Otomatis</p>
        </div>
        <div class="content">
            <div class="info-box">
                <h2>👤 Informasi Peserta</h2>
                <p><strong>NIM:</strong> {current_student.get('nim', 'Tidak tersedia')}</p>
                <p><strong>Nama:</strong> {current_student.get('name', 'Tidak tersedia')}</p>
                <p><strong>Waktu Mulai:</strong> {exam_start_time.strftime('%d %B %Y, %H:%M:%S') if exam_start_time else 'Tidak tercatat'}</p>
                <p><strong>Waktu Selesai:</strong> {datetime.datetime.now().strftime('%d %B %Y, %H:%M:%S')}</p>
                <p><strong>Durasi Ujian:</strong> {exam_duration}</p>
            </div>
            
            <h2>📈 Ringkasan Pelanggaran</h2>
    """

    if total_all == 0:
        body += '''
            <div class="no-violation-box">
                <h2 style="color: #10b981; margin: 0;">✅ SEMPURNA</h2>
                <p style="font-size: 18px; margin: 10px 0;">Tidak ada pelanggaran terdeteksi selama ujian</p>
                <p style="color: #6b7280;">Mahasiswa menjalankan ujian dengan baik dan mengikuti semua protokol.</p>
            </div>
        '''
    elif total_all < 3:
        body += f'<p class="safe">✅ BAIK - {total_all} pelanggaran ringan terdeteksi</p>'
    elif total_all < 5:
        body += f'<p class="warning">⚠️ PERINGATAN - {total_all} pelanggaran terdeteksi</p>'
    else:
        body += f'<p class="danger">🚨 PERHATIAN SERIUS - {total_all} pelanggaran terdeteksi</p>'

    body += f"""
        <div class="info-box">
            <p><strong>Total Pelanggaran Kamera:</strong> {total_camera}</p>
            <p><strong>Total Tab Switch:</strong> {total_tab}</p>
            <p><strong>Total Keseluruhan:</strong> {total_all}</p>
        </div>
    """

    if total_camera > 0:
        body += "<h2>🎥 Detail Pelanggaran dari Kamera</h2>"
        for i, v in enumerate(violations_list, 1):
            try:
                timestamp = datetime.datetime.fromisoformat(v['timestamp']).strftime('%d %B %Y, %H:%M:%S')
            except:
                timestamp = v.get('timestamp', 'Waktu tidak tercatat')
            
            vtype = v.get('violation_type', 'Tidak diketahui')
            desc = v.get('description', 'Tidak ada deskripsi')
            
            body += f"""
                <div class="violation-item">
                    <span class="violation-number">{i}</span>
                    <span class="violation-type">{vtype}</span>
                    <div class="violation-desc">{desc}</div>
                    <div class="violation-time">⏰ Waktu: {timestamp}</div>
                </div>
            """
    else:
        body += """
            <div class="info-box">
                <p style="color: #10b981; font-weight: bold;">✅ Tidak ada pelanggaran dari kamera terdeteksi</p>
            </div>
        """

    if total_tab > 0:
        body += "<h2>🖥️ Detail Aktivitas Tab Switching</h2>"
        for i, v in enumerate(tab_violations, 1):
            try:
                timestamp = datetime.datetime.fromisoformat(v['timestamp']).strftime('%d %B %Y, %H:%M:%S')
            except:
                timestamp = v.get('timestamp', 'Waktu tidak tercatat')
            
            duration_ms = v.get('duration', 0)
            duration = f" (selama {duration_ms / 1000:.1f} detik)" if duration_ms > 0 else ""
            
            body += f"""
                <div class="violation-item">
                    <span class="violation-number">{i}</span>
                    <span class="violation-type">Tab Switch</span>
                    <div class="violation-desc">Mahasiswa meninggalkan halaman ujian{duration}</div>
                    <div class="violation-time">⏰ Waktu: {timestamp}</div>
                </div>
            """
    else:
        body += """
            <div class="info-box">
                <p style="color: #10b981; font-weight: bold;">✅ Tidak ada aktivitas tab switching terdeteksi</p>
            </div>
        """

    body += """
        <hr style="margin: 30px 0; border: none; border-top: 2px solid #e5e7eb;">
        <p style="color: #6b7280; font-size: 13px; text-align: center;">
            📧 Email ini dikirim otomatis oleh Sistem AI Proctoring<br>
            Buffer: Menoleh 2 detik | Wajah Hilang 3 detik | 2 Wajah 2 detik | HP/Buku 2 detik<br>
            <strong>Catatan:</strong> Laporan ini dikirim untuk setiap ujian yang selesai, baik ada pelanggaran maupun tidak.
        </p>
        </div>
    </body>
    </html>
    """

    msg.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
            server.starttls()
            server.login(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["sender_password"])
            server.send_message(msg)
        print(f"📧 Email berhasil terkirim ke {EMAIL_CONFIG['supervisor_email']}")
        return True
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return False

def generate_frames():
    global camera
    
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        for i in range(1, 5):
            camera = cv2.VideoCapture(i)
            if camera.isOpened():
                print(f"✅ Camera found at index {i}")
                break
    
    if not camera.isOpened():
        print("❌ No camera available")
        return

    while is_exam_active:
        ret, frame = camera.read()
        if not ret:
            break

        try:
            frame, face_count, direction, angle = detector.detect_face_and_pose(frame)
            objects = detector.detect_objects(frame)
            detector.check_violations(face_count, direction, objects, frame)
            
            cv2.putText(frame, f"Wajah: {face_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Arah: {direction} ({angle:.1f}deg)", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            cv2.putText(frame, f"Pelanggaran: {len(violations_list)}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.putText(frame, f"Tab Switch: {len(tab_violations)}", (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
            
            if len(objects) > 0:
                cv2.putText(frame, f"Objek: {', '.join(objects)}", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

            ret, buffer = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")
        except Exception as e:
            print(f"❌ Error: {e}")
            break

    if camera:
        camera.release()

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/video")
def video():
    return Response(generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start", methods=["POST"])
def start_exam():
    global is_exam_active, violations_list, tab_violations, current_student, exam_start_time
    
    data = request.json
    
    # VALIDASI: NIM dan Nama HARUS diisi
    nim = data.get("nim", "").strip()
    name = data.get("name", "").strip()
    
    if not nim or not name:
        return jsonify({
            "status": "error",
            "message": "NIM dan Nama harus diisi!"
        }), 400
    
    current_student = {
        "nim": nim,
        "name": name
    }
    
    violations_list = []
    tab_violations = []
    is_exam_active = True
    exam_start_time = datetime.datetime.now()
    
    print(f"✅ Exam started for {current_student['name']} ({current_student['nim']}) at {exam_start_time.strftime('%H:%M:%S')}")
    return jsonify({
        "status": "started",
        "student": current_student
    })

@app.route("/stop", methods=["POST"])
def stop_exam():
    global is_exam_active

    is_exam_active = False
    
    total_violations = len(violations_list) + len(tab_violations)
    
    if total_violations == 0:
        print("⏹️ Exam stopped - NO VIOLATIONS DETECTED")
        print("📧 Sending email report (clean exam)...")
    else:
        print(f"⏹️ Exam stopped - {total_violations} violation(s) detected")
        print("📧 Sending email report...")

    try:
        email_sent = send_email()
        if email_sent:
            print("✅ Email successfully sent to supervisor")
        else:
            print("❌ Failed to send email")
    except Exception as e:
        print(f"❌ Email error: {e}")
        email_sent = False

    return jsonify({
        "status": "stopped",
        "email_sent": email_sent,
        "total_violations": total_violations,
        "message": "Email laporan telah dikirim ke pengawas" if email_sent else "Gagal mengirim email"
    })

@app.route("/get_violations")
def get_violations():
    return jsonify({
        "violations": violations_list,
        "tab_violations": tab_violations,
        "total": len(violations_list) + len(tab_violations)
    })

@app.route("/log_tab_switch", methods=["POST"])
def log_tab_switch():
    global tab_violations
    
    if not is_exam_active:
        return jsonify({
            "status": "error",
            "message": "Exam is not active"
        })
    
    data = request.json or {}
    
    violation = {
        "timestamp": datetime.datetime.now().isoformat(),
        "type": data.get("type", "TAB_SWITCH"),
        "duration": data.get("duration", 0)
    }
    
    tab_violations.append(violation)
    print(f"📱 Tab switch logged at {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    return jsonify({
        "status": "success",
        "total_switches": len(tab_violations)
    })

if __name__ == "__main__":
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("="*60)
    print("🚀 AI PROCTORING SYSTEM - CLEAN VERSION")
    print("="*60)
    print("📍 http://localhost:5000")
    print("⚙️  DELAYS (SUPER CEPAT!):")
    print("   • Wajah Hilang: 3 detik")
    print("   • Menoleh Kiri/Kanan: 2 detik")
    print("   • 2 Wajah: 2 detik")
    print("   • HP/Buku: 2 detik")
    print("="*60)
    print("⚠️  PENTING: Ganti EMAIL_CONFIG dengan email Anda!")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)