# techtitans_updated.py
import time
import sys
import os
import glob
import cv2
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget,
                             QTableWidget, QGridLayout, QFrame, QScrollArea,
                             QHeaderView, QSpacerItem, QSizePolicy, QTableWidgetItem,
                             QDialog, QFormLayout, QLineEdit, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage, QPalette, QColor
from PIL import Image
import numpy as np


# Centralized data handling functions
def save_student_to_csv(roll_no, name, address):
    """Appends a new student record to the students.csv file."""
    file_exists = os.path.isfile('students.csv')
    with open('students.csv', 'a', newline='') as csvfile:
        fieldnames = ['RollNo', 'Name', 'Address']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'RollNo': roll_no, 'Name': name, 'Address': address})

def load_students_from_csv():
    """Loads all student records from students.csv."""
    if not os.path.isfile('students.csv'):
        return []
    with open('students.csv', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

# ---------------------- FACE RECOGNITION FUNCTIONS ----------------------

def DrawBoundary(img, classifier, scaleFactor, minNeighbors, clf):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
    for (x, y, w, h) in features:
        roi_gray = gray_image[y:y + h, x:x + w]
        try:
            id_, pred = clf.predict(roi_gray)
            confidence = int(100 * (1 - pred / 300))
        except Exception:
            id_, pred, confidence = -1, 0, 0

        if confidence > 78:
            label = f"User {id_} ({confidence}%)"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img, []

def recognize(img, clf, face_cascade):
    if clf is None or face_cascade is None:
        return img
    img_with_boxes, _ = DrawBoundary(img, face_cascade, 1.2, 10, clf)
    return img_with_boxes

# ---------------------- WIDGET CLASSES (No Changes) ----------------------
class StatCard(QFrame):
    def __init__(self, title, value, color="#3498db"):
        super().__init__()
        self.setFixedHeight(120)
        self.setStyleSheet(f"""
            QFrame {{ background-color: white; border-radius: 10px; border-left: 4px solid {color}; }}
            QLabel {{ background: transparent; border: none; }}
        """)
        layout = QVBoxLayout(); layout.setSpacing(5); layout.setContentsMargins(20, 20, 20, 20)
        self.value_label = QLabel(str(value)); self.value_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self.value_label.setStyleSheet("color: #2c3e50;"); self.value_label.setAlignment(Qt.AlignCenter)
        title_label = QLabel(title); title_label.setFont(QFont("Segoe UI", 10))
        title_label.setStyleSheet("color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label); layout.addWidget(title_label); self.setLayout(layout)
    def set_value(self, value):
        self.value_label.setText(str(value))

class MenuButton(QPushButton):
    def __init__(self, text, icon=""):
        super().__init__()
        self.setText(f"  {icon}  {text}"); self.setFont(QFont("Segoe UI", 11, QFont.Medium)); self.setFixedHeight(50)
        self.setCursor(Qt.PointingHandCursor); self.setStyleSheet(""" ... """)

class Sidebar(QFrame):
    menuClicked = pyqtSignal(str)
    def __init__(self): super().__init__(); self.setFixedWidth(280); self.setStyleSheet(""" ... """); layout = QVBoxLayout(); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0); layout.addWidget(self.create_header()); layout.addWidget(self.create_menu()); self.setLayout(layout)
    def create_header(self): header = QFrame(); header.setFixedHeight(170); header.setStyleSheet("border-bottom: 1px solid rgba(255, 255, 255, 0.1);"); layout = QVBoxLayout(); layout.setAlignment(Qt.AlignCenter); layout.setContentsMargins(25, 20, 25, 30); icon_label = QLabel("👤"); icon_label.setFont(QFont("Segoe UI", 24)); icon_label.setFixedSize(60, 60); icon_label.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3498db, stop:1 #2980b9); border-radius: 12px; color: white;"); icon_container = QWidget(); icon_container_layout = QHBoxLayout(); icon_container_layout.setContentsMargins(0,0,0,0); icon_container_layout.addStretch(); icon_container_layout.addWidget(icon_label); icon_container_layout.addStretch(); icon_container.setLayout(icon_container_layout); title = QLabel("Face Attendance"); title.setAlignment(Qt.AlignCenter); title.setFont(QFont("Segoe UI", 16, QFont.Bold)); title.setStyleSheet("color: white;"); subtitle = QLabel("SYSTEM V1.0"); subtitle.setAlignment(Qt.AlignCenter); subtitle.setFont(QFont("Segoe UI", 10)); subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.7); letter-spacing: 1px;"); layout.addWidget(icon_container); layout.addWidget(title); layout.addWidget(subtitle); header.setLayout(layout); return header
    def create_menu(self): menu_widget = QFrame(); layout = QVBoxLayout(); layout.setContentsMargins(0, 20, 0, 0); layout.setSpacing(5); menu_items = [("Dashboard", "📊"), ("Mark Attendance", "✓"), ("View Records", "📋"), ("Settings", "⚙️")]; self.menu_buttons = []; [self.menu_buttons.append(btn) or layout.addWidget(btn) for text, icon in menu_items for btn in [MenuButton(text, icon)] if btn.clicked.connect(lambda checked, t=text: self.menuClicked.emit(t))]; layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)); logout_btn = MenuButton("Log Out", "🚪"); logout_btn.clicked.connect(lambda: self.menuClicked.emit("Log Out")); logout_btn.setStyleSheet(logout_btn.styleSheet() + "QPushButton:hover { background: rgba(231, 76, 60, 0.2); }"); layout.addWidget(logout_btn); layout.addSpacing(20); menu_widget.setLayout(layout); return menu_widget
    def set_active_button(self, text): [btn.setStyleSheet(btn.styleSheet() + "QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3498db, stop:1 #2980b9); }") if text in btn.text() else btn.setStyleSheet(MenuButton(btn.text(), "").styleSheet()) for btn in self.menu_buttons]

class TopBar(QFrame):
    def __init__(self): super().__init__(); self.setFixedHeight(70); self.setStyleSheet("QFrame { background: white; border-bottom: 1px solid #e1e8ed; }"); layout = QHBoxLayout(); layout.setContentsMargins(30, 0, 30, 0); self.title_label = QLabel("Dashboard"); self.title_label.setFont(QFont("Segoe UI", 20, QFont.Bold)); self.title_label.setStyleSheet("color: #2c3e50; border: none;"); layout.addWidget(self.title_label); layout.addStretch(); user_label = QLabel("Welcome, Admin"); user_label.setFont(QFont("Segoe UI", 12)); user_label.setStyleSheet("color: #2c3e50; border: none;"); avatar = QLabel("A"); avatar.setFixedSize(40, 40); avatar.setAlignment(Qt.AlignCenter); avatar.setFont(QFont("Segoe UI", 14, QFont.Bold)); avatar.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3498db, stop:1 #2980b9); border-radius: 20px; color: white;"); layout.addWidget(user_label); layout.addSpacing(15); layout.addWidget(avatar); self.setLayout(layout)
    def set_title(self, title): self.title_label.setText(title)

class CameraWidget(QLabel):
    def __init__(self, clf=None, face_cascade=None, width=400, height=300):
        super().__init__(); self.clf = clf; self.face_cascade = face_cascade; self.setFixedSize(width, height); self.setAlignment(Qt.AlignCenter); self.setStyleSheet("QLabel { background: #ecf0f1; border: 2px dashed #bdc3c7; border-radius: 10px; color: #7f8c8d; font-size: 14px; }"); self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW); [print("Error: Cannot open camera") if not self.cap.isOpened() else None]; self.timer = QTimer(); self.timer.timeout.connect(self.update_frame); self.timer.start(30)
    def update_frame(self): ret, frame = self.cap.read(); [self.setPixmap(QPixmap.fromImage(QImage(cv2.cvtColor(recognize(frame, self.clf, self.face_cascade) if self.clf and self.face_cascade else frame, cv2.COLOR_BGR2RGB).data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)).scaled(self.width(), self.height(), Qt.KeepAspectRatio)) if ret and frame is not None else None]
    def close(self): self.timer.stop(); [self.cap.release() if self.cap and self.cap.isOpened() else None]; super().close()

class AddStudentPopup(QDialog):
    def __init__(self, face_cascade, cap):
        super().__init__()
        self.face_cascade = face_cascade
        self.cap = cap
        
        self.img_id = 0
        self.frames_needed = 20
        self.st_name = ""
        self.st_roll = ""
        self.st_address = ""

        self.setWindowTitle("Add New Student")
        self.setFixedSize(500, 600)

        layout = QVBoxLayout()

        self.preview_label = QLabel("Camera will appear here...")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(480, 360)
        self.preview_label.setStyleSheet("""
            QLabel { background-color: black; border: 1px solid #555; border-radius: 8px; }
        """)
        layout.addWidget(self.preview_label)

        form_layout = QFormLayout()
        self.stNameEdit = QLineEdit()
        self.stRollNoEdit = QLineEdit()
        self.stAddressEdit = QLineEdit()
        form_layout.addRow("Student Name:", self.stNameEdit)
        form_layout.addRow("Roll Number:", self.stRollNoEdit)
        form_layout.addRow("Address:", self.stAddressEdit)
        layout.addLayout(form_layout)

        self.status_label = QLabel("Fill details and press 'Start Capture'.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-weight: bold; color: #3498db;")
        layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.capture_btn = QPushButton("Start Capture")
        self.capture_btn.setFixedHeight(36)
        self.capture_btn.setStyleSheet("QPushButton { background-color: #27ae60; color: white; border-radius: 6px; } QPushButton:hover { background-color: #2ecc71; }")
        self.capture_btn.clicked.connect(self.start_capture_process)
        btn_row.addWidget(self.capture_btn)

        cancel_btn = QPushButton("Close")
        cancel_btn.setFixedHeight(36)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self.process_frame)
    
    def start_capture_process(self):
        self.st_name = self.stNameEdit.text().strip()
        self.st_roll = self.stRollNoEdit.text().strip()
        self.st_address = self.stAddressEdit.text().strip()

        if not self.st_name or not self.st_roll:
            QMessageBox.warning(self, "Input Error", "Please fill in both Student Name and Roll Number.")
            return

        self.capture_btn.setEnabled(False)
        self.stNameEdit.setEnabled(False)
        self.stRollNoEdit.setEnabled(False)
        self.stAddressEdit.setEnabled(False)

        self.img_id = 0
        os.makedirs("facedata", exist_ok=True)
        
        self.capture_timer.start(50)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.status_label.setText("Error: Cannot read from camera.")
            return

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if len(faces) > 0:
            self.status_label.setText(f"Capturing... {self.img_id + 1}/{self.frames_needed}")
            (x, y, w, h) = faces[0]
            cropped_face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(cropped_face, (200, 200))
            file_name = f"user.{self.st_roll}.{self.img_id + 1}.jpg"
            cv2.imwrite(os.path.join("facedata", file_name), face_resized)
            self.img_id += 1
        else:
            self.status_label.setText("No face detected. Please face the camera.")

        self.display_frame(frame)

        if self.img_id >= self.frames_needed:
            self.finish_capture()

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.preview_label.setPixmap(pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio))

    def finish_capture(self):
        self.capture_timer.stop()
        self.preview_label.setText("Capture Complete!")
        save_student_to_csv(self.st_roll, self.st_name, self.st_address)
        
        QMessageBox.information(self, "Success", f"Captured images for {self.st_name}.\nNow training the model.")
        self.status_label.setText("Training model... Please wait.")
        QApplication.processEvents()

        if self.train_classifier("facedata"):
            QMessageBox.information(self, "Training Complete", "New student added and model updated.")
            self.accept()
        else:
            QMessageBox.warning(self, "Training Error", "Could not train the model.")
            self.reject()

    def reject(self):
        self.capture_timer.stop()
        super().reject()

    def train_classifier(self, facedata_dir):
        path = [os.path.join(facedata_dir, f) for f in os.listdir(facedata_dir)]
        faces, ids = [], []
        for image in path:
            try:
                img = Image.open(image).convert('L')
                ImageNp = np.array(img, 'uint8')
                id_ = int(os.path.split(image)[1].split(".")[1])
                faces.append(ImageNp)
                ids.append(id_)
            except Exception as e: print(f"Skipping file {image}: {e}")
        
        if not faces or not ids: return False

        ids = np.array(ids)
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)
        clf.write("classifier.xml")
        print("Classifier trained and saved successfully!")
        return True


# ---------------------- DELETE STUDENT POPUP  ----------------------
class DeleteStudentPopup(QDialog):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Delete Student"); self.setFixedSize(360, 150)
        layout = QFormLayout(); self.roll_input = QLineEdit(); layout.addRow("Roll Number to delete:", self.roll_input)
        self.status = QLabel(""); layout.addRow(self.status)
        btn_row = QHBoxLayout(); del_btn = QPushButton("Delete"); del_btn.setFixedHeight(34)
        del_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; border-radius: 6px; } QPushButton:hover { background-color: #c0392b; }")
        del_btn.clicked.connect(self.delete_by_roll); btn_row.addWidget(del_btn)
        cancel_btn = QPushButton("Cancel"); cancel_btn.setFixedHeight(34); cancel_btn.clicked.connect(self.reject); btn_row.addWidget(cancel_btn)
        layout.addRow(btn_row); self.setLayout(layout)
    def delete_by_roll(self):
        roll = self.roll_input.text().strip()
        if not roll: self.status.setText("Enter a roll number."); return
        matches = glob.glob(os.path.join("facedata", f"user.{roll}.*.jpg"))
        if not matches: self.status.setText("No student face files found."); return
        for p in matches:
            try: os.remove(p)
            except Exception as e: print(f"Error deleting file: {p}, {e}")
        QMessageBox.information(self, "Deleted", f"Deleted {len(matches)} face image file(s).")
        self.accept()

# ---------------------- DASHBOARD PAGE  ----------------------
class DashboardPage(QScrollArea):
    def __init__(self, face_cascade, main_window):
        super().__init__(); self.face_cascade = face_cascade; self.main_window = main_window
        self.setWidgetResizable(True); self.setStyleSheet("QScrollArea { border: none; background: #f8fafc; }")
        content = QWidget(); layout = QVBoxLayout(content); layout.setContentsMargins(30, 30, 30, 30); layout.setSpacing(30)
        stats_layout = QGridLayout(); stats_layout.setSpacing(20)
        self.total_students_card = StatCard("Total Students", "0", "#3498db"); self.present_card = StatCard("Present Today", "0", "#27ae60"); self.absent_card = StatCard("Absent Today", "0", "#e74c3c")
        stats_layout.addWidget(self.total_students_card, 0, 0); stats_layout.addWidget(self.present_card, 0, 1); stats_layout.addWidget(self.absent_card, 0, 2)
        stats_widget = QWidget(); stats_widget.setLayout(stats_layout); layout.addWidget(stats_widget)
        actions_frame = QFrame(); actions_frame.setStyleSheet("QFrame { background: white; border-radius: 10px; }"); a_layout = QVBoxLayout(actions_frame); a_layout.setContentsMargins(30, 30, 30, 30); a_layout.setSpacing(12); title_label = QLabel("Student Management"); title_label.setFont(QFont("Segoe UI", 16, QFont.Bold)); title_label.setStyleSheet("color: #2c3e50;"); a_layout.addWidget(title_label, alignment=Qt.AlignCenter)
        btn_row = QHBoxLayout(); add_btn = QPushButton("Add Student"); add_btn.setFixedHeight(40); add_btn.setStyleSheet("QPushButton { background-color: #2d88c5; color: white; border-radius: 8px; font-size: 14px; } QPushButton:hover { background-color: #415162; }"); add_btn.clicked.connect(self.open_add_student_popup); btn_row.addWidget(add_btn); delete_btn = QPushButton("Delete Student"); delete_btn.setFixedHeight(40); delete_btn.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; border-radius: 8px; font-size: 14px; } QPushButton:hover { background-color: #c0392b; }"); delete_btn.clicked.connect(self.open_delete_student_popup); btn_row.addWidget(delete_btn)
        a_layout.addLayout(btn_row); layout.addWidget(actions_frame); self.setWidget(content); self.update_stats()
    def update_stats(self):
        students = load_students_from_csv(); self.total_students_card.set_value(len(students)); self.absent_card.set_value(len(students))
    def open_add_student_popup(self):
        popup = AddStudentPopup(self.face_cascade, self.main_window.attendance_page.camera_widget.cap)
        if popup.exec_() == QDialog.Accepted:
            self.main_window.reload_classifier(); self.main_window.records_page.load_student_data(); self.update_stats()
    def open_delete_student_popup(self):
        popup = DeleteStudentPopup()
        if popup.exec_() == QDialog.Accepted:
            self.main_window.reload_classifier(); self.main_window.records_page.load_student_data(); self.update_stats()

# ---------------------- OTHER PAGES  ----------------------
class AttendancePage(QWidget):
    def __init__(self, clf, face_cascade):
        super().__init__(); layout = QVBoxLayout(self); layout.setAlignment(Qt.AlignCenter); title = QLabel("Mark Attendance"); title.setAlignment(Qt.AlignCenter); title.setStyleSheet("font-size: 24px; font-weight: bold;"); layout.addWidget(title); self.camera_widget = CameraWidget(clf, face_cascade, width=700, height=520); layout.addWidget(self.camera_widget, alignment=Qt.AlignCenter)

class RecordsPage(QScrollArea):
    def __init__(self):
        super().__init__(); self.setWidgetResizable(True); content = QWidget(); layout = QVBoxLayout(content); layout.setContentsMargins(30, 30, 30, 30); title = QLabel("Student Records"); title.setFont(QFont("Segoe UI", 24, QFont.Bold)); title.setStyleSheet("color: #2c3e50; margin-bottom: 20px;"); layout.addWidget(title); self.table = QTableWidget(); self.table.setColumnCount(3); self.table.setHorizontalHeaderLabels(["Roll Number", "Student Name", "Address"]); self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); self.table.setAlternatingRowColors(True); self.table.setSelectionBehavior(QTableWidget.SelectRows); self.table.setEditTriggers(QTableWidget.NoEditTriggers); layout.addWidget(self.table); content.setLayout(layout); self.setWidget(content); self.load_student_data()
    def load_student_data(self):
        students = load_students_from_csv(); self.table.setRowCount(len(students))
        for row, data in enumerate(students):
            self.table.setItem(row, 0, QTableWidgetItem(data.get('RollNo', ''))); self.table.setItem(row, 1, QTableWidgetItem(data.get('Name', ''))); self.table.setItem(row, 2, QTableWidgetItem(data.get('Address', '')))

class SettingsPage(QScrollArea):
    def __init__(self): super().__init__(); self.setWidgetResizable(True); content = QWidget(); layout = QVBoxLayout(content); layout.setContentsMargins(30, 30, 30, 30); layout.setAlignment(Qt.AlignCenter); title = QLabel("Settings"); title.setFont(QFont("Segoe UI", 24, QFont.Bold)); title.setStyleSheet("color: #2c3e50;"); title.setAlignment(Qt.AlignCenter); layout.addWidget(title); settings_label = QLabel("Application settings will be displayed here"); settings_label.setAlignment(Qt.AlignCenter); settings_label.setStyleSheet("color: #7f8c8d; font-size: 16px; margin-top: 50px;"); layout.addWidget(settings_label); content.setLayout(layout); self.setWidget(content)

# ---------------------- MAIN WINDOW  ----------------------
class FaceAttendanceSystem(QMainWindow):
    def __init__(self): super().__init__(); self.initUI()
    def initUI(self):
        self.setWindowTitle("Face Attendance System"); self.setGeometry(100, 100, 1400, 900); self.setStyleSheet("QMainWindow { background: #f8fafc; }")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty(): QMessageBox.critical(self, "Error", f"Failed to load Haar Cascade from: {cascade_path}"); sys.exit()
        self.clf = None; self.reload_classifier()
        central_widget = QWidget(); self.setCentralWidget(central_widget); main_layout = QHBoxLayout(central_widget); main_layout.setContentsMargins(0, 0, 0, 0); main_layout.setSpacing(0); self.sidebar = Sidebar(); self.sidebar.menuClicked.connect(self.handle_menu_click); main_layout.addWidget(self.sidebar)
        content_layout = QVBoxLayout(); content_layout.setContentsMargins(0, 0, 0, 0); content_layout.setSpacing(0); self.top_bar = TopBar(); content_layout.addWidget(self.top_bar); self.stacked_widget = QStackedWidget()
        self.dashboard_page = DashboardPage(self.face_cascade, self); self.attendance_page = AttendancePage(self.clf, self.face_cascade); self.records_page = RecordsPage(); self.settings_page = SettingsPage()
        self.stacked_widget.addWidget(self.dashboard_page); self.stacked_widget.addWidget(self.attendance_page); self.stacked_widget.addWidget(self.records_page); self.stacked_widget.addWidget(self.settings_page)
        content_layout.addWidget(self.stacked_widget); main_layout.addLayout(content_layout); self.show_page("Dashboard")
    def reload_classifier(self):
        try:
            self.clf = cv2.face.LBPHFaceRecognizer_create(); classifier_file = "classifier.xml"
            if os.path.exists(classifier_file): self.clf.read(classifier_file); print("Classifier reloaded."); [setattr(self.attendance_page.camera_widget, 'clf', self.clf) if hasattr(self, 'attendance_page') else None]
            else: self.clf = None; print("classifier.xml not found.")
        except Exception as e: print(f"Error loading recognizer: {e}"); self.clf = None
    def handle_menu_click(self, menu_text): [self.close() if menu_text == "Log Out" else self.show_page(menu_text)]
    def show_page(self, page_name):
        self.top_bar.set_title(page_name); self.sidebar.set_active_button(page_name)
        page_mapping = {"Dashboard":0, "Mark Attendance":1, "View Records":2, "Settings":3}
        if page_name in page_mapping: self.stacked_widget.setCurrentIndex(page_mapping[page_name])

# ---------------------- RUN ----------------------
def main():
    app = QApplication(sys.argv); app.setStyle('Fusion'); palette = QPalette(); palette.setColor(QPalette.Window, QColor(248, 250, 252)); app.setPalette(palette)
    window = FaceAttendanceSystem(); window.show(); sys.exit(app.exec_())

if __name__ == "__main__":
    main()