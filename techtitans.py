import sys
import os
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget,
                             QTableWidget, QGridLayout, QFrame, QScrollArea,
                             QHeaderView, QSpacerItem, QSizePolicy, QTableWidgetItem,
                             QDialog, QFormLayout, QLineEdit)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage, QPalette, QColor

# ---------------------- FACE RECOGNITION FUNCTIONS ----------------------
def DrawBoundary(img, classifier, scaleFactor, minNeighbors, clf):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
    coords = []

    for (x, y, w, h) in features:
        roi_gray = gray_image[y:y + h, x:x + w]
        id_, pred = clf.predict(roi_gray)
        confidence = int(100 * (1 - pred / 300))

        if confidence > 78:
            label = f"User {id_} ({confidence}%)"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        # Draw on the original color image
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        coords = [x, y, w, h]

    return img, coords  # return color image now

def recognize(img, clf, face_cascade):
    img_with_boxes, _ = DrawBoundary(img, face_cascade, 1.2, 10, clf)
    return img_with_boxes  # return color image

# ---------------------- WIDGET CLASSES ----------------------
class StatCard(QFrame):
    def __init__(self, title, value, color="#3498db"):
        super().__init__()
        self.setFixedHeight(120)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border-radius: 10px;
                border-left: 4px solid {color};
            }}
            QLabel {{
                background: transparent;
                border: none;
            }}
        """)
        layout = QVBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(20, 20, 20, 20)

        value_label = QLabel(str(value))
        value_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
        value_label.setStyleSheet("color: #2c3e50;")
        value_label.setAlignment(Qt.AlignCenter)

        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 10))
        title_label.setStyleSheet("color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px;")
        title_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(value_label)
        layout.addWidget(title_label)
        self.setLayout(layout)

class MenuButton(QPushButton):
    def __init__(self, text, icon=""):
        super().__init__()
        self.setText(f"  {icon}  {text}")
        self.setFont(QFont("Segoe UI", 11, QFont.Medium))
        self.setFixedHeight(50)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding-left: 25px;
                border: none;
                border-radius: 8px;
                margin: 3px 15px;
                color: white;
                background: transparent;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.1);
                transform: translateX(3px);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.2);
            }
        """)

class Sidebar(QFrame):
    menuClicked = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(280)
        self.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #2c3e50, stop:1 #34495e);
                border: none;
            }
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.create_header())
        layout.addWidget(self.create_menu())
        self.setLayout(layout)

    def create_header(self):
        header = QFrame()
        header.setFixedHeight(170)
        header.setStyleSheet("border-bottom: 1px solid rgba(255, 255, 255, 0.1);")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(25, 20, 25, 30)

        icon_label = QLabel("👤")
        icon_label.setFont(QFont("Segoe UI", 24))
        icon_label.setFixedSize(60, 60)
        icon_label.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                      stop:0 #3498db, stop:1 #2980b9);
            border-radius: 12px;
            color: white;
        """)
        icon_container = QWidget()
        icon_container_layout = QHBoxLayout()
        icon_container_layout.setContentsMargins(0, 0, 0, 0)
        icon_container_layout.addSpacing(75)
        icon_container_layout.addWidget(icon_label)
        icon_container_layout.addStretch()
        icon_container.setLayout(icon_container_layout)

        title = QLabel("Face Attendance")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: white;")
        subtitle = QLabel("SYSTEM V1.0")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.7); letter-spacing: 1px;")

        layout.addWidget(icon_container)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        header.setLayout(layout)
        return header

    def create_menu(self):
        menu_widget = QFrame()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 20, 0, 0)
        layout.setSpacing(5)

        menu_items = [("Dashboard", "📊"),
                      ("Mark Attendance", "✓"),
                      ("View Records", "📋"),
                      ("Settings", "⚙️")]
        self.menu_buttons = []
        for text, icon in menu_items:
            btn = MenuButton(text, icon)
            btn.clicked.connect(lambda checked, t=text: self.menuClicked.emit(t))
            self.menu_buttons.append(btn)
            layout.addWidget(btn)

        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        logout_btn = MenuButton("Log Out", "🚪")
        logout_btn.clicked.connect(lambda: self.menuClicked.emit("Log Out"))
        logout_btn.setStyleSheet(logout_btn.styleSheet() + """
            QPushButton:hover {
                background: rgba(231, 76, 60, 0.2);
            }
        """)
        layout.addWidget(logout_btn)
        layout.addSpacing(20)
        menu_widget.setLayout(layout)
        return menu_widget

    def set_active_button(self, text):
        for btn in self.menu_buttons:
            if text in btn.text():
                btn.setStyleSheet(btn.styleSheet() + """
                    QPushButton {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                                  stop:0 #3498db, stop:1 #2980b9);
                    }
                """)
            else:
                btn.setStyleSheet(MenuButton(btn.text(), "").styleSheet())

class TopBar(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(70)
        self.setStyleSheet("""
            QFrame {
                background: white;
                border-bottom: 1px solid #e1e8ed;
            }
        """)
        layout = QHBoxLayout()
        layout.setContentsMargins(30, 0, 30, 0)
        self.title_label = QLabel("Dashboard")
        self.title_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.title_label.setStyleSheet("color: #2c3e50; border: none;")
        layout.addWidget(self.title_label)
        layout.addStretch()

        user_label = QLabel("Welcome, Admin")
        user_label.setFont(QFont("Segoe UI", 12))
        user_label.setStyleSheet("color: #2c3e50; border: none;")
        avatar = QLabel("A")
        avatar.setFixedSize(40, 40)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setFont(QFont("Segoe UI", 14, QFont.Bold))
        avatar.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                      stop:0 #3498db, stop:1 #2980b9);
            border-radius: 20px;
            color: white;
        """)
        layout.addWidget(user_label)
        layout.addSpacing(15)
        layout.addWidget(avatar)
        self.setLayout(layout)

    def set_title(self, title):
        self.title_label.setText(title)

# ---------------------- CAMERA WIDGET ----------------------
class CameraWidget(QLabel):
    def __init__(self, clf=None, face_cascade=None, width=400, height=300):
        super().__init__()
        self.clf = clf
        self.face_cascade = face_cascade
        self.setFixedSize(width, height)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background: #ecf0f1;
                border: 2px dashed #bdc3c7;
                border-radius: 10px;
                color: #7f8c8d;
                font-size: 14px;
            }
        """)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            if self.clf and self.face_cascade:
                frame_to_display = recognize(frame, self.clf, self.face_cascade)
            else:
                frame_to_display = frame  # show original color if no classifier
    
            # Convert BGR to RGB for PyQt display
            rgb_frame = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.width(), self.height(), Qt.KeepAspectRatio
            ))

# add student  
class AddStudentPopup(QDialog):
    def __init__(self, face_cascade, clf):
        super().__init__()
        self.face_cascade = face_cascade
        self.clf = clf
        self.setWindowTitle("Add New Student")
        self.setFixedSize(400, 250)

        layout = QFormLayout()
        self.stName = QLineEdit()
        self.stRollNo = QLineEdit()
        self.stAddress = QLineEdit()
        layout.addRow("Student Name:", self.stName)
        layout.addRow("Roll Number:", self.stRollNo)
        layout.addRow("Address:", self.stAddress)

        self.status_label = QLabel("")
        layout.addRow(self.status_label)

        capture_btn = QPushButton("Capture Face")
        capture_btn.clicked.connect(self.capture_face)
        layout.addRow(capture_btn)

        self.setLayout(layout)

    def capture_face(self):
        st_name = self.stName.text()
        st_roll = self.stRollNo.text()
        st_address = self.stAddress.text()

        if not st_name or not st_roll:
            self.status_label.setText("Please fill all fields!")
            return

        cap = cv2.VideoCapture(0)
        face_folder = "faces"
        os.makedirs(face_folder, exist_ok=True)

        face_exists = False
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 10)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                try:
                    id_, pred = self.clf.predict(roi_gray)
                    confidence = int(100 * (1 - pred / 300))
                    if confidence > 78:
                        self.status_label.setText(f"Student's Face already exists with name {st_name}")
                        face_exists = True
                        break
                except:
                    pass

                if not face_exists:
                    file_name = f"{st_name}_{st_roll}.jpg"
                    cv2.imwrite(os.path.join(face_folder, file_name), roi_gray)
                    self.status_label.setText(f"Face captured for {st_name}")
                    face_exists = True
                    break

            cv2.imshow("Capture Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or face_exists:
                break

        cap.release()
        cv2.destroyAllWindows()

# ---------------------- DASHBOARD PAGE ----------------------
class DashboardPage(QScrollArea):
    def __init__(self, clf=None, face_cascade=None):
        super().__init__()
        self.clf = clf
        self.face_cascade = face_cascade
        self.setWidgetResizable(True)
        self.setStyleSheet("QScrollArea { border: none; background: #f8fafc; }")
        
        content = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(30)
        
        # Stats cards
        stats_layout = QGridLayout()
        stats_layout.setSpacing(20)
        
        cards = [
            ("Total Students", "156", "#3498db"),
            ("Present Today", "142", "#27ae60"),
            ("Absent Today", "14", "#e74c3c"),
            ("Attendance Rate", "91%", "#f39c12")
        ]
        
        for i, (title, value, color) in enumerate(cards):
            card = StatCard(title, value, color)
            stats_layout.addWidget(card, 0, i)
        
        stats_widget = QWidget()
        stats_widget.setLayout(stats_layout)
        layout.addWidget(stats_widget)
        
        # Camera section
        camera_section = QFrame()
        camera_section.setStyleSheet("QFrame { background: white; border-radius: 10px; }")
        cam_layout = QVBoxLayout()
        cam_layout.setContentsMargins(30, 30, 30, 30)
        cam_layout.setSpacing(20)

        cam_title = QLabel("Face Recognition Camera")
        cam_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        cam_title.setStyleSheet("color: #2c3e50; border: none;")
        cam_title.setAlignment(Qt.AlignCenter)
        cam_layout.addWidget(cam_title)

        # Add camera widget
        self.camera_widget = CameraWidget(clf, face_cascade, width=500, height=400)
        cam_layout.addWidget(self.camera_widget, alignment=Qt.AlignCenter)

        # -> Add Student button
        add_student_btn = QPushButton("Add Student")
        add_student_btn.setFixedHeight(40)
        add_student_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        add_student_btn.clicked.connect(self.open_add_student_popup)
        cam_layout.addWidget(add_student_btn, alignment=Qt.AlignCenter)

        camera_section.setLayout(cam_layout)
        layout.addWidget(camera_section)

        # Recent records table
        table_section = self.create_table_section()
        layout.addWidget(table_section)
        
        content.setLayout(layout)
        self.setWidget(content)

    def open_add_student_popup(self):
        popup = AddStudentPopup(self.face_cascade, self.clf)
        popup.exec_()

    def create_table_section(self):
        section = QFrame()
        section.setStyleSheet("""
            QFrame { background: white; border-radius: 10px; }
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel("Recent Attendance Records")
        header.setFont(QFont("Segoe UI", 14, QFont.Bold))
        header.setStyleSheet("""
            QLabel {
                background: #f8fafc;
                color: #2c3e50;
                padding: 20px 25px;
                border-bottom: 1px solid #e1e8ed;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
            }
        """)

        table = QTableWidget(3, 4)
        table.setHorizontalHeaderLabels(["Student Name", "Time", "Date", "Status"])
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.verticalHeader().setVisible(False)
        
        data = [
            ["John Doe", "09:15 AM", "2024-12-19", "Present"],
            ["Jane Smith", "09:12 AM", "2024-12-19", "Present"],
            ["Mike Johnson", "--", "2024-12-19", "Absent"]
        ]
        for row, row_data in enumerate(data):
            for col, value in enumerate(row_data):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row, col, item)

        layout.addWidget(header)
        layout.addWidget(table)
        section.setLayout(layout)
        return section

# ---------------------- OTHER PAGES ----------------------
class AttendancePage(QWidget):
    def __init__(self, clf, face_cascade):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Mark Attendance")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        self.camera_widget = CameraWidget(clf, face_cascade, width=500, height=400)
        layout.addWidget(self.camera_widget, alignment=Qt.AlignCenter)

class RecordsPage(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 30, 30, 30)
        title = QLabel("Attendance Records")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; margin-bottom: 20px;")
        layout.addWidget(title)

        table = QTableWidget(10, 5)
        table.setHorizontalHeaderLabels(["ID", "Student Name", "Time", "Date", "Status"])
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(table)
        content.setLayout(layout)
        self.setWidget(content)

class SettingsPage(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setAlignment(Qt.AlignCenter)
        title = QLabel("Settings")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title.setStyleSheet("color: #2c3e50;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        settings_label = QLabel("Application settings will be displayed here")
        settings_label.setAlignment(Qt.AlignCenter)
        settings_label.setStyleSheet("color: #7f8c8d; font-size: 16px; margin-top: 50px;")
        layout.addWidget(settings_label)
        content.setLayout(layout)
        self.setWidget(content)

# ---------------------- MAIN WINDOW ----------------------
class FaceAttendanceSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Face Attendance System")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet("QMainWindow { background: #f8fafc; }")

        # Load face cascade and classifier
        self.face_cascade = cv2.CascadeClassifier('C:/Users/AVI SHARMA/Documents/Assignment/haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            print("Error loading haarcascade_frontalface_default.xml")
        self.clf = cv2.face.LBPHFaceRecognizer_create()
        self.clf.read("C:/Users/AVI SHARMA/Documents/Assignment/classifier.xml")  # Ensure this file exists

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.menuClicked.connect(self.handle_menu_click)
        main_layout.addWidget(self.sidebar)

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        self.top_bar = TopBar()
        content_layout.addWidget(self.top_bar)

        self.stacked_widget = QStackedWidget()
        self.dashboard_page = DashboardPage(self.clf, self.face_cascade)
        self.attendance_page = AttendancePage(self.clf, self.face_cascade)
        self.records_page = RecordsPage()
        self.settings_page = SettingsPage()

        self.stacked_widget.addWidget(self.dashboard_page)
        self.stacked_widget.addWidget(self.attendance_page)
        self.stacked_widget.addWidget(self.records_page)
        self.stacked_widget.addWidget(self.settings_page)

        content_layout.addWidget(self.stacked_widget)
        content_widget = QWidget()
        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget)
        central_widget.setLayout(main_layout)
        self.show_page("Dashboard")

    def handle_menu_click(self, menu_text):
        if menu_text == "Log Out":
            self.close()
        else:
            self.show_page(menu_text)

    def show_page(self, page_name):
        self.top_bar.set_title(page_name)
        self.sidebar.set_active_button(page_name)
        page_mapping = {"Dashboard":0,"Mark Attendance":1,"View Records":2,"Settings":3}
        if page_name in page_mapping:
            self.stacked_widget.setCurrentIndex(page_mapping[page_name])

# ---------------------- RUN ----------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(248, 250, 252))
    app.setPalette(palette)
    window = FaceAttendanceSystem()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
