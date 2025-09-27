-> Face Attendance System
A Face Recognition Attendance Management System built using PyQt5 and OpenCV. The application provides a dashboard, attendance marking, and records management interface with real-time face recognition.

-> Features
Dashboard with statistics (total students, present/absent count, attendance rate)
Real-time face recognition using Haar Cascade and LBPH recognizer
Add new students with face capture and record storage
View attendance records in a tabular format
Settings page for configuration
Modern UI with sidebar, top bar, and stat cards
Tech Stack
Python 3.8+
PyQt5 (UI framework)
OpenCV (Face detection and recognition)
LBPH (Local Binary Patterns Histograms recognizer)

-> Project Structure
main.py - Main application entry
faces/ - Captured student faces
haarcascade_frontalface_default.xml - Haar Cascade classifier
classifier.xml - Trained LBPH face recognizer model


-> Install dependencies
pip install -r requirements.txt
requirements.txt should contain:
opencv-contrib-python
PyQt5


-> Place required model files
Download haarcascade_frontalface_default.xml from the OpenCV GitHub repository
Train and generate classifier.xml (LBPH face recognizer)

-> Usage
Run the application
python main.py


-> Key details
Press q to stop face capture while adding students
Faces with confidence above 78% are recognized as known users
Other faces are marked as Unknown

-> Future Improvements
Database integration (SQLite/MySQL) for storing records
Cloud syncing of attendance data

Role-based authentication (Admin/Teacher/Student)
