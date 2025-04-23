# Attendance_Via_Face_Recognition
---
## Group Members

Jiya Patel - KU2407U412
<br/>Jiya Chaudhari - KU2407U411
<br/>Gopesh Jha - KU2407U402

---

## Objective Of The Project

**Attendance via Face Detection**

---

## Project Overview

This system captures student's face through webcam,recognizes them and compares with preloaded images, records the attendance and adds student's name in the attendance sheet.

---

## Tools and Libraries used

<br/>-> Visual Studio Code
<br/>-> Python 
<br/>-> Face Recognition Library - Detects and compares with known faces
<br/>-> OpenCV - For face detection and preprocessing 
<br/>-> TensorFlow - Model Training 
<br/>-> Dlib - Face encoding and landmark detection
<br/>-> Face Recognition - Simple face recognition using Dlib

---

## Features

<br/>=> DNN Face Detection: Accurate detection using OpenCV’s deep learning model.
<br/>=> LBPH Recognition: Reliable face recognition with good performance in varied lighting.
<br/>=> Multi-Scale & Augmentation: Improves accuracy using rotated, brightened, and enhanced images.
<br/>=> Real-Time Webcam: Detects and matches faces live via webcam.
<br/>=> Step-by-Step Matching: Recognizes one face at a time with manual controls.
<br/>=> Attendance Controls: Keys for next, skip, retrain, mark, and quit (n, y, r, q).
<br/>=> CSV Logging: Saves name, time, and confidence in an attendance file.
<br/>=> Retraining Support: Easily retrain model with updated data.

---

## Challenges faced

<br/>Low-Quality Images – Affected face matching accuracy.
<br/>Webcam Access Issues – Errors if camera was blocked or in use.
<br/>Control Flow Complexity – Managing per-student logic (next, skip, quit) needed clear structure.
<br/>Attendance Output – Required saving attendance to file after session.

---

## Conclusion 

<br/>Face recognition technology provides a modern, contactless, and efficient solution for automating attendance.
<br/>It saves time, improves accuracy, and prevents proxy attendance, making it ideal for educational, corporate, and public settings.
<br/>Despite a few challenges like lighting and privacy concerns, the system offers significant potential for smart and secure attendance management.




