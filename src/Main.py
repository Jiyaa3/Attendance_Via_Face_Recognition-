import cv2
import numpy as np
import os
import urllib.request
import csv
from datetime import datetime
from cv2.face import LBPHFaceRecognizer_create
import pickle

# Configuration 
PHOTOS_DIR = r"C:\Users\gopes\Desktop\AI END SEM\Photo"
ATTENDANCE_DIR = r"C:\Users\gopes\Desktop\AI END SEM\Attendance"
MODELS_DIR = r"C:\Users\gopes\Desktop\AI END SEM\Models"
PROTOTXT = "deploy.prototxt"
CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"
MODEL_URL = "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/"

# Settings
CONFIDENCE_THRESHOLD = 65
MIN_FACE_SIZE = (60, 60)
LBPH_PARAMS = {"radius": 2, "neighbors": 10, "grid_x": 10, "grid_y": 10}
ENABLE_FEATURES = {"histogram_eq": True, "data_augment": True, "multi_scale": True, "use_saved": True}

# Global variables
label_dict, photo_dict, name_dict = {}, {}, {}
attendance_recorded = set()
current_status = "Waiting for first person..."
last_attendance_time, processing_person = None, False

def setup():
    """Setup directories and models"""
    for dir_path in [PHOTOS_DIR, ATTENDANCE_DIR, MODELS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Download models if needed
    if not os.path.exists(PROTOTXT):
        urllib.request.urlretrieve(f"{MODEL_URL}{PROTOTXT}", PROTOTXT)
    if not os.path.exists(CAFFEMODEL):
        urllib.request.urlretrieve(
            "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            CAFFEMODEL
        )
    
    # Initialize attendance file
    today = datetime.now().strftime("%Y-%m-%d")
    csv_file = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            csv.writer(file).writerow(["Name", "Photo Filename", "Time", "Date", "Confidence"])
    else:
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                if len(row) > 0:
                    attendance_recorded.add(row[0])

def detect_faces(frame, net):
    """Detect faces using DNN"""
    if frame is None or frame.size == 0:
        return []
        
    faces = []
    h, w = frame.shape[:2]
    scales = [1.0]
    if ENABLE_FEATURES["multi_scale"]:
        scales.append(0.75)
    
    for scale in scales:
        if scale != 1.0:
            scaled = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            sh, sw = scaled.shape[:2]
        else:
            scaled = frame
            sh, sw = h, w
            
        blob = cv2.dnn.blobFromImage(cv2.resize(scaled, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([sw, sh, sw, sh])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Scale back if necessary
                if scale != 1.0:
                    startX, startY = int(startX / scale), int(startY / scale)
                    endX, endY = int(endX / scale), int(endY / scale)
                
                # Validation checks
                if (endX - startX) < MIN_FACE_SIZE[0] or (endY - startY) < MIN_FACE_SIZE[1]:
                    continue
                    
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                
                if startX >= endX or startY >= endY:
                    continue
                    
                face = frame[startY:endY, startX:endX]
                if face.size > 0:
                    faces.append((face, (startX, startY, endX, endY)))
        
        if faces:  # If faces found at this scale, no need to try others
            break
    
    return faces

def preprocess_faces(faces, augment=False):
    """Process detected faces for recognition"""
    processed = []
    for face in faces:
        # Convert to grayscale
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face.copy()
        
        # Apply histogram equalization
        if ENABLE_FEATURES["histogram_eq"]:
            gray = cv2.equalizeHist(gray)
            
        # Apply noise reduction
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Resize for consistency
        gray = cv2.resize(gray, (100, 100))
        processed.append(gray)
        
        # Generate augmented versions if requested
        if augment and ENABLE_FEATURES["data_augment"]:
            h, w = gray.shape[:2]
            center = (w // 2, h // 2)
            
            # Add rotated versions
            for angle in [-5, 5]:
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                processed.append(cv2.warpAffine(gray, rotation_matrix, (w, h)))
            
            # Add brightness variations
            for alpha in [0.9, 1.1]:
                processed.append(cv2.convertScaleAbs(gray, alpha=alpha, beta=0))
    
    return processed

def load_or_train_model(net):
    """Load existing model or train a new one"""
    global label_dict, photo_dict, name_dict
    
    # Try loading saved model
    model_path = os.path.join(MODELS_DIR, "lbph_model.yml")
    dict_path = os.path.join(MODELS_DIR, "face_data.pkl")
    
    if ENABLE_FEATURES["use_saved"] and os.path.exists(model_path) and os.path.exists(dict_path):
        recognizer = LBPHFaceRecognizer_create()
        recognizer.read(model_path)
        with open(dict_path, 'rb') as f:
            label_dict, photo_dict, name_dict = pickle.load(f)
        print("Loaded saved model")
        return recognizer
    
    # Train new model
    faces, labels = [], []
    current_label = 0
    
    # Process each person's directory
    for person_dir_name in os.listdir(PHOTOS_DIR):
        person_dir = os.path.join(PHOTOS_DIR, person_dir_name)
        if not os.path.isdir(person_dir):
            continue
            
        # Format name from directory
        formatted_name = person_dir_name.replace('_', ' ').title()
        print(f"Processing: {formatted_name}")
        
        label_dict[current_label] = formatted_name
        name_dict[current_label] = formatted_name
        photo_dict[current_label] = []
        person_faces = 0
        
        # Process each image
        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            image = cv2.imread(os.path.join(person_dir, img_name))
            if image is None:
                continue
            
            # Detect and process faces
            detected = detect_faces(image, net)
            if not detected:
                continue
                
            for face, _ in detected:
                photo_dict[current_label].append(img_name)
                processed = preprocess_faces([face], augment=True)
                for p_face in processed:
                    faces.append(p_face)
                    labels.append(current_label)
                    person_faces += 1
        
        if person_faces > 0:
            print(f"Added {person_faces} samples for {formatted_name}")
            current_label += 1
        else:
            del label_dict[current_label]
            del name_dict[current_label]
            del photo_dict[current_label]
    
    if not faces:
        print("No faces found in training data!")
        return None
    
    # Create and train recognizer
    print(f"Training on {len(faces)} samples for {len(set(labels))} people...")
    recognizer = LBPHFaceRecognizer_create(**LBPH_PARAMS)
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    
    # Save model for future use
    recognizer.write(model_path)
    with open(dict_path, 'wb') as f:
        pickle.dump((label_dict, photo_dict, name_dict), f)
    
    return recognizer

def record_attendance(name, label, confidence):
    """Record attendance in CSV file"""
    global current_status, last_attendance_time, processing_person
    
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    if name not in attendance_recorded:
        csv_file = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")
        photo = photo_dict.get(label, ["N/A"])[0] if label in photo_dict else "N/A"
        
        with open(csv_file, mode='a', newline='') as file:
            csv.writer(file).writerow([name, photo, current_time, today, f"{confidence:.2f}"])
        
        attendance_recorded.add(name)
        last_attendance_time = current_time
        current_status = f"{name} recorded at {current_time}. Press 'n' for next."
    else:
        current_status = f"{name} already present today. Press 'n' for next."
    
    processing_person = False
    return name not in attendance_recorded

def main():
    """Main application function"""
    global current_status, processing_person
    
    setup()
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
    recognizer = load_or_train_model(net)
    if not recognizer:
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return
    
    consecutive_detections = []
    detection_threshold = 3
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        
        # Display status
        cv2.putText(display, current_status + (f" (Last: {last_attendance_time})" if last_attendance_time else ""), 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, "n: next | r: retrain | q: quit", 
                   (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        if not processing_person:
            detected = detect_faces(frame, net)
            
            if detected:
                # Get largest face
                largest = max(detected, key=lambda item: (item[1][2]-item[1][0])*(item[1][3]-item[1][1]))
                face, (x1, y1, x2, y2) = largest
                
                # Process for recognition
                processed = preprocess_faces([face])[0]
                label, confidence = recognizer.predict(processed)
                person_name = name_dict.get(label, "Unknown")
                
                # Track consecutive detections for stability
                consecutive_detections.append((person_name, confidence))
                if len(consecutive_detections) > detection_threshold:
                    consecutive_detections.pop(0)
                
                if len(consecutive_detections) == detection_threshold:
                    names = [d[0] for d in consecutive_detections]
                    most_common = max(set(names), key=names.count)
                    count = names.count(most_common)
                    
                    if count >= detection_threshold * 0.6:
                        avg_conf = sum([d[1] for d in consecutive_detections if d[0] == most_common]) / count
                        person_name, confidence = most_common, avg_conf
                        processing_person = True
                
                # Determine status and color
                if person_name == "Unknown":
                    status, color = "Unknown", (0, 0, 255)
                elif confidence < CONFIDENCE_THRESHOLD:
                    status, color = "Recognized", (0, 255, 0)
                else:
                    status, color = "Low Confidence", (0, 165, 255)
                
                # Handle attendance marking
                if processing_person:
                    if person_name in attendance_recorded:
                        status = "Already Present"
                    elif confidence < CONFIDENCE_THRESHOLD and person_name != "Unknown":
                        record_attendance(person_name, label, confidence)
                        status = "Marked Present"
                    elif person_name == "Unknown":
                        current_status = "Unknown person. Press 'n' for next."
                        processing_person = False
                    else:
                        current_status = f"Low confidence ({confidence:.2f}). 'n': next, 'y': mark anyway"
                
                # Draw UI elements
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                y_pos = y1 - 10 if y1 > 15 else y1 + 15
                cv2.putText(display, f"{person_name}", (x1, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(display, f"{status} ({confidence:.2f})", (x1, y_pos + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow("Face Recognition Attendance", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            processing_person = False
            current_status = "Waiting for next person..."
            consecutive_detections = []
        elif key == ord('y') and person_name != "Unknown" and processing_person:
            record_attendance(person_name, label, confidence)
        elif key == ord('r'):
            if os.path.exists(os.path.join(MODELS_DIR, "lbph_model.yml")):
                os.remove(os.path.join(MODELS_DIR, "lbph_model.yml"))
            recognizer = load_or_train_model(net)
            current_status = "Model retrained. Ready for next person."
            consecutive_detections = []
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()