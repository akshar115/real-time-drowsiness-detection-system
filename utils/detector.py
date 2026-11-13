import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_v2
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_input_v3

class DrowsinessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.load_models("custom_cnn")
        
    def load_models(self, model_type):
        self.current_model_type = model_type
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, 'model')
        eye_model_path = os.path.join(model_dir, f'eye_model_{model_type}.h5')
        mouth_model_path = os.path.join(model_dir, f'mouth_model_{model_type}.h5')
        
        if os.path.exists(eye_model_path):
            self.eye_model = load_model(eye_model_path)
            print(f"Loaded {eye_model_path}")
        else:
            self.eye_model = None
            print(f"Warning: {eye_model_path} not found.")
            
        if os.path.exists(mouth_model_path):
            self.mouth_model = load_model(mouth_model_path)
            print(f"Loaded {mouth_model_path}")
        else:
            self.mouth_model = None
            print(f"Warning: {mouth_model_path} not found.")
            
        # State tracking
        self.consecutive_frames_drowsy = 0
        self.drowsy_threshold_frames = 15
        
        # Standard landmarks for EAR (Eye Aspect Ratio) calculation
        self.LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
        
        # Indices for extracting bounding boxes for CNN models
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.FACE_INDICES = list(range(468)) # All landmarks to extract full face

    def calculate_ear(self, landmarks, eye_indices, w, h):
        """Calculates the Eye Aspect Ratio (EAR) given specific eye landmarks."""
        pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
        # Vertical distances
        p2_p6 = np.linalg.norm(pts[1] - pts[5])
        p3_p5 = np.linalg.norm(pts[2] - pts[4])
        # Horizontal distance
        p1_p4 = np.linalg.norm(pts[0] - pts[3])
        if p1_p4 == 0:
            return 0
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)

    def extract_roi(self, image, landmarks, indices, w, h, padding=15):
        """Extracts a bounding box region of interest (ROI) from an image based on landmarks."""
        xs = [int(landmarks[i].x * w) for i in indices]
        ys = [int(landmarks[i].y * h) for i in indices]
        
        x_min, x_max = max(0, min(xs) - padding), min(w, max(xs) + padding)
        y_min, y_max = max(0, min(ys) - padding), min(h, max(ys) + padding)
        
        roi = image[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return None
        
        # Resize to match MobileNetV2 input size
        roi = cv2.resize(roi, (128, 128))
        # Ensure it is RGB as expected by MobileNet preprocess
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        return roi

    def process_frame(self, frame):
        """Processes a single BGR frame, calculates EAR, and runs CNN models on ROIs."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        status = "ALERT"
        eye_score = 0.0      # Probability eyes are CLOSED (0 to 1)
        mouth_score = 0.0    # Probability of YAWN (0 to 1)
        ear = 0.0
        face_detected = False
        
        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 1. EAR Calculation
            left_ear = self.calculate_ear(landmarks, self.LEFT_EYE_POINTS, w, h)
            right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE_POINTS, w, h)
            ear = (left_ear + right_ear) / 2.0
            
            # 2. ROI Extraction
            left_eye_roi = self.extract_roi(frame, landmarks, self.LEFT_EYE_INDICES, w, h)
            right_eye_roi = self.extract_roi(frame, landmarks, self.RIGHT_EYE_INDICES, w, h)
            face_roi = self.extract_roi(frame, landmarks, self.FACE_INDICES, w, h, padding=20)
            
            # 3. Model Predictions (Using exact preprocessing as training)
            prep_fn = preprocess_input_v3 if getattr(self, 'current_model_type', '') == 'mobilenet_v3' else preprocess_input_v2
            
            if self.eye_model and left_eye_roi is not None and right_eye_roi is not None:
                l_eye_prep = prep_fn(np.expand_dims(left_eye_roi, axis=0).astype(np.float32))
                r_eye_prep = prep_fn(np.expand_dims(right_eye_roi, axis=0).astype(np.float32))
                
                # In train.py: Closed=0, Open=1
                # pred is probability of Open. We want probability of Closed (eye_score)
                l_pred_open = self.eye_model.predict(l_eye_prep, verbose=0)[0][0]
                r_pred_open = self.eye_model.predict(r_eye_prep, verbose=0)[0][0]
                
                l_pred_closed = 1.0 - l_pred_open
                r_pred_closed = 1.0 - r_pred_open
                
                eye_score = float((l_pred_closed + r_pred_closed) / 2.0)
                
            if self.mouth_model and face_roi is not None:
                # Isolate the face to remove background noise that confuses the model
                mouth_prep = prep_fn(np.expand_dims(face_roi, axis=0).astype(np.float32))
                
                # In train.py: no_yawn=0, Yawn=1
                m_pred_yawn = self.mouth_model.predict(mouth_prep, verbose=0)[0][0]
                
                # Moving Average Filter to completely eliminate UI fluctuation
                if not hasattr(self, 'mouth_history'):
                    self.mouth_history = []
                self.mouth_history.append(float(m_pred_yawn))
                if len(self.mouth_history) > 10:
                    self.mouth_history.pop(0)
                    
                mouth_score = sum(self.mouth_history) / len(self.mouth_history)
                
            # 4. Status Determination logic
            # Primary signal: EAR (highly reliable for drowsiness)
            if ear < 0.25:
                self.consecutive_frames_drowsy += 1
            else:
                # Also fall back to the CNN eye model if EAR isn't definitive but model is highly confident eyes are closed
                if eye_score > 0.8:
                    self.consecutive_frames_drowsy += 1
                else:
                    self.consecutive_frames_drowsy = max(0, self.consecutive_frames_drowsy - 2) # Reduce faster if awake
                
            # Final Status priority: DROWSY > YAWN > ALERT
            if mouth_score > 0.70:
                status = "YAWN"
                
            if self.consecutive_frames_drowsy >= self.drowsy_threshold_frames:
                status = "DROWSY"
                
        # Drowsiness percentage based on consecutive frames to threshold (cap at 100)
        drowsiness_pct = min(100, int((self.consecutive_frames_drowsy / self.drowsy_threshold_frames) * 100))
        
        return {
            "status": status,
            "eye_score": round(eye_score, 3),
            "mouth_score": round(mouth_score, 3),
            "drowsiness_pct": drowsiness_pct,
            "ear": round(ear, 3),
            "face_detected": face_detected
        }
