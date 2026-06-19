import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import threading  # <-- New core built-in library for parallel background processing

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

with open('sign_model.pkv', 'rb') as f:
    model = pickle.load(f)

actions = np.array(['Hello', 'Thank_You', 'Emergency'])

# --- THREAD-SAFE BACKGROUND VOICE FUNCTION ---
def speak_text(text):
    """Initializes and destroys the engine inside a separate thread to prevent hardware locks."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 175)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass  # Quietly catch any backend overlaps

def extract_hand_keypoints(result):
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        for idx, hand_handedness in enumerate(result.handedness):
            label = hand_handedness[0].category_name
            landmarks = result.hand_landmarks[idx]
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            
            if label == 'Left': rh = coords   
            elif label == 'Right': lh = coords 
    return np.concatenate([lh, rh])

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.15
)

sequence = []        
sentence = []        
last_spoken = None  

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    print("Parallel Multi-Threaded Voice Engine Active. Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = landmarker.detect(mp_image)
        keypoints = extract_hand_keypoints(detection_result)
        
        sequence.append(keypoints)
        sequence = sequence[-30:] 
        
        lh_on = False
        rh_on = False
        if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
            for h in detection_result.handedness:
                if h[0].category_name == 'Left': rh_on = True   
                if h[0].category_name == 'Right': lh_on = True  
        
        if len(sequence) == 30:
            input_data = np.array(sequence).flatten().reshape(1, -1)
            res = model.predict_proba(input_data)[0]
            predicted_class = np.argmax(res)
            confidence = res[predicted_class]
            
            is_valid_gesture = False
            if predicted_class == 0 and rh_on and not lh_on: is_valid_gesture = True      
            elif predicted_class == 1 and rh_on and not lh_on: is_valid_gesture = True    
            elif predicted_class == 2 and lh_on and rh_on: is_valid_gesture = True        

            if confidence > 0.30 and is_valid_gesture:
                detected_word = actions[predicted_class]
                
                if len(sentence) == 0 or sentence[-1] != detected_word:
                    sentence.append(detected_word)
                
                # --- ASYNCHRONOUS BACKGROUND AUDIO LAUNCH ---
                if detected_word != last_spoken:
                    spoken_phrase = detected_word.replace('_', ' ') 
                    
                    # Fire the voice off to its own background lane so it never freezes the webcam loop
                    threading.Thread(target=speak_text, args=(spoken_phrase,), daemon=True).start()
                    
                    last_spoken = detected_word  
            else:
                if not lh_on and not rh_on:
                    last_spoken = None

            if len(sentence) > 5: sentence = sentence[-5:]

        # UI Text Overlays
        cv2.rectangle(frame, (0,0), (640, 45), (20, 20, 20), -1)
        cv2.putText(frame, f"TRANSLATION: {' -> '.join(sentence)}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Left: {'OK' if lh_on else 'HIDDEN'} | Right: {'OK' if rh_on else 'HIDDEN'}", (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imshow('Sign Language AI - Dedicated Hand Mode', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()