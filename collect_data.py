import cv2
import numpy as np
import os
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['Hello', 'Thank_You', 'Emergency'])
no_sequences = 30
sequence_length = 30

for action in actions: 
    for sequence in range(no_sequences):
        try: os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except: pass

def extract_hand_keypoints(result):
    # Extracts left and right hand data seamlessly (21 landmarks * 3 coordinates = 63 per hand)
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    
    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        for idx, hand_handedness in enumerate(result.handedness):
            label = hand_handedness[0].category_name
            landmarks = result.hand_landmarks[idx]
            coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            
            if label == 'Left':
                lh = coords
            elif label == 'Right':
                rh = coords
                
    return np.concatenate([lh, rh]) # Exact total shape: 126 elements

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.2
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    print("Hands Data Collector Active. Ready...")
    
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret: break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                detection_result = landmarker.detect(mp_image)
                keypoints = extract_hand_keypoints(detection_result)
                
                if frame_num == 0: 
                    cv2.putText(frame, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(frame, f'Collecting "{action}" Video {sequence} Frame {frame_num}', (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)
                
                np.save(os.path.join(DATA_PATH, action, str(sequence), str(frame_num)), keypoints)
                if cv2.waitKey(10) & 0xFF == ord('q'): break
                
    cap.release()
    cv2.destroyAllWindows()