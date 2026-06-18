import cv2
import mediapipe as mp
import numpy as np

# Import the new Tasks API components
BaseOptions = mp.tasks.BaseOptions
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def extract_keypoints(result):
    """
    Extracts coordinates from the modern Tasks result object.
    Maintains a static output shape of 1662 features per frame.
    """
    # 1. Pose (33 landmarks x 4 channels: X, Y, Z, visibility)
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        # Loop directly over the pose list items
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks]).flatten()
    else:
        pose = np.zeros(33 * 4)
        
    # 2. Face Mesh (468 landmarks x 3 channels: X, Y, Z)
    if result.face_landmarks and len(result.face_landmarks) > 0:
        face = np.array([[lm.x, lm.y, lm.z] for lm in result.face_landmarks]).flatten()
    else:
        face = np.zeros(468 * 3)
        
    # 3. Left Hand (21 landmarks x 3 channels: X, Y, Z)
    if result.left_hand_landmarks and len(result.left_hand_landmarks) > 0:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks]).flatten()
    else:
        lh = np.zeros(21 * 3)
        
    # 4. Right Hand (21 landmarks x 3 channels: X, Y, Z)
    if result.right_hand_landmarks and len(result.right_hand_landmarks) > 0:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks]).flatten()
    else:
        rh = np.zeros(21 * 3)
        
    return np.concatenate([pose, face, lh, rh])

# Configure options for the landmarker
options = HolisticLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='holistic_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO, 
    min_face_detection_confidence=0.5,
    min_pose_detection_confidence=0.5,
    min_hand_landmarks_confidence=0.5
)

# Start Video Capture
cap = cv2.VideoCapture(0)

with HolisticLandmarker.create_from_options(options) as landmarker:
    print("Modern Tasks Pipeline Active. Press 'q' to quit.")
    
    # Flag to debug-print the type only once
    debugged = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Safe monotonic timestamp generation for Video Mode
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if frame_timestamp_ms == 0:
            frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        # Run inference
        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        # Debugging step: print the structure of the data once so we see it
        if detection_result.pose_landmarks and not debugged:
            print("--- DATA FORMAT DEBUG ---")
            print(f"Type of pose_landmarks: {type(detection_result.pose_landmarks)}")
            print(f"Length of pose_landmarks: {len(detection_result.pose_landmarks)}")
            print(f"Type of first element: {type(detection_result.pose_landmarks[0])}")
            print("-------------------------")
            debugged = True

        # Safely convert tracking data to flat feature array
        keypoints = extract_keypoints(detection_result)
        
        # Display current vector matrix statistics on frame
        cv2.putText(frame, f"Vector Shape: {keypoints.shape}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Real-time state validation boxes
        has_lh = "LH: OK" if detection_result.left_hand_landmarks else "LH: Missing"
        has_rh = "RH: OK" if detection_result.right_hand_landmarks else "RH: Missing"
        cv2.putText(frame, f"{has_lh} | {has_rh}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

        # Render UI frame window
        cv2.imshow('Sign Language AI - Modern Pipeline', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()