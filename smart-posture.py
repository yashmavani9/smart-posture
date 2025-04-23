import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)



# Helper function to calculate angle between three points
def calculate_angle(a, b, c):
    """
    Calculate the angle formed by the three points: a, b, c
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point (vertex)
    c = np.array(c)  # Third point
    
    # Calculate angle using cosine rule
    ab = a - b
    bc = c - b
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Ensure angle is between -1 and 1
    return np.degrees(angle)

import numpy as np

def calculate_horizontal_deviation(shoulder_left, shoulder_right):
    """
    Calculate the acute angle between the shoulder line (from left to right shoulder)
    and the horizontal axis (ideal 180 degrees).
    """
    # Calculate the vector from left shoulder to right shoulder
    vector = np.array(shoulder_right) - np.array(shoulder_left)
    
    # Calculate the angle of this vector with respect to the horizontal axis (x-axis)
    angle = np.arctan2(vector[1], vector[0])  # Angle in radians
    
    # Convert angle from radians to degrees
    angle_deg = np.degrees(angle)
    
    # Ensure the angle is between 0° and 180°, and return the acute angle
    if angle_deg < 0:
        angle_deg += 180  # Adjust for negative angles (i.e., make them positive)
    
    # If the angle is greater than 90°, return the complementary acute angle
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    
    return angle_deg


def calculate_posture_score(angle_diff, shoulder_deviation):
    """
    Calculate a posture score as a percentage based on deviation from ideal posture.
    """
    # Ideal values for angles and deviation
    ideal_angle_diff = 0  # Ideal difference between corresponding angles (perfect symmetry)
    ideal_shoulder_deviation = 0  # Ideal deviation from horizontal

    # Maximum acceptable deviation thresholds for scoring
    max_angle_diff = 15  # Maximum difference in angles for good posture
    max_shoulder_deviation = 10  # Maximum shoulder deviation for good posture

    # Normalize deviations to a score between 0 and 100
    angle_score = max(0, 100 - (abs(angle_diff) / max_angle_diff) * 100)
    shoulder_score = max(0, 100 - (abs(shoulder_deviation) / max_shoulder_deviation) * 100)

    # Combine the scores (weighted average, if needed)
    posture_score = (angle_score + shoulder_score) / 2

    return posture_score

# Function to process the posture analysis and draw lines/angles on the image
def analyze_posture(image):
    # Convert the image to RGB for pose detection
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get pose landmarks
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        # Extract relevant landmarks
        left_eye = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y]
        right_eye = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].y]

        left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        right_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]

        left_ear = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y]
        right_ear = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].y]
        
        

        # Convert to pixel coordinates
        height, width, _ = image.shape
        left_eye = [int(left_eye[0] * width), int(left_eye[1] * height)]
        right_eye = [int(right_eye[0] * width), int(right_eye[1] * height)]
        left_shoulder = [int(left_shoulder[0] * width), int(left_shoulder[1] * height)]
        right_shoulder = [int(right_shoulder[0] * width), int(right_shoulder[1] * height)]
        right_ear = [int(right_ear[0] * width), int(right_ear[1] * height)]
        left_ear = [int(left_ear[0] * width), int(left_ear[1] * height)]
        z1 = [(right_eye[0]+left_eye[0])//2,(right_eye[1]+left_eye[1])//2]
        z2 = [(right_shoulder[0]+left_shoulder[0]//2),(right_shoulder[1]+left_shoulder[1]//2)]

        # Calculate angles x and y
        angle_x1 = calculate_angle(left_eye, left_shoulder, right_shoulder)
        angle_x2 = calculate_angle(right_eye, right_shoulder, left_shoulder)
        angle_x3 = calculate_angle(left_ear,left_shoulder,right_shoulder)
        angle_x4 = calculate_angle(right_ear,right_shoulder,left_shoulder)

        # Calculate shoulder line deviation from horizontal (180 degrees)
        shoulder_deviation = calculate_horizontal_deviation(left_shoulder, right_shoulder)

        # Output posture feedback
        print(f"Angle x: {angle_x1:.2f}°")
        print(f"Angle y: {angle_x2:.2f}°")
        print(f"Angle y: {angle_x3:.2f}°")
        print(f"Angle y: {angle_x4:.2f}°")
        print(f"Shoulder deviation from horizontal: {shoulder_deviation:.2f}°")

        # Draw lines and angles on the image
        # cv2.line(image, left_eye, left_shoulder, (255, 0, 0), 2)  # Left eye to left shoulder
        # cv2.line(image, left_ear, left_shoulder, (255, 0, 0), 2)  # Left ear to left shoulder
        # cv2.line(image, left_shoulder, right_shoulder, (255, 0, 0), 2)  # Left shoulder to right shoulder
        # cv2.line(image, right_eye, right_shoulder, (255, 0, 0), 2)  # Right eye to right shoulder
        # cv2.line(image, right_ear, right_shoulder, (255, 0, 0), 2)  # Right ear to right shoulder
        # cv2.line(image, right_shoulder, left_shoulder, (255, 0, 0), 2)  # Right shoulder to left shoulder


        # Put text showing angles on the image
        # cv2.putText(image, f"Angle x1: {angle_x1:.2f}°", (left_shoulder[0] + 10, left_shoulder[1] - 10), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.putText(image, f"Angle x2: {angle_x2:.2f}°", (right_shoulder[0] + 10, right_shoulder[1] - 10), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.putText(image, f"Angle x3: {angle_x2:.2f}°", (right_ear[0] + 10, right_ear[1] - 10), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.putText(image, f"Angle x4: {angle_x2:.2f}°", (left_ear[0] + 10, left_ear[1] - 10), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.putText(image, f"Deviation: {shoulder_deviation:.2f}°", (50, 50), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
        cv2.line(image, midpoint, left_shoulder, (255, 0, 0), 2)  
        cv2.line(image, midpoint, right_shoulder, (255, 0, 0), 2) 
        cv2.line(image, left_eye, right_eye, (255, 0, 0), 2) 
        # cv2.line(image,z1,z2,(255, 0, 0), 2)
        # Process the image using Face Mesh
    results_face = face_mesh.process(image_rgb)

    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:

        # Draw a line representing the midline on the face
            # Get the coordinates of the midline landmarks (nose tip and chin)
            nose_tip = face_landmarks.landmark[1]   # Nose tip index
            chin = face_landmarks.landmark[152]     # Chin index

# Convert to pixel coordinates
            nose_tip_coords = [int(nose_tip.x * width), int(nose_tip.y * height)]
            chin_coords = [int(chin.x * width), int(chin.y * height)]

# Draw a line representing the midline on the face
            cv2.line(image, nose_tip_coords, chin_coords, (0, 255, 0), 2)
            cv2.line(image, nose_tip_coords, chin_coords, (0, 255, 0), 2)


        a = math.pow((angle_x1-angle_x2),2)
        b = math.pow((angle_x3-angle_x4),2)
        angle_diff = math.sqrt((math.pow((angle_x1 - angle_x2), 2) + math.pow((angle_x3 - angle_x4), 2)) / 2)
        posture_score = calculate_posture_score(angle_diff, shoulder_deviation)

        print(f"Posture Score: {posture_score:.2f}%")
        cv2.putText(image, f"Posture Score: {posture_score:.2f}%", (50, height - 80),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_AA)

        
        # Determine posture quality
        if posture_score >= 75:
            posture_feedback = "Good posture!"
            color = (0, 255, 0)
        elif posture_score >= 55:
            posture_feedback = "Fair posture."
            color = (0, 255, 255)
        else:
            posture_feedback = "Poor posture!"
            color = (0, 0, 255)
        
        cv2.putText(image, posture_feedback, (50, height - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)

    else:
        # Pose landmarks not detected feedback
        posture_feedback = "Pose landmarks not detected."
        cv2.putText(image, posture_feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
# Open webcam and start posture analysis
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    analyze_posture(frame)

    # Display the resulting frame
    cv2.imshow('Posture Analysis', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
