import sys
import numpy as np
import mediapipe as mp
import cv2

def main():
    img_path = r"C:\Users\Moises James Q. Sy\.gemini\antigravity-ide\brain\a42a9980-4ade-4ad7-b3c1-a8aa0435fe6a\.user_uploaded\media_1787172322301.png"
    img = cv2.imread(img_path)
    if img is None:
        print("Could not load image.")
        return
        
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if not results.pose_landmarks:
        print("No pose found.")
        return
        
    ls = results.pose_landmarks.landmark[11]
    rs = results.pose_landmarks.landmark[12]
    nose = results.pose_landmarks.landmark[0]
    
    shoulder_width = float(np.sqrt((rs.x - ls.x) ** 2 + (rs.y - ls.y) ** 2))
    dx_sh = rs.x - ls.x
    dy_sh = rs.y - ls.y
    shoulder_tilt_deg = float(np.degrees(np.arctan2(dy_sh, dx_sh)))
    shoulder_z_diff = abs(ls.z - rs.z)
    
    print(f"Shoulder Width: {shoulder_width:.4f}")
    print(f"Shoulder Z Diff: {shoulder_z_diff:.4f}")
    print(f"Z Diff / Width Ratio: {shoulder_z_diff / shoulder_width:.4f}")
    print(f"Shoulder Tilt (deg): {shoulder_tilt_deg:.2f}")

if __name__ == '__main__':
    main()
