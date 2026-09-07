import cv2
import sys
import numpy as np
from src.pose_detector import PoseDetector

def main():
    img_path = r"C:\Users\Moises James Q. Sy\.gemini\antigravity-ide\brain\a42a9980-4ade-4ad7-b3c1-a8aa0435fe6a\.user_uploaded\media_1787171060935.png"
    img = cv2.imread(img_path)
    if img is None:
        print("Could not load image.")
        sys.exit(1)
        
    detector = PoseDetector(use_tasks_api=False)
    pose_data, results = detector.process_frame(img)
    
    if results is None:
        print("No results.")
        sys.exit(1)
        
    try:
        out_img = detector.draw_skeleton(img, results, is_lateral=True)
        print("Draw skeleton success.")
    except Exception as e:
        print(f"ERROR inside draw_skeleton: {e}")
        
if __name__ == '__main__':
    main()
