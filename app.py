import streamlit as st
import cv2
import pickle
import numpy as np
from pose_module import PoseDetector

# Load model
model = pickle.load(open('trainer_model.pkl', 'rb'))
detector = PoseDetector()

st.title("🤖 AI Virtual Fitness Trainer")
run = st.checkbox('Start Camera')

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = detector.detect_pose(frame, draw=True)
    landmarks = detector.extract_landmarks()

    if landmarks is not None:
        prediction = model.predict([landmarks])
        cv2.putText(frame, f'Exercise: {prediction[0]}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

else:
    cap.release()
    cv2.destroyAllWindows()
