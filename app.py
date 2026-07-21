
import os
import pickle
import time

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import tensorflow as tf

MODEL_PATH = "model/sign_cnn_model.h5"
ENCODER_PATH = "model/label_encoder.pkl"
NUM_LANDMARKS = 21
NUM_AXES = 3
CONFIDENCE_THRESHOLD = 0.75

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, le


def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype="float32")


def speak(text):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.warning(f"Text-to-speech not available: {e}")


def main():
    st.set_page_config(page_title="AI Sign Language Recognition", layout="wide")
    st.title("🤟 AI Sign Language Recognition System")
    st.caption("Real-time webcam-based sign language to text (MediaPipe + CNN)")

    if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)):
        st.error(
            "Model files not found. Please run `data_collection.py` to collect data "
            "and `train_model.py` to train the model first."
        )
        return

    model, le = load_model_and_encoder()

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Settings")
        run = st.checkbox("Start Webcam", value=False)
        enable_tts = st.checkbox("Enable Text-to-Speech", value=False)
        st.subheader("Recognized Text")
        text_placeholder = st.empty()
        confidence_placeholder = st.empty()

    with col1:
        frame_placeholder = st.empty()

    sentence = ""
    last_pred = None
    last_pred_time = 0
    STABLE_TIME = 1.2  # seconds a prediction must stay stable before adding to sentence

    if run:
        cap = cv2.VideoCapture(0)
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Could not access webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            predicted_label = None
            confidence = 0.0

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = extract_landmarks(hand_landmarks)
                input_tensor = landmarks.reshape(1, NUM_LANDMARKS, NUM_AXES, 1)

                preds = model.predict(input_tensor, verbose=0)[0]
                idx = np.argmax(preds)
                confidence = float(preds[idx])
                predicted_label = le.inverse_transform([idx])[0]

                cv2.putText(
                    frame,
                    f"{predicted_label} ({confidence*100:.1f}%)",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

            # Stable-sign-to-sentence logic
            if predicted_label and confidence >= CONFIDENCE_THRESHOLD:
                if predicted_label == last_pred:
                    if time.time() - last_pred_time >= STABLE_TIME:
                        if not sentence.endswith(predicted_label):
                            sentence += predicted_label
                            last_pred_time = time.time() + 999  # avoid double-add until sign changes
                else:
                    last_pred = predicted_label
                    last_pred_time = time.time()

            frame_placeholder.image(frame, channels="BGR")
            text_placeholder.markdown(f"### `{sentence}`")
            confidence_placeholder.progress(min(max(confidence, 0.0), 1.0))

            if enable_tts and sentence and sentence[-1] == predicted_label:
                pass  # hook: call speak(sentence) on word/sentence boundary as needed

        cap.release()
    else:
        st.info("Check 'Start Webcam' to begin recognition.")

    if st.button("Clear Text"):
        sentence = ""

    if st.button("🔊 Speak Sentence") and sentence:
        speak(sentence)


if __name__ == "__main__":
    main()