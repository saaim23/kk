import streamlit as st
import cv2
import tensorflow as tf
import numpy as np

# lets load the trained model
@st.cache_resource 
def load_model():
    return tf.keras.models.load_model("colorization_model.keras")

model = load_model()

# lets set the title and stuff
st.title("Real-Time Video Colorization")
st.sidebar.title("Settings")
mode = st.sidebar.radio("Choose Mode", ("Grayscale", "Colorized"))

#this one should be in app 
st.markdown(
    """
    - **Grayscale**: Displays the webcam feed in grayscale.
    - **Colorized**: Displays the colorized output based on the trained model.
    """
)
#this one captire the webcam if it ask for perms just allow it 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Webcam not detected.")
else:
    # App main loop
    stframe = st.empty()  # Placeholder for video frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam feed not available.")
            break
                
        # resize 
        frame_resized = cv2.resize(frame, (32, 32))
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        gray_frame_expanded = gray_frame.reshape(1, 32, 32, 1) / 255.0

        if mode == "Colorized":
            # predict colorized frame
            colorized_frame = model.predict(gray_frame_expanded)[0]
            colorized_frame = np.clip(colorized_frame, 0, 1) * 255
            colorized_frame = cv2.resize(colorized_frame.astype("uint8"), (640, 480))
            frame_to_display = cv2.cvtColor(colorized_frame, cv2.COLOR_RGB2BGR)
        else:  # grayscale
            gray_frame_large = cv2.resize(gray_frame, (640, 480))
            frame_to_display = cv2.cvtColor(gray_frame_large, cv2.COLOR_GRAY2BGR)

        # this will display 
        stframe.image(frame_to_display, channels="BGR")

    cap.release()
