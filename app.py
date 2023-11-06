import streamlit as st
import cv2
from PIL import Image as PilImage
from PIL import ImageDraw
import numpy as np
import io
import base64

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_write_on_faces(image):
    text_to_write = "Happy faces, this person has a retention rate of 69% in a balanced way."

    # Convert the uploaded image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Convert the OpenCV image to a Pillow image
    pil_image = PilImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    for (x, y, w, h) in faces:
        # Write text on the detected face
        draw.text((x, y - 10), text_to_write, fill=(255, 0, 0, 0))

    # Convert the Pillow image back to OpenCV format
    image_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return image_with_text

st.title("Face Detection and Text Writing")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    if st.button("Process Image"):
        input_image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), -1)
        result_image = detect_and_write_on_faces(input_image)

        st.image(result_image, caption="Processed Image", use_column_width=True)

        # Allow the user to download the processed image as a JPEG file
        output_buffer = io.BytesIO()
        PilImage.fromarray(result_image).save(output_buffer, format="JPEG")
        st.markdown("### Download Processed Image")
        st.markdown(
            f"Download your processed image [here](data:file/jpeg;base64,{base64.b64encode(output_buffer.getvalue()).decode()})"
        )
