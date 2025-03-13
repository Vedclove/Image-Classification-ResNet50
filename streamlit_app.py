import streamlit as st
import requests
from PIL import Image
import io
import os
from dotenv import load_dotenv
load_dotenv()
# Flask API URL
FLASK_API_URL = f"http://{os.getenv('VM_IP')}:8080/predict"

# Streamlit UI
st.title("🐶🐱 Image Classification - ResNet50")

st.write("Upload an image **or** paste an image URL to get predictions from the model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# URL input with tooltip
st.markdown("""
    <style>
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #ccc;
        cursor: help;
    }
    </style>
""", unsafe_allow_html=True)

image_url = st.text_input(
    "Or paste an image URL here:",
    help="1. Visit google to an Cat or Dog image\n2. Right click on the image and select 'Copy image address'\n3. Paste the URL and Enter"
)

image = None  # Initialize image variable

# If user uploads a file
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

# If user enters a URL
elif image_url:
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Ensure valid response
        image = Image.open(io.BytesIO(response.content))
        st.image(image, caption="Image from URL", use_container_width=True)
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading image: {e}")

# Process and send the image to Flask API if available
if image is not None and st.button("Predict"):
    st.write("🚀 Sending image to the model...")

    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    # Send to Flask API
    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
    response = requests.post(FLASK_API_URL, files=files)

    # Display the prediction result
    if response.status_code == 200:
        prediction = response.json()
        st.success(f"**Predicted Class:** {prediction['predicted_class']} 🎉")
    else:
        st.error("❌ Error in prediction. Check if the Flask API is running.")