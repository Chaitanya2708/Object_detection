import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

# Title of the app
st.title("Object Detection Model")
st.write("Upload an image and detect objects using Facebook's DETR model.")

# Load object detection pipeline
@st.cache_resource
def load_model():
    return pipeline("object-detection", model="facebook/detr-resnet-50")

obj_detector = load_model()

# Function to draw boxes on image
def draw_boxes_on_image(pil_image, detections):
    image = pil_image.copy()
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()

    for det in detections:
        box = det["box"]
        label = f"{det['label']} ({det['score']:.2f})"
        x, y, w, h = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        draw.rectangle([(x, y), (w, h)], outline="black", width=6)
        draw.text((x, y - 60), label, fill="black", font=font)

    return image

# Upload image via Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running object detection..."):
        detections = obj_detector(image)
        annotated_image = draw_boxes_on_image(image, detections)

    st.image(annotated_image, caption="Detected Objects", use_column_width=True)
