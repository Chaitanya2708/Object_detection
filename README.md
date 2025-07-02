#Object Detection Web App using Streamlit & DETR
This is a simple web application built with Streamlit that allows users to upload an image and run object detection using Facebook’s DETR (DEtection TRansformer) model.

It uses Hugging Face's transformers library to load the pretrained detr-resnet-50 model and detects objects in images with bounding boxes and confidence scores.
 Features
Upload any .jpg, .jpeg, or .png image

Detects multiple objects in the image

Draws bounding boxes and labels on the image

Runs entirely in the browser via Streamlit interface

Tech Stack
Streamlit — UI framework

Hugging Face Transformers — for loading DETR model

Pillow (PIL) — for image processing

Installation
Clone the repository or copy the code into a new file:


git clone https://github.com/your-username/streamlit-detr-object-detection.git
cd streamlit-detr-object-detection

Create a virtual environment (optional but recommended):


python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
If you don't have a requirements.txt, install manually:

pip install streamlit transformers pillow torch

Running the App

streamlit run app.py

Once running, the app will open in your browser at http://localhost:8501.
