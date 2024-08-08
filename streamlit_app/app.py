# prompt: create an app using streamlit, using all above code where take input image from user, segment the image and show it. create a table below with the caption with the same layout as above.

pip install streamlit
pip install ultralytics

import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import os
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Load the captioning model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_caption = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def segment_image(image, output_dir):
    results = model(image, save=False)
    segmented_images = []

    for idx, det in enumerate(results[0].boxes.data):
        class_id = int(det[5])
        class_name = model.names[class_id]
        xmin, ymin, xmax, ymax, conf, _ = det

        cropped_img = image[int(ymin):int(ymax), int(xmin):int(xmax)]
        output_filename = os.path.join(output_dir, f"{class_name}{idx+1}.jpg")
        cv2.imwrite(output_filename, cropped_img)
        segmented_images.append((output_filename, class_name))

    return segmented_images

def generate_caption(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model_caption.generate(**inputs)
    caption = processor.batch_decode(out, skip_special_tokens=True)[0]
    return caption

# Streamlit app
st.title("Image Segmentation and Captioning")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button("Segment and Caption"):
        temp_dir = "temp_segmentation"
        os.makedirs(temp_dir, exist_ok=True)

        # Segment the image
        segmented_images = segment_image(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), temp_dir)

        # Display segmented images and captions
        table_data = []
        seq_number = 1
        for image_path, class_name in segmented_images:
            caption = generate_caption(image_path)
            st.image(Image.open(image_path), caption=f"{class_name}", use_column_width=True)
            table_data.append([seq_number, os.path.basename(image_path), caption])
            seq_number += 1

        # Display the table
        st.markdown("### Captioning Results")
        st.markdown("| Seq Number | Segmented Image | Caption |")
        st.markdown("|---|---|---|")
        for row in table_data:
            st.markdown(f"| {row[0]} | {row[1]} | {row[2]} |")
