import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# Example placeholder functions for captioning and segmentation
def generate_caption(image):
    # Placeholder: Replace with your image captioning model
    return "A group of people standing around a park."

def segment_image(image):
    # Placeholder: Replace with your segmentation model
    # Generates a fake mask (for demo)
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    mask[30:200, 50:270] = 1  # Example mask region
    return mask

def overlay_mask(image, mask):
    color = (255, 0, 0)  # Red
    img_array = np.array(image).copy()
    img_array[mask == 1] = [255, 0, 0]   # Overlay mask
    return Image.fromarray(img_array)

# Streamlit interface
st.set_page_config(page_title="Image Captioning & Segmentation", layout="centered")

st.title("Image Captioning and Segmentation System")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Process Image"):
        with st.spinner("Generating caption..."):
            caption = generate_caption(image)
        with st.spinner("Segmenting image..."):
            mask = segment_image(image)
            segmented_image = overlay_mask(image, mask)
        
        st.subheader("Generated Caption:")
        st.write(f"*{caption}*")

        st.subheader("Segmentation Result:")
        st.image(segmented_image, caption="Segmented Image", use_column_width=True)

        st.subheader("Segmentation Mask:")
        fig, ax = plt.subplots()
        ax.imshow(mask, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)

