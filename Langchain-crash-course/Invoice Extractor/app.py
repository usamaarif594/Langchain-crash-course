import streamlit as st
import google.generativeai as genai
from PIL import Image
from pathlib import Path

# Configure API key
api = 'AIzaSyB1p0yVl-BSf7-AGGTdMBBHb5tjsKlEv4s'
genai.configure(api_key=api)

# Define the generative model
model = genai.GenerativeModel("gemini-1.5-flash")

def get_genai_response(input_prompt, image, user_prompt):
    image_info = image_format(image)
    response = model.generate_content([input_prompt, image_info[0], user_prompt])
    return response.text

def image_format(image):
    image_parts = [
        {
            "mime_type": image.type,  # Use the mime type from the uploaded file
            "data": image.getvalue()
        }
    ]
    return image_parts

# Define the initial prompt
input_prompt = """
You are a specialist in comprehending receipts.
Input images in the form of receipts will be provided to you,
and your task is to respond to questions based on the content of the input image.
"""

# Get user prompt
user_prompt = st.text_input('Enter your question about the receipt')

# File uploader for image
uploaded_image = st.file_uploader("Upload a receipt image", type=["jpeg", "png", "jpg"],accept_multiple_files=True)

if st.button('Get Response'):
    if uploaded_image is not None:
        response = get_genai_response(input_prompt, uploaded_image, user_prompt)
        st.write(response)
    else:
        st.write("Please upload a receipt image to get a response.")
