import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

# Load Environment Variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Generative AI Model
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

def generate_notes(topic, language="english"):
    # Create a prompt for note generation
    prompt = f"Create comprehensive, well-structured notes about {topic} in {language} language. Organize the notes with clear headings and bullet points."
    
    response = model.generate_content(prompt)
    return response.text

# Streamlit app for notes generation
st.set_page_config(page_title="Topic-Wise Notes Generator", layout="centered")
st.header("Topic-Wise Notes Generator using Gemini AI")

st.write("Enter any topic and get well-structured, student-friendly notes.")

# Input section
notes_topic = st.text_input("Enter your topic here:", placeholder="e.g., Operating System")
notes_language = st.selectbox("Select Language", ["English", "Hindi", "Spanish", "French", "German"])

generate_btn = st.button("Generate Notes")

if generate_btn and notes_topic:
    with st.spinner('Generating your notes...'):
        notes = generate_notes(notes_topic, notes_language)
        st.subheader("Generated Notes:")
        st.write(notes)
        
        # Add download button
        st.download_button(
            label="Download Notes",
            data=notes,
            file_name=f"{notes_topic.replace(' ', '_')}_notes.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("Powered by Google Gemini AI | Created by Group 62")