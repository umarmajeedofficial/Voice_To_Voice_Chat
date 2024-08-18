import os
import whisper
import streamlit as st
from groq import Groq
from gtts import gTTS
import tempfile
from pydub import AudioSegment

# Set up Groq API key
os.environ['GROQ_API_KEY'] = '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'
groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

# Load Whisper model
whisper_model = whisper.load_model("base")

# Function to process audio
def process_audio(audio_file):
    # Transcribe audio using Whisper
    result = whisper_model.transcribe(audio_file)
    user_text = result['text']
    
    # Generate response using Llama 8b model with Groq API
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_text,
            }
        ],
        model="llama3-8b-8192",
    )
    response_text = chat_completion.choices[0].message.content
    
    # Convert response text to speech using gTTS
    tts = gTTS(text=response_text, lang='en')
    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts.save(audio_file.name)
    
    return response_text, audio_file.name

# Streamlit UI
st.title("Audio Transcription and Response")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        audio_segment = AudioSegment.from_file(uploaded_file)
        audio_segment.export(tmp.name, format="wav")
        
        # Process audio and get response
        response_text, response_audio = process_audio(tmp.name)
        
        # Display response
        st.text_area("Response", response_text)

        # Play response audio
        st.audio(response_audio, format="audio/mp3")

