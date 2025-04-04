import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import librosa
import tempfile
import string
from transformers import BertTokenizer

st.markdown(
    """
    <style>
    /* Set overall background color */
    body {
        background-color: #f0f2f6;
    }
    
    /* Customize Streamlit container background */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* Customize buttons */
    div.stButton > button {
        background-color: #2a9d8f;
        color: white;
        border: none;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 5px;
    }
    div.stButton > button:hover {
        background-color: #21867a;
    }
    
    /* Style file uploader container */
    .stFileUploader {
        background-color: #f4a261;
        border: 2px dashed #264653;
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Style text preview box */
    .text-preview {
        background-color: #f9c74f;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define label mapping and modality names
all_labels = ['angry', 'disgust', 'fear', 'happy','neutral','sad','surprise']
idx_to_label = {i: label for i, label in enumerate(all_labels)}
modality_names = ["Image", "Audio", "Text"]

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the traced model
@st.cache(allow_output_mutation=True)
def load_traced_model():
    model = torch.jit.load(r"C:\Users\umara\Downloads\expression_modal_traced\expression_modal_traced.pt", map_location=device)
    model.to(device)
    model.eval()
    return model

model = load_traced_model()

# Define processing functions for each modality
def process_image(file_obj):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(file_obj).convert("RGB")
    tensor_image = transform(image).unsqueeze(0)
    return tensor_image

def process_audio(file_obj, target_length=42240):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
        tmp.write(file_obj.read() if hasattr(file_obj, 'read') else file_obj)
        tmp_path = tmp.name
    y, sr = librosa.load(tmp_path, sr=16000)
    y = librosa.util.normalize(y)
    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    audio_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    return audio_tensor

def process_text_from_file(file_obj):
    text = file_obj.read().decode('utf-8')
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    inputs = tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    return inputs['input_ids']

def process_text_from_input(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    inputs = tokenizer(text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    return inputs['input_ids']

# Streamlit App UI with colorful styling
st.markdown("<h1 style='text-align: center; color: #2a9d8f;'>Multimodal Emotion Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #264653; font-size:18px;'>Provide your input (image, audio, text) for emotion prediction.</p>", unsafe_allow_html=True)

# Image Input: Upload or Capture
st.markdown("<h3 style='color:#e9c46a;'>Image Input</h3>", unsafe_allow_html=True)
image_input_method = st.radio("Choose Image Input Method", ("Upload", "Capture"))
if image_input_method == "Upload":
    image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="img_upload")
elif image_input_method == "Capture":
    image_file = st.camera_input("Capture an Image")
else:
    image_file = None

# Show preview for image input
if image_file is not None:
    if image_input_method == "Upload":
        image_file.seek(0)
        img_bytes = image_file.read()
        st.image(img_bytes, caption="Uploaded Image", use_column_width=True)
    else:
        # For camera input, Streamlit shows the captured image automatically.
        pass

# Audio Input: Upload or Record
st.markdown("<h3 style='color:#e9c46a;'>Audio Input</h3>", unsafe_allow_html=True)
audio_input_method = st.radio("Choose Audio Input Method", ("Upload", "Record"))
if audio_input_method == "Upload":
    audio_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3"], key="audio_upload")
elif audio_input_method == "Record":
    try:
        from st_audiorec import st_audiorec
        st.write("Audio Recorder Initialized. Click 'Stop' when you're finished recording.")
        audio_file = st_audiorec()
    except ImportError as e:
        st.error(f"Failed to import st_audiorec: {e}")
        audio_file = None
    except Exception as e:
        st.error(f"Error with audio recorder: {e}")
        audio_file = None
else:
    audio_file = None

# Show preview for audio input
if audio_file is not None and audio_input_method == "Upload":
    audio_file.seek(0)
    st.audio(audio_file.read())

# Text Input: Upload or Type
st.markdown("<h3 style='color:#e9c46a;'>Text Input</h3>", unsafe_allow_html=True)
text_input_method = st.radio("Choose Text Input Method", ("Upload", "Type"))
if text_input_method == "Upload":
    text_file = st.file_uploader("Upload a Text File", type=["txt"], key="text_upload")
    text_data = None
elif text_input_method == "Type":
    text_data = st.text_area("Type your text here")
    text_file = None
else:
    text_file = None
    text_data = None

# Show preview for text input (for upload option)
if text_file is not None and text_input_method == "Upload":
    text_file.seek(0)
    text_content = text_file.read().decode('utf-8')
    st.markdown(f"<div style='background-color:#f9c74f; padding:10px; border-radius:5px;'><pre>{text_content}</pre></div>", unsafe_allow_html=True)

# Predict Button and Model Inference
if st.button("Predict Emotion"):
    # Process image input
    if image_file is not None:
        img_tensor = process_image(image_file).to(device)
    else:
        img_tensor = torch.zeros((1, 3, 224, 224), device=device)
    
    # Process audio input
    if audio_file is not None:
        audio_tensor = process_audio(audio_file).to(device)
    else:
        audio_tensor = torch.zeros((1, 42240), device=device)
    
    # Process text input
    if text_file is not None:
        text_tensor = process_text_from_file(text_file).to(device)
    elif text_data:
        text_tensor = process_text_from_input(text_data).to(device)
    else:
        text_tensor = torch.zeros((1, 128), dtype=torch.long, device=device)
    
    with st.spinner("Processing..."):
        with torch.no_grad():
            outputs, weights = model(img_tensor, audio_tensor, text_tensor)
            probs = F.softmax(outputs, dim=1)
            predicted_idx = outputs.argmax(dim=1).item()
            predicted_emotion = idx_to_label[predicted_idx]
            confidence = probs[0, predicted_idx].item()
            weights_np = weights.cpu().numpy()[0]
            if image_file is None:
                weights_np[0] = 0
            if audio_file is None:
                weights_np[1] = 0
            if (text_file is None) and (not text_data):
                weights_np[2] = 0
            total = weights_np.sum()
            if total > 0:
                weights_np = weights_np / total
            dominant_idx = np.argmax(weights_np)
            dominant_modality = modality_names[dominant_idx]
    
    st.success(f"Predicted Emotion: **{predicted_emotion}**")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
    st.write("### Modality Weights (Adjusted for Missing Modalities)")
    st.write(f"Image: **{weights_np[0]:.2f}**")
    st.write(f"Audio: **{weights_np[1]:.2f}**")
    st.write(f"Text: **{weights_np[2]:.2f}**")
    st.info(f"Prediction primarily based on the **{dominant_modality}** modality.")
