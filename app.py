import streamlit as st
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

st.set_page_config(
    page_title="Advanced AI Image Caption Generator",
    page_icon="🤖",
    layout="wide"
)

st.title("🚀 Advanced Multimodal AI Image Captioning System")
st.markdown("Comparing a custom **ResNet152 + BiLSTM + Multi-Head Attention** model (Flickr8k) with a **Spatial BLIP Transformer**.")

@st.cache_resource
def load_models():
    model_path = "/content/drive/MyDrive/image caption/model_caption_generator_multihead_15.keras"
    tokenizer_path = "/content/drive/MyDrive/image caption/tokenizer.pkl"

    custom_model = load_model(model_path, compile=False)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    base_resnet = ResNet152(weights="imagenet")
    resnet_extractor = Model(inputs=base_resnet.inputs, outputs=base_resnet.layers[-2].output)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    return custom_model, tokenizer, resnet_extractor, blip_processor, blip_model, device

with st.spinner("Loading AI models into Cloud GPU... Please wait ⏳"):
    custom_model, tokenizer, resnet_extractor, blip_processor, blip_model, device = load_models()

index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

st.sidebar.header("⚙️ Advanced Settings")
max_length = st.sidebar.slider("Max Caption Length", min_value=10, max_value=50, value=34)
temperature = st.sidebar.slider("Sampling Temperature", min_value=0.1, max_value=1.0, value=0.5)

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(image, caption="Selected Target Image", use_column_width=True)

    with col2:
        if st.button("Generate & Analyze Captions 🚀"):
            with st.spinner("Generating captions..."):
                img_resized = image.resize((224, 224))
                arr = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)
                arr = preprocess_input(arr)
                features = resnet_extractor.predict(arr, verbose=0)

                in_text = "startseq"
                word_confidences = []

                for _ in range(max_length):
                    seq = tokenizer.texts_to_sequences([in_text])[0]
                    seq = pad_sequences([seq], maxlen=max_length, padding="post")
                    preds = custom_model.predict([features, seq], verbose=0)[0]

                    scaled = np.log(preds + 1e-8) / temperature
                    probs = np.exp(scaled) / np.sum(np.exp(scaled))

                    next_idx = int(np.argmax(probs))
                    confidence = float(probs[next_idx])

                    word = index_to_word.get(next_idx)
                    if word is None or word == "endseq":
                        break
                    in_text += " " + word
                    word_confidences.append((word, confidence))

                custom_caption = in_text.replace("startseq", "").strip()

                inputs = blip_processor(image, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = blip_model.generate(**inputs, max_new_tokens=40)
                blip_caption = blip_processor.decode(out[0], skip_special_tokens=True).strip()

            st.success("Analysis Complete!")

            tab1, tab2 = st.tabs(["🤖 Model Captions", "📊 Confidence Metrics"])

            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    st.info("**Custom ResNet152 + BiLSTM**")
                    st.write(custom_caption)
                    st.code(custom_caption, language="text")
                with c2:
                    st.success("**Spatial BLIP Model**")
                    st.write(blip_caption)
                    st.code(blip_caption, language="text")

            with tab2:
                st.markdown("#### Word-by-Word Prediction Confidence (Custom Model)")
                if word_confidences:
                    for w, c in word_confidences:
                        st.write(f"**{w}** — `{c * 100:.2f}%`")
                else:
                    st.write("No confidence metrics available.")
else:
    st.info("👈 Upload an image from the sidebar to get started.")
