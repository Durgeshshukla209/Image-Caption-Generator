import gc
import os
import pickle
import urllib.request
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

st.set_page_config(
    page_title="Advanced AI Image Caption Generator", page_icon="🤖", layout="wide"
)

st.title("🚀 Advanced Multimodal AI Image Captioning System")
st.markdown(
    "Comparing a custom **ResNet152 + BiLSTM + Multi-Head Attention** model"
    " (Flickr8k) with a **Spatial BLIP Transformer**."
)

MODEL_PATH = "model_caption_generator_multihead_15.keras"
MODEL_URL = (
    "https://github.com/Durgeshshukla209/Image-Caption-Generator/releases/"
    "download/v1.0/model_caption_generator_multihead_15.keras"
)
TOKENIZER_PATH = "tokenizer.pkl"


@st.cache_resource
def load_custom_assets():
    if not os.path.exists(MODEL_PATH):
        with st.spinner(
            "Downloading model_caption_generator_multihead_15.keras from GitHub"
            " Release v1.0..."
        ):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    custom_model = load_model(MODEL_PATH, compile=False)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    base_resnet = ResNet152(weights="imagenet")
    resnet_extractor = Model(
        inputs=base_resnet.inputs, outputs=base_resnet.layers[-2].output
    )
    del base_resnet
    gc.collect()

    index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
    return custom_model, tokenizer, index_to_word, resnet_extractor


@st.cache_resource
def load_blip_assets():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", torch_dtype=dtype
    ).to(device)
    blip_model.eval()
    return blip_processor, blip_model, device


def _get_expected_seq_len(model, fallback):
    try:
        inp_shape = model.input_shape
        if isinstance(inp_shape, (list, tuple)) and len(inp_shape) >= 2:
            seq_shape = inp_shape[1]
            if isinstance(seq_shape, (list, tuple)) and len(seq_shape) >= 2 and seq_shape[1]:
                return int(seq_shape[1])
    except Exception:
        pass
    return int(fallback)


def _predict_step(model, features, padded):
    try:
        out = model.predict([features, padded], verbose=0)
    except ValueError:
        input_names = getattr(model, "input_names", None)
        if input_names and len(input_names) >= 2:
            out = model.predict(
                {input_names[0]: features, input_names[1]: padded}, verbose=0
            )
        else:
            raise
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out[0].astype(np.float64)


def generate_custom_caption(
    model,
    tokenizer,
    index_to_word,
    features,
    max_length,
    temperature=0.25,
    beam_width=2,
    rep_penalty=0.25,
):
    start_id = tokenizer.word_index.get("startseq")
    end_id = tokenizer.word_index.get("endseq")
    if start_id is None:
        return "", []

    expected_len = _get_expected_seq_len(model, max_length)
    step_limit = min(int(max_length), expected_len)

    beams = [([start_id], 0.0, [])]
    completed = []

    for _ in range(step_limit):
        candidates = []

        for seq, score, confs in beams:
            padded = pad_sequences(
                [seq[:expected_len]], maxlen=expected_len, padding="post"
            )
            preds = _predict_step(model, features, padded)
            del padded

            if start_id < len(preds):
                preds[start_id] = 0.0

            if len(seq) > 1 and seq[-1] < len(preds):
                preds[seq[-1]] = 0.0

            for prev_id in set(seq[1:]):
                if prev_id < len(preds):
                    preds[prev_id] *= rep_penalty

            temp = max(float(temperature), 1e-2)
            scaled = np.log(np.maximum(preds, 1e-12)) / temp
            scaled -= np.max(scaled)
            probs = np.exp(scaled)
            probs_sum = np.sum(probs)
            probs = probs / probs_sum if probs_sum > 0 else np.ones_like(probs) / len(probs)
            del preds, scaled

            top_ids = np.argsort(probs)[-beam_width:]
            for idx in top_ids:
                idx = int(idx)
                prob = float(probs[idx])
                if prob <= 0.0:
                    continue

                new_seq = seq + [idx]
                new_score = score + np.log(max(prob, 1e-12))
                word = index_to_word.get(idx)

                if end_id is not None and idx == end_id:
                    completed.append((new_seq, new_score, confs))
                elif word:
                    candidates.append((new_seq, new_score, confs + [(word, prob)]))
            del probs

        if not candidates:
            break

        beams = sorted(
            candidates,
            key=lambda item: item[1] / max(len(item[0]) - 1, 1),
            reverse=True,
        )[:beam_width]

        if len(completed) >= beam_width:
            break

    pool = completed if completed else beams
    if not pool:
        return "", []

    best_seq, _, best_confs = max(
        pool,
        key=lambda item: item[1] / max(len(item[0]) - 1, 1),
    )

    words = [
        index_to_word.get(tok_id, "")
        for tok_id in best_seq
        if tok_id not in (start_id, end_id)
    ]
    caption = " ".join(w for w in words if w).strip()
    return caption, best_confs


st.sidebar.header("⚙️ Advanced Settings")
model_mode = st.sidebar.radio(
    "Model Execution Mode (saves RAM)",
    ["Spatial BLIP Only", "Custom ResNet152 + BiLSTM Only", "Both (High RAM)"],
    index=0,
)
max_length = st.sidebar.slider(
    "Max Caption Length", min_value=10, max_value=34, value=18
)
temperature = st.sidebar.slider(
    "Sampling Temperature", min_value=0.1, max_value=1.0, value=0.25
)

uploaded_file = st.sidebar.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(
            image, caption="Selected Target Image", use_container_width=True
        )

    with col2:
        if st.button("Generate & Analyze Captions 🚀"):
            custom_caption = None
            word_confidences = []
            blip_caption = None

            with st.spinner("Running selected model(s)..."):
                if model_mode in ("Custom ResNet152 + BiLSTM Only", "Both (High RAM)"):
                    custom_model, tokenizer, index_to_word, resnet_extractor = load_custom_assets()
                    img_resized = image.resize((224, 224))
                    arr =
