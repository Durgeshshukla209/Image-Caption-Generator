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
def load_models():
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

  device = "cuda" if torch.cuda.is_available() else "cpu"
  blip_processor = BlipProcessor.from_pretrained(
      "Salesforce/blip-image-captioning-base"
  )
  blip_model = BlipForConditionalGeneration.from_pretrained(
      "Salesforce/blip-image-captioning-base"
  ).to(device)
  blip_model.eval()

  return (
      custom_model,
      tokenizer,
      resnet_extractor,
      blip_processor,
      blip_model,
      device,
  )


def generate_custom_caption(
    model,
    tokenizer,
    index_to_word,
    features,
    max_length,
    temperature=0.2,
    beam_width=3,
    rep_penalty=0.25,
):
  start_id = tokenizer.word_index.get("startseq")
  end_id = tokenizer.word_index.get("endseq")
  if start_id is None:
    return "", []

  beams = [([start_id], 0.0, [])]
  completed = []

  for _ in range(max_length):
    candidates = []

    for seq, score, confs in beams:
      padded = pad_sequences([seq], maxlen=max_length, padding="post")
      preds = model.predict([features, padded], verbose=0)[0].astype(np.float64)

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


with st.spinner("Loading AI models... Please wait ⏳"):
  (
      custom_model,
      tokenizer,
      resnet_extractor,
      blip_processor,
      blip_model,
      device,
  ) = load_models()

index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

st.sidebar.header("⚙️ Advanced Settings")
max_length = st.sidebar.slider(
    "Max Caption Length", min_value=10, max_value=50, value=34
)
temperature = st.sidebar.slider(
    "Sampling Temperature", min_value=0.1, max_value=1.0, value=0.5
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
      with st.spinner("Generating captions..."):
        img_resized = image.resize((224, 224))
        arr = np.expand_dims(
            np.array(img_resized, dtype=np.float32), axis=0
        )
        arr = preprocess_input(arr)
        features = resnet_extractor.predict(arr, verbose=0)

        custom_caption, word_confidences = generate_custom_caption(
            custom_model,
            tokenizer,
            index_to_word,
            features,
            max_length=max_length,
            temperature=temperature,
            beam_width=3,
            rep_penalty=0.25,
        )

        inputs = blip_processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
          out = blip_model.generate(**inputs, max_new_tokens=40)
        blip_caption = blip_processor.decode(
            out[0], skip_special_tokens=True
        ).strip()

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
