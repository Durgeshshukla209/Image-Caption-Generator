# 🚀 Advanced Multimodal AI Image Captioning System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-SUBDOMAIN.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-HuggingFace_BLIP-EE4C2C?logo=pytorch&logoColor=white)

An end-to-end multimodal image captioning web application built with **Streamlit**. The system benchmarks a custom **ResNet152 + BiLSTM + Multi-Head Attention** sequence model (trained on **Flickr8k**) side-by-side against a pretrained **Salesforce BLIP Vision-Language Transformer** (`Salesforce/blip-image-captioning-base`), featuring interactive temperature scaling and word-level prediction confidence metrics.

---

## ✨ Key Features

- **Dual-Model Side-by-Side Comparison:** Generates captions concurrently from the custom CNN-RNN-Attention network and the pretrained Spatial BLIP Transformer.
- **Interactive Decoding Controls:** Dynamically tune **Max Caption Length** (`10–50` tokens, default `34`) and **Sampling Temperature** (`0.1–1.0`, default `0.5`) from the sidebar.
- **Word-by-Word Confidence Inspection:** Displays token-level softmax probability scores (`%`) for every generated word in the custom model.
- **Automated Cloud Weight Fetching:** Automatically downloads the trained `.keras` checkpoint (`~125 MB`) from GitHub Releases (`v1.0`) on startup and caches all models in memory via `@st.cache_resource`.

---

## 🧠 System Architecture & Inference Pipeline

1. **Image Preprocessing & Visual Encoding:**
   - Uploaded images (`JPG`, `JPEG`, `PNG`) are converted to RGB and resized to `224 × 224`.
   - **Custom Branch:** Passed through an ImageNet-pretrained **ResNet152** backbone (penultimate pooling layer) to extract deep visual feature embeddings.
   - **Transformer Branch:** Processed by `BlipProcessor` into Vision Transformer (ViT) patch embeddings.
2. **Sequence Generation & Attention:**
   - **Custom Branch:** Token sequences initialized with `startseq` are padded to `max_length` and fused with visual features inside a **BiLSTM + Multi-Head Attention** decoder until `endseq` is predicted.
   - **Transformer Branch:** `BlipForConditionalGeneration` autoregressively decodes captions (`max_new_tokens=40`) on CPU/CUDA.
3. **Temperature-Scaled Confidence Scoring:**
   - Raw logits/probabilities are scaled using $\text{softmax}(\log(p + \epsilon) / T)$ to inspect token-level confidence across varying temperatures.

| Dimension | Custom Sequence Model | Pretrained Transformer Baseline |
| :--- | :--- | :--- |
| **Visual Encoder** | **ResNet152** (ImageNet pretrained) | **Vision Transformer (ViT)** |
| **Language Decoder** | **BiLSTM + Multi-Head Attention** | **BLIP Conditional Text Decoder** |
| **Training Dataset** | **Flickr8k** (custom tokenizer) | Large-scale web image-text pairs |
| **Checkpoint Size** | `~125 MB` (`model_caption_generator_multihead_15.keras`) | Hugging Face Hub (`blip-image-captioning-base`) |
| **Framework** | TensorFlow / Keras | PyTorch + Hugging Face `transformers` |

---

## 📊 Dataset & Evaluation Benchmarks

- **Dataset:** **Flickr8k** (8,000 images with 5 reference captions per image, tokenized with `startseq` and `endseq` boundary markers).
- **Vocabulary Size:** `[Insert vocab size from tokenizer.pkl, e.g., 8,494]` words | **Max Sequence Length:** `34` tokens.

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ResNet152 + BiLSTM + Multi-Head Attention** | `[XX.X]` | `[XX.X]` | `[XX.X]` | `[XX.X]` | Trained on Flickr8k (15 epochs) |
| **Salesforce BLIP (`base`)** | `[XX.X]` | `[XX.X]` | `[XX.X]` | `[XX.X]` | Zero-shot / pretrained baseline |

*(Update the bracketed BLEU values above with the exact numbers from your evaluation notebook/project report.)*

---

## 📂 Repository Structure

```text
Image-Caption-Generator/
├── app.py                  # Streamlit UI, GitHub Release weight loader & dual inference pipeline
├── tokenizer.pkl           # Serialized Keras Tokenizer fitted on Flickr8k captions
├── requirements.txt        # Python dependencies for local and Streamlit Cloud environments
└── README.md               # Project documentation
```

> **Model Weights Hosting:** `model_caption_generator_multihead_15.keras` is hosted under [GitHub Releases (`v1.0`)](https://github.com/Durgeshshukla209/Image-Caption-Generator/releases/tag/v1.0) and fetched automatically at runtime.

---

## 🖼️ Application Screenshots

*(Drag and drop your Streamlit Cloud screenshots directly into the GitHub editor here to showcase the Caption Comparison and Word-Level Confidence tabs.)*

---

## 🛠️ Local Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Durgeshshukla209/Image-Caption-Generator.git](https://github.com/Durgeshshukla209/Image-Caption-Generator.git)
   cd Image-Caption-Generator
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
   On first launch, `app.py` will automatically download `model_caption_generator_multihead_15.keras` from Release `v1.0` into the working directory.

---

## ☁️ Live Deployment

- **Platform:** Streamlit Community Cloud
- **Repository:** `Durgeshshukla209/Image-Caption-Generator` (`main` branch, `app.py`)
- **Live App:** [Open Live Demo](https://YOUR-SUBDOMAIN.streamlit.app)

---

## 🔮 Future Improvements

- Add **Beam Search decoding** alongside temperature-scaled greedy sampling for higher BLEU-4 fluency.
- Visualize **Multi-Head Attention heatmaps** overlaid directly onto the input image regions.
- Export batch captioning results as downloadable JSON/CSV reports.

---

## 👨‍💻 Author

**Durgesh Shukla**  
GitHub: [@Durgeshshukla209](https://github.com/Durgeshshukla209)
