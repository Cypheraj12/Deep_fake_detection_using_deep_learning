import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64
import os

# =========================
# ⚙️ CONFIG
# =========================
IMG_SIZE = 128

# 🔊 SET YOUR FAKE DETECTION AUDIO FILE HERE
AUDIO_FAKE = r"C:\Users\Asus\fake\AUD-20260403-WA0010.mp3"  # ← change to your filename
# Set to None to use default beep: AUDIO_FAKE = None

st.set_page_config(
    page_title="Deepfake Detector AI",
    page_icon="🧠",
    layout="centered"
)
st.title("🧠 Deepfake Detection System")
st.markdown("Upload an image and detect whether it is **Real, Fake, or Uncertain** 🚀")

# =========================
# 🎛️ MODEL SELECTION
# =========================
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["MobileNetV2 (Best)", "Custom CNN"]
)
model_path = (
    "deepfake_mobilenetv2.h5"
    if model_choice == "MobileNetV2 (Best)"
    else "deepfake_detector.h5"
)

# =========================
# 📦 LOAD MODEL
# =========================
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(model_path)
st.sidebar.success("Model Loaded ✅")

# =========================
# 📦 LOAD AUDIO ONCE (cached)
# =========================
@st.cache_resource
def load_audio_b64(path):
    if path is None:
        return None
    if not os.path.exists(path):
        st.sidebar.warning(f"Audio file not found: {path}")
        return None
    ext = path.rsplit(".", 1)[-1].lower()
    mime_map = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg"}
    mime_type = mime_map.get(ext, "audio/mpeg")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return b64, mime_type

fake_audio = load_audio_b64(AUDIO_FAKE)

# show audio status in sidebar
st.sidebar.markdown("---")
st.sidebar.header("🔊 Audio Status")
if fake_audio:
    st.sidebar.success("Fake alert audio loaded ✅")
else:
    st.sidebar.info("No audio file — default beep will play")

# =========================
# 🔊 AUTOPLAY AUDIO
# =========================
def autoplay_audio(audio_data=None):
    if audio_data is not None:
        b64, mime_type = audio_data
        src = f"data:{mime_type};base64,{b64}"
        html = f"""
        <script>
        (function() {{
            var src = "{src}";
            function playInFrame() {{
                var a = new Audio(src);
                a.volume = 1.0;
                a.play().catch(function() {{ playViaParent(); }});
            }}
            function playViaParent() {{
                window.parent.postMessage({{ type: 'PLAY_AUDIO', src: src }}, '*');
            }}
            if (!window.parent._audioListenerAdded) {{
                window.parent._audioListenerAdded = true;
                window.parent.addEventListener('message', function(e) {{
                    if (e.data && e.data.type === 'PLAY_AUDIO') {{
                        new Audio(e.data.src).play().catch(function(){{}});
                    }}
                }});
            }}
            playInFrame();
            playViaParent();
        }})();
        </script>
        """
    else:
        # Default beep if no audio file
        html = """
        <script>
        (function() {
            function beep(ctx) {
                [660, 520, 400].forEach(function(freq, i) {
                    var osc = ctx.createOscillator();
                    var gain = ctx.createGain();
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    osc.type = 'sawtooth';
                    osc.frequency.setValueAtTime(freq, ctx.currentTime + i * 0.35);
                    gain.gain.setValueAtTime(0.5, ctx.currentTime + i * 0.35);
                    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + i * 0.35 + 0.3);
                    osc.start(ctx.currentTime + i * 0.35);
                    osc.stop(ctx.currentTime + i * 0.35 + 0.3);
                });
            }
            try {
                var c = new (window.AudioContext || window.webkitAudioContext)();
                c.resume().then(function() { beep(c); });
            } catch(e) {}
            try {
                var p = new (window.parent.AudioContext || window.parent.webkitAudioContext)();
                p.resume().then(function() { beep(p); });
            } catch(e) {}
        })();
        </script>
        """
    st.components.v1.html(html, height=0)

# =========================
# 📤 UPLOAD IMAGE
# =========================
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# 🧠 PREPROCESS
# =========================
def preprocess(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# 🚀 PREDICTION LOGIC
# =========================
def predict_image(img_array):
    prediction = model.predict(img_array)[0][0]
    real_prob = float(prediction)
    fake_prob = float(1 - prediction)

    if real_prob >= 0.75:
        label = "✅ Real Image (High Confidence)"
        status = "REAL"
    elif real_prob <= 0.25:
        label = "❌ Fake Image (High Confidence)"
        status = "FAKE"
    else:
        label = "⚠️ Uncertain Prediction"
        status = "UNCERTAIN"

    return real_prob, fake_prob, label, status

# =========================
# 🎯 UI OUTPUT
# =========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("🔍 Analysis")
        if st.button("Run Detection 🚀"):
            input_data = preprocess(img)
            real_prob, fake_prob, label, status = predict_image(input_data)

            st.markdown("### 📊 Result")
            if status == "REAL":
                st.success(label)
            elif status == "FAKE":
                st.error(label)
                autoplay_audio(fake_audio)   # 🔊 plays your audio file
            else:
                st.warning(label)

            st.markdown("### 📈 Confidence Scores")
            st.write(f"Real Probability: {real_prob:.2f}")
            st.progress(real_prob)
            st.write(f"Fake Probability: {fake_prob:.2f}")
            st.progress(fake_prob)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("⚡ Built using TensorFlow + Streamlit | Deepfake Detection AI")