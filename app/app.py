import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="AI Health Diagnosis", layout="wide")

# ---------------- DARK THEME STYLE ---------------- #
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.markdown("""
<h1 style='text-align:center; color:#00ADB5;'>🧠 AI Alzheimer’s Diagnosis System</h1>
<p style='text-align:center; color:gray;'>Upload MRI scan for AI-based detection with explainability</p>
""", unsafe_allow_html=True)

# ---------------- PATIENT INPUT ---------------- #
st.sidebar.header("👤 Patient Details")
patient_name = st.sidebar.text_input("Enter Patient Name")
patient_age = st.sidebar.number_input("Enter Age", min_value=1, max_value=120, step=1)

# ---------------- LOAD MODEL ---------------- #
MODEL_PATH = "artifacts/model.h5"

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Train model first.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- GRAD-CAM ---------------- #
def get_gradcam(model, image):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer("last_conv").output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

# ---------------- FILE UPLOAD ---------------- #
uploaded_file = st.file_uploader("📤 Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼 Original MRI")
        st.image(image, width='stretch')

    # Preprocess
    resized = cv2.resize(image, (128,128))
    normalized = resized / 255.0
    input_image = np.expand_dims(normalized, axis=0)

    # Prediction
    prediction = model.predict(input_image)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    classes = ["Non-Demented", "Moderate Demented"]
    result = classes[class_idx]

    # Grad-CAM
    heatmap = get_gradcam(model, input_image)
    gradcam_img = overlay_heatmap(resized, heatmap)

    with col2:
        st.subheader("🔥 Model Attention (Grad-CAM)")
        st.image(gradcam_img, width='stretch')

    # ---------------- RESULT CARD ---------------- #
    st.markdown(f"""
    <div style="background-color:#1f2937;padding:20px;border-radius:12px">
        <h2 style="color:#00ADB5;">Prediction: {result}</h2>
        <h3 style="color:white;">Confidence: {confidence:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- REPORT CONTENT ---------------- #
    if result == "Non-Demented":
        diagnosis = "No significant signs of Alzheimer’s detected."
        recommendation = "Maintain healthy lifestyle and routine checkups."
    else:
        diagnosis = "Signs of Moderate Alzheimer’s detected."
        recommendation = "Consult neurologist immediately."

    report_text = f"""
AI HEALTH DIAGNOSIS REPORT

Patient Name: {patient_name}
Age: {patient_age}

Prediction: {result}
Confidence: {confidence:.2f}%

Diagnosis:
{diagnosis}

Recommendation:
{recommendation}

Note:
This AI system assists diagnosis but does not replace medical professionals.
"""

    st.subheader("📄 Diagnosis Report")
    st.text(report_text)

    # ---------------- PDF GENERATION ---------------- #
    def create_pdf(text):
        file_path = "report.pdf"
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()

        content = []
        for line in text.split("\n"):
            content.append(Paragraph(line, styles["Normal"]))
            content.append(Spacer(1, 10))

        doc.build(content)

        return file_path

    pdf_file = create_pdf(report_text)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📥 Download PDF Report",
            data=f,
            file_name="Alzheimer_Report.pdf",
            mime="application/pdf"
        )