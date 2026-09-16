# ============================================================
# app.py
# Streamlit UI for the Skin Cancer Classification model
# ============================================================

import streamlit as st
from PIL import Image

from inference import (
    create_model, predict, preprocess_image,
    CLASS_LABELS, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES, DEVICE
)
from explainability import generate_gradcam, generate_shap_analysis

st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🩺",
    layout="centered"
)


@st.cache_resource
def load_model():
    return create_model()


st.title("🩺 Skin Cancer Classification")
st.write(
    "Upload a dermatoscopic image to classify it into one of 7 HAM10000 "
    "lesion types using a ResNet18 model."
)

model = load_model()

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width="stretch")

    with st.spinner("Running inference..."):
        result = predict(image, model=model)

    st.subheader(f"Prediction: {result['label']} ({result['class']})")
    st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")

    st.write("### Class probabilities")
    sorted_probs = dict(
        sorted(
            result["probabilities"].items(),
            key=lambda item: item[1],
            reverse=True
        )
    )
    st.bar_chart({CLASS_LABELS[k]: v for k, v in sorted_probs.items()})

    for class_name, prob in sorted_probs.items():
        st.write(f"**{CLASS_LABELS[class_name]}**: {prob * 100:.2f}%")

    st.write("### Explainability")
    _, tensor = preprocess_image(image)
    predicted_idx = CLASS_TO_IDX[result["class"]]

    gradcam_tab, shap_tab = st.tabs(["Grad-CAM", "SHAP"])

    with gradcam_tab:
        if st.button("Generate Grad-CAM heatmap"):
            with st.spinner("Computing Grad-CAM..."):
                target_layer = model.backbone.layer4[-1]
                overlay, _ = generate_gradcam(
                    model, image, tensor, target_layer,
                    target_class=predicted_idx
                )
            st.image(
                overlay,
                caption="Grad-CAM: regions influencing the prediction",
                width="stretch"
            )

    with shap_tab:
        st.caption("SHAP can take longer to compute than Grad-CAM.")
        if st.button("Generate SHAP heatmap"):
            class_codes = [IDX_TO_CLASS[i] for i in range(NUM_CLASSES)]
            with st.spinner("Computing SHAP values..."):
                overlay, _, class_scores = generate_shap_analysis(
                    model, image, tensor, DEVICE, class_codes,
                    target_class=predicted_idx
                )
            st.image(
                overlay,
                caption="SHAP: pixel contribution to the prediction",
                width="stretch"
            )

            st.write("#### SHAP class contribution")
            st.caption(
                "Positive = evidence supports that class, "
                "negative = evidence argues against it."
            )
            sorted_scores = dict(
                sorted(
                    class_scores.items(),
                    key=lambda item: item[1],
                    reverse=True
                )
            )
            st.bar_chart({
                CLASS_LABELS[code]: score
                for code, score in sorted_scores.items()
            })

    st.caption(
        "⚠️ This tool is for educational purposes only and is not a "
        "substitute for professional medical diagnosis."
    )
