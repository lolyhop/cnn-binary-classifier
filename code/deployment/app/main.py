import base64
import io
import os
import typing as tp

import requests
import streamlit as st
from PIL import Image

API_URL: str = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Cats vs Dogs Classifier", page_icon="🐱🐶", layout="centered"
)


def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def predict_image(image: Image.Image) -> tp.Dict[str, tp.Any]:
    try:
        img_base64: str = encode_image_to_base64(image)
        payload: tp.Dict[str, str] = {"image": img_base64}

        response = requests.post(f"{API_URL}/predict", json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API request failed with status {response.status_code}"}

    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to the API"}
    except Exception as e:
        return {"error": str(e)}


def render_result(result: tp.Dict[str, tp.Any]) -> None:
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return

    probabilities: tp.Dict[str, float] = result.get("probabilities", {})
    prediction: str = result.get("prediction", "Unknown")
    confidence: float = max(probabilities.values()) if probabilities else 0.0

    st.markdown("<br>", unsafe_allow_html=True)

    emoji: str = "🐱" if prediction.lower() == "cat" else "🐶"
    animal: str = "Cat" if prediction.lower() == "cat" else "Dog"

    st.markdown(
        f"<h2 style='text-align: center;'>{emoji} It's a {animal}!</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<h3 style='text-align: center;'>Confidence: {confidence:.1%}</h3>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(
        "<h1 style='text-align: center;'>🐱🐶 Cats vs Dogs Classifier</h1>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    _, col_center, _ = st.columns([1, 1, 1])
    with col_center:
        upload_button: bool = st.button(
            "Upload Image", type="primary", use_container_width=True
        )

    if uploaded_file is not None and upload_button:
        image: Image.Image = Image.open(uploaded_file)

        with st.spinner("Analyzing image..."):
            image_resized: Image.Image = image.convert("RGB").resize((224, 224))
            result: tp.Dict[str, tp.Any] = predict_image(image_resized)

        st.markdown("<br><br>", unsafe_allow_html=True)

        _, col_center, _ = st.columns([1, 2, 1])
        with col_center:
            st.image(image, use_container_width=True)
            render_result(result)


if __name__ == "__main__":
    main()
