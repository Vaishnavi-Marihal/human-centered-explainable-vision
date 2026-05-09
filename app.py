import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os

from classifier import load_image, classify
from uncertainty import get_uncertainty_summary
from gradcam import GradCAM
from query_interpreter import interpret_query, get_intent_description
from explanation_engine import ExplanationEngine

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Human-Centered Explainable Vision",
    page_icon="🔍",
    layout="wide",
)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------

st.title("Human-Centered Explainable Vision System")
st.markdown(
    "Upload an image. Ask the model to explain its prediction. "
    "Receive structured visual and textual explanations."
)
st.markdown("---")

# ------------------------------------------------------------------
# Session state initialisation
# ------------------------------------------------------------------

if "classified" not in st.session_state:
    st.session_state.classified = False
if "labels" not in st.session_state:
    st.session_state.labels = None
if "probs" not in st.session_state:
    st.session_state.probs = None
if "indices" not in st.session_state:
    st.session_state.indices = None
if "full_probs" not in st.session_state:
    st.session_state.full_probs = None
if "pil_image" not in st.session_state:
    st.session_state.pil_image = None
if "tensor" not in st.session_state:
    st.session_state.tensor = None

# ------------------------------------------------------------------
# Initialise engine once
# ------------------------------------------------------------------

@st.cache_resource
def get_engine():
    return ExplanationEngine()

engine = get_engine()

# ------------------------------------------------------------------
# Layout: two columns
# ------------------------------------------------------------------

col1, col2 = st.columns([1, 1])

# ------------------------------------------------------------------
# Left column: image upload + classification
# ------------------------------------------------------------------

with col1:
    st.subheader("1. Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Upload any image for classification and explanation"
    )

    if uploaded_file is not None:
        # Save to temp file so load_image can read it
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".jpg"
        ) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load and classify
        pil_image, tensor = load_image(tmp_path)
        labels, probs, indices, full_probs = classify(tensor)
        uncertainty = get_uncertainty_summary(full_probs)

        # Store in session state
        st.session_state.classified = True
        st.session_state.labels = labels
        st.session_state.probs = probs
        st.session_state.indices = indices
        st.session_state.full_probs = full_probs
        st.session_state.pil_image = pil_image
        st.session_state.tensor = tensor

        # Clean up temp file
        os.unlink(tmp_path)

        # Display image
        st.image(pil_image, caption="Uploaded Image", width=400)

        # Display prediction summary
        st.subheader("2. Classification Result")

        band_colors = {
            "high": "🟢",
            "cautious": "🟡",
            "ambiguous": "🔴",
        }

        st.markdown(
            f"**Prediction:** {labels[0]}"
        )
        st.markdown(
            f"**Confidence:** {probs[0]*100:.1f}%"
        )
        st.markdown(
            f"**Uncertainty:** "
            f"{band_colors[uncertainty['band']]} "
            f"{uncertainty['band'].upper()} "
            f"(entropy: {uncertainty['entropy']})"
        )
        st.markdown(
            f"*{uncertainty['message']}*"
        )

        # Top-5 bar chart
        st.subheader("Top-5 Predictions")
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.barh(
            labels[:5][::-1],
            [p * 100 for p in probs[:5][::-1]],
            color="steelblue"
        )
        ax.set_xlabel("Probability (%)")
        ax.set_xlim(0, 100)
        for i, (label, prob) in enumerate(
            zip(labels[:5][::-1], probs[:5][::-1])
        ):
            ax.text(
                prob * 100 + 0.5, i,
                f"{prob*100:.1f}%",
                va="center", fontsize=8
            )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ------------------------------------------------------------------
# Right column: query interface + explanation
# ------------------------------------------------------------------

with col2:
    st.subheader("3. Query the Model")

    if not st.session_state.classified:
        st.info("Upload an image first to enable explanations.")
    else:
        # Example queries
        st.markdown("**Example queries:**")
        examples = [
            "Why did you classify this?",
            "How confident are you?",
            "Which part of the image mattered most?",
            "What else could this be?",
            "Why this instead of the second option?",
        ]
        for ex in examples:
            st.markdown(f"- *{ex}*")

        st.markdown("")

        # Query input
        query = st.text_input(
            "Ask a question about the prediction:",
            placeholder="e.g. Why did you classify this image this way?"
        )

        if query:
            intent, matched_pattern = interpret_query(query)

            # Show detected intent
            st.markdown(
                f"**Detected intent:** `{intent}`  \n"
                f"*{get_intent_description(intent)}*"
            )
            st.markdown("---")

            # Generate explanation
            with st.spinner("Generating explanation..."):
                result = engine.explain(
                    intent,
                    st.session_state.pil_image,
                    st.session_state.tensor,
                    st.session_state.labels,
                    st.session_state.probs,
                    st.session_state.indices,
                    st.session_state.full_probs,
                )

            # Display textual explanation
            st.subheader("Explanation")
            st.markdown(result["text"])

            # Display visual explanation
            if result.get("visual") is not None:
                if intent == "comparison":
                    # Side by side for comparison intent
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(
                            result["visual"],
                            caption=result.get(
                                "visual_label", "Grad-CAM: Top-1"
                            ),
                            width=400,
                        )
                    with c2:
                        if result.get("visual2") is not None:
                            st.image(
                                result["visual2"],
                                caption=result.get(
                                    "visual2_label", "Grad-CAM: Top-2"
                                ),
                                width=400,
                            )
                else:
                    st.image(
                        result["visual"],
                        caption=result.get("visual_label", ""),
                        width=400,
                    )

            # Display bar chart for confidence and alternative intents
            if result.get("extras", {}).get("bar_data") and intent != "confidence":
                bar_data = result["extras"]["bar_data"]
                fig, ax = plt.subplots(figsize=(5, 2.5))
                ax.barh(
                    list(bar_data.keys())[::-1],
                    list(bar_data.values())[::-1],
                    color="steelblue"
                )
                ax.set_xlabel("Probability (%)")
                ax.set_xlim(0, 100)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------

st.markdown("---")
st.markdown(
    "**Human-Centered Explainable Vision System** — "
    "Built to investigate the gap between model confidence "
    "and human interpretability. "
    "Grad-CAM implemented from Selvaraju et al. (2017)."
)