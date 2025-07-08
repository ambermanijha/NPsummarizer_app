# To launch this app, run:
# streamlit run streamlit_app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Path to your pretrained T5-small model
MODEL_PATH = r"C:\Users\amber\OneDrive\Desktop\Super Folder\Research\Rakesh_Sir\NepaliLLM\t5-small-pretrained2"

@st.cache_resource
def load_summarizer(model_path):
    """
    Load tokenizer and model, initialize summarization pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    # Use torch to check GPU
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=device,
    )
    return summarizer

# System prompt template for Nepali summarization
SYSTEM_PROMPT = """
तपाईं एक कुशल संक्षेपण मोडेल हुनुहुन्छ। कृपया तलको पाठलाई छोटकरीमा र मुख्य बुँदाहरू मिलाएर नेपालीमा मात्र संक्षेप गर्नुहोस्।
सीमा: मिनिमम {min_len} टोकन, अधिकतम {max_len} टोकन।
"""

st.set_page_config(page_title="Nepali Summarizer", layout="wide")
st.title("🇳🇵 Nepali Text Summarization with T5-small")

# Load pipeline
summarizer = load_summarizer(MODEL_PATH)

# Input text area
text = st.text_area("Enter Nepali text to summarize:", height=300)

# Length controls
col1, col2 = st.columns(2)
with col1:
    min_len = st.slider("Minimum summary length", min_value=10, max_value=100, value=20)
with col2:
    max_len = st.slider("Maximum summary length", min_value=50, max_value=300, value=100)

# Summarize
if st.button("Generate Summary"):
    if not text.strip():
        st.warning("कृपया संक्षेपणको लागि पाठ प्रविष्ट गर्नुहोस्।")
    else:
        prompt = SYSTEM_PROMPT.format(min_len=min_len, max_len=max_len) + text
        with st.spinner("Summarizing..."):
            result = summarizer(
                prompt,
                max_length=max_len,
                min_length=min_len,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
            )
        summary = result[0]['summary_text']
        st.subheader("Summary:")
        st.write(summary)

# Footer
st.markdown("---")
st.caption("Model: Nepali T5-small pretrained | Powered by Streamlit & Hugging Face Transformers")

