import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set page configuration
st.set_page_config(page_title="Medical Q&A Assistant", layout="wide")

# Initialize session state
if 'question' not in st.session_state:
    st.session_state.question = ""

@st.cache_resource
def load_model():
    """Load model and tokenizer (cached to avoid reloading)"""
    tokenizer = AutoTokenizer.from_pretrained("lekhana123456/medical-gemma-2b-merged")

    model = AutoModelForCausalLM.from_pretrained(
        "lekhana123456/medical-gemma-2b-merged",
        device_map="auto",              # Automatically selects GPU
        load_in_8bit=True,              # Enable 8-bit loading via bitsandbytes
        torch_dtype=torch.float16       # Efficient dtype for GPU
    )
    return model, tokenizer

def generate_answer(question, model, tokenizer):
    """Generates a medical answer using the fine-tuned model."""
    prompt = (
        "Context: You are a medical expert. Only return the single most appropriate next step.\n"
        f"Question: {question}\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        try:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            answer = generated_text.split("Answer:")[-1].strip()
            return answer
        except Exception as e:
            return f"Error generating response: {str(e)}. Try a different question or restart the application."

# App title and description
st.title("Medical Question Answering")
st.markdown("""
This application uses a fine-tuned Gemma-2B model to answer medical questions.
Enter your medical question below and click 'Generate Answer'.
""")

# Input text area for the question
question = st.text_area("Enter your medical question:", value=st.session_state.question, height=100)

# Button to trigger the generation
if st.button("Generate Answer"):
    if question.strip():
        with st.spinner("Generating answer... This may take a moment on GPU."):
            try:
                model, tokenizer = load_model()
                answer = generate_answer(question, model, tokenizer)
                st.markdown("### Answer:")
                st.markdown(answer)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question first.")

# Sidebar content
st.sidebar.header("About")
st.sidebar.info("""
This app uses a fine-tuned version of the Gemma-2B model for medical Q&A.
Optimized for GPU with bitsandbytes 8-bit loading.
""")

# GPU info in sidebar
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    st.sidebar.success(f"Running on GPU: {gpu_name}")
else:
    st.sidebar.warning("GPU not available. Running on CPU.")

# Example questions
st.sidebar.header("Example Questions")
example_questions = [
    "What is the antidote for opioid overdose?",
    "What are the first-line treatments for hypertension?",
    "How do you diagnose appendicitis?",
    "What are the contraindications for beta blockers?"
]

st.sidebar.markdown("Try one of these examples:")
for example in example_questions:
    if st.sidebar.button(example):
        st.session_state.question = example
        st.rerun()



