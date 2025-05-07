import streamlit as st
import os
import torch
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Constants
FAISS_SAVE_PATH = "faiss_index_uploaded_data"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TINYLLAMA_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Change to TinyLlama

# Set page configuration
st.set_page_config(page_title="The Diplomat", layout="wide")

# Hide Streamlit menu and footer
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Main app title
st.title("The Diplomat Chatbot")
st.markdown("Ask questions about the documents mentioned. The system will retrieve relevant context and give you an answer.")

@st.cache_resource
def load_embeddings_model(model_name, device):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
        return embeddings
    except Exception as e:
        st.error(f"🚨 Error loading embedding model: {e}")
        st.stop()

@st.cache_resource
def load_faiss_index(save_path, _embeddings):
    if not os.path.exists(save_path):
        st.error(f"🚨 Error: FAISS index directory '{save_path}' not found.")
        st.error("Please ensure the data vectorization script ran successfully and created the index in the correct location.")
        st.stop()
    try:
        db = FAISS.load_local(
            save_path,
            _embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={"k": 3})
        return retriever
    except Exception as e:
        st.error(f"🚨 Error loading FAISS index: {e}")
        st.error("Ensure the embedding model used here matches the one used for creating the index.")
        st.stop()

@st.cache_resource
def load_tinyllama_model(model_name, device):
    """Load the TinyLlama model from Hugging Face."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Display loading message
        with st.spinner("Loading TinyLlama model... This may take a moment."):
            # Configure model loading with appropriate parameters for TinyLlama
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True
            )
        
        # Create a text generation pipeline
        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.95,
            repetition_penalty=1.15,
            device=0 if device == "cuda" else -1
        )
        
        # Wrap the pipeline in a LangChain HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        
        return llm
    except Exception as e:
        st.error(f"🚨 Error initializing TinyLlama model: {e}")
        st.error("Please check your internet connection and ensure you have enough system resources.")
        st.stop()

def get_rag_chain(_retriever, _llm):
    template = """
    <|system|>
    You are an assistant for question-answering tasks.
    Use only the following pieces of retrieved context to answer the question.
    If you don't know the answer from the context, just say that you don't know.
    Do not make up an answer. Keep the answer concise.
    </|system|>
    
    <|user|>
    Context:
    {context}
    
    Question:
    {question}
    </|user|>
    
    <|assistant|>
    """
    # Note: Adjusted the template to match TinyLlama's chat format
    
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": _retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | _llm
        | StrOutputParser()
    )
    return rag_chain

# Device detection for GPU/CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    st.sidebar.success(f"⚡ Using GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
else:
    st.sidebar.info("💻 Using CPU - TinyLlama will run slower without GPU acceleration")

# Memory info if using GPU
if device == "cuda":
    with st.sidebar:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        st.info(f"GPU Memory: {gpu_memory:.2f} GB")
        st.info("Note: TinyLlama-1.1B requires approximately 2GB VRAM")

# Load models with progress indicators
with st.spinner("Loading models..."):
    # Step 1: Load embedding model
    with st.status("Loading embedding model...", expanded=False) as status:
        embeddings_model = load_embeddings_model(EMBEDDING_MODEL_NAME, device)
        status.update(label="✅ Embedding model loaded", state="complete")
    
    # Step 2: Load FAISS index
    with st.status("Loading vector database...", expanded=False) as status:
        retriever = load_faiss_index(FAISS_SAVE_PATH, embeddings_model)
        status.update(label="✅ Vector database loaded", state="complete")
    
    # Step 3: Load TinyLlama model
    with st.status("Loading TinyLlama model...", expanded=False) as status:
        llm = load_tinyllama_model(TINYLLAMA_MODEL_NAME, device)
        status.update(label="✅ TinyLlama model loaded", state="complete")

# Create RAG chain
if retriever and llm:
    rag_chain = get_rag_chain(retriever, llm)
    st.success("✅ All models loaded successfully!")
else:
    st.error("🚨 RAG chain could not be initialized due to previous errors.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a welcome message
    welcome_message = "Welcome to The Diplomat Chatbot powered by TinyLlama! Ask me questions about the documents, and I'll try to help."
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and response
if query := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            with st.spinner("TinyLlama is thinking... 🤔"):
                # Show a progress bar to indicate processing
                progress_bar = st.progress(0)
                for i in range(100):
                    # Simulate thinking process
                    progress_bar.progress(i + 1)
                    if i < 90:  # Make the last 10% wait for the actual response
                        time.sleep(0.01)
                
                # Get actual response
                answer = rag_chain.invoke(query)
                full_response = answer
                
                # Clean up TinyLlama's output (remove any trailing conversation markers)
                if "</|assistant|>" in full_response:
                    full_response = full_response.split("</|assistant|>")[0]

            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"🚨 An error occurred: {e}")
            full_response = "Sorry, I encountered an error while processing your request. TinyLlama might need more resources or there might be an issue with the query."
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar content
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    """
    This app uses a RAG (Retrieval-Augmented Generation) pipeline with:
    
    - **Model**: TinyLlama 1.1B Chat
    - **Embeddings**: all-MiniLM-L6-v2
    - **Vector Store**: FAISS
    
    TinyLlama is a compact but powerful language model that can run on modest hardware.
    """
)

# Add model information
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.markdown("""
- **TinyLlama**: 1.1 billion parameters
- **Context Length**: 2048 tokens
- **Training**: Fine-tuned for chat applications
- **GitHub**: [TinyLlama Project](https://github.com/jzhang38/TinyLlama)
""")

# Add a reset button to clear chat history
if st.sidebar.button("Reset Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# Add a missing import at the top
import time
