import streamlit as st
import pypdf
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import time

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📚💬", layout="wide")

st.title("📚💬 PDF Q&A Chatbot")
st.write("Upload a PDF document, and then ask questions about its content.")

# --- Configuration ---
# You can choose a different sentence transformer model.
# 'all-MiniLM-L6-v2' is small and fast, good for local.
# 'all-mpnet-base-v2' is larger and more accurate.
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 500  # Characters per text chunk
CHUNK_OVERLAP = 50 # Overlap between chunks to maintain context
RELEVANCE_THRESHOLD = 0.65 # Adjust this value (0 to 1) for how strict the "relevant" match is.
                            # Lower value = more matches, potentially less accurate.
                            # Higher value = fewer matches, more precise.

# >>>>>> IMPORTANT FIX: Define MAX_ALLOWED_DISTANCE here <<<<<<
MAX_ALLOWED_DISTANCE = 1.2 # Adjust this based on experimentation for your model
# >>>>>> END OF IMPORTANT FIX <<<<<<

# --- Session State Initialization ---
if "pdf_text_chunks" not in st.session_state:
    st.session_state.pdf_text_chunks = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "embedding_model" not in st.session_state:
    # Load model only once
    st.session_state.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please upload a PDF to start answering questions."}]

# --- PDF Processing Functions ---

@st.spinner("Extracting text from PDF...")
def extract_text_from_pdf(uploaded_file):
    reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

@st.spinner("Chunking text...")
def chunk_text(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

@st.spinner("Generating embeddings and building index...")
def process_pdf_for_qa(uploaded_file):
    full_text = extract_text_from_pdf(uploaded_file)
    if not full_text.strip():
        st.error("Could not extract text from the PDF. It might be an image-only PDF or corrupted.")
        return [], None

    chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        st.error("No text chunks generated. PDF might be empty or problematic.")
        return [], None

    st.session_state.pdf_text_chunks = chunks

    # Generate embeddings for all chunks
    embeddings = st.session_state.embedding_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32') # FAISS requires float32

    # Create a FAISS index
    dimension = embeddings.shape[1] # Dimension of the embeddings
    index = faiss.IndexFlatL2(dimension) # L2 distance (Euclidean) for similarity
    index.add(embeddings)

    st.session_state.faiss_index = index
    st.success(f"PDF processed! Loaded {len(chunks)} text chunks.")
    return chunks, index

# --- Chatbot Response Logic ---

def get_answer_from_pdf(query):
    if not st.session_state.faiss_index:
        return "Please upload and process a PDF first."

    query_embedding = st.session_state.embedding_model.encode([query]).astype('float32')

    # Search the FAISS index for the most similar chunks
    # k=3 means retrieve the top 3 most similar chunks
    distances, indices = st.session_state.faiss_index.search(query_embedding, k=3)

    relevant_chunks = []
    highest_score = 0
    
    # We'll use a threshold directly on the distance.
    # Empirically, for 'all-MiniLM-L6-v2' embeddings, L2 distance:
    # < 0.5 - 0.7: Very close semantic match
    # 0.7 - 1.0: Good semantic match
    # 1.0 - 1.3: Moderate semantic match
    # > 1.3: Weak or no semantic match

    for i in range(len(indices[0])):
        idx = indices[0][i]
        dist = distances[0][i]

        if dist < MAX_ALLOWED_DISTANCE: # Only consider chunks below the distance threshold
            relevant_chunks.append(st.session_state.pdf_text_chunks[idx])
            if (MAX_ALLOWED_DISTANCE - dist) > highest_score: # Simple way to track highest "relevance"
                highest_score = (MAX_ALLOWED_DISTANCE - dist)

    if relevant_chunks:
        combined_context = "\n\n".join(relevant_chunks)
        
        return (f"Based on the document, here's what I found:\n\n"
                f"\"...{combined_context[:500]}...\"" # Show a snippet of context
                f"\n\n**(Relevance score: {highest_score/(MAX_ALLOWED_DISTANCE):.2f})**" # Display a derived relevance score
                f"\n\n(Note: For more complex answers, a larger language model would be needed.)")
    else:
        return ("I couldn't find a direct answer to your question in the uploaded document. "
                "Please rephrase your question, or consider asking higher authorities for assistance. "
                f"*(Lowest distance found: {distances[0][0]:.2f}, which is above threshold.)*")


# --- Streamlit UI Layout ---

# Sidebar for PDF upload and info
with st.sidebar:
    st.header("1. Upload PDF Document")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_pdf is not None:
        if st.button("Process PDF"):
            st.session_state.pdf_text_chunks = []
            st.session_state.faiss_index = None
            st.session_state.messages = [{"role": "assistant", "content": "Hello! Please upload a PDF to start answering questions."}] # Clear chat history

            chunks, index = process_pdf_for_qa(uploaded_pdf)
            if index is not None:
                st.session_state.faiss_index = index
                st.session_state.pdf_text_chunks = chunks
                st.session_state.messages.append({"role": "assistant", "content": f"PDF '{uploaded_pdf.name}' processed! You can now ask questions."})
            else:
                 st.session_state.messages.append({"role": "assistant", "content": f"Failed to process PDF '{uploaded_pdf.name}'. Please try another file."})

    st.markdown("---")
    st.header("2. How it Works")
    st.markdown("""
    This bot extracts text from your PDF, splits it into chunks, and converts each chunk into numerical "embeddings" using a pre-trained language model.
    
    When you ask a question, your question is also converted into an embedding. The bot then finds the most semantically similar chunks in the PDF's index using FAISS.
    
    If relevant content is found, it provides the snippet. Otherwise, it suggests escalating.
    """)
    st.markdown("---")
    st.info(f"**Embedding Model:** {EMBEDDING_MODEL_NAME}\n\n**Chunk Size:** {CHUNK_SIZE} chars\n\n**Relevance Threshold (FAISS L2 Distance Max):** {MAX_ALLOWED_DISTANCE}")


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the PDF..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.spinner("Thinking..."):
        bot_response = get_answer_from_pdf(prompt)

    # Display bot response
    with st.chat_message("assistant"):
        # Simulate typing for a better user experience
        message_placeholder = st.empty()
        full_response = ""
        for chunk in bot_response.split():
            full_response += chunk + " "
            time.sleep(0.02) # Adjust typing speed
            message_placeholder.markdown(full_response + "▌") # Typing cursor
        message_placeholder.markdown(full_response) # Final message

    # Add bot message to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})