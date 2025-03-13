import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from github import Github
from nbconvert import PythonExporter
import nbformat
import json
import os
from dotenv import load_dotenv
import concurrent.futures
import time
from typing import List, Optional

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone vector database with error handling
# Initialize Pinecone vector database with error handling
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "codebase-rag"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="gcp", region="us-east-1")
        )
    pinecone_index = pc.Index(index_name)
except Exception as e:
    st.error(f"Failed to initialize Pinecone: {str(e)}")
    pinecone_index = None

SUPPORTED_EXTENSIONS = {
    '.py', '.js', '.tsx', '.jsx', '.java',
    '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h', '.md', '.txt', '.ipynb'
}

# Cache the sentence transformer model to prevent reloading on each use
# Cache the sentence transformer model to prevent reloading on each use
@st.cache_resource
def load_sentence_transformer(model_name="sentence-transformers/all-mpnet-base-v2"):
    return SentenceTransformer(model_name)

def get_huggingface_embeddings(text: str, model=None) -> List[float]:
    """
    Generates embeddings for the given text using a sentence transformer model.
    Includes timeout and error handling for reliability.
    
    Args:
        text (str): The text to generate embeddings for
        model (SentenceTransformer, optional): Pre-loaded model instance
    
    Returns:
        List[float]: Vector embedding of the input text
        
    Raises:
        TimeoutError: If embedding generation takes too long
        Exception: For other embedding generation failures
    """
    """
    Generates embeddings for the given text using a sentence transformer model.
    Includes timeout and error handling for reliability.
    
    Args:
        text (str): The text to generate embeddings for
        model (SentenceTransformer, optional): Pre-loaded model instance
    
    Returns:
        List[float]: Vector embedding of the input text
        
    Raises:
        TimeoutError: If embedding generation takes too long
        Exception: For other embedding generation failures
    """
    if model is None:
        model = load_sentence_transformer()
    
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(model.encode, text)
            return future.result(timeout=30)  # 30 second timeout
    except concurrent.futures.TimeoutError:
        raise TimeoutError("Embedding generation timed out")
    except Exception as e:
        raise Exception(f"Failed to generate embeddings: {str(e)}")

def fetch_repo_content(repo_url: str, github_token: str, file_limit: int = 50) -> List[str]:
    """
    Fetches and processes files from a GitHub repository with progress tracking.
    Handles multiple file types and includes error handling for robustness.
    
    Args:
        repo_url (str): URL of the GitHub repository to process
        github_token (str): GitHub authentication token
        file_limit (int): Maximum number of files to process
    
    Returns:
        List[str]: List of file contents as strings
        
    Raises:
        ValueError: If GitHub token is missing
        Exception: For repository access or processing failures
    """
    """
    Fetches and processes files from a GitHub repository with progress tracking.
    Handles multiple file types and includes error handling for robustness.
    
    Args:
        repo_url (str): URL of the GitHub repository to process
        github_token (str): GitHub authentication token
        file_limit (int): Maximum number of files to process
    
    Returns:
        List[str]: List of file contents as strings
        
    Raises:
        ValueError: If GitHub token is missing
        Exception: For repository access or processing failures
    """
    if not github_token:
        raise ValueError("GitHub token is not provided")

    try:
        g = Github(github_token)
        repo_name = repo_url.split("github.com/")[1]
        repo = g.get_repo(repo_name)
    except Exception as e:
        raise Exception(f"Failed to access repository: {str(e)}")

    content = []
    count = 0
    progress_bar = st.progress(0)

    def fetch_folder(folder_path=""):
        """
        Recursive helper function to traverse repository folders and fetch file contents.
        
        Args:
            folder_path (str): Current folder path in the repository
        """
        """
        Recursive helper function to traverse repository folders and fetch file contents.
        
        Args:
            folder_path (str): Current folder path in the repository
        """
        nonlocal count
        try:
            files = repo.get_contents(folder_path)
            for file in files:
                if count >= file_limit:
                    return
                
                if file.type == "file" and any(file.name.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    try:
                        if file.name.endswith(".ipynb"):
                            notebook_data = file.decoded_content.decode("utf-8")
                            python_code = convert_notebook_to_python(notebook_data)
                            content.append(python_code)
                        else:
                            content.append(file.decoded_content.decode("utf-8"))
                        count += 1
                        progress_bar.progress(min(count / file_limit, 1.0))
                    except Exception as e:
                        st.warning(f"Skipping file {file.path}: {str(e)}")
                elif file.type == "dir":
                    fetch_folder(file.path)
        except Exception as e:
            st.warning(f"Error accessing {folder_path}: {str(e)}")

    try:
            fetch_folder()
    finally:
        progress_bar.empty()

    if not content:
        raise Exception("No valid files found in repository")

    return content

def convert_notebook_to_python(notebook_data):
    """Convert Jupyter notebook to Python code."""
    notebook = nbformat.reads(notebook_data, as_version=4)
    python_exporter = PythonExporter()
    python_code, _ = python_exporter.from_notebook_node(notebook)
    return python_code

def chunk_text(text, max_chunk_size=30000, overlap=500):
    """
    Split text into chunks that fit within Pinecone's metadata size limits.
    
    Args:
        text (str): The text to chunk
        max_chunk_size (int): Maximum size of each chunk in characters
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Look for a good breaking point (newline) to avoid cutting in the middle of a line
        breakpoint = text.rfind('\n', start + max_chunk_size - overlap, end)
        if breakpoint == -1:  # No good breakpoint found, use the maximum size
            chunks.append(text[start:end])
            start = end - overlap
        else:
            chunks.append(text[start:breakpoint])
            start = breakpoint
    
    return chunks

def embed_and_store_repo(repo_content: List[str], namespace: str):
    """
    Processes repository content by generating embeddings and storing them in Pinecone.
    Handles large files by chunking them before storage.
    
    Args:
        repo_content (List[str]): List of file contents to process
        namespace (str): Namespace for storing vectors in Pinecone
    """
    batch_size = 10
    batch = []
    total_chunks = 0  # We'll count chunks instead of files
    
    # First, estimate the total number of chunks
    for text in repo_content:
        total_chunks += len(chunk_text(text))
    
    progress_bar = st.progress(0)
    chunk_count = 0
    model = load_sentence_transformer()

    for file_idx, text in enumerate(repo_content):
        # Split large files into chunks
        chunks = chunk_text(text)
        
        for chunk_idx, chunk in enumerate(chunks):
            try:
                embedding = get_huggingface_embeddings(chunk, model)
                vector_id = f"{namespace}-file{file_idx}-chunk{chunk_idx}"
                
                # Add metadata about chunking for better retrieval
                metadata = {
                    "text": chunk,
                    "file_index": file_idx,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks)
                }
                
                batch.append({"id": vector_id, "values": embedding, "metadata": metadata})

                if len(batch) >= batch_size:
                    pinecone_index.upsert(vectors=batch, namespace=namespace)
                    batch = []
                
                chunk_count += 1
                progress_bar.progress(min(chunk_count / total_chunks, 1.0))
                
            except Exception as e:
                st.error(f"Error embedding file {file_idx}, chunk {chunk_idx}: {str(e)}")

    if batch:
        try:
            pinecone_index.upsert(vectors=batch, namespace=namespace)
        except Exception as e:
            st.error(f"Error in final batch upload: {str(e)}")
    
    progress_bar.empty()

def perform_rag(query, namespace):
    """
    Performs Retrieval-Augmented Generation to answer queries about the codebase.
    Combines vector search with LLM processing for intelligent responses.
    
    Args:
        query (str): User's question about the codebase
        namespace (str): Namespace to search in Pinecone
    
    Returns:
        str: Generated response based on relevant code context
    """
    """
    Performs Retrieval-Augmented Generation to answer queries about the codebase.
    Combines vector search with LLM processing for intelligent responses.
    
    Args:
        query (str): User's question about the codebase
        namespace (str): Namespace to search in Pinecone
    
    Returns:
        str: Generated response based on relevant code context
    """
    try:
        raw_query_embedding = get_huggingface_embeddings(query)

        # Query Pinecone
        top_matches = pinecone_index.query(
            vector=raw_query_embedding.tolist(),
            top_k=5,  # Increased to 5 to get more context chunks
            include_metadata=True,
            namespace=namespace
        )

        # Safely retrieve matched contexts
        contexts = [
            item['metadata']['text'] for item in top_matches['matches']
            if 'metadata' in item and 'text' in item['metadata']
        ]

        if not contexts:
            return "No relevant context found for your query."

        # Limit total context length
        max_total_context = 8000  # Adjust as needed for your LLM
        limited_contexts = []
        current_length = 0
        
        for ctx in contexts:
            # Skip duplicate chunks (can happen with overlapping chunks)
            if ctx in limited_contexts:
                continue
                
            ctx_length = len(ctx)
            if current_length + ctx_length > max_total_context:
                # If adding this chunk would exceed our limit, we'll truncate it
                available_space = max_total_context - current_length
                if available_space > 500:  # Only add if we have reasonable space
                    limited_contexts.append(ctx[:available_space] + "...")
                break
            else:
                limited_contexts.append(ctx)
                current_length += ctx_length

        # Prepare the augmented query
        augmented_query = (
            "<CONTEXT>\n" + "\n\n-------\n\n".join(limited_contexts) +
            "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
        )

        system_prompt = """You are an expert Senior Software Engineer.
        Answer any questions I have about the codebase, based on the code provided.
        Always consider all of the context provided when forming a response. If the context doesn't contain enough information, say so.
        Be concise and focus on the most relevant parts of the code to answer the question.
        """

        llm_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )

        return llm_response.choices[0].message.content

    except Exception as e:
        return f"Error performing RAG: {str(e)}"

# Initialize OpenAI Client
try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY")
    )
except Exception as e:
    st.error(f"Failed to initialize Groq client: {str(e)}")
    client = None

# Streamlit Frontend
st.title("Database Chatbot")

repo_url = st.text_input("Enter GitHub Repository URL:")
namespace = repo_url

# Repository loading and processing interface
# Repository loading and processing interface
if st.button("Load Repository"):
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        st.error("GitHub token is not configured.")
    elif not repo_url:
        st.error("Please provide a GitHub repository URL.")
    else:
        try:
            with st.spinner("🔄 Fetching repository content..."):
                repo_content = fetch_repo_content(repo_url, github_token)

            with st.spinner("🛠️ Embedding repository content..."):
                embed_and_store_repo(repo_content, namespace)

            st.success("✅ Repository content has been processed!")
            
        except Exception as e:
            st.error(f"Failed to process repository: {str(e)}")

# Chat interface initialization and management
# Chat interface initialization and management
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat history

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
# Handle new user input
if prompt := st.chat_input("Ask about the repository!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = perform_rag(prompt, namespace)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

