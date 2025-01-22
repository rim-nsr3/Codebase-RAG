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

# Initialize Pinecone with error handling
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

# Cache the model to prevent reloading
@st.cache_resource
def load_sentence_transformer(model_name="sentence-transformers/all-mpnet-base-v2"):
    return SentenceTransformer(model_name)

def get_huggingface_embeddings(text: str, model=None) -> List[float]:
    """Generate embeddings with timeout and error handling."""
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
    """Fetch files from GitHub with timeout and progress tracking."""
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

def embed_and_store_repo(repo_content: List[str], namespace: str):
    """Embed and store content with progress tracking and error handling."""
    batch_size = 10
    batch = []
    total_files = len(repo_content)
    progress_bar = st.progress(0)

    model = load_sentence_transformer()

    for idx, text in enumerate(repo_content):
        try:
            embedding = get_huggingface_embeddings(text, model)
            vector_id = f"{namespace}-{idx}"
            batch.append({"id": vector_id, "values": embedding, "metadata": {"text": text}})

            if len(batch) >= batch_size:
                pinecone_index.upsert(vectors=batch, namespace=namespace)
                batch = []
            
            progress_bar.progress(min((idx + 1) / total_files, 1.0))
            
        except Exception as e:
            st.error(f"Error embedding file {idx}: {str(e)}")

    if batch:
        try:
            pinecone_index.upsert(vectors=batch, namespace=namespace)
        except Exception as e:
            st.error(f"Error in final batch upload: {str(e)}")
    
    progress_bar.empty()

def perform_rag(query, namespace):
    """Perform Retrieval-Augmented Generation (RAG) with error handling."""
    try:
        raw_query_embedding = get_huggingface_embeddings(query)

        # Query Pinecone
        top_matches = pinecone_index.query(
            vector=raw_query_embedding.tolist(),
            top_k=5,
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

        # Prepare the augmented query
        augmented_query = (
            "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) +
            "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
        )

        system_prompt = """You are an expert Senior Software Engineer.
        Answer any questions I have about the codebase, based on the code provided.
        Always consider all of the context provided when forming a response. Think step by step to not make errors. Act natural in your responses.
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

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the repository!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = perform_rag(prompt, namespace)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
