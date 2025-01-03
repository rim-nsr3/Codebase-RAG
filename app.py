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

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone
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

SUPPORTED_EXTENSIONS = {
    '.py', '.js', '.tsx', '.jsx', '.java',
    '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h', '.md', '.txt', '.ipynb'
}

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    """Generate embeddings for a given text."""
    model = SentenceTransformer(model_name)
    return model.encode(text)

def fetch_repo_content(repo_url, github_token, file_limit=50):
    """Fetch files from a GitHub repository."""
    g = Github(github_token)
    repo_name = repo_url.split("github.com/")[1]
    repo = g.get_repo(repo_name)

    content = []
    count = 0

    def fetch_folder(folder_path=""):
        nonlocal count
        for file in repo.get_contents(folder_path):
            if count >= file_limit:
                break
            if file.type == "file" and any(file.name.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                try:
                    if file.name.endswith(".ipynb"):
                        # Convert notebook to Python
                        notebook_data = file.decoded_content.decode("utf-8")
                        python_code = convert_notebook_to_python(notebook_data)
                        content.append(python_code)
                    else:
                        content.append(file.decoded_content.decode("utf-8"))
                    count += 1
                except Exception as e:
                    st.warning(f"Skipping file {file.path}: {str(e)}")
            elif file.type == "dir":
                fetch_folder(file.path)

    fetch_folder()
    return content

def convert_notebook_to_python(notebook_data):
    """Convert Jupyter notebook to Python code."""
    notebook = nbformat.reads(notebook_data, as_version=4)
    python_exporter = PythonExporter()
    python_code, _ = python_exporter.from_notebook_node(notebook)
    return python_code

def embed_and_store_repo(repo_content, namespace):
    """Embed and store content in Pinecone in batches."""
    batch_size = 10
    batch = []

    for idx, text in enumerate(repo_content):
        try:
            embedding = get_huggingface_embeddings(text)
            vector_id = f"{namespace}-{idx}"
            batch.append({"id": vector_id, "values": embedding, "metadata": {"text": text}})

            if len(batch) >= batch_size:
                pinecone_index.upsert(vectors=batch, namespace=namespace)
                batch = []
        except Exception as e:
            st.error(f"Error embedding file {idx}: {str(e)}")

    if batch:
        pinecone_index.upsert(vectors=batch, namespace=namespace)

def perform_rag(query, namespace):
    """Perform Retrieval-Augmented Generation (RAG)."""
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

    # Prepare the augmented query
    augmented_query = (
        "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) +
        "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
    )

    system_prompt = """You are a Senior Software Engineer.
    Answer any questions I have about the codebase, based on the code provided.
    Always consider all of the context provided when forming a response.
    """

    llm_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return llm_response.choices[0].message.content

# Initialize OpenAI Client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# Streamlit Frontend
st.title("Database Chatbot")

repo_url = st.text_input("Enter GitHub Repository URL:")
namespace = repo_url

if st.button("Load Repository"):
    github_token = os.getenv("GITHUB_TOKEN")
    if not repo_url:
        st.error("Please provide a GitHub repository URL.")
    else:
        with st.spinner("🔄 Fetching repository content..."):
            repo_content = fetch_repo_content(repo_url, github_token)

        with st.spinner("🛠️ Embedding repository content..."):
            embed_and_store_repo(repo_content, namespace)

        st.success("✅ Repository content has been processed!")

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
