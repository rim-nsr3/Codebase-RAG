# Codebase RAG

An AI-powered repository navigator and code explainer that uses Retrieval-Augmented Generation to help you quickly understand any GitHub repository.
...
## How It Works

1. The application fetches files from a GitHub repository using the GitHub API
2. Code files are processed, with special handling for Jupyter notebooks (converted to Python)
3. Text is chunked to maintain context while staying within vector database limits
4. Embeddings are generated using Sentence Transformers and stored in Pinecone
5. User questions are converted to embeddings and used to query the vector database
6. Relevant code snippets are retrieved and sent to the LLM along with the user's question
7. The LLM generates a contextual, code-aware response

## Setup Instructions

### Prerequisites

- Python 3.7+
- A Pinecone API key
- A GROQ API key (for LLama 3.1 access)
- A GitHub personal access token

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
GITHUB_TOKEN=your_github_token
```

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/codebase-rag.git
cd codebase-rag
```

2. Install the required packages:

3. Run the Streamlit app:
```
streamlit run app.py
```

## Usage

1. Enter a GitHub repository URL in the text input
2. Click "Load Repository" to process the repository (this may take a few minutes depending on repository size)
3. Once loaded, ask questions about the codebase in the chat interface
4. The system will retrieve relevant code snippets and provide explanations
# Codebase RAG

An AI-powered repository navigator and code explainer that uses Retrieval-Augmented Generation to help you quickly understand any GitHub repository.
...
## How It Works

1. The application fetches files from a GitHub repository using the GitHub API
2. Code files are processed, with special handling for Jupyter notebooks (converted to Python)
3. Text is chunked to maintain context while staying within vector database limits
4. Embeddings are generated using Sentence Transformers and stored in Pinecone
5. User questions are converted to embeddings and used to query the vector database
6. Relevant code snippets are retrieved and sent to the LLM along with the user's question
7. The LLM generates a contextual, code-aware response

## Setup Instructions

### Prerequisites

- Python 3.7+
- A Pinecone API key
- A GROQ API key (for LLama 3.1 access)
- A GitHub personal access token

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
GITHUB_TOKEN=your_github_token
```

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/codebase-rag.git
cd codebase-rag
```

2. Install the required packages:

3. Run the Streamlit app:
```
streamlit run app.py
```

## Usage

1. Enter a GitHub repository URL in the text input
2. Click "Load Repository" to process the repository (this may take a few minutes depending on repository size)
3. Once loaded, ask questions about the codebase in the chat interface
4. The system will retrieve relevant code snippets and provide explanations
