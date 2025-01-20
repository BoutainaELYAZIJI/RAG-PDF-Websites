# Chat with PDFs and Websites 👨‍🔧

This Streamlit application allows users to interact with PDF documents and website content using conversational AI powered by GroqCloud's generative AI.

## Features

- **Upload PDF Files**: Users can upload and process multiple PDF documents.
- **Scrape Websites**: Extracts and processes text from a website and its linked pages.
- **Search and Chat**: Users can ask questions related to the uploaded content (PDFs or websites) and receive AI-generated responses using FAISS vector search and GroqCloud's generative AI.
- **Memory-Enhanced Conversations**: Keeps track of chat history to improve the interaction flow.
- **Clear Chat**: Easily reset the conversation with a single button.

## Requirements

- Python 3.8+
- Libraries specified in `requirements.txt`
- GroqCloud API Key configured via environment variable `GROQCLOUD_API_KEY`

## Installation and Setup

1. **Clone the repository**:

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:

    Create a `.env` file in the root directory and add your GroqCloud API Key:

    ```makefile
    API_KEY=<your_groqcloud_api_key>
    ```

## Usage

1. **Run the application**:

    ```bash
    streamlit run app.py
    ```

2. **Interact with the application**:

    - **For PDFs**: Upload files using the sidebar and process them to extract and index text.
    - **For Websites**: Enter a URL in the sidebar to scrape and index content.
    - **Chat**: Enter questions in the chat interface to retrieve answers based on the uploaded content.
    - **Clear Chat**: Reset the chat conversation using the "Clear Chat" button.

## About

This project demonstrates the integration of PDF and website content processing, semantic search using FAISS, and conversational AI powered by GroqCloud's API. The intuitive interface is designed for exploring and querying information from multiple sources.

## Credits

- **Streamlit**: Interactive web application framework.
- **PyPDF2**: Library for reading and processing PDF files.
- **BeautifulSoup**: Web scraping and text extraction tool.
- **FAISS**: Efficient vector search library for semantic search.
- **SentenceTransformers**: Library for generating sentence embeddings.
- **GroqCloud Generative AI**: API used for generating AI responses based on user queries and contextual data.

## Dependencies

Here are the main dependencies for the project:

- `streamlit`
- `PyPDF2`
- `faiss-cpu`
- `sentence-transformers`
- `langchain`
- `beautifulsoup4`
- `requests`
- `python-dotenv`
- `scrapy`

Install them using the `requirements.txt` file.
