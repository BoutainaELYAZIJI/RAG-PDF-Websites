# Chat with PDFs and Websites 👨‍🔧

This Streamlit application allows users to interact with PDF documents and website content using conversational AI powered by GroqCloud's generative AI. Users can dynamically select from various supported models to customize their AI experience.

## Screenshots

<img src="https://github.com/BoutainaELYAZIJI/RAG-PDF-Websites/blob/main/image1.png">
<img src="https://github.com/BoutainaELYAZIJI/RAG-PDF-Websites/blob/main/image2.png">

## Features

- **Upload PDF Files**: Users can upload and process multiple PDF documents.
- **Scrape Websites**: Extracts and processes text from a website and its linked pages.
- **Search and Chat**: Users can ask questions related to the uploaded content (PDFs or websites) and receive AI-generated responses using FAISS vector search and GroqCloud's generative AI.
- **Memory-Enhanced Conversations**: Keeps track of chat history to improve the interaction flow.
- **Clear Chat**: Easily reset the conversation with a single button.
- **Dynamic Model Selection**: Choose from a list of supported models provided by GroqCloud to tailor the conversational AI behavior.
<img src="https://github.com/BoutainaELYAZIJI/RAG-PDF-Websites/blob/main/model selection.png">

## Supported Models

Those are one of models that GroqCloud currently supports :

### Production Models
| MODEL ID               | DEVELOPER   | CONTEXT WINDOW (TOKENS) | MAX OUTPUT TOKENS |
|------------------------|-------------|--------------------------|-------------------|
| distil-whisper-large-v3-en | HuggingFace | -                        | -                 | 
| gemma2-9b-it           | Google      | 8,192                    | -                 | 
| llama-3.3-70b-versatile | Meta        | 128k                     | 32,768            |
| llama-3.1-8b-instant   | Meta        | 128k                     | 8,192             | 
| llama-guard-3-8b       | Meta        | 8,192                    | -                 | 


## Requirements

- Python 3.12.3
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
    Or add your GroqCloud API Key in App.py :
    ```makefile
    api_key = "your_groqcloud_api_key"
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
    - **Model Selection**: Choose a GroqCloud model to customize your AI experience.
    - **Clear Chat**: Reset the chat conversation using the "Clear Chat" button.

