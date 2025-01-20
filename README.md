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

## Supported Models

GroqCloud currently supports the following models:

### Production Models
| MODEL ID               | DEVELOPER   | CONTEXT WINDOW (TOKENS) | MAX OUTPUT TOKENS | MODEL CARD LINK  |
|------------------------|-------------|--------------------------|-------------------|------------------|
| distil-whisper-large-v3-en | HuggingFace | -                        | -                 | [Card](#)        |
| gemma2-9b-it           | Google      | 8,192                    | -                 | [Card](#)        |
| llama-3.3-70b-versatile | Meta        | 128k                     | 32,768            | [Card](#)        |
| llama-3.1-8b-instant   | Meta        | 128k                     | 8,192             | [Card](#)        |
| llama-guard-3-8b       | Meta        | 8,192                    | -                 | [Card](#)        |

### Preview Models
| MODEL ID               | DEVELOPER   | CONTEXT WINDOW (TOKENS) | MAX OUTPUT TOKENS | MODEL CARD LINK  |
|------------------------|-------------|--------------------------|-------------------|------------------|
| llama-3.3-70b-specdec  | Meta        | 8,192                    | -                 | [Card](#)        |
| llama-3.2-1b-preview   | Meta        | 128k                     | 8,192             | [Card](#)        |

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

---

### Streamlit Integration

You can add a dropdown in the Streamlit app to let the user select the model dynamically. Here's an updated section of your app code:

```python
# Dropdown for selecting a model
st.sidebar.header("Model Selection")
supported_models = [
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-guard-3-8b",
    "llama-3.3-70b-specdec",
    "llama-3.2-1b-preview",
]
default_model = "llama-3.1-8b-instant"  # Default model
selected_model = st.sidebar.selectbox("Choose a GroqCloud model:", supported_models, index=supported_models.index(default_model))

# Use the selected model in API calls
response = generate_text_groq(query=user_query, context=context, model=selected_model)
