import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup   
from urllib.parse import urljoin
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# GroqCloud API Configuration
# Load environment variables from the .env file
load_dotenv()

# Access the API key
api_key = "your_groqcloud_api_key"
# api_key = os.getenv("API_KEY")

groq_generation_url = "https://api.groq.com/openai/v1/chat/completions"

# Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from PDF
def process_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Chunk text
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

# Scrape Website and Linked Pages
def scrape_website_and_links(main_url, max_pages=10):
    visited_urls = set()
    website_data = []

    def scrape_page(url):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            metadata = {
                "title": soup.title.string.strip() if soup.title else "No title found",
                "description": soup.find("meta", attrs={"name": "description"})["content"].strip()
                if soup.find("meta", attrs={"name": "description"})
                else "No description found",
                "keywords": soup.find("meta", attrs={"name": "keywords"})["content"].strip()
                if soup.find("meta", attrs={"name": "keywords"})
                else "No keywords found",
            }

            body_content = soup.get_text(separator="\n").strip()
            formatted_text = "\n".join(
                line.strip() for line in body_content.splitlines() if line.strip()
            )

            return {"url": url, "metadata": metadata, "content": formatted_text}

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    print(f"Scraping main page: {main_url}")
    main_page_data = scrape_page(main_url)
    if main_page_data:
        website_data.append(main_page_data)
        visited_urls.add(main_url)

    print("Extracting links...")
    try:
        response = requests.get(main_url)
        soup = BeautifulSoup(response.text, "html.parser")
        links = [urljoin(main_url, a['href']) for a in soup.find_all('a', href=True)]

        for link in links:
            if link not in visited_urls and len(visited_urls) < max_pages:
                print(f"Scraping linked page: {link}")
                page_data = scrape_page(link)
                if page_data:
                    website_data.append(page_data)
                    visited_urls.add(link)

    except Exception as e:
        print(f"Error extracting links: {e}")

    return website_data

# Generate Embeddings
def generate_embeddings(text_chunks):
    return embedding_model.encode(text_chunks, show_progress_bar=True).tolist()

# Create and Store FAISS Index
def store_embeddings(text_chunks, embeddings):
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    embedding_matrix = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    vector_store = FAISS(
        index=index,
        docstore={str(i): doc for i, doc in enumerate(documents)},
        index_to_docstore_id={i: str(i) for i in range(len(documents))},
        embedding_function=None,
    )
    return vector_store

# Query FAISS Index
def query_faiss_index(vector_store, query, embedding_model, k=3):
    query_embedding = embedding_model.encode([query])[0].astype("float32")
    distances, indices = vector_store.index.search(np.array([query_embedding]), k)
    docs = [vector_store.docstore[str(idx)] for idx in indices[0] if str(idx) in vector_store.docstore]
    return docs

# Generate Response Using GroqCloud
def generate_text_groq(query, context, model="gemma2-9b-it"):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"},
        ],
    }
    response = requests.post(groq_generation_url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"GroqCloud API error: {response.text}")

# Query with Memory and Template
def query_with_memory_and_template(query, context, memory, template):
    # Format the context and query with the template
    prompt = template.format(context=context, query=query)

    # Generate the response using GroqCloud API
    answer = generate_text_groq(query=query, context=context, model="llama-3.1-8b-instant")

    # Update the memory with user input and AI response
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(answer)

    return answer, memory.chat_memory.messages  # Return chat history


# Streamlit App
def main():
    st.set_page_config(page_title="AI Chatbot with GroqCloud", layout="wide")
    st.title("💬 AI Chatbot with GroqCloud 🚀")

    # Initialize memory in session state
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # Store user/AI messages

    # Sidebar for data sources and model selection
    st.sidebar.header("📂 Data Sources")
    data_sources = st.sidebar.multiselect("Choose data sources:", ["PDF", "Website"], default=[])
    
    # Model selection
    st.sidebar.header("🤖 Model Selection")
    model_options = [
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768"
    ]
    selected_model = st.sidebar.selectbox("Choose an AI model:", model_options, index=0)

    if "PDF" in data_sources:
        pdf_docs = st.sidebar.file_uploader("📄 Upload your PDF files:", accept_multiple_files=True, type=["pdf"])

    if "Website" in data_sources:
        website_url = st.sidebar.text_input("🔗 Enter website URL:")

    # Process data when the button is clicked
    if st.sidebar.button("📊 Process Data"):
        all_data = []

        # Process PDFs
        if "PDF" in data_sources and pdf_docs:
            with st.spinner("📄 Processing PDF files..."):
                pdf_text = process_pdf(pdf_docs)
                pdf_chunks = chunk_text(pdf_text)
                all_data.extend(pdf_chunks)

        # Process Website
        if "Website" in data_sources and website_url:
            with st.spinner("🔗 Extracting content from the website..."):
                scraped_pages = scrape_website_and_links(website_url, max_pages=5)
                for page in scraped_pages:
                    website_chunks = chunk_text(page["content"])
                    all_data.extend(website_chunks)

        # Generate embeddings and store in FAISS
        if all_data:
            with st.spinner("🔍 Creating data index..."):
                embeddings = generate_embeddings(all_data)
                vector_store = store_embeddings(all_data, embeddings)
                st.session_state["vector_store"] = vector_store
                st.success("✅ Data processed and indexed successfully!")

    # Section for the chatbot
    st.write("### 🤖 Chatbot")

    # Display chat history incrementally
    chat_placeholder = st.container()
    with chat_placeholder:
        for message in st.session_state["chat_history"]:
            if message["role"] == "user":
                st.markdown(
                    f"""
                    <div style="background-color: #ffffff; padding: 10px; border-radius: 5px; margin: 10px 0; border: 1px solid #ddd;">
                        <strong>👤 You:</strong> {message['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif message["role"] == "assistant":
                st.markdown(
                    f"""
                    <div style="background-color: #f2f2f2; padding: 10px; border-radius: 5px; margin: 10px 0; border: 1px solid #ddd;">
                        <strong>🤖 AI:</strong> {message['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Add a "Clear Chat" button
    if st.button("🧹 Clear Chat"):
        st.session_state["chat_history"] = []  # Clear chat history
        st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.success("Chat cleared successfully!")

    # User input for asking a question
    user_query = st.text_input("Ask a question:", key="user_input", placeholder="Enter your question here...")
    if st.button("Send"):
        if user_query:
            # Add user's question to memory
            memory = st.session_state["memory"]
            memory.chat_memory.add_user_message(user_query)

            # Retrieve context based on documents
            vector_store = st.session_state.get("vector_store", None)
            context = ""
            if vector_store:
                docs = query_faiss_index(vector_store, user_query, embedding_model, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])

            # Call GroqCloud API to generate a response
            try:
                response = generate_text_groq(query=user_query, context=context, model=selected_model)
                memory.chat_memory.add_ai_message(response)

                # Add the new user message and AI response to chat history
                st.session_state["chat_history"].append({"role": "user", "content": user_query})
                st.session_state["chat_history"].append({"role": "assistant", "content": response})

                # Display only the new response
                with chat_placeholder:
                    st.markdown(
                        f"""
                        <div style="background-color: #ffffff; padding: 10px; border-radius: 5px; margin: 10px 0; border: 1px solid #ddd;">
                            <strong>👤 You:</strong> {user_query}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                        <div style="background-color: #f2f2f2; padding: 10px; border-radius: 5px; margin: 10px 0; border: 1px solid #ddd;">
                            <strong>🤖 AI:</strong> {response}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"Error calling the API: {e}")

if __name__ == "__main__":
    main()
