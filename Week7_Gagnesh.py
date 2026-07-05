import streamlit as st
import cohere
import fitz
import os
import uuid
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec

# ==========================================
# 1. VectorStore Module
# ==========================================
class VectorStore:
    def __init__(self, cohere_api_key: str, pinecone_api_key: str):
        self.co = cohere.Client(cohere_api_key)
        self.pinecone_api_key = pinecone_api_key
        self.chunks = []
        self.embeddings = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        
    def ingest_pdf(self, pdf_path: str):
        text = ""
        with fitz.open(pdf_path) as pdf:
            for page_num in range(pdf.page_count):
                page = pdf.load_page(page_num)
                text += page.get_text("text")
        self._split_and_index(text)

    def ingest_txt(self, txt_path: str):
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self._split_and_index(text)

    def ingest_huggingface(self, dataset_name: str, split: str = "train", column: str = "question"):
        dataset = load_dataset(dataset_name, split=split)
        # Using a small subset to prevent timeout or hitting rate limits during demo
        text = "\n\n".join([str(item[column]) for item in dataset.select(range(min(20, len(dataset))))])
        self._split_and_index(text)

    def _split_and_index(self, text: str):
        self.split_text(text)
        self.embed_chunks()
        self.index_chunks()

    def split_text(self, text: str, chunk_size=1000):
        self.chunks = []
        sentences = text.split(". ")
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                self.chunks.append(current_chunk)
                current_chunk = sentence + ". "
        if current_chunk:
            self.chunks.append(current_chunk)

    def embed_chunks(self, batch_size=90):
        self.embeddings = []
        total_chunks = len(self.chunks)
        for i in range(0, total_chunks, batch_size):
            batch = self.chunks[i:min(i + batch_size, total_chunks)]
            batch_embeddings = self.co.embed(
                texts=batch, input_type="search_document", model="embed-english-v3.0"
            ).embeddings
            self.embeddings.extend(batch_embeddings)

    def index_chunks(self):
        pc = Pinecone(api_key=self.pinecone_api_key)
        index_name = 'rag-qa-bot'
        
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=len(self.embeddings[0]),
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        self.index = pc.Index(index_name)
        chunks_metadata = [{'text': chunk} for chunk in self.chunks]
        ids = [str(i) for i in range(len(self.chunks))]
        self.index.upsert(vectors=zip(ids, self.embeddings, chunks_metadata))

    def retrieve(self, query: str) -> list:
        query_emb = self.co.embed(
            texts=[query], model="embed-english-v3.0", input_type="search_query"
        ).embeddings
        res = self.index.query(vector=query_emb[0], top_k=self.retrieve_top_k, include_metadata=True)
        docs_to_rerank = [match['metadata']['text'] for match in res['matches']]
        
        if not docs_to_rerank:
            return []
            
        rerank_results = self.co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v2.0"
        )
        return [res['matches'][result.index]['metadata'] for result in rerank_results.results]


# ==========================================
# 2. Chatbot Module
# ==========================================
class Chatbot:
    def __init__(self, vectorstore, cohere_api_key: str):
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())
        self.co = cohere.Client(cohere_api_key)

    def respond(self, user_message: str):
        retrieved_docs = self.vectorstore.retrieve(user_message)
        
        if retrieved_docs:
            response = self.co.chat_stream(
                message=user_message,
                model="command-r",
                documents=retrieved_docs,
                conversation_id=self.conversation_id,
            )
        else:
            response = self.co.chat_stream(
                message=user_message,
                model="command-r",
                conversation_id=self.conversation_id,
            )
        return response, retrieved_docs


# ==========================================
# 3. Streamlit Application (Frontend)
# ==========================================
def main():
    st.title("Enterprise RAG System 🤖")
    st.write("Upload a custom document or load a HuggingFace dataset, input your API keys, and ask questions!")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None

    with st.sidebar:
        st.header("Configuration ⚙️")
        cohere_api_key = st.text_input("Cohere API Key", type="password")
        pinecone_api_key = st.text_input("Pinecone API Key", type="password")
        
        st.header("Data Source 📄")
        source_type = st.radio("Select Source Type:", ("PDF", "TXT", "HuggingFace Dataset"))
        
        uploaded_file = None
        hf_dataset_name = None
        
        if source_type in ("PDF", "TXT"):
            file_ext = "pdf" if source_type == "PDF" else "txt"
            uploaded_file = st.file_uploader(f"Upload a {source_type} file", type=file_ext)
        else:
            hf_dataset_name = st.text_input("Dataset Name", value="vectara/open_ragbench")
            
        process_btn = st.button("Process Data")

    if process_btn and cohere_api_key and pinecone_api_key:
        if source_type in ("PDF", "TXT") and not uploaded_file:
            st.warning("Please upload a file first.")
        else:
            with st.spinner("Processing Document & Initializing Vector Store..."):
                vectorstore = VectorStore(cohere_api_key, pinecone_api_key)
                
                try:
                    if source_type == "PDF":
                        temp_path = "temp_uploaded.pdf"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.read())
                        vectorstore.ingest_pdf(temp_path)
                    elif source_type == "TXT":
                        temp_path = "temp_uploaded.txt"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.read())
                        vectorstore.ingest_txt(temp_path)
                    else:
                        vectorstore.ingest_huggingface(hf_dataset_name)
                    
                    st.session_state["vectorstore"] = vectorstore
                    st.success("Data successfully processed and indexed in Pinecone!")
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")

    user_query = st.chat_input("Ask a question based on the ingested document...")

    if user_query:
        if not st.session_state.get("vectorstore"):
            st.error("Please process a document first in the sidebar.")
        else:
            for message in st.session_state["chat_history"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            with st.chat_message("user"):
                st.markdown(user_query)
                
            with st.spinner("Generating response..."):
                chatbot = Chatbot(st.session_state["vectorstore"], cohere_api_key)
                response_stream, retrieved_docs = chatbot.respond(user_query)
                
                accumulated_response = ""
                for event in response_stream:
                    if event.event_type == "text-generation":
                        accumulated_response += event.text
                        
                with st.chat_message("assistant"):
                    st.markdown(accumulated_response)
                    with st.expander("View Retrieved Context"):
                        st.json(retrieved_docs)
                        
                st.session_state["chat_history"].append({"role": "user", "content": user_query})
                st.session_state["chat_history"].append({"role": "assistant", "content": accumulated_response})

if __name__ == "__main__":
    main()
