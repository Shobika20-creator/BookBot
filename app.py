from flask import Flask, request, jsonify, render_template
import os

# ✅ UPDATED IMPORTS (Lightweight for Render)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🚀 Optimized embedding model (Uses only ~100MB RAM)
embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vector_db = None


@app.route('/')
def home():
    return render_template("index.html")


# 📌 Upload PDF
@app.route('/upload', methods=['POST'])
def upload_pdf():
    global vector_db

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split text
    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = splitter.split_documents(documents)

    # Create FAISS vector DB
    vector_db = FAISS.from_documents(texts, embedding_model)

    return jsonify({"message": "PDF uploaded and processed successfully"})


# 📌 Ask Question
@app.route('/ask', methods=['POST'])
def ask():
    global vector_db

    if vector_db is None:
        return jsonify({"error": "Upload a PDF first"}), 400

    data = request.get_json()
    question = data.get("question")

    # Retrieve relevant chunks
    docs = vector_db.similarity_search(question, k=3)

    # 🔥 GROQ LLM (Using environment variable for production)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return jsonify({"error": "GROQ_API_KEY environment variable not set"}), 500

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant"
    )

    chain = load_qa_chain(llm, chain_type="stuff")

    result = chain.invoke({"input_documents": docs, "question": question})

    return jsonify({"answer": result["output_text"]})


if __name__ == '__main__':
    print("🔥 Starting Flask server...")
    port = int(os.environ.get("PORT", 5000)); app.run(host="0.0.0.0", port=port, debug=False)
