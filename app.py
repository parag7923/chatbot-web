import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create upload folder if not exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key="AIzaSyBpZUY9sQrxkPFc64bVlkaO8D-K0s84KlY")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(pages, embeddings)
    return db

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"})
    
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process PDF
        db = process_pdf(file_path)
        app.config['db'] = db  # Store db in app config (can be retrieved later)
        return jsonify({"status": "success"})
    
    return jsonify({"status": "error", "message": "Invalid file format"})

@app.route('/get_answer', methods=['GET'])
def get_answer():
    query = request.args.get('query')
    db = app.config.get('db')

    if db is not None and query:
        docs = db.similarity_search(query)
        relevant_search = "\n".join([x.page_content for x in docs])
        gemini_prompt = ("Use the following pieces of context to answer the question. "
                         "If you don't know the answer, just say you don't know the answer. Don't make it up.")
        input_prompt = f"{gemini_prompt}\nContext: {relevant_search}\nUser Question: {query}"
        result = llm.invoke(input_prompt)
        return jsonify({"answer": result.content})
    
    return jsonify({"answer": "No answer available. Please upload a PDF first."})

if __name__ == '__main__':
    app.run(debug=True)
