# 🚀 RAG System - Setup & Run Instructions

## 📋 Prerequisites
- Python 3.8+
- OpenAI API Key
- Milvus/Zilliz Cloud credentials

## 🔧 Installation

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

## 📊 Setup

### Step 2: Configure Environment Variables
Make sure your `.env` file has:
```
MILVUS_ENDPOINT=your_milvus_endpoint
MILVUS_API_KEY=your_milvus_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Step 3: Create Vector Database (One-time setup)
```bash
python process_pdf.py
```
This will:
- Read `Ahex Technologies.pdf`
- Create intelligent chunks
- Generate embeddings
- Store in Milvus collection `ahex_technology`

## 🎯 Running the Application

### Option 1: Run API and UI Separately (Recommended)

#### Terminal 1 - Start FastAPI Backend:
```bash
python api.py
```
- API will run on: `http://localhost:8000`
- Check API docs: `http://localhost:8000/docs`

#### Terminal 2 - Start Streamlit UI:
```bash
streamlit run app.py
```
- UI will open automatically in browser
- Default: `http://localhost:8501`

### Option 2: Run Streamlit Only (Direct Mode)
If you don't want to use FastAPI, you can integrate the logic directly in Streamlit.

## 🎨 Using the Application

### Mode 1: Vector Database (Ahex Technologies Policies)
1. Select "Vector Database (Ahex Technologies Policies)"
2. Enter your question (e.g., "How many work from home days are allowed?")
3. Click "🔍 Query Vector Database"
4. View answer, validation results, and retrieved sources

### Mode 2: Custom Context
1. Select "Custom Context"
2. **Upload a document** (PDF or DOCX) **OR** paste your own text in the context area
   - Supported formats: `.pdf`, `.docx`
   - Text is automatically extracted and displayed in the text area
   - You can edit the extracted text if needed
3. Enter your question about that text
4. Click "🔍 Query Custom Context"
5. View answer and validation results

## 📁 Project Structure
```
zilliz/
├── .env                    # Environment variables
├── api.py                  # FastAPI backend
├── app.py                  # Streamlit UI
├── process_pdf.py          # Vector DB setup (run once)
├── requirements.txt        # Dependencies
├── Ahex Technologies.docx # Source Word
└── Ahex Technologies.pdf  # Source PDF
```

## 🔍 API Endpoints

### GET /
Health check and available endpoints

### POST /query/vector
Query vector database mode
```json
{
  "question": "How many paid leaves do I get?"
}
```

### POST /query/custom
Query with custom context
```json
{
  "question": "What is the main topic?",
  "context": "Your custom text here..."
}
```

## ✅ Features Implemented
- ✓ Prompt Constraints (Context, Format, Behavior, Length, Safety)
- ✓ Retrieval Grounding (Vector DB with Milvus)
- ✓ Validation Prompts (JSON, Context, Hallucination checks)
- ✓ Two query modes (Vector DB & Custom Context)
- ✓ File upload support (PDF & DOCX) in Custom Context mode
- ✓ Automatic text extraction from uploaded documents
- ✓ Beautiful Streamlit UI
- ✓ FastAPI REST API
- ✓ Automatic validation skipping for out-of-scope questions

## 🐛 Troubleshooting

### Issue: "Cannot connect to API"
**Solution:** Make sure FastAPI is running on port 8000
```bash
python api.py
```

### Issue: "Collection not found"
**Solution:** Run the setup script first
```bash
python process_pdf.py
```

### Issue: "OpenAI API Error"
**Solution:** Check your API key in `.env` file

## 📝 Notes
- Validation is automatically skipped when answer is not found in context
