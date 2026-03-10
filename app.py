import streamlit as st
import requests
import json
from docx import Document
from PyPDF2 import PdfReader
import io

# Page config
st.set_page_config(
    page_title="RAG System - Ahex Technologies",
    page_icon="🤖",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"

# Title
st.title("🤖 RAG System with Validation")
st.markdown("---")

# Mode selection
mode = st.radio(
    "Select Query Mode:",
    ["Vector Database (Ahex Technologies Policies)", "Custom Context"],
    help="Choose between querying the vector database or providing your own context"
)

st.markdown("---")

# Input fields based on mode
if mode == "Vector Database (Ahex Technologies Policies)":
    st.subheader("📚 Vector Database Mode")
    st.info("Query the Ahex Technologies Employee Policy Handbook stored in the vector database")
    
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., How many work from home days are allowed per month?",
        key="vector_question"
    )
    
    if st.button("🔍 Query Vector Database", type="primary"):
        if not question:
            st.error("Please enter a question")
        else:
            with st.spinner("Searching vector database and generating answer..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query/vector",
                        json={"question": question}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.subheader("💬 Answer")
                            answer_data = result['answer']
                            
                            if answer_data['source_found']:
                                st.success(answer_data['answer'])
                                st.caption(f"Confidence: {answer_data['confidence']}")
                            else:
                                st.warning(answer_data['answer'])
                                st.caption("This question is outside the scope of employee policies")
                        
                        with col2:
                            st.subheader("✅ Validation")
                            if result['validation_skipped']:
                                st.info("Validation skipped (out of scope)")
                            elif result['passed']:
                                st.success("✅ Validation Passed")
                            else:
                                st.error("❌ Validation Failed")
                            
                            if not result['validation_skipped']:
                                with st.expander("Validation Details"):
                                    st.json(result['validation'])
                        
                        # Full response
                        with st.expander("🔍 View Full Response"):
                            st.json(result)
                    
                    else:
                        st.error(f"API Error: {response.status_code}")
                        st.error(response.text)
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure the FastAPI server is running on port 8000")
                    st.code("python api.py")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

else:  # Custom Context Mode
    st.subheader("📝 Custom Context Mode")
    st.info("Provide your own context and ask questions based on it")
    
    # Initialize session state for context
    if 'custom_context_text' not in st.session_state:
        st.session_state.custom_context_text = ""
    
    # File upload option
    uploaded_file = st.file_uploader("Upload a document (PDF or DOCX)", type=["pdf", "docx"])
    
    if uploaded_file:
        try:
            if uploaded_file.type == "application/pdf":
                pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
                st.session_state.custom_context_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
                st.success(f"✅ Extracted text from PDF ({len(st.session_state.custom_context_text)} characters)")
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = Document(io.BytesIO(uploaded_file.read()))
                st.session_state.custom_context_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                st.success(f"✅ Extracted text from DOCX ({len(st.session_state.custom_context_text)} characters)")
        except Exception as e:
            st.error(f"❌ Error extracting text: {str(e)}")
    
    # Text input section
    st.markdown("### Or paste your text directly:")
    context = st.text_area(
        "Enter your context:",
        value=st.session_state.custom_context_text,
        placeholder="Paste your text here or upload a file above...",
        height=200
    )
    
    # Show text stats
    if context.strip():
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Characters", len(context))
        with col_stat2:
            st.metric("Words", len(context.split()))
    
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the main topic discussed?",
        key="custom_question"
    )
    
    if st.button("🔍 Query Custom Context", type="primary"):
        if not question:
            st.error("Please enter a question")
        elif not context.strip():
            st.error("Please provide context (upload file or paste text)")
        else:
            with st.spinner("Generating answer from your context..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query/custom",
                        json={"question": question, "context": context}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.subheader("💬 Answer")
                            answer_data = result['answer']
                            
                            if answer_data['source_found']:
                                st.success(answer_data['answer'])
                                st.caption(f"Confidence: {answer_data['confidence']}")
                            else:
                                st.warning(answer_data['answer'])
                                st.caption("Answer not found in provided context")
                        
                        with col2:
                            st.subheader("✅ Validation")
                            if result['validation_skipped']:
                                st.info("Validation skipped (out of scope)")
                            elif result['passed']:
                                st.success("✅ Validation Passed")
                            else:
                                st.error("❌ Validation Failed")
                            
                            if not result['validation_skipped']:
                                with st.expander("Validation Details"):
                                    st.json(result['validation'])
                        
                        # Full response
                        with st.expander("🔍 View Full Response"):
                            st.json(result)
                    
                    else:
                        st.error(f"API Error: {response.status_code}")
                        st.error(response.text)
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure the FastAPI server is running on port 8000")
                    st.code("python api.py")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This RAG system implements:
    - **Prompt Constraints**: Control LLM behavior
    - **Retrieval Grounding**: Use real data
    - **Validation Prompts**: Verify accuracy
    
    ### Modes:
    1. **Vector DB**: Query Ahex Technologies policies
    2. **Custom Context**: Use your own text
    
    ### How to Run:
    ```bash
    # Start API
    python api.py
    
    # Start UI (in another terminal)
    streamlit run app.py
    ```
    """)
    
    st.markdown("---")
    st.caption("Built with FastAPI + Streamlit + OpenAI + Milvus")
