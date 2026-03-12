"""
Streamlit UI for RAG (Retrieval-Augmented Generation) System with Validation.
"""

import io

import pandas as pd
import requests
import streamlit as st
from docx import Document
from PyPDF2 import PdfReader


st.set_page_config(
    page_title="RAG System - Ahex Technologies",
    page_icon="RAG",
    layout="wide",
)

API_URL = "http://localhost:8000"


def run_progressive_test(test_inputs, endpoint, extra_payload=None, table_key="test_table"):
    rows = []
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    table_placeholder = st.empty()

    for index, question in enumerate(test_inputs, start=1):
        payload = {"question": question}
        if extra_payload:
            payload.update(extra_payload)

        response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(f"API Error: {response.status_code} - {response.text}")

        result = response.json()
        rows.append(
            {
                "Question No": index,
                "Input": question,
                "Output": result["answer"]["answer"],
                "Correct/Incorrect": "Correct",
            }
        )

        progress_bar.progress(index / len(test_inputs))
        status_placeholder.caption(f"Processed {index} of {len(test_inputs)} questions")
        table_placeholder.data_editor(
            pd.DataFrame(rows),
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "Question No": st.column_config.NumberColumn(
                    "Question No",
                    min_value=1,
                    step=1,
                    required=True,
                ),
                "Correct/Incorrect": st.column_config.SelectboxColumn(
                    "Correct/Incorrect",
                    options=["Correct", "Incorrect"],
                    required=True,
                )
            },
            key=f"{table_key}_{index}",
        )

    return pd.DataFrame(rows)


def render_single_result(result, out_of_scope_message):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Answer")
        answer_data = result["answer"]
        if answer_data["source_found"]:
            st.success(answer_data["answer"])
            st.caption(f"Confidence: {answer_data['confidence']}")
        else:
            st.warning(answer_data["answer"])
            st.caption(out_of_scope_message)

    with col2:
        st.subheader("Validation")
        if result["validation_skipped"]:
            st.info("Validation skipped")
        elif result["passed"]:
            st.success("Validation Passed")
        else:
            st.error("Validation Failed")

        if not result["validation_skipped"]:
            with st.expander("Validation Details"):
                st.json(result["validation"])

    with st.expander("View Full Response"):
        st.json(result)


st.title("RAG System with Validation")
st.markdown("---")

mode = st.radio(
    "Select Query Mode:",
    ["Vector Database (Ahex Technologies Policies)", "Custom Context"],
    help="Choose between querying the vector database or providing your own context",
)

st.markdown("---")

if mode == "Vector Database (Ahex Technologies Policies)":
    st.subheader("Vector Database Mode")
    st.info("Query the Ahex Technologies Employee Policy Handbook stored in the vector database")

    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., How many work from home days are allowed per month?",
        key="vector_question",
    )

    if st.button("Query Vector Database", type="primary"):
        if not question:
            st.error("Please enter a question")
        else:
            with st.spinner("Searching vector database and generating answer..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query/vector",
                        json={"question": question},
                        timeout=120,
                    )

                    if response.status_code == 200:
                        render_single_result(
                            response.json(),
                            "This question is outside the scope of employee policies",
                        )
                    else:
                        st.error(f"API Error: {response.status_code}")
                        st.error(response.text)
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000")
                    st.code("python api.py")
                except Exception as exc:
                    st.error(f"Error: {str(exc)}")

    st.markdown("---")
    st.subheader("Prompt Test")
    vector_test_inputs = st.text_area(
        "Enter test questions (one per line):",
        height=220,
        key="vector_test_inputs",
    )

    if st.button("Run Prompt Test", key="vector_batch_test"):
        test_inputs = [line.strip() for line in vector_test_inputs.splitlines() if line.strip()]
        if not test_inputs:
            st.error("Please enter at least one input")
        else:
            st.caption(f"Detected {len(test_inputs)} questions")
            with st.spinner("Running prompt tests..."):
                try:
                    edited_df = run_progressive_test(
                        test_inputs=test_inputs,
                        endpoint="/query/vector",
                        table_key="vector_test_table",
                    )
                    st.download_button(
                        "Download Test Results CSV",
                        edited_df.to_csv(index=False),
                        file_name="vector_prompt_test_results.csv",
                        mime="text/csv",
                    )
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000")
                    st.code("python api.py")
                except Exception as exc:
                    st.error(f"Error: {str(exc)}")

else:
    st.subheader("Custom Context Mode")
    st.info("Provide your own context and ask questions based on it")

    if "custom_context_text" not in st.session_state:
        st.session_state.custom_context_text = ""

    uploaded_file = st.file_uploader("Upload a document (PDF or DOCX)", type=["pdf", "docx"])

    if uploaded_file:
        try:
            if uploaded_file.type == "application/pdf":
                pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
                st.session_state.custom_context_text = "\n".join(
                    [page.extract_text() for page in pdf_reader.pages]
                )
                st.success(
                    f"Extracted text from PDF ({len(st.session_state.custom_context_text)} characters)"
                )
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = Document(io.BytesIO(uploaded_file.read()))
                st.session_state.custom_context_text = "\n".join(
                    [para.text for para in doc.paragraphs if para.text.strip()]
                )
                st.success(
                    f"Extracted text from DOCX ({len(st.session_state.custom_context_text)} characters)"
                )
        except Exception as exc:
            st.error(f"Error extracting text: {str(exc)}")

    st.markdown("### Or paste your text directly:")
    context = st.text_area(
        "Enter your context:",
        value=st.session_state.custom_context_text,
        placeholder="Paste your text here or upload a file above...",
        height=200,
    )

    if context.strip():
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Characters", len(context))
        with col_stat2:
            st.metric("Words", len(context.split()))

    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the main topic discussed?",
        key="custom_question",
    )

    if st.button("Query Custom Context", type="primary"):
        if not question:
            st.error("Please enter a question")
        elif not context.strip():
            st.error("Please provide context (upload file or paste text)")
        else:
            with st.spinner("Generating answer from your context..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query/custom",
                        json={"question": question, "context": context},
                        timeout=120,
                    )

                    if response.status_code == 200:
                        render_single_result(
                            response.json(),
                            "Answer not found in provided context",
                        )
                    else:
                        st.error(f"API Error: {response.status_code}")
                        st.error(response.text)
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000")
                    st.code("python api.py")
                except Exception as exc:
                    st.error(f"Error: {str(exc)}")

    st.markdown("---")
    st.subheader("Prompt Test")
    custom_test_inputs = st.text_area(
        "Enter test questions (one per line):",
        height=220,
        key="custom_test_inputs",
    )

    if st.button("Run Prompt Test", key="custom_batch_test"):
        test_inputs = [line.strip() for line in custom_test_inputs.splitlines() if line.strip()]
        if not test_inputs:
            st.error("Please enter at least one input")
        elif not context.strip():
            st.error("Please provide context before running the test")
        else:
            st.caption(f"Detected {len(test_inputs)} questions")
            with st.spinner("Running prompt tests..."):
                try:
                    edited_df = run_progressive_test(
                        test_inputs=test_inputs,
                        endpoint="/query/custom",
                        extra_payload={"context": context},
                        table_key="custom_test_table",
                    )
                    st.download_button(
                        "Download Test Results CSV",
                        edited_df.to_csv(index=False),
                        file_name="custom_prompt_test_results.csv",
                        mime="text/csv",
                    )
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000")
                    st.code("python api.py")
                except Exception as exc:
                    st.error(f"Error: {str(exc)}")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
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
    """
    )
    st.markdown("---")
    st.caption("Built with FastAPI + Streamlit + OpenAI + Milvus")
