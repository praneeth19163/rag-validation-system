import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymilvus import MilvusClient
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

app = FastAPI(title="RAG System API")

# Initialize clients
milvus_client = MilvusClient(
    uri=os.getenv("MILVUS_ENDPOINT"),
    token=os.getenv("MILVUS_API_KEY")
)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

COLLECTION_NAME = "ahex_technology"

# Request models
class VectorQueryRequest(BaseModel):
    question: str

class CustomContextRequest(BaseModel):
    question: str
    context: str

# ============================================
# SHARED FUNCTIONS
# ============================================
def generate_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def generate_answer(prompt, system_message):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=400
    )
    return response.choices[0].message.content

def validate_answer(validation_prompt):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a strict validation assistant."},
            {"role": "user", "content": validation_prompt}
        ],
        temperature=0.1,
        max_tokens=300
    )
    return response.choices[0].message.content

# ============================================
# VECTOR DB MODE FUNCTIONS
# ============================================
def retrieve_from_vector_db(query, top_k=3):
    query_embedding = generate_embedding(query)
    results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=top_k,
        output_fields=["heading", "content"]
    )
    return results[0]

def build_vector_context(search_results):
    context_parts = []
    for i, result in enumerate(search_results):
        context_parts.append(
            f"[Policy Section {i+1}]\n"
            f"Title: {result['entity']['heading']}\n"
            f"Content: {result['entity']['content']}"
        )
    return "\n\n".join(context_parts)

def create_vector_prompt(query, context):
    return f"""You are an HR assistant for Ahex Technologies helping employees understand company policies.

PROMPT CONSTRAINTS (MUST FOLLOW):
1. CONTEXT CONSTRAINT: Answer ONLY using the provided policy context below. Do NOT use any external knowledge.
2. FORMAT CONSTRAINT: Return your answer as valid JSON in this EXACT format:
   {{
     "answer": "",
     "source_found": true/false,
     "confidence": "high/medium/low"
   }}
3. BEHAVIOR CONSTRAINT: If the answer is not found in the context, set source_found to false and answer to "Not found in provided policy documents"
4. LENGTH CONSTRAINT: Keep the answer concise and under 150 words
5. SAFETY CONSTRAINT: Only answer questions related to Ahex Technologies employee policies

CONTEXT (Retrieved from Ahex Technologies Employee Handbook):
{context}

EMPLOYEE QUESTION:
{query}

RESPONSE (JSON only):"""

# ============================================
# CUSTOM CONTEXT MODE FUNCTIONS
# ============================================
def create_custom_prompt(query, context):
    return f"""You are an AI assistant helping users answer questions based on provided context.

PROMPT CONSTRAINTS (MUST FOLLOW):
1. CONTEXT CONSTRAINT: Answer ONLY using the provided context below. Do NOT use any external knowledge.
2. FORMAT CONSTRAINT: Return your answer as valid JSON in this EXACT format:
   {{
     "answer": "",
     "source_found": true/false,
     "confidence": "high/medium/low"
   }}
3. BEHAVIOR CONSTRAINT: If the answer is not found in the context, set source_found to false and answer to "Not found in provided context"
4. LENGTH CONSTRAINT: Keep the answer concise and under 150 words

CONTEXT:
{context}

QUESTION:
{query}

RESPONSE (JSON only):"""

# ============================================
# VALIDATION FUNCTION (SHARED)
# ============================================
def create_validation_prompt(context, answer_json):
    return f"""You are a validation assistant. Verify if an AI-generated answer is correct and grounded.

ORIGINAL CONTEXT:
{context}

AI-GENERATED ANSWER TO VALIDATE:
{answer_json}

VALIDATION TASKS:
1. JSON VALIDATION: Check if the output is valid JSON with all required fields
2. CONTEXT VERIFICATION: Verify every factual claim is supported by the context
3. HALLUCINATION DETECTION: Check if answer contains information NOT in the context
4. COMPLETENESS: Check if answer adequately addresses the question

Return validation result as JSON:
{{
  "valid": true/false,
  "json_valid": true/false,
  "hallucination_detected": true/false,
  "context_grounded": true/false,
  "reason": "detailed explanation"
}}

VALIDATION RESULT (JSON only):"""

# ============================================
# API ENDPOINTS
# ============================================
@app.get("/")
def root():
    return {"message": "RAG System API", "endpoints": ["/query/vector", "/query/custom"]}

@app.post("/query/vector")
def query_vector_db(request: VectorQueryRequest):
    """Query using vector database (Ahex Technologies policies)"""
    try:
        # Step 1: Retrieve from vector DB
        search_results = retrieve_from_vector_db(request.question, top_k=3)
        context = build_vector_context(search_results)
        
        retrieved_sources = [
            {
                "heading": result['entity']['heading'],
                "similarity": float(result['distance'])
            }
            for result in search_results
        ]
        
        # Step 2: Generate answer
        prompt = create_vector_prompt(request.question, context)
        answer_text = generate_answer(prompt, "You are a helpful HR assistant for Ahex Technologies.")
        
        answer_json = json.loads(answer_text)
        source_found = answer_json.get('source_found', False)
        
        # Step 3: Validation (skip if source not found)
        if not source_found:
            return {
                "answer": answer_json,
                "validation": {"valid": True, "reason": "Out of scope - correctly handled"},
                "passed": True,
                "sources": retrieved_sources,
                "validation_skipped": True
            }
        
        validation_prompt = create_validation_prompt(context, answer_text)
        validation_text = validate_answer(validation_prompt)
        validation_json = json.loads(validation_text)
        
        return {
            "answer": answer_json,
            "validation": validation_json,
            "passed": validation_json.get("valid", False),
            "sources": retrieved_sources,
            "validation_skipped": False
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/custom")
def query_custom_context(request: CustomContextRequest):
    """Query using user-provided custom context"""
    try:
        # Step 1: Use provided context directly
        context = request.context
        
        # Step 2: Generate answer
        prompt = create_custom_prompt(request.question, context)
        answer_text = generate_answer(prompt, "You are a helpful AI assistant.")
        
        answer_json = json.loads(answer_text)
        source_found = answer_json.get('source_found', False)
        
        # Step 3: Validation (skip if source not found)
        if not source_found:
            return {
                "answer": answer_json,
                "validation": {"valid": True, "reason": "Out of scope - correctly handled"},
                "passed": True,
                "validation_skipped": True
            }
        
        validation_prompt = create_validation_prompt(context, answer_text)
        validation_text = validate_answer(validation_prompt)
        validation_json = json.loads(validation_text)
        
        return {
            "answer": answer_json,
            "validation": validation_json,
            "passed": validation_json.get("valid", False),
            "validation_skipped": False
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
