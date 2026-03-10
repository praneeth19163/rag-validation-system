import os
from pymilvus import MilvusClient, DataType
from dotenv import load_dotenv
from openai import OpenAI
import PyPDF2
import re

load_dotenv()

# Initialize clients
milvus_client = MilvusClient(
    uri=os.getenv("MILVUS_ENDPOINT"),
    token=os.getenv("MILVUS_API_KEY")
)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# def smart_chunk_text(text):
#     """Chunk text based on headings and related content"""
#     # Split by common heading patterns (lines with all caps, numbered sections, etc.)
#     lines = text.split('\n')
#     chunks = []
#     current_chunk = []
#     current_heading = ""
    
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
            
#         # Detect headings (all caps, numbered, or short lines ending with colon)
#         is_heading = (
#             (line.isupper() and len(line.split()) <= 10) or
#             re.match(r'^\d+\.?\s+[A-Z]', line) or
#             (len(line) < 60 and line.endswith(':'))
#         )
        
#         if is_heading and current_chunk:
#             # Save previous chunk
#             chunks.append({
#                 'heading': current_heading,
#                 'content': '\n'.join(current_chunk)
#             })
#             current_chunk = []
#             current_heading = line
        
#         current_chunk.append(line)
    
#     # Add last chunk
#     if current_chunk:
#         chunks.append({
#             'heading': current_heading,
#             'content': '\n'.join(current_chunk)
#         })
    
#     return chunks
def smart_chunk_text(text):
    """Chunk text by main section headings only (15 chunks for 15 sections)"""
    lines = text.split('\n')
    chunks = []
    current_heading = ""
    current_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Only match MAIN headings like "1. Introduction", "5. Leave Policy"
        # Must start with a number followed by dot and a capital letter word
        is_main_heading = bool(re.match(r'^\d{1,2}\.\s+[A-Z][a-z]', line))

        if is_main_heading:
            # Save the previous chunk before starting a new one
            if current_lines:
                full_text = f"{current_heading}. {' '.join(current_lines)}"
                chunks.append({
                    "heading": current_heading,
                    "content": full_text.strip()
                })
            # Start new chunk
            current_heading = line
            current_lines = []
        else:
            # Everything else belongs to the current section
            current_lines.append(line)

    # Don't forget the last section
    if current_lines:
        full_text = f"{current_heading}. {' '.join(current_lines)}"
        chunks.append({
            "heading": current_heading,
            "content": full_text.strip()
        })

    return chunks
def generate_embedding(text):
    """Generate embedding using OpenAI"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def create_collection():
    """Create Milvus collection for Ahex Technologies"""
    collection_name = "ahex_technology"
    
    # Drop if exists
    if collection_name in milvus_client.list_collections():
        milvus_client.drop_collection(collection_name)
    
    # Create collection
    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="heading", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=5000)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=1536)
    
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
    
    milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    
    return collection_name

def main():
    pdf_path = "Ahex Technologies.pdf"
    
    print("📄 Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    
    print("✂️ Chunking text based on structure...")
    chunks = smart_chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    
    print("🗄️ Creating collection...")
    collection_name = create_collection()
    
    print("🔢 Generating embeddings and inserting...")
    data = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}: {chunk['heading'][:50]}...")
        embedding = generate_embedding(chunk['content'])
        data.append({
            "heading": chunk['heading'][:500],
            "content": chunk['content'][:5000],
            "embedding": embedding
        })
    
    milvus_client.insert(collection_name=collection_name, data=data)
    print(f"✅ Successfully inserted {len(data)} chunks into '{collection_name}' collection")

if __name__ == "__main__":
    main()
