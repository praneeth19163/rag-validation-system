"""
Simple Milvus Connection Test Script.

Tests connectivity to Milvus/Zilliz Cloud and lists existing collections.
Uses credentials from .env file.
"""

import os
from pymilvus import MilvusClient
from dotenv import load_dotenv

load_dotenv()

client = MilvusClient(
    uri=os.getenv("MILVUS_ENDPOINT"),
    token=os.getenv("MILVUS_API_KEY")
)

try:
    version = client.get_server_version()
    print("✅ Connected to Milvus. Server version:", version)

    collections = client.list_collections()
    print("📂 Existing collections:", collections)

except Exception as e:
    print("❌ Connection failed:", e)
