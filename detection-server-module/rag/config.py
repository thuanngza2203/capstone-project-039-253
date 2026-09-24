from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
# `python -m rag.ingest` không đi qua package app, nên tự nạp .env của module.
load_dotenv(BASE_DIR.parent / '.env')
DATA_DIR = BASE_DIR / 'data'
CHROMA_DIR = BASE_DIR / 'chroma_db'
COLLECTION_NAME = 'plant_disease_vi'

CHUNK_SIZE = int(os.getenv('RAG_CHUNK_SIZE','1000'))
CHUNK_OVERLAP = int(os.getenv('RAG_CHUNK_OVERLAP','150'))
TOP_K = int(os.getenv('RAG_TOP_K','5'))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL','AITeamVN/Vietnamese_Embedding')
EMBEDDING_DEVICE = os.getenv('EMBEDDING_DEVICE','cpu')
