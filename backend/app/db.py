# backend/app/db.py
import os
from pymongo import MongoClient

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)

# Use a database named 'rag_chatbot'
db = client.rag_chatbot