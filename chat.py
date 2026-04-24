import os
import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# 1. Connect to the FREE Gemini API and the saved Database
chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    collection = chroma_client.get_collection(name="toddlerbot_memory")
except Exception:
    print("⚠️ Database not found! Did you run rag_engine.py first?")
    exit()

# 2. Ask a Question!
print("🤖 ToddlerBot AI (Free Tier) is ready.")
user_question = input("Ask a question about the ToddlerBot: ")

# 3. Search the Database
print("\nSearching lab notes...")
results = collection.query(
    query_texts=[user_question],
    n_results=1 
)

found_text = results['documents'][0][0]
source_file = results['metadatas'][0][0]['source']

# 4. Have Gemini generate an answer
print("Thinking...\n")
system_instructions = f"""
You are the ToddlerBot AI assistant. Answer the user's question using ONLY the provided lab notes. 
If the answer is not in the notes, say "I don't have that in my records."

LAB NOTES:
{found_text}
"""

# Call the Gemini model
response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents=[system_instructions, user_question]
)

# 5. Print the final answer
print("--------------------------------------------------")
print(response.text)
print(f"\n(Source: {source_file})")
print("--------------------------------------------------")