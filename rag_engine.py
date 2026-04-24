import os
import chromadb

# 1. Start the Local Database (Permanently saved to a folder!)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    chroma_client.delete_collection(name="toddlerbot_memory")
except Exception:
    pass

collection = chroma_client.create_collection(name="toddlerbot_memory")

# 2. Read the actual ToddlerBot documents
docs_folder = "docs"
documents = []
metadatas = []
ids = []

print(f"Scanning the '{docs_folder}' folder for documentation...")

for filename in os.listdir(docs_folder):
    if filename.endswith(".rst") or filename.endswith(".txt") or filename.endswith(".md"):
        filepath = os.path.join(docs_folder, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                documents.append(file.read())
                metadatas.append({"source": filename}) 
                ids.append(filename)
        except Exception as e:
            print(f"Skipped {filename} because it couldn't be read: {e}")

# 3. Save everything to the vector database
if len(documents) > 0:
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"✅ Success! Loaded {len(documents)} official files into the local memory.")
else:
    print("⚠️ No text files found in the docs folder!")