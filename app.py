import os
import sys
import subprocess
import streamlit as st
import chromadb
from dotenv import load_dotenv
from google import genai

# --- 1. Setup and API Key ---
load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# --- 2. Page Configuration ---
st.set_page_config(page_title="ToddlerBot AI", page_icon="🤖")
st.title("🤖 ToddlerBot Knowledge Assistant")
st.caption("Ask me anything about building ToddlerBot!")

# --- 3. Bulletproof Database Connection ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    # Try to grab the collection
    collection = chroma_client.get_collection(name="toddlerbot_memory")
except Exception as e:
    # If it fails, the database is empty or broken. Force a rebuild and spill the logs!
    with st.spinner("Memory missing! Forcing engine to rebuild..."):
        result = subprocess.run(
            [sys.executable, "rag_engine.py"], 
            capture_output=True, 
            text=True
        )
        st.error("Engine forced to run. Here are the internal logs:")
        
        st.markdown("**What the engine thought it did (Standard Output):**")
        st.code(result.stdout or "No output printed.")
        
        st.markdown("**The hidden crash reason (Standard Error):**")
        st.code(result.stderr or "No errors thrown.")
        
        st.stop()

# --- 4. Conversation Memory ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. Chat Interface & Logic ---
if prompt := st.chat_input("Ask a question about ToddlerBot..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Search ChromaDB for 3 most relevant docs
    results = collection.query(
        query_texts=[prompt],
        n_results=3
    )
    
    # Safety check: Ensure documents were actually found
    if results['documents'] and results['documents'][0]:
        found_texts = "\n\n---\n\n".join(results['documents'][0])
        sources = [m['source'] for m in results['metadatas'][0]]
    else:
        found_texts = "No relevant documents found in the database."
        sources = ["None"]

    # Build conversation history for Gemini
    history = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in st.session_state.messages[:-1]
    ])

    # Prompt template
    system_prompt = f"""
You are the ToddlerBot AI assistant for the NYCU replication team.
Answer using ONLY the provided documents below.
If the answer is not in the documents, ask the user a follow-up question to get more context.
Never make things up.

CONVERSATION HISTORY:
{history}

RELEVANT DOCUMENTS:
{found_texts}
"""

    # Call Gemini
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Updated SDK implementation using GenerateContentConfig
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt
                )
            )
            answer = response.text
            st.markdown(answer)
            
            # Display unique sources (prevents listing the same file 3 times)
            if sources != ["None"]:
                unique_sources = list(set(sources))
                st.caption(f"Sources: {', '.join(unique_sources)}")

    # Save assistant response to memory
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
