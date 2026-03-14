import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import uuid

# --- UI Setup ---
st.set_page_config(page_title="Enterprise Support AI", layout="wide")
st.title("🛠️ Enterprise Support AI: RAG System")
st.write("Powered by Endee Vector Database")

# --- Load AI Model ---
@st.cache_resource
def load_ai_model():
    # This model converts our text into mathematical vectors
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_ai_model()
ENDEE_URL = "http://localhost:8080" # The local connection to Endee

# --- Feature 1: Uploading Support Tickets ---
st.subheader("1. Ingest Past Support Tickets")
st.info("Paste historical customer support tickets here (one per line). The AI will learn from them.")
ticket_data = st.text_area("Example: 'User cannot reset password due to email delay.'", height=150)

if st.button("Store Tickets in Endee"):
    if ticket_data:
        tickets = ticket_data.split('\n')
        success_count = 0
        
        with st.spinner("Processing and vectorizing tickets..."):
            for ticket in tickets:
                if not ticket.strip(): continue
                
                # Convert text to vector
                vector = model.encode(ticket).tolist()
                
                # Send to Endee Database
                payload = {
                    "id": str(uuid.uuid4()),
                    "vector": vector,
                    "metadata": {"type": "support_ticket", "content": ticket}
                }
                try:
                    res = requests.post(f"{ENDEE_URL}/api/insert", json=payload)
                    if res.status_code == 200: success_count += 1
                except Exception as e:
                    st.error("Make sure the Endee Database is running!")
                    break
                    
        st.success(f"Successfully learned {success_count} support tickets!")

st.divider()

# --- Feature 2: Smart AI Search (RAG) ---
st.subheader("2. AI Support Assistant (Search)")
st.write("Ask a question. The system will find the most relevant past tickets to help you solve it.")
query = st.text_input("Enter new customer issue:")

if st.button("Find Solutions via Endee"):
    if query:
        with st.spinner("Searching vector space for similar past issues..."):
            query_vector = model.encode(query).tolist()
            
            try:
                res = requests.post(f"{ENDEE_URL}/api/search", json={"vector": query_vector, "top_k": 3})
                
                if res.status_code == 200:
                    results = res.json().get('results', [])
                    st.write("### Recommended Solutions based on Past Tickets:")
                    for i, r in enumerate(results):
                        # Extracting the text from the Endee response
                        past_ticket = r.get('metadata', {}).get('content', 'No content found')
                        st.success(f"**Similar Case {i+1}:** {past_ticket}")
                else:
                    st.warning("No similar cases found.")
            except:
                st.error("Failed to connect to Endee. Is the database active?")
                