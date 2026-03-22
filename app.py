import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import uuid

# --- 1. Page Configuration (Must be first) ---
st.set_page_config(page_title="Endee Support AI", page_icon="🛠️", layout="wide")

# --- 2. Left Sidebar (Project Details) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8633/8633104.png", width=80) # Generic AI Support icon
    st.title("About This Project")
    st.markdown("""
    **Enterprise Support AI** is a Retrieval-Augmented Generation (RAG) system built for the ENDEE evaluation.
    
    It empowers customer support agents by instantly finding solutions from historical data based on the *semantic meaning* of a new ticket.
    """)
    
    st.divider()
    
    st.subheader("⚙️ System Architecture")
    st.markdown("""
    - **Frontend:** Streamlit
    - **Embedding Model:** `all-MiniLM-L6-v2`
    - **Vector Database:** Endee (Localhost:8080)
    """)
    
    st.divider()
    st.caption("Developed for the 2026 ENDEE On-Campus Drive.")

# --- 3. AI Model & DB Setup ---
@st.cache_resource
def load_ai_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_ai_model()
ENDEE_URL = "http://localhost:8080"

# --- 4. Main UI Area ---
st.title("🛠️ Enterprise Support AI Dashboard")
st.markdown("Welcome to the support portal. Use the tabs below to navigate the system.")

# Create clean tabs for the interface
tab_search, tab_ingest = st.tabs(["🔍 Smart Ticket Search (Agent View)", "🗄️ Database Ingestion (Admin View)"])

# --- TAB 1: SEARCH (What the user sees first) ---
with tab_search:
    st.markdown("### Find Historical Solutions")
    st.write("Describe the customer's issue below. The AI will search the Endee database for similar past cases.")
    
    query = st.text_input("Customer Issue:", placeholder="E.g., The application crashes when I try to export to PDF...")
    
    if st.button("Search Endee Database", type="primary"):
        if query:
            with st.spinner("Searching vector space..."):
                query_vector = model.encode(query).tolist()
                try:
                    res = requests.post(f"{ENDEE_URL}/api/search", json={"vector": query_vector, "top_k": 3})
                    if res.status_code == 200:
                        results = res.json().get('results', [])
                        st.subheader("Recommended Solutions:")
                        for i, r in enumerate(results):
                            past_ticket = r.get('metadata', {}).get('content', 'No content found')
                            # Display results in nice looking callout boxes
                            st.info(f"**Historical Match {i+1}:**\n\n{past_ticket}")
                    else:
                        st.warning("No similar cases found.")
                except:
                    st.error("🚨 Connection Failed: Ensure the Endee Vector Database is actively running on your system.")
        else:
            st.warning("Please enter a query first.")

# --- TAB 2: INGESTION (Where data is uploaded) ---
with tab_ingest:
    st.markdown("### Train the AI with Past Tickets")
    st.write("Upload historical support tickets. The system will vectorize them and store them securely in Endee.")
    
    ticket_data = st.text_area("Paste historical tickets (one per line):", height=200, 
                               placeholder="User cannot reset password due to email delay.\nScreen goes black upon startup.")
    
    if st.button("Process & Store in Endee"):
        if ticket_data:
            tickets = ticket_data.split('\n')
            success_count = 0
            with st.spinner("Vectorizing and sending to database..."):
                for ticket in tickets:
                    if not ticket.strip(): continue
                    vector = model.encode(ticket).tolist()
                    payload = {"id": str(uuid.uuid4()), "vector": vector, "metadata": {"type": "support_ticket", "content": ticket}}
                    try:
                        res = requests.post(f"{ENDEE_URL}/api/insert", json=payload)
                        if res.status_code == 200: success_count += 1
                    except Exception as e:
                        st.error("🚨 Connection Failed: Ensure the Endee Vector Database is actively running on your system.")
                        break
            if success_count > 0:
                st.success(f"Successfully vectorized and stored {success_count} support tickets!")
        else:
            st.warning("Please paste some text to ingest.")
            