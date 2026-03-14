# Enterprise Support AI: RAG System powered by Endee

## Project Overview
This project demonstrates an advanced enterprise application of the Endee Vector Database: an AI-driven Customer Support Retrieval-Augmented Generation (RAG) system. By embedding historical support tickets, businesses can automatically discover common points of failure and empower support agents to find instant solutions based on past data.

## System Design and Technical Approach
1. **Frontend:** A clean, interactive dashboard built with `Streamlit`.
2. **Embedding Architecture:** Utilizes the `all-MiniLM-L6-v2` model to transform unstructured text (customer complaints/tickets) into high-dimensional vector representations.
3. **Vector Storage & Retrieval:** **Endee** acts as the core engine. It stores the complex vector data and performs lightning-fast Nearest Neighbor semantic searches to find historical tickets that match the meaning of a new customer query, even if the exact keywords differ.

## How Endee is Used
* **Ingestion:** When a batch of historical tickets is uploaded, the application vectorizes them and uses an HTTP POST request to store the vectors and their corresponding text metadata directly into the Endee database.
* **Semantic Search:** When an agent types a new customer issue, the system vectorizes the query and asks Endee to retrieve the top `k` most semantically similar past tickets, providing instant context for a resolution.

## Setup Instructions
1. Clone this repository to your local machine.
2. Ensure you have Python installed. Create a virtual environment and activate it.
3. Run `pip install -r requirements.txt` to install dependencies.
4. Ensure your local instance of the **Endee Vector Database** is actively running on port 8080.
5. Start the application by running: `streamlit run app.py`