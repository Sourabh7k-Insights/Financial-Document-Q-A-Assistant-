
# 📊 Financial Document Analyst Chatbot
A Streamlit-based web app enabling users to upload financial documents (PDF) and interactively query financial metrics via natural language. The app extracts meaningful data and answers user questions about revenue, expenses, profits, and more.

# 🚀  Features
Clean, intuitive web interface for document upload and interactive chat

Supports PDF uploads

Context-aware Q&A on financial metrics from uploaded documents

Displays extracted financial information in readable format

Real-time feedback on processing status and chatbot responses

# 🛠️ Setup Instructions
### 1. Clone the Repository

git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

### 2. Create and Activate a Virtual Environment (optional but recommended)

python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Set Up Environment Variables


Create a .env file in your project root:
HF_TOKEN=your-huggingface-token
Never upload your real .env file—share a .env.example safely.

### 5. Install Ollama and Pull Model


Make sure Ollama is installed locally.
Pull the desired model (example for Llama 3):
ollama pull llama3

### 6. Run the Streamlit Application

streamlit run app.py
### 7. Access the App


Open your browser and go to http://localhost:8501



