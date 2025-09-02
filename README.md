# GHG Emissions Analysis Agent
This project is an AI-powered agent designed to serve as an expert assistant for carbon emissions analysis. It provides a simple conversational interface to analyze and interpret emissions data from various sources, including company-specific CSV files, the GHG Protocol knowledge base, and competitor reports.

## Core Capabilities 
The agent is built with a tool-based architecture to provide a range of specialized functions:

- **Knowledge Base Q&A**: Answers general questions about emissions accounting by referencing the GHG Protocol.
- **Data Validation**: Audits the company's emissions data against the protocol's guidelines, checking both activity classification and the plausibility of emission factors.
- **Comparative Analysis**: Benchmarks the company's emissions against pre-processed peer data, allowing for granular comparisons down to the sub-scope level.
- **Text-to-SQL Analysis**: Converts natural language questions into SQL queries to perform specific calculations and data retrieval from the company's CSVs.
- **Automated Reporting**: Generates dynamic summary reports with key insights for any combination of emission scopes.

## Technical Architecture
The system is architected as an agent that uses a tool-based design to ensure reliability and prevent hallucination. The core of the agent is a Router powered by Google's Gemini 1.5 Flash model, which directs user queries to the appropriate specialized tool.

- **Core Logic**: Python  
- **AI Model**: Google Gemini 1.5 Flash  
- **Frameworks**: LangChain (for RAG), Pandas, pandasql  
- **UI**: Streamlit  

## Setup and Installation
Follow these steps to get the agent running on your local machine.

### 1. Prerequisites
- Python 3.8+  
- An environment variable set for your Google API Key.

### 2. Project Structure

```
.
├── knowledge_base/
│   ├── ghg-protocol-revised.pdf
│   ├── peer1_emissions_report.pdf
│   └── peer2_emissions_report.pdf
├── emissions_data/
│   ├── scope1.csv
│   ├── scope2.csv
│   └── scope3.csv
├── ghg_analysis_agent.py
├── extract_peer_data.py
├── evaluate_agent.py
├── evaluation_dataset.csv
└── requirements.txt
```

### 3. Installation
1. Clone the repository (if applicable) or set up your folder.  
2. Install dependencies:  
   Open your terminal in the project root and run:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Set your Google API Key:  
   You must set your API key as an environment variable.  

   On macOS/Linux:  
   ```bash
   export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
   ```  

   On Windows (Command Prompt):  
   ```bash
   set GOOGLE_API_KEY="YOUR_API_KEY_HERE"
   ```  

## How to Use the Agent
Running the agent is a two-step process.

### Step 1: Pre-process Peer Data (One-Time Step)
First, we must run the script to extract data from the peer PDF reports. This script uses Gemini's multi-modal capabilities to read the tables (even if they are images) and save them as clean CSV files.

```bash
python extract_peer_data.py
```

This will create `peer1_data.csv` and `peer2_data.csv` inside the `emissions_data` folder. You only need to run this step once, or whenever your peer reports are updated.

### Step 2: Launch the Streamlit Application
Now you can start the interactive agent.

```bash
streamlit run ghg_analysis_agent.py
```

Web browser will open with the application interface, ready for you to ask questions.

## Evaluation
The project includes a robust, automated evaluation framework to ensure the agent's accuracy and reliability.

- **Evaluation Dataset (`evaluation_dataset.csv`)**: A curated list of test questions with expected outcomes.  
- **Automated Script (`evaluate_agent.py`)**: A script that runs all test queries against the agent.  
- **LLM-as-a-Judge**: The script uses Gemini to programmatically score the agent's responses on Factual Accuracy and Relevance, saving the results to `evaluation_results.csv`.  

To run the evaluation, execute the following command:  

```bash
python evaluate_agent.py
```
