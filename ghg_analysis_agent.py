import asyncio
import re
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandasql as ps


load_dotenv()


def _ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

_ensure_event_loop()

# --- Page Configuration ---
st.set_page_config(
    page_title="GHG Emissions Analysis Agent",
    page_icon="🌍",
    layout="wide"
)


# --- API Key Configuration ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (FileNotFoundError, KeyError):
    try:
        GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
        genai.configure(api_key=GOOGLE_API_KEY)
    except KeyError:
        st.error("🚨 GOOGLE_API_KEY not found! Please set it in your Streamlit secrets or as an environment variable.")
        st.stop()
        
# --- File Path Definitions ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")
EMISSIONS_DATA_DIR = os.path.join(BASE_DIR, "emissions_data")

# --- Caching and Data Loading ---
@st.cache_resource
def load_and_index_ghg_protocol():
    """Loads the GHG Protocol PDF, splits it into chunks, and creates a searchable vector store."""
    protocol_path = os.path.join(KNOWLEDGE_BASE_DIR, "ghg-protocol-revised.pdf")
    if not os.path.exists(protocol_path):
        st.error(f"{protocol_path} not found. Please ensure it's in the 'knowledge_base' directory.")
        return None
    try:
        st.info("Loading and indexing the GHG Protocol document... (This may take a moment)")
        loader = PyPDFLoader(protocol_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(docs, embeddings)
        st.success("GHG Protocol indexed successfully!")
        return vector_store
    except Exception as e:
        st.error(f"Failed to load or index GHG Protocol: {e}")
        return None

@st.cache_data
def load_company_data():
    """Loads the company's scope 1, 2, and 3 emissions data from CSV files."""
    data = {}
    try:
        data['scope1'] = pd.read_csv(os.path.join(EMISSIONS_DATA_DIR, 'scope1.csv'))
        data['scope2'] = pd.read_csv(os.path.join(EMISSIONS_DATA_DIR, 'scope2.csv'))
        data['scope3'] = pd.read_csv(os.path.join(EMISSIONS_DATA_DIR, 'scope3.csv'))
        return data
    except FileNotFoundError as e:
        st.error(f"Error: Make sure `{os.path.basename(e.filename)}` is in the 'emissions_data' directory.")
        return None

@st.cache_data
def load_peer_data():
    """Loads the pre-processed peer data from the generated CSV files."""
    data = {}
    peer1_path = os.path.join(EMISSIONS_DATA_DIR, 'peer1_data.csv')
    peer2_path = os.path.join(EMISSIONS_DATA_DIR, 'peer2_data.csv')
    if os.path.exists(peer1_path):
        data['peer1'] = pd.read_csv(peer1_path)
    if os.path.exists(peer2_path):
        data['peer2'] = pd.read_csv(peer2_path)
    return data


# --- Agent Tools ---
def run_protocol_qa_tool(query: str, vector_store):
    """Answers questions based solely on the GHG Protocol document."""
    st.info("Using GHG Protocol Knowledge Base to answer...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = f"""
    You are an expert on the GHG Protocol. Answer the user's question based *only* on the provided context.
    If the context does not contain the answer, state that clearly. Be concise and clear in your explanation.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    ANSWER:
    """
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    return response.text

def run_validation_tool(query: str, vector_store, company_data):
    """Validates company's emissions data and emission factors against the GHG Protocol."""
    st.info("Validating company data and emission factors against GHG Protocol...")
    
    scopes_to_validate = []
    query_lower = query.lower()
    
    if "all scopes" in query_lower:
        scopes_to_validate = ["scope1", "scope2", "scope3"]
    else:
        if "scope 1" in query_lower: scopes_to_validate.append("scope1")
        if "scope 2" in query_lower: scopes_to_validate.append("scope2")
        if "scope 3" in query_lower: scopes_to_validate.append("scope3")
    
    if not scopes_to_validate:
        return "Could not determine which scope(s) to validate. Please specify 'Scope 1', 'Scope 2', 'Scope 3', or 'all scopes' in your query."
    
    final_analysis = []

    for scope in scopes_to_validate:
        if scope not in company_data:
            final_analysis.append(f"### Validation for {scope.replace('scope', 'Scope ')}\n\nCompany data for {scope} is not available.")
            continue

        df = company_data[scope]
        
        # Determine the correct column name for activity type based on the file
        activity_col = 'Activity_Type' if scope == 'scope1' else \
                       'Energy_Type' if scope == 'scope2' else \
                       'Activity_Description' if scope == 'scope3' else None

        if not activity_col or activity_col not in df.columns or 'Emission_Factor' not in df.columns:
            final_analysis.append(f"### Validation for {scope.replace('scope', 'Scope ')}\n\nCould not find required 'Activity' or 'Emission_Factor' columns in the {scope} data.")
            continue
        
        # Group activities and collect all associated emission factors
        activities_and_factors = df.groupby(activity_col)['Emission_Factor'].apply(list).to_dict()
        
        # Format this data for the prompt
        formatted_data = "\n".join([f"- {activity}: {', '.join(map(str, sorted(list(set(factors)))))}" for activity, factors in activities_and_factors.items()])

        retriever = vector_store.as_retriever()
        search_query = f"guidance, definitions, and calculation methodologies for {scope.replace('scope', 'Scope ')} emissions"
        relevant_docs = retriever.get_relevant_documents(search_query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""
        You are a GHG emissions auditor. A user wants to validate their {scope.replace('scope', 'Scope ')} emission calculations.

        Here is the official guidance from the GHG Protocol:
        ---
        {context}
        ---
        
        Here is a list of unique 'activity types' and their associated 'Emission Factors' recorded in the company's {scope.replace('scope', 'Scope ')} data file:
        ---
        {formatted_data}
        ---
        
        Based *only* on the GHG Protocol guidance, perform an analysis:
        **Classification Analysis:** For each activity, analyze if it is correctly classified under {scope.replace('scope', 'Scope ')}.
        Provide a step-by-step reasoning for your conclusions. Structure your response clearly with headings for each part of the analysis.
        Assume per unit of activity is in - kg CO2e/ton
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        final_analysis.append(f"### Analysis for {scope.replace('scope', 'Scope ')}\n\n{response.text}")

    return "\n\n---\n\n".join(final_analysis)

def run_comparison_tool(query: str, company_data, peer_data):
    """Compares the company's emissions with peers for any specified scope or sub-scope."""
    st.info("Comparing company data with peers...")

    # Extract potential comparison keywords from the query. More flexible.
    # Looks for things like "Scope 1", "Scope 3.6", "Business Travel", "all scopes"
    keywords = re.findall(r'scope\s?\d(\.\d{1,2})?|business travel|employee commuting|purchased goods|all scopes', query, re.IGNORECASE)
    
    if not keywords:
        return "Could not determine what to compare. Please specify a scope (e.g., 'Scope 1'), a sub-category (e.g., 'Business Travel'), or 'all scopes'."

    if "all scopes" in [k.lower() for k in keywords]:
        keywords = ["Scope 1", "Scope 2", "Scope 3"]

    final_analysis = []

    for keyword in set(keywords): # Use set to avoid duplicate processing
        keyword_lower = keyword.lower()
        my_company_total = 0
        
        # --- Calculate My Company's Total for the Keyword ---
        if "scope 1" in keyword_lower:
            my_company_total = company_data['scope1']['CO2e_Tonnes'].sum()
            topic = "Scope 1"
        elif "scope 2" in keyword_lower:
            my_company_total = company_data['scope2']['CO2e_Tonnes'].sum()
            topic = "Scope 2"
        elif "scope 3" in keyword_lower and not any(char.isdigit() for char in keyword_lower.replace("scope 3","")):
            my_company_total = company_data['scope3']['CO2e_Tonnes'].sum()
            topic = "Scope 3"
        else: # It's a sub-category search
            topic = keyword.title()
            # Search within the 'Activity_Description' column of scope3 data
            mask = company_data['scope3']['Activity_Description'].str.contains(keyword, case=False, na=False)
            my_company_total = company_data['scope3'][mask]['CO2e_Tonnes'].sum()

        # --- Find Peer Data Matching the Keyword ---
        peer_comparison_text = []
        for peer, df in peer_data.items():
            # Search for the keyword in the column headers of the peer's dataframe
            matching_cols = [col for col in df.columns if keyword_lower in col.lower()]
            if matching_cols:
                peer_value = df[matching_cols].sum().sum()
                if peer_value > 0:
                    peer_comparison_text.append(f"- {peer.capitalize()}: {peer_value:,.2f} tonnes\n")

        analysis_title = f"Comparison for {topic}"
        if not peer_comparison_text:
            final_analysis.append(f"### {analysis_title}\n\nOur Company Total: **{my_company_total:,.2f} tonnes**.\n\nNo comparable data found for this category in the peer reports.")
            continue
        
        prompt = f"""
        You are a sustainability analyst. Provide a brief analysis comparing our company's emissions with its peers for {analysis_title}.
        Derive one or two key insights from the comparison. Keep the tone professional and data-driven.

        Here is the data:
        - Our Company's Emissions for {topic}: {my_company_total:,.2f} tonnes
        Peer Data for {topic}:
        {''.join(peer_comparison_text)}
        
        ANALYSIS AND INSIGHTS for {analysis_title}:
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        final_analysis.append(f"### {analysis_title}\n\n{response.text}")

    return "\n\n---\n\n".join(final_analysis)

def run_summary_report_tool(query: str, company_data):
    """
    Generates a flexible summary report based on specified scopes in the query.
    Defaults to a full report if no scopes are specified.
    """
    st.info("Generating summary report...")
    
    scope1 = company_data['scope1']
    scope2 = company_data['scope2']
    scope3 = company_data['scope3']
    env = {'scope1': scope1, 'scope2': scope2, 'scope3': scope3}

    query_lower = query.lower()
    requested_scopes = []
    if "scope 1" in query_lower: requested_scopes.append("scope1")
    if "scope 2" in query_lower: requested_scopes.append("scope2")
    if "scope 3" in query_lower: requested_scopes.append("scope3")
    
    # If no specific scope is mentioned, default to all scopes
    if not requested_scopes:
        requested_scopes = ["scope1", "scope2", "scope3"]

    try:
        data_for_prompt = []
        s1_total = s2_total = s3_total = 0

        if "scope1" in requested_scopes:
            s1_total = ps.sqldf("SELECT SUM(CO2e_Tonnes) FROM scope1", env).iloc[0, 0]
            s1_breakdown = ps.sqldf("SELECT Fuel_Type, SUM(CO2e_Tonnes) as Total FROM scope1 GROUP BY Fuel_Type ORDER BY Total DESC", env)
            data_for_prompt.append(f"- Total Scope 1 Emissions: {s1_total:,.2f} tCO2e")
            data_for_prompt.append(f"- Scope 1 Breakdown by Fuel Type:\n{s1_breakdown.to_string(index=False)}")

        if "scope2" in requested_scopes:
            s2_total = ps.sqldf("SELECT SUM(CO2e_Tonnes) FROM scope2", env).iloc[0, 0]
            data_for_prompt.append(f"- Total Scope 2 Emissions: {s2_total:,.2f} tCO2e")

        if "scope3" in requested_scopes:
            s3_total = ps.sqldf("SELECT SUM(CO2e_Tonnes) FROM scope3", env).iloc[0, 0]
            s3_breakdown = ps.sqldf("SELECT Activity_Description, SUM(CO2e_Tonnes) as Total FROM scope3 GROUP BY Activity_Description ORDER BY Total DESC LIMIT 5", env)
            data_for_prompt.append(f"- Total Scope 3 Emissions: {s3_total:,.2f} tCO2e")
            data_for_prompt.append(f"- Top 5 Contributors to Scope 3 Emissions:\n{s3_breakdown.to_string(index=False)}")

        total_emissions = s1_total + s2_total + s3_total
        if len(requested_scopes) > 1:
             data_for_prompt.insert(0, f"- Overall Total Emissions for Requested Scopes: {total_emissions:,.2f} tCO2e")


        prompt = f"""
        You are a senior sustainability analyst preparing a high-level summary report for management based on the user's request.
        Based on the following data points, generate a concise report.

        The report should include:
        1.  An executive summary of the total emissions for the requested scope(s).
        2.  A breakdown of emissions if available.
        3.  Key insights or observations, such as identifying the largest emission sources within the requested scope(s).
        4.  Use clear headings and bullet points for readability.

        --- DATA FOR ANALYSIS ---
        {chr(10).join(data_for_prompt)}
        ---

        Generate the report now.
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"An error occurred while generating the report: {e}"

def run_sql_query_tool(query: str, company_data):
    """
    Uses an LLM to convert a natural language question into a SQL query,
    executes it against the company's dataframes, and returns the result.
    """
    st.info("Performing data analysis query on company emissions...")
    
    scope1 = company_data['scope1']
    scope2 = company_data['scope2']
    scope3 = company_data['scope3']
    
    scope1_schema = ", ".join(scope1.columns)
    scope2_schema = ", ".join(scope2.columns)
    scope3_schema = ", ".join(scope3.columns)

    few_shot_prompt = f"""
    You are an expert at converting natural language questions into SQL queries.
    You will be querying pandas dataframes named 'scope1', 'scope2', and 'scope3'.
    
    Here are the schemas for the tables:
    - scope1: {scope1_schema}
    - scope2: {scope2_schema}
    - scope3: {scope3_schema}

    Based on the user's question, generate SINGLE/MULTIPLE, valid SQL query.
    Respond with ONLY the SQL query, without any explanation or formatting.

    --- FEW-SHOT EXAMPLES ---
    
    Question: "What is the total scope 1 emission from the Manufacturing Plant?"
    SQL Query:
    SELECT SUM(CO2e_Tonnes) FROM scope1 WHERE Facility = 'Manufacturing Plant';

    Question: "Show me the breakdown of scope 3 emissions by activity description, sorted from highest to lowest."
    SQL Query:
    SELECT Activity_Description, SUM(CO2e_Tonnes) as TotalEmissions FROM scope3 GROUP BY Activity_Description ORDER BY TotalEmissions DESC;

    Question: "how I can calculate my business travel emissions?"
    SQL Query:
    SELECT SUM(CO2e_Tonnes) FROM scope3 WHERE Activity_Description LIKE '%Business Travel%';
    
    --- END OF EXAMPLES ---

    Question: "{query}"
    SQL Query:
    """

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(few_shot_prompt)
    sql_query = response.text.strip().replace("`", "").replace("sql", "")
    
    st.info(f"Generated SQL Query: `{sql_query}`")

    try:
        # **FIX:** Use an explicit environment dictionary instead of locals() for robustness.
        env = {'scope1': scope1, 'scope2': scope2, 'scope3': scope3}
        result_df = ps.sqldf(sql_query, env)
        
        if result_df.empty:
            return "The query ran successfully but returned no results."
            
        summary_prompt = f"""
        The user asked the following question: "{query}"
        The following SQL query was executed: "{sql_query}"
        The query returned this data:
        {result_df.to_string()}

        Provide a brief, natural language summary of the result.
        """
        summary_response = model.generate_content(summary_prompt)

        return f"{summary_response.text}\n\n**Query Result:**\n\n" + result_df.to_markdown(index=False)

    except Exception as e:
        return f"There was an error executing the query: {e}. Please try rephrasing your question."

def run_fallback_tool(query: str, vector_store, company_data):
    """
    A fallback tool that searches both the GHG protocol and company data
    to provide a helpful answer when other tools aren't a direct match.
    """
    st.info("No specific tool matched. Using general knowledge search...")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(query)
    protocol_context = "\n\n".join([doc.page_content for doc in relevant_docs])

    data_context_list = []
    for scope_name, df in company_data.items():
        df_str = df.astype(str).apply(lambda x: ' '.join(x), axis=1)
        if any(keyword in df_str.str.cat(sep=' ').lower() for keyword in query.lower().split()):
            data_context_list.append(f"Found mentions in your {scope_name} data.")

    data_context = "\n".join(data_context_list) if data_context_list else "No direct matches found in your company's emissions data files."

    prompt = f"""
    You are a helpful GHG Emissions assistant. The user asked a question that didn't fit a specific tool.
    Your task is to provide a helpful response by combining information from the GHG Protocol and a summary of relevant company data.

    User Question: "{query}"

    --- GHG Protocol Context ---
    {protocol_context}

    --- Company Data Context ---
    {data_context}
    ---

    Based on all the context above, provide a comprehensive answer to the user's question.
    If the protocol context is relevant, explain it. If the data context is relevant, mention it.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

# --- Router ---
def get_intent(query: str, peer_data_available: bool):
    """Uses Gemini to classify the user's intent and select one or more tools."""
    
    tool_descriptions = [
        "TOOL_SUMMARY_REPORT: Use for broad requests for summaries, reports, or key insights about the company's emissions, whether for all scopes or specific ones (e.g., 'scope 1 report'). Keywords: 'summary', 'report', 'insights', 'overview'.",
        "TOOL_SQL_QUERY: Use for specific, analytical questions about the company's own data that require calculations, aggregations, or filtering. Keywords: 'calculate', 'total', 'sum', 'average', 'breakdown', 'what is the', 'how can I calculate'.",
        "TOOL_PROTOCOL_QA: Use for general questions about GHG accounting, definitions, or methodologies from the GHG Protocol document. Keywords: 'what is', 'explain', 'define'. Does not require company data.",
        "TOOL_VALIDATION: Use when the user asks to validate or check their company's data against the protocol. Keywords: 'is my', 'are our', 'valid', 'correct'.",
        "TOOL_FALLBACK: Use this as a last resort if no other tool is a clear match for the user's query.",
    ]
    
    if peer_data_available:
        tool_descriptions.insert(0, "TOOL_COMPARISON: Use when comparing emissions against other companies. Keywords: 'compare', 'stack up', 'against others', 'all scopes', 'business travel'.")
    
    prompt = f"""
    Based on the user's query, identify all the appropriate tools to use. A query can sometimes require multiple tools.
    Respond with a comma-separated list of tool names (e.g., 'TOOL_PROTOCOL_QA, TOOL_SQL_QUERY').
    If no other tool is a clear fit, respond with 'TOOL_FALLBACK'.

    Here are the available tools:
    ---
    {chr(10).join(tool_descriptions)}
    ---

    User Query: "{query}"

    Selected Tool(s):
    """
    
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    
    tool_names = [tool.strip() for tool in response.text.strip().replace("`", "").upper().split(',')]
    print(f"Query: '{query}' -> Selected Tools: '{tool_names}'") 
    return tool_names


# --- Main App UI ---
st.title("🌍 GHG Emissions Analysis Agent")
st.markdown("Ask me to analyze your company's emissions data, validate it against the GHG Protocol, or compare it with peers.")

vector_store = load_and_index_ghg_protocol()
company_data = load_company_data()
peer_data = load_peer_data()

if vector_store and company_data:
    st.sidebar.success("All data loaded successfully!")
    st.sidebar.subheader("Your Company's Data")
    st.sidebar.metric("Scope 1 Emissions (tCO2e)", f"{company_data['scope1']['CO2e_Tonnes'].sum():,.0f}")
    st.sidebar.metric("Scope 2 Emissions (tCO2e)", f"{company_data['scope2']['CO2e_Tonnes'].sum():,.0f}")
    st.sidebar.metric("Scope 3 Emissions (tCO2e)", f"{company_data['scope3']['CO2e_Tonnes'].sum():,.0f}")
    
    if peer_data:
        st.sidebar.subheader("Peer Data Loaded")
        for peer, df in peer_data.items():
            with st.sidebar.expander(f"View {peer.capitalize()} Data"):
                st.dataframe(df)
    else:
        st.sidebar.warning("Peer data not loaded. Run `extract_peer_data.py` first.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    st.markdown("---")
    st.write("Or, try one of these questions:")
    cols = st.columns(4)
    q1 = "Should employee business travel be classified as Scope 1 or Scope 3? Explain and show me how to calculate it."
    q2 = "Is my scope 2 calculation valid according to the protocol?"
    q3 = "How does my Business Travel emission compare with peers?"
    q4 = "What is the total CO2e from Diesel in Scope 1?"
    
    if cols[0].button(q1, use_container_width=True): st.session_state.user_input = q1
    if cols[1].button(q2, use_container_width=True): st.session_state.user_input = q2
    if cols[2].button(q3, use_container_width=True, disabled=not bool(peer_data)): st.session_state.user_input = q3
    if cols[3].button(q4, use_container_width=True): st.session_state.user_input = q4
        
    prompt = st.chat_input("What would you like to know?")
    if "user_input" in st.session_state and st.session_state.user_input:
        prompt = st.session_state.user_input
        st.session_state.user_input = ""

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                intents = get_intent(prompt, peer_data_available=bool(peer_data))
                
                full_response = []
                
                tool_order = ["TOOL_PROTOCOL_QA", "TOOL_VALIDATION", "TOOL_SQL_QUERY", "TOOL_COMPARISON", "TOOL_FALLBACK"]
                sorted_intents = sorted(intents, key=lambda x: tool_order.index(x) if x in tool_order else 99)

                for intent in sorted_intents:
                    response_part = ""
                    if "TOOL_PROTOCOL_QA" in intent:
                        response_part = run_protocol_qa_tool(prompt, vector_store)
                    elif "TOOL_VALIDATION" in intent:
                        response_part = run_validation_tool(prompt, vector_store, company_data)
                    elif "TOOL_COMPARISON" in intent:
                        response_part = run_comparison_tool(prompt, company_data, peer_data)
                    elif "TOOL_SQL_QUERY" in intent:
                        response_part = run_sql_query_tool(prompt, company_data)
                    elif "TOOL_SUMMARY_REPORT" in intent:
                        response_part = run_summary_report_tool(prompt, company_data)
                    elif "TOOL_FALLBACK" in intent:
                        response_part = run_fallback_tool(prompt, vector_store, company_data)
                    else:
                        response_part = f"I'm sorry, I could not determine how to answer that. The tool '{intent}' is not recognized."
                    
                    full_response.append(response_part)

                final_response = "\n\n---\n\n".join(full_response)
                st.markdown(final_response, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
else:
    st.warning("Application is not ready. Please check for errors above regarding data loading.")

