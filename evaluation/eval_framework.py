import pandas as pd
import google.generativeai as genai
import os
import sys
import json
import warnings

# Suppress pandasql experimental warning
warnings.filterwarnings("ignore", category=UserWarning, module="pandasql")

# --- Setup: Add app directory to path to allow imports from app.py ---
# This assumes this script is in the same directory as app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Attempt to import the core logic from the Streamlit application file
    from ghg_analysis_agent import (
        load_and_index_ghg_protocol,
        load_company_data,
        load_peer_data,
        get_intent,
        run_protocol_qa_tool,
        run_validation_tool,
        run_comparison_tool,
        run_sql_query_tool,
        run_summary_report_tool,
        run_fallback_tool
    )
except ImportError as e:
    print("="*80)
    print(f"ERROR: Could not import functions from app.py -> {e}")
    print("Please ensure 'app.py' is in the same directory as this script and that its functions")
    print("(e.g., load_company_data, get_intent, run_sql_query_tool) can be imported.")
    print("You may need to restructure app.py to separate the core logic from the Streamlit UI code.")
    print("="*80)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)


# --- Configuration ---
EVALUATION_DATASET_PATH = 'evaluation_dataset.csv'
RESULTS_PATH = 'evaluation_results.csv'

# Configure the Gemini API Key
try:
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    print("="*80)
    print("ERROR: GOOGLE_API_KEY environment variable not set.")
    print("Please set your API key to run this script.")
    print("For example: export GOOGLE_API_KEY='your_api_key_here'")
    print("="*80)
    sys.exit(1)

# --- Core Evaluation Logic ---

def get_ai_evaluation(query, expected_outcome, actual_response):
    """
    Uses Gemini to score the agent's response against the expected outcome.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are an evaluation expert for an AI agent. Your task is to score the agent's response based on a user query and an expected outcome.
    Provide your evaluation in a JSON format.

    **User Query:**
    "{query}"

    **Expected Outcome:**
    "{expected_outcome}"

    **Actual Agent Response:**
    "{actual_response}"

    **Evaluation Criteria:**
    1.  **Factual_Accuracy (score from 0.0 to 1.0):** How factually correct is the agent's response compared to the expected outcome? A score of 1.0 means all facts (numbers, classifications, etc.) are perfect. A score of 0.0 means all facts are incorrect.
    2.  **Relevance_Completeness (score from 0.0 to 1.0):** How well does the agent's response address all parts of the user's query and align with the expected outcome? A score of 1.0 means it is perfectly relevant and complete.

    **Reasoning:**
    Provide a brief, one-sentence reasoning for your scores.

    **Your Response MUST be a single JSON object with the keys "Factual_Accuracy", "Relevance_Completeness", and "Reasoning".**
    
    Example JSON Output:
    {{
      "Factual_Accuracy": 1.0,
      "Relevance_Completeness": 0.8,
      "Reasoning": "The agent correctly calculated the total emissions but did not provide the qualitative insight mentioned in the expected outcome."
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean the response to ensure it's valid JSON
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_text)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  - Warning: Could not parse AI evaluation. Error: {e}")
        return {
            "Factual_Accuracy": 0.0,
            "Relevance_Completeness": 0.0,
            "Reasoning": f"Error during AI evaluation: {e}"
        }

def run_agent_on_query(query, resources):
    """
    Runs the full agent pipeline for a single query.
    """
    intents = get_intent(query, peer_data_available=bool(resources['peer_data']))
    
    full_response = []
    tool_order = ["TOOL_PROTOCOL_QA", "TOOL_VALIDATION", "TOOL_SUMMARY_REPORT", "TOOL_SQL_QUERY", "TOOL_COMPARISON", "TOOL_FALLBACK"]
    sorted_intents = sorted(intents, key=lambda x: tool_order.index(x) if x in tool_order else 99)

    for intent in sorted_intents:
        if intent == "TOOL_PROTOCOL_QA":
            response_part = run_protocol_qa_tool(query, resources['vector_store'])
        elif intent == "TOOL_VALIDATION":
            response_part = run_validation_tool(query, resources['vector_store'], resources['company_data'])
        elif intent == "TOOL_COMPARISON":
            response_part = run_comparison_tool(query, resources['company_data'], resources['peer_data'])
        elif intent == "TOOL_SQL_QUERY":
            response_part = run_sql_query_tool(query, resources['company_data'])
        elif intent == "TOOL_SUMMARY_REPORT":
            response_part = run_summary_report_tool(query, resources['company_data'])
        elif intent == "TOOL_FALLBACK":
             response_part = run_fallback_tool(query, resources['vector_store'], resources['company_data'])
        else:
            response_part = f"Error: Tool '{intent}' not recognized."
        full_response.append(response_part)
        
    return ", ".join(intents), "\n\n---\n\n".join(full_response)

def main():
    """Main function to run the evaluation suite."""
    print("--- Starting GHG Agent Evaluation ---")
    
    # 1. Load resources
    print("Loading resources (this may take a moment)...")
    vector_store = load_and_index_ghg_protocol()
    company_data = load_company_data()
    peer_data = load_peer_data()
    
    if not all([vector_store, company_data]):
        print("ERROR: Failed to load necessary data. Exiting evaluation.")
        return

    resources = {
        'vector_store': vector_store,
        'company_data': company_data,
        'peer_data': peer_data
    }
    print("Resources loaded successfully.")

    # 2. Load evaluation dataset
    try:
        eval_df = pd.read_csv(EVALUATION_DATASET_PATH)
    except FileNotFoundError:
        print(f"ERROR: Evaluation dataset not found at '{EVALUATION_DATASET_PATH}'.")
        return
        
    results = []
    total_queries = len(eval_df)

    # 3. Loop through dataset and run tests
    for i, row in eval_df.iterrows():
        print(f"\n[{i+1}/{total_queries}] Testing Query: \"{row['Query']}\"")
        
        # Run the agent to get the actual tool and response
        actual_tools, actual_response = run_agent_on_query(row['Query'], resources)
        print(f"  - Agent selected tool(s): {actual_tools}")

        # Evaluate the agent's response using the AI evaluator
        evaluation_scores = get_ai_evaluation(row['Query'], row['Expected_Outcome'], actual_response)
        print(f"  - AI Evaluation complete.")

        # Compare tool selection
        tool_selection_correct = (actual_tools == row['Expected_Tool'])
        
        results.append({
            'Query': row['Query'],
            'Expected_Tool': row['Expected_Tool'],
            'Actual_Tool': actual_tools,
            'Tool_Selection_Correct': tool_selection_correct,
            'Actual_Response': actual_response,
            'Factual_Accuracy': evaluation_scores['Factual_Accuracy'],
            'Relevance_Completeness': evaluation_scores['Relevance_Completeness'],
            'AI_Reasoning': evaluation_scores['Reasoning']
        })

    # 4. Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n--- Evaluation Complete ---")
    print(f"Results saved to '{RESULTS_PATH}'")

    # 5. Print summary report
    avg_accuracy = results_df['Factual_Accuracy'].mean()
    avg_relevance = results_df['Relevance_Completeness'].mean()
    tool_accuracy = results_df['Tool_Selection_Correct'].mean()

    print("\n--- Summary Report ---")
    print(f"Total Queries Tested: {total_queries}")
    print(f"Tool Selection Accuracy: {tool_accuracy:.2%}")
    print(f"Average Factual Accuracy: {avg_accuracy:.2%}")
    print(f"Average Relevance & Completeness: {avg_relevance:.2%}")
    print("------------------------\n")


if __name__ == "__main__":
    main()
