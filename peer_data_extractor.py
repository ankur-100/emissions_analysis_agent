import google.generativeai as genai
import os
import pypdfium2 as pdfium
from PIL import Image
import io
import pandas as pd

# --- Configuration ---
# IMPORTANT: Set your Google API key as an environment variable before running.
# For example, in your terminal: export GOOGLE_API_KEY="your_api_key_here"
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("="*80)
    print("ERROR: GOOGLE_API_KEY environment variable not set.")
    print("Please set your API key to run this script.")
    print("For example: export GOOGLE_API_KEY='your_api_key_here'")
    print("="*80)
    exit()


# Define base directory to locate files relative to the script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")
EMISSIONS_DATA_DIR = os.path.join(BASE_DIR, "emissions_data")

# Define the peer reports, their location, and the pages with data tables.
PEER_REPORTS_CONFIG = {
    "peer1": {
        "filename": os.path.join(KNOWLEDGE_BASE_DIR, "peer1_emissions_report.pdf"),
        "page_number": 49, # As specified in the context
        "output_csv": os.path.join(BASE_DIR, "peer1_data.csv")
    },
    "peer2": {
        "filename": os.path.join(KNOWLEDGE_BASE_DIR, "peer2_emissions_report.pdf"),
        "page_number": 18, # As specified in the context, contains an image table
        "output_csv": os.path.join(BASE_DIR, "peer2_data.csv")
    }
}

# --- Main Extraction Logic ---

def render_pdf_page_to_image(pdf_path, page_number):
    """
    Renders a specific page of a PDF file into a PIL Image object.
    Page numbers are 1-based for user-friendliness, converting to 0-based for the library.
    """
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        if page_number > len(pdf):
            print(f"Error: Page number {page_number} is out of range for {pdf_path}.")
            return None
        # Convert 1-based page number to 0-based index for the library
        page = pdf.get_page(page_number - 1)
        bitmap = page.render(scale=3) # Higher scale for better resolution
        return bitmap.to_pil()
    except Exception as e:
        print(f"Could not render PDF {pdf_path} on page {page_number}. Error: {e}")
        return None

def extract_data_with_gemini(image: Image):
    """
    Uses Gemini 1.5 Flash to extract tabular data from an image into a detailed,
    wide-format CSV.
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are a data extraction specialist. Your task is to analyze an image of an emissions data table and convert it into a single-row CSV string.

    INSTRUCTIONS:
    1.  Identify the data for the **most recent year** in the table.
    2.  From that year's data, extract the values for:
        - Scope 1 emissions.
        - Scope 2 emissions. If the table shows two types (e.g., 'Market-based' and 'Location-based'), extract both.
        - All available Scope 3 sub-categories (e.g., '3.1 purchased goods', '3.6 business travel', etc.).
    3.  Format your output as a **single-row CSV string**.
    4.  The first line must be the header row with descriptive names. For Scope 3, include the number and a short description (e.g., `Scope 3.1 - Purchased Goods`).
    5.  The second line must contain ONLY the corresponding numerical values. Remove all commas, units (like tCO2e), and any other text.
    6.  Provide ONLY the raw CSV string in your response. Do not include introductory text, explanations, or code formatting backticks.

    EXAMPLE OUTPUT FORMAT:
    Scope 1,Scope 2 - Market Based,Scope 2 - Location Based,Scope 3.1 - Purchased Goods and Services,Scope 3.6 - Business Travel
    449,8545,101018,155206,16957
    """
    
    print("Sending image to Gemini for detailed analysis...")
    response = model.generate_content([prompt, image])
    
    try:
        # Clean up the response to ensure it's a pure CSV string
        clean_response = response.text.strip()
        if clean_response.startswith("```csv"):
            clean_response = clean_response.replace("```csv\n", "").replace("```", "")
        print("Received and cleaned response from Gemini.")
        return clean_response
    except Exception as e:
        print(f"Error processing Gemini response: {e}")
        print("Full response was:\n", response.text)
        return None

def main():
    """
    Main function to iterate through peer reports, extract data, and save to CSVs.
    """
    print("Starting peer emissions data extraction process...")
    
    # Create output directory if it doesn't exist
    os.makedirs(EMISSIONS_DATA_DIR, exist_ok=True)
    
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"ERROR: The directory '{KNOWLEDGE_BASE_DIR}' was not found.")
        print("Please create it and place your PDF reports inside.")
        return

    for peer, config in PEER_REPORTS_CONFIG.items():
        pdf_path = config["filename"]
        output_csv_path = config["output_csv"]
        
        if not os.path.exists(pdf_path):
            print(f"\n--- SKIPPING: {peer} ---")
            print(f"File not found: {pdf_path}. Please make sure it exists in your 'knowledge_base' directory.")
            continue
            
        print(f"\n--- Processing: {peer} ({pdf_path}) ---")
        
        page_image = render_pdf_page_to_image(pdf_path, config["page_number"])
        
        if page_image:
            csv_data = extract_data_with_gemini(page_image)
            
            if csv_data:
                try:
                    # Use io.StringIO to treat the string as a file for pandas
                    df = pd.read_csv(io.StringIO(csv_data))
                    df.to_csv(output_csv_path, index=False)
                    print(f"SUCCESS: Data for {peer} extracted and saved to {output_csv_path}")
                    print("Extracted Data Preview:")
                    print(df.head().to_string())
                except Exception as e:
                    print(f"ERROR: Could not parse the Gemini response into a valid CSV for {peer}. Error: {e}")
                    print("Received data was:\n", csv_data)

    print("\nExtraction process complete.")

if __name__ == "__main__":
    main()