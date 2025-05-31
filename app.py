import streamlit as st
import google.generativeai as genai
from PIL import Image
import fitz  # PyMuPDF (for PDF text and image extraction)
# from pdf2image import convert_from_bytes # Alternative for PDF to image
import io
import pandas as pd
import csv # For more robust CSV parsing/generation if needed

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Data Extrtr By Sumit Yadav")

# --- Helper Functions ---

def get_gemini_api_key():
    """Gets Gemini API key from Streamlit secrets or user input."""
    # Try to get from Streamlit secrets first (for deployed apps)
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (FileNotFoundError, KeyError):
        # Fallback to sidebar input for local development
        api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password", key="api_key_input")
        if not api_key:
            st.sidebar.warning("Please enter your Gemini API Key to proceed.")
            st.stop()
        return api_key

def extract_text_from_pdf_page(pdf_bytes, page_num):
    """Extracts text from a single page of a PDF."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if page_num < len(doc):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            doc.close()
            return text
        doc.close()
        return ""
    except Exception as e:
        st.error(f"Error extracting text from PDF page {page_num + 1}: {e}")
        return ""

def convert_pdf_page_to_image(pdf_bytes, page_num):
    """Converts a single PDF page to a PIL Image."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if page_num < len(doc):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=200)  # Higher DPI for better OCR
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            doc.close()
            return img
        doc.close()
        return None
    except Exception as e:
        st.error(f"Error converting PDF page {page_num + 1} to image: {e}")
        return None

def get_pdf_page_count(pdf_bytes):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0

def call_gemini_api(api_key, prompt_parts, model_name="gemini-2.5-flash-preview-05-20"):
    """Calls the Gemini API with the given prompt parts."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        st.error(f"Gemini API call failed: {e}")
        # You might want to inspect response.prompt_feedback if available
        # For example: if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
        # st.error(f"Prompt Feedback: {response.prompt_feedback}")
        return None

def parse_llm_response_to_df(response_text, headers_from_prompt):
    """
    Parses LLM's CSV-like response into a Pandas DataFrame.
    Assumes pipe ('|') as delimiter and first line is header.
    """
    if not response_text:
        return pd.DataFrame()

    lines = response_text.strip().split('\n')
    if not lines:
        st.warning("LLM response was empty after stripping.")
        return pd.DataFrame()

    # Use pipe as delimiter as requested in the prompt
    delimiter = '|'

    # The first line from LLM *should* be the header
    llm_headers_raw = lines[0].split(delimiter)
    llm_headers = [h.strip() for h in llm_headers_raw]

    # Validate if LLM headers match expected (or use LLM headers directly)
    # For simplicity, let's try to use the LLM's headers.
    # If you want strict matching to your prompt_headers, add checks here.
    if not llm_headers or not all(llm_headers): # check for empty headers
        st.warning(f"LLM did not return valid headers. Got: {llm_headers_raw}")
        # Fallback or error
        return pd.DataFrame()

    data_rows = []
    if len(lines) > 1:
        for line in lines[1:]:
            if line.strip(): # Skip empty lines
                values = [v.strip() for v in line.split(delimiter)]
                if len(values) == len(llm_headers):
                    data_rows.append(dict(zip(llm_headers, values)))
                else:
                    st.warning(f"Row column count mismatch. Expected {len(llm_headers)}, got {len(values)}. Row: '{line}'")
                    # Attempt to pad or truncate if desired, or just skip
                    # For now, skipping malformed rows

    if not data_rows:
        st.warning("No valid data rows parsed from LLM response.")
        return pd.DataFrame()

    return pd.DataFrame(data_rows)


# --- Streamlit App UI ---
st.title("📄 Document Data Extractor by Sumit Yadav🤖")
st.markdown("Upload a PDF or Image, specify the fields to extract, and let Gemini do the work!")

# --- API Key Input ---
API_KEY = get_gemini_api_key() # Ensure API key is available

# --- User Inputs ---
uploaded_file = st.file_uploader("1. Upload a PDF or Image file", type=["pdf", "png", "jpg", "jpeg"])

default_fields = "S.No., Date, Product Name, Model No., Make, Supplier, Invoice No., Project, Unit, Rate, Received date, Location, Units Left, Cost, Total Cost, Payment Status, Material Age (Days), Comments"
target_fields_str = st.text_area(
    "2. Specify fields to extract (comma-separated, these will be your CSV headers):",
    value=default_fields,
    height=100,
    help="These fields will be requested from the AI. The AI will attempt to create CSV headers based on these."
)
headers_for_prompt = [h.strip() for h in target_fields_str.split(',') if h.strip()]

# PDF Processing Options
pdf_processing_method = "text_and_image" # Default to combined
if uploaded_file and uploaded_file.type == "application/pdf":
    pdf_processing_method = st.radio(
        "PDF Processing Method:",
        ("text_priority", "image_per_page", "text_and_image_per_page"),
        index=2, # Default to text_and_image_per_page
        help="""
        - **text_priority**: Extracts all text from PDF. Good for text-heavy, clean PDFs.
        - **image_per_page**: Converts each PDF page to an image. Good for scanned/image-based PDFs.
        - **text_and_image_per_page**: Sends both text and image for each page. Most comprehensive but uses more tokens.
        """
    )

if st.button("🚀 Extract Data", type="primary") and uploaded_file and API_KEY and headers_for_prompt:
    if not headers_for_prompt:
        st.error("Please specify the fields you want to extract.")
    else:
        with st.spinner("Processing document and querying Gemini... This might take a moment."):
            prompt_parts = []
            document_description = f"The document is a '{uploaded_file.type}'. "

            if uploaded_file.type == "application/pdf":
                file_bytes = uploaded_file.getvalue()
                num_pages = get_pdf_page_count(file_bytes)
                st.info(f"Processing PDF with {num_pages} page(s) using '{pdf_processing_method}' method...")

                if pdf_processing_method == "text_priority":
                    all_text = ""
                    for i in range(num_pages):
                        page_text = extract_text_from_pdf_page(file_bytes, i)
                        all_text += f"\n--- PDF Page {i+1} Text ---\n{page_text}"
                    if all_text:
                        prompt_parts.append(all_text)
                    else:
                        st.warning("No text could be extracted using text_priority. Try 'image_per_page'.")
                        st.stop()

                elif pdf_processing_method == "image_per_page":
                    for i in range(num_pages):
                        st.write(f"Converting PDF Page {i+1} to image...")
                        img = convert_pdf_page_to_image(file_bytes, i)
                        if img:
                            prompt_parts.append(f"--- Content from PDF Page {i+1} (Image) ---")
                            prompt_parts.append(img)
                        else:
                            st.warning(f"Could not convert PDF page {i+1} to image.")
                    if not any(isinstance(p, Image.Image) for p in prompt_parts):
                        st.error("No images could be extracted from the PDF.")
                        st.stop()

                elif pdf_processing_method == "text_and_image_per_page":
                    for i in range(num_pages):
                        st.write(f"Processing PDF Page {i+1} (text and image)...")
                        page_text = extract_text_from_pdf_page(file_bytes, i)
                        img = convert_pdf_page_to_image(file_bytes, i)
                        prompt_parts.append(f"--- Content from PDF Page {i+1} ---")
                        if page_text:
                            prompt_parts.append(f"[Text from page {i+1}]:\n{page_text}")
                        if img:
                            prompt_parts.append(f"[Image of page {i+1}]:")
                            prompt_parts.append(img)
                        if not page_text and not img:
                             st.warning(f"Could not extract text or image from PDF page {i+1}.")


            elif uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
                document_description = "The document is an image. "
                img = Image.open(uploaded_file)
                prompt_parts.append("--- Document Image ---")
                prompt_parts.append(img)
            else:
                st.error("Unsupported file type.")
                st.stop()

            # Construct the main instruction prompt
            # Using pipe delimiter for CSV as commas can be in data
            instruction_prompt = f"""
            {document_description}
            Your task is to extract information based on the following fields: {', '.join(headers_for_prompt)}.
            The document might contain one or more items or records. For each distinct item/record, extract its details.
            If the document is an invoice or quotation with line items, each line item should be a separate record.
            Global information (like a main Invoice No. or Supplier that applies to all items) should be repeated for each item if it makes sense for the requested fields, or handled appropriately.

            Format your entire output as a pipe-delimited ('|') string, suitable for CSV parsing.
            The VERY FIRST line of your output MUST be the header row, using EXACTLY these field names, pipe-delimited:
            "{'|'.join(headers_for_prompt)}"

            Each subsequent line should represent one extracted record/item, with values also pipe-delimited.
            If a value for a specific field is not found for an item, use "N/A" or leave it blank, but ensure the correct number of delimiters to maintain column structure.
            Ensure all text content from the document is considered.
            Extract the data now.
            """
            prompt_parts.append(instruction_prompt) # Add the main instruction prompt at the end

            # Call Gemini API
            st.write("Sending request to Gemini...")
            gemini_response_text = call_gemini_api(API_KEY, prompt_parts)

            if gemini_response_text:
                st.subheader("Raw Response from Gemini:")
                st.text_area("LLM Output", gemini_response_text, height=200)

                st.subheader("Parsed Data:")
                df = parse_llm_response_to_df(gemini_response_text, headers_for_prompt)

                if not df.empty:
                    st.dataframe(df)

                    # --- Download Buttons ---
                    csv_export = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download data as CSV",
                        data=csv_export,
                        file_name='extracted_data.csv',
                        mime='text/csv',
                    )

                    txt_export = df.to_string(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download data as TXT",
                        data=txt_export,
                        file_name='extracted_data.txt',
                        mime='text/plain',
                    )
                else:
                    st.warning("Could not parse structured data from Gemini's response based on the expected format. Check the raw response.")
            else:
                st.error("Failed to get a response from Gemini.")

# --- Instructions & Tips ---
st.sidebar.title("💡 How to Use")
st.sidebar.info(
    """
    1.  Enter your Gemini API Key.
    2.  Upload a PDF or image file containing the data.
    3.  Verify or customize the 'Fields to extract'. These will be your table headers.
    4.  If uploading a PDF, choose a processing method. 'text_and_image_per_page' is often best for mixed or scanned PDFs.
    5.  Click 'Extract Data'.
    6.  Review the raw and parsed output. Download as CSV or TXT.
    """
)
st.sidebar.title("⚠️ Important Notes")
st.sidebar.warning(
    """
    -   **Accuracy:** AI extraction is not always 100% perfect. Always verify critical data.
    -   **Prompting:** The quality of extraction heavily depends on the clarity of the 'Fields to extract' and the internal prompt structure.
    -   **Complex Layouts:** Very complex tables or layouts might be challenging. Gemini 1.5 Pro is good, but not infallible.
    -   **Token Limits & Cost:** Processing large PDFs (especially page-by-page as images) consumes more tokens and may incur higher API costs.
    -   **API Key Security:** For deployed apps, use Streamlit Secrets to store your API key. Do not hardcode it.
    """
)
