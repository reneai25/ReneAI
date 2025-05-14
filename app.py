import typesense
import streamlit as st
import os
from google.genai import types
from google import genai
from pdf_to_markdown import pdf_to_markdown
from rag_model import answer_question
from qdrant_client import QdrantClient
from qdrantdb import create_qdrant_collection, add_documents_to_qdrant
from typsesensedb import setup_typesense_collection, add_to_typesense, hybrid_search
from svg_to_graph import process_svg
from streamlit_modal import Modal
import json
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(layout="wide")
load_dotenv()
collection_name = "blueprints"
vector_size = 768 

for key, default in {
    "selected_file": None,
    "processed": False,
    "document_name": None,
    "extracted_text": "",
    "extracted_images": {},
    "categories": None,
    "image_analysis": {},
    "db_stored": False,
    "selected_gnn_image_path": None,
    "selected_folder": None,
}.items():
    st.session_state.setdefault(key, default)
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333") # Default for local
if 'qdrant_client' not in st.session_state:
    st.session_state.qdrant_client = QdrantClient(url=qdrant_url)
if 'typesense_client' not in st.session_state:
    st.session_state.typesense_client = typesense.Client({
        'api_key': os.getenv('TYPESENSE_API_KEY'),
        'nodes': [{
            'host': 'localhost',
            'port': 8108,
            'protocol': 'http'
        }],
        'connection_timeout_seconds':2
    })

if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None



# Initialize view-related session state keys at a higher scope (once per session)
st.session_state.setdefault("view_text", False)
st.session_state.setdefault("view_images", False)
st.session_state.setdefault("image_category", None)
st.session_state.setdefault("formatted", False)
st.session_state.setdefault("view_pdf", False) 

modal = Modal(title="Upload PDF", key="upload_modal")

SAMPLE_DIR = "samples"
CATEGORIES = ["floor plans", "site plans", "elevation plans", "details", "company logo","other"] # Define CATEGORIES at a broader scope

api_key = os.getenv("API_KEY")
client = genai.Client(api_key=api_key)
# Remove duplicate CSS blocks and combine into one with proper tab spacing
st.markdown("""
<style>
    /* Increase main app title font size */
    h1 {
        font-size: 3rem !important;
        font-weight: 700 !important;
    }
    
    /* Style tab titles with increased size and spacing */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.25rem;
        font-weight: 600;
        padding: 0 1rem;
    }
    
    /* Add spacing for h3 headers */
    h3 {
        margin-right: 1rem !important;
        margin-bottom: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)
st.session_state.setdefault("show_pass_key", True)
st.title("Rene AI")
pass_key_input = None # Initialize to None

if st.session_state.show_pass_key:
    pass_key_input = st.text_input("Enter passkey", type="password", key="pass_key_field") # Added a unique key
    if pass_key_input == os.getenv("PASS_KEY"): # Ensure "PASS_KEY" matches your .env file
        st.session_state.show_pass_key = False
        st.rerun() # Force a rerun to hide the input and show content
    elif pass_key_input and pass_key_input != "": # Show error only if something was typed and it's wrong
        st.error("Invalid passkey. Please try again.")
if not st.session_state.show_pass_key:
    # Now create the tabs
    tab1, tab2 = st.tabs(["AI Plan Check Review and Quantity Takeoff", "Spatial GNN Reasoning"])
    with tab1:

        # PDF Upload Button and Modal Logic (remains largely the same)
        col_upload_btn1, col_upload_btn2, col_upload_btn3 = st.columns([1, 2, 1])
        with col_upload_btn2:
            if st.button("Upload PDF", use_container_width=True):
                st.session_state.show_modal = True
                modal.open()
                st.experimental_rerun()

        if modal.is_open() and st.session_state.show_modal:
            with modal.container():
                st.markdown("## Choose or Upload a PDF")
                # ... (modal content: sample choice, uploader, proceed button) ...
                # (Ensure this part correctly updates st.session_state.selected_file and reruns)
                col_modal1, col_modal2 = st.columns(2)
                with col_modal1:
                    samples = [f for f in reversed(os.listdir(SAMPLE_DIR)) if f.endswith(".pdf")]
                    sample_choice = st.radio("Select a sample file", ["None"] + samples, index=0)
                with col_modal2:
                    uploaded = st.file_uploader("Upload a PDF file", type="pdf", key="modal_upload")
                if st.button("Proceed"):
                    new_selection = None
                    if uploaded is not None:
                        new_selection = uploaded
                    elif sample_choice != "None":
                        new_selection = os.path.join(SAMPLE_DIR, sample_choice)
                    
                    if new_selection:
                        if st.session_state.selected_file != new_selection:
                            st.session_state.selected_file = new_selection
                            st.session_state.processed = False # Mark for reprocessing
                            st.session_state.categories = None # Reset categories
                            st.session_state.extracted_text = ""
                            st.session_state.extracted_images = {}
                            st.session_state.document_name = None # Reset document name
                            st.session_state.image_analysis = {} # Reset image_analysis
                            st.session_state.formatted = False # Reset formatted flag
                            st.session_state.db_stored = False
                        modal.close()
                        st.session_state.show_modal = False
                        st.experimental_rerun()
                    else:
                        st.error("Please select a sample or upload a file.")
                        # st.stop() # Not needed if modal stays open

        # Display selected file name (guarded)
        if st.session_state.selected_file:
            current_file_name = ""
            if hasattr(st.session_state.selected_file, 'name'):
                current_file_name = st.session_state.selected_file.name
            else:
                current_file_name = os.path.basename(str(st.session_state.selected_file))
            st.write("Selected file:", current_file_name)

        # --- Step 1: Process PDF (extract text & images) if a new file is selected or not yet processed ---
        if st.session_state.selected_file and \
        (st.session_state.selected_file != st.session_state.document_name or \
            not st.session_state.processed):
            
            # If it's a genuinely new file different from the last processed one
            if st.session_state.selected_file != st.session_state.document_name:
                st.session_state.processed = False
                st.session_state.categories = None
                st.session_state.extracted_text = ""
                st.session_state.extracted_images = {}

            if not st.session_state.processed: # Proceed only if not marked as processed
                with st.spinner("Processing PDF (extracting text & images)..."):
                    markdown, images = pdf_to_markdown(
                        st.session_state.selected_file, api_key
                    )
                    st.session_state.extracted_text = markdown
                    st.session_state.extracted_images = images
                    st.session_state.processed = True
                    st.session_state.document_name = st.session_state.selected_file # Update to current file
                st.success("PDF content extracted!")

        # --- Step 2: Classify Images if processed and not yet categorized ---
        if st.session_state.processed and \
        st.session_state.extracted_images and \
        st.session_state.categories is None: # Check if categories need to be generated
            with st.spinner("Classifying images and generating summaries..."): # Updated spinner text
                buckets = {cat: [] for cat in CATEGORIES}
                all_image_summaries = [] # To store summaries

                for idx, (fname, img) in enumerate(st.session_state.extracted_images.items()):
                    prompt = f"""
                        You are an expert in architectural plan analysis and classification.
                        For the following blueprint image (filename: {fname}):

                        1.  **Classify** it into exactly ONE of the categories listed below.
                        2.  Provide an **accurate** and **detailed** **in-depth** **explanation** of the image without leaving any detail out and also do not hallucinate.

                        Categories for Classification:
                        - "floor plans": Drawings showing a building's layout from a top-down view, detailing rooms, walls, doors, windows, and other interior features of a specific floor. This category is *only* for the insides of a floor.
                        - "site plans": Drawings showing the entire property, including building footprint, landscaping, driveways, walkways, and other external features. The image must be a site plan and not an image that has site plan written in it. This category specifies the surroundings of the construction area.
                        - "elevation plans": Drawings showing the building's exterior views (e.g., front, side, rear), specifying height-related information like story heights, overall building height, and vertical dimensions.
                        - "details": Drawings providing magnified or in-depth views of specific architectural components, connections, or construction techniques. These are often separate, detailed explanations of a small part.
                        - "company logo": Images primarily displaying a company's logo, name or other branding elements, often found on title blocks or cover sheets.
                        - "other": If the image does not fit into any of the above categories, classify it as "other".

                        Respond in the following JSON format:
                        {{
                            "category": "CHOSEN_CATEGORY_NAME_FROM_LIST_ABOVE",
                            "summary": "ALL_THE_DETAILS_OF_THE_IMAGE_CONTENT"
                        }}

                        Example for a floor plan:
                        {{
                            "category": "floor plans",
                            "summary": "This image shows the layout of the first floor, including room dimensions and door placements. It has 5 rooms in total, 2 bathrooms, 1 kitchen, 1 living room, 1 dining room and 1 bedroom. The living room is in the center of the floor plan and the bedrooms are on the sides. The kitchen is in the north-east corner and the bathrooms are in the south-east corner. The floor plan is 100 feet by 100 feet."
                        }}

                        Ensure the category name is one of the exact names provided.
                    """
                    resp = client.models.generate_content(
                        model="gemini-1.5-flash", # Ensure this is the correct and desired model
                        contents=[prompt, img],
                        # Optional: Specify response_mime_type if the model supports it directly for JSON
                        # generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
                    )
                    
                    try:
                        # Attempt to parse the response as JSON
                        # The model's output might have ```json ... ``` around it.
                        raw_response_text = resp.text.strip()
                        if raw_response_text.startswith("```json"):
                            raw_response_text = raw_response_text[7:]
                            if raw_response_text.endswith("```"):
                                raw_response_text = raw_response_text[:-3]
                        
                        response_data = json.loads(raw_response_text)
                        chosen_category = response_data.get("category", "").strip().lower()
                        image_summary = response_data.get("summary", "").strip()

                        if chosen_category in CATEGORIES:
                            buckets[chosen_category].append(fname)
                        
                        if image_summary: # Add summary if present
                            st.session_state.extracted_text += "\n\n"+image_summary
                            st.session_state.image_analysis[fname]= image_summary

                    except (json.JSONDecodeError, AttributeError, KeyError) as e:
                        # Fallback if JSON parsing fails or structure is wrong
                        # Try to extract category with a simpler split if possible, or just log error
                        print(f"Error parsing LLM response for {fname}: {e}. Response text: {resp.text[:200]}")
                        # Simple fallback for category if response is just the category name
                        chosen_category_fallback = resp.text.strip().lower()
                        if chosen_category_fallback in CATEGORIES:
                            buckets[chosen_category_fallback].append(fname)


                st.session_state.categories = buckets
                
            st.success("Image classification and summarization complete!")

        # --- Step 3: Format the extracted text for readability ---
        if st.session_state.extracted_text and not st.session_state.formatted: # Check if there's text to format
            with st.spinner("Formatting extracted text for readability..."):
                formatting_prompt = f"""
                        You are a text formatting expert. Your task is to reformat the following text to improve its readability and consistency.

                        Please adhere to these rules STRICTLY:
                        1.  PRESERVE ALL ORIGINAL CONTENT: Do not add, remove, or change any of the factual information or meaning from the original text.
                        2.  NO NEW INFORMATION: Do not introduce any sentences, phrases, or words that were not present in the original text.
                        3.  IMPROVE STRUCTURE:
                            - Ensure consistent paragraph spacing.
                            - Correct any awkward line breaks or spacing issues.
                            - Maintain and, if possible, clarify existing Markdown structures (like headings, lists). Do not convert plain text to Markdown unless it's clearly intended (e.g., existing list-like structures).
                        4.  CLEAN WHITESPACE: Remove excessive or unnecessary whitespace.
                        5.  NO HALLUCINATION: Do not invent any content. Your output should solely be a better-formatted version of the input.

                        Here is the text to reformat:
                        ---
                        {st.session_state.extracted_text}
                        ---
                    """
                try:
                    formatting_resp = client.models.generate_content(
                        model="gemini-1.5-flash", # Or your preferred model for text tasks
                        contents=[formatting_prompt]
                    )
                    formatted_text = formatting_resp.text.strip()
                    st.session_state.extracted_text = formatted_text # Update with formatted text
                    st.session_state.formatted = True
                    st.success("Text formatting complete!")
                except Exception as e:
                    st.warning(f"Could not format the extracted text: {e}")
                    # Keep the unformatted text if formatting fails

        if st.session_state.formatted and \
        st.session_state.extracted_text and \
        not st.session_state.db_stored: # Ensure text is formatted and not yet stored

            doc_name_for_db = "unknown_document"
            if hasattr(st.session_state.document_name, 'name'):
                doc_name_for_db = st.session_state.document_name.name
            elif isinstance(st.session_state.document_name, str):
                doc_name_for_db = os.path.basename(st.session_state.document_name)
            
            # Prepare document structure if needed (assuming simple text for now)
            # Your DB functions might expect a list of texts or specific document objects.
            # For simplicity, let's assume they can take the raw text and a document ID/name.
            # You might need to create a unique ID for each document chunk if your DB requires it.
            document_to_store = {
                "id": doc_name_for_db, # Using filename as a simple ID
                "text": st.session_state.extracted_text,
                # Add any other metadata you want to store
            }

            # Store in Qdrant
            try:
                create_qdrant_collection(st.session_state.qdrant_client, collection_name, vector_size)
                with st.spinner("Storing text in Qdrant DB..."):
                    # Assuming your add_documents_to_qdrant can take a list of such dicts
                    # and the qdrant_client is already initialized and available in session_state
                    texts_to_embed = [st.session_state.extracted_text]
                    embedding_response = client.models.embed_content(
                        model="models/embedding-001",
                        contents=texts_to_embed,
                        config=types.EmbedContentConfig(task_type="retrieval_document")
                    )
                    add_documents_to_qdrant(
                        client=st.session_state.qdrant_client,
                        collection_name=collection_name,
                        documents=texts_to_embed, # The text(s) themselves
                        embedding_response=embedding_response # The embedding results
                    )
                    st.success(f"Text from '{doc_name_for_db}' stored in Qdrant DB.")
            except Exception as e:
                st.error(f"Error storing text in Qdrant DB: {e}")

            # Store in Typesense
            try:
                with st.spinner("Storing text in Typesense DB..."):
                    # Assuming your add_to_typesense can take a list of such dicts
                    # and the typesense_client is already initialized and available in session_state
                    setup_typesense_collection(
                        client=st.session_state.typesense_client, 
                        collection_name=collection_name # Use global collection_name
                    )
                    add_to_typesense(
                        client=st.session_state.typesense_client,
                        collection_name=collection_name, # Correct argument order
                        text=st.session_state.extracted_text,      # Correct argument order
                        image_analyses=st.session_state.image_analysis # Correct argument order
                    )
                    st.success(f"Text from '{doc_name_for_db}' stored in Typesense DB.")
            except Exception as e:
                st.error(f"Error storing text in Typesense DB: {e}")
            
            st.session_state.db_stored = True # 

        # --- Step 4: UI for Viewing Text and Images (shown if PDF processing is complete) ---
        if st.session_state.processed:
            btn_col1, btn_col2, btn_col3 = st.columns(3) # Use different variable names for button columns

            with btn_col1:
                if st.button("View Extracted Text"):
                    st.session_state.view_text = not st.session_state.view_text
                    if st.session_state.view_text:
                        st.session_state.view_images = False
                        st.session_state.view_pdf = False
            with btn_col2:
                if st.button("View Extracted Images"):
                    st.session_state.view_images = not st.session_state.view_images
                    if st.session_state.view_images:
                        st.session_state.view_text = False
                        st.session_state.view_pdf = False
            with btn_col3:
                if st.button("View PDF"):
                    st.session_state.view_pdf = not st.session_state.view_pdf
                    st.session_state.view_text = False # Reset other views
                    st.session_state.view_images = False

            if st.session_state.view_text:
                st.markdown("### Extracted Markdown Text")
                if st.session_state.extracted_text:
                    st.markdown(st.session_state.extracted_text)
                else:
                    st.info("No text extracted or available.")

            if st.session_state.view_pdf:
                st.markdown("### View Original PDF")
                import base64
                try:
                    # Determine how to read the PDF file
                    if hasattr(st.session_state.selected_file, 'read'):
                        # Uploaded file (from st.file_uploader)
                        pdf_bytes = st.session_state.selected_file.read()
                        st.session_state.selected_file.seek(0)  # Reset file pointer
                    else:
                        # Sample file (path from SAMPLE_DIR)
                        with open(st.session_state.selected_file, 'rb') as f:
                            pdf_bytes = f.read()
                    
                    # Encode PDF bytes to base64
                    pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
                    pdf_data_uri = f"data:application/pdf;base64,{pdf_base64}"
                    
                    # Create a clickable link to open PDF in a new tab
                    pdf_display_name = os.path.basename(st.session_state.selected_file.name if hasattr(st.session_state.selected_file, 'name') else st.session_state.selected_file)
                    st.markdown(
                        f"""
                        <a href="{pdf_data_uri}" target="_blank" download="{pdf_display_name}">
                            Click here to view {pdf_display_name} in a new tab
                        </a>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Optional: Embed a PDF viewer in the app for immediate viewing
                    st.markdown("Or view the PDF below:")
                    pdf_viewer_html = f"""
                    <iframe src="{pdf_data_uri}" width="100%" height="600px" style="border:none;"></iframe>
                    """
                    st.markdown(pdf_viewer_html, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Could not load PDF for viewing: {e}")

            if st.session_state.view_images:
                cats = st.session_state.categories or {}
                seg = [cat for cat in cats if cats.get(cat)] # Ensure cat exists and has images
                
                if not seg:
                    st.info("No categorized images available to display or no images extracted.")
                else:
                    st.markdown("### Select category")
                    if "image_category" not in st.session_state or st.session_state.image_category not in seg:
                        st.session_state.image_category = seg[0]
                    
                    choice = st.radio("Category", options=seg, key="image_category")
                    
                    files_in_category = []
                    if choice and choice in cats:
                        files_in_category = cats[choice]

                    if not files_in_category:
                        if choice:
                            st.info(f"No images in **{choice.replace('_',' ').title()}**.")
                        else:
                            st.info("Please select a category.") # Should not happen if seg is populated
                    else:
                        st.markdown(f"## {choice.replace('_',' ').title()}")
                        for fname_in_cat in files_in_category:
                            img_data = st.session_state.extracted_images.get(fname_in_cat)
                            if img_data:
                                st.image(img_data, caption=fname_in_cat, use_container_width=True) # Corrected caption
                            else:
                                st.warning(f"Missing image data for {fname_in_cat}")
        st.session_state.setdefault("view_qa", False)
        st.session_state.setdefault("current_question", None)
        st.session_state.setdefault("current_answer", None)
        st.session_state.setdefault("custom_question_input", "")
        if st.session_state.db_stored:
            if st.button("Ask Questions About the Document"):
                st.session_state.view_qa = not st.session_state.view_qa
                if st.session_state.view_qa:
                    st.session_state.view_text = False
                    st.session_state.view_images = False
                    st.session_state.current_question = None # Reset QA state
                    st.session_state.current_answer = None
                else: # If toggling QA off
                    st.session_state.current_question = None
                    st.session_state.current_answer = None


        if st.session_state.view_qa and st.session_state.db_stored:
            st.markdown("### Ask Questions About the Document")

            sample_questions = [
                "The lot shall include an existing single family dwelling.",
                "What does the project describe?",
                "Is the ADU limited to two stories maximum?",
                "What is the total area for ADU? DOes it exceed 1,200 sf?",
                "Are there any specific safety features or requirements noted?"
            ]

            st.markdown("#### Try these sample questions:")
            sample_q_cols = st.columns(len(sample_questions) if len(sample_questions) <= 3 else 3) # Max 3 sample Qs per row
            for i, question_text_sample in enumerate(sample_questions):
                if sample_q_cols[i % len(sample_q_cols)].button(question_text_sample, key=f"sample_q_{i}"):
                    st.session_state.current_question = question_text_sample
                    st.session_state.current_answer = None 
                    st.session_state.custom_question_input_key = "" # Clear custom input
                    with st.spinner(f"Thinking about: '{question_text_sample}'..."):
                        try:
                            # 1. Embed the question
                            embedding_resp = client.models.embed_content(
                                model="models/embedding-001",
                                contents=[question_text_sample],
                                config=types.EmbedContentConfig(task_type="retrieval_query")
                            )
                            question_embedding_values = embedding_resp.embeddings[0].values

                            # 2. Get context using hybrid_search from typsesensedb.py
                            context = hybrid_search(
                                query=question_text_sample,
                                qdrant_client=st.session_state.qdrant_client,
                                typesense_client=st.session_state.typesense_client,
                                collection_name=collection_name,
                                query_embedding_values=question_embedding_values # Pass the fresh question embedding
                            )

                            # 3. Generate answer using rag_model.answer_question
                            if context and not "Failed to retrieve information" in context:
                                answer = answer_question(
                                    query=question_text_sample,
                                    context=context,
                                    gemini_api_key=api_key # Pass the API key as expected by your function
                                )
                            elif context: # hybrid_search returned an error message
                                answer = context 
                            else:
                                answer = "No relevant context was found to answer the question."
                                
                            st.session_state.current_answer = answer
                        except Exception as e:
                            st.session_state.current_answer = f"Error processing question: {e}"
                    # st.experimental_rerun() 
                    if st.session_state.current_question and st.session_state.current_answer:
                        st.markdown(f"**Question:** {st.session_state.current_question}")
                        st.markdown(f"**Answer:**")
                        st.markdown(st.session_state.current_answer)
                    elif st.session_state.current_question and st.session_state.current_answer is None:
                        # This state occurs if a sample question was clicked, spinner showed, but then rerun happened before answer set.
                        # The spinner itself usually covers this. If issues persist, a "Loading..." message could be added.
                        pass

            st.markdown("---")
            st.markdown("#### Or ask your own question:")

            # Using st.chat_input for a cleaner interface
            user_custom_query = st.chat_input("Ask your question here...", key="custom_chat_input")

            if user_custom_query:
                st.session_state.current_question = user_custom_query
                st.session_state.current_answer = None 
                with st.spinner(f"Thinking about: '{user_custom_query}'..."):
                    try:
                        # 1. Embed the question
                        embedding_resp = client.models.embed_content(
                            model="models/embedding-001",
                            contents=[user_custom_query],
                            config=types.EmbedContentConfig(task_type="retrieval_query")
                        )
                        question_embedding_values = embedding_resp.embeddings[0].values

                        # 2. Get context using hybrid_search
                        context = hybrid_search(
                            query=user_custom_query,
                            qdrant_client=st.session_state.qdrant_client,
                            typesense_client=st.session_state.typesense_client,
                            collection_name=collection_name,
                            query_embedding_values=question_embedding_values # Pass the fresh question embedding
                        )

                        # 3. Generate answer
                        if context and not "Failed to retrieve information" in context:
                            answer = answer_question(
                                query=user_custom_query,
                                context=context,
                                gemini_api_key=api_key
                            )
                        elif context: # hybrid_search returned an error message
                            answer = context
                        else:
                            answer = "No relevant context was found to answer the question."
                            
                        st.session_state.current_answer = answer
                    except Exception as e:
                        st.session_state.current_answer = f"Error processing custom question: {e}"
                # st.experimental_rerun() # Rerun to display the new answer

                if st.session_state.current_question and st.session_state.current_answer:
                    st.markdown(f"**Question:** {st.session_state.current_question}")
                    st.markdown(f"**Answer:**")
                    st.markdown(st.session_state.current_answer)
                elif st.session_state.current_question and st.session_state.current_answer is None and not user_custom_query:
                    # This state occurs if a sample question was clicked, spinner showed, but then rerun happened before answer set.
                    # The spinner itself usually covers this. If issues persist, a "Loading..." message could be added.
                    pass

    with tab2:
        base_image_path = os.path.join("samples", "images")
        
        image_subfolders = []
        if os.path.exists(base_image_path) and os.path.isdir(base_image_path):
            image_subfolders = [f for f in os.listdir(base_image_path) if os.path.isdir(os.path.join(base_image_path, f))]

        if not image_subfolders:
            st.info("No image subfolders found in 'streamlit_app/samples/images/'.")
        else:
            st.write("Click on a button to view the F1_original.png image from the respective folder:")
            
            buttons_per_row = 3 
            cols = st.columns(buttons_per_row)
            button_idx = 0

            for folder in image_subfolders:
                image_file_path = os.path.join(base_image_path, folder, "F1_original.png")
                if os.path.exists(image_file_path):
                    col = cols[button_idx % buttons_per_row]
                    if col.button(image_file_path):
                        st.session_state.selected_folder = os.path.join(base_image_path, folder)
                        st.session_state.selected_gnn_image_path = image_file_path
                        # No explicit rerun needed, button click handles it.
                    button_idx += 1
            
            if button_idx == 0: # If no F1_original.png files were found to create buttons
                st.warning("No 'F1_original.png' files found to display as buttons.")

        # Display the selected image in a two-column layout
        if st.session_state.selected_gnn_image_path:
            st.markdown("---") # Visual separator
            st.subheader("Selected Image")
            col_left, col_right = st.columns([2,3])
            with col_left:
                try:
                    st.image(st.session_state.selected_gnn_image_path, caption=f"Displaying: {os.path.basename(st.session_state.selected_gnn_image_path)} from {os.path.basename(os.path.dirname(st.session_state.selected_gnn_image_path))}", use_container_width=True)
                except Exception as e:
                    st.error(f"Could not load image {st.session_state.selected_gnn_image_path}: {e}")
            with col_right:
                svg_path = os.path.join(st.session_state.selected_folder, "model.svg")
                if os.path.exists(svg_path):
                    try:
                        with st.spinner("Processing SVG and generating graph..."):
                            ax = process_svg(svg_path)
                            if ax and hasattr(ax, 'figure'):
                                st.pyplot(ax.figure, use_container_width=True)
                            else:
                                st.warning("Graph visualization could not be generated.")
                    except Exception as e:
                        st.error(f"Error processing SVG file '{svg_path}': {e}")
                else:
                    st.warning(f"SVG file 'model.svg' not found in folder: {os.path.basename(st.session_state.selected_folder)}")
