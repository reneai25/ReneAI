# pdf_to_markdown.py

from marker.models import create_model_dict
from marker.converters.pdf import PdfConverter
import tempfile
import os

def pdf_to_markdown(uploaded_file, api_key):
    """
    Accepts either:
      - a file path (str) to an existing PDF
      - a file-like object with getvalue() or read() methods
    Returns: (markdown_text, images_dict)
    """
    # Determine the PDF path
    if isinstance(uploaded_file, str):
        pdf_path = uploaded_file
    else:
        # Read bytes from the file-like object
        if hasattr(uploaded_file, 'getvalue'):
            data = uploaded_file.getvalue()
        else:
            data = uploaded_file.read()
        # Write to a temporary file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        tmp.write(data)
        tmp_path = tmp.name
        tmp.close()
        pdf_path = tmp_path

    # Initialize and run the converter
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config={
            "yolo_model_path": "marker/model/yolo/best.pt",
            "output_format": "json",
            "use_llm": True,
            "gemini_api_key": api_key,
            "redo_inline_math": True,
        }
    )
    text_obj, images = converter(pdf_path)

    # Cleanup the temp file if we created one
    if not isinstance(uploaded_file, str):
        try:
            os.remove(pdf_path)
        except OSError:
            pass

    return text_obj.markdown, images
