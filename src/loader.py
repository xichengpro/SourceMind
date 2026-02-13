import os
import requests
import tempfile
import base64
from typing import List, Tuple, Optional
import fitz # pymupdf
import pymupdf4llm # New high-quality extractor
from langchain_community.document_loaders import ArxivLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.prompts import VLM_PARSING_PROMPT
from src.model_utils import get_llm, get_vlm_llm

def download_arxiv_pdf(arxiv_id: str) -> str:
    """Download Arxiv PDF to a temp file."""
    # Construct PDF URL
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download PDF from {pdf_url}, status code: {response.status_code}")
    
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(response.content)
        return f.name

def extract_images_from_pdf(pdf_path: str, output_dir: str = "temp/figures") -> List[str]:
    """Extract images from a PDF file and save them to output_dir."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_paths = []
    
    try:
        doc = fitz.open(pdf_path)
        for page_index, page in enumerate(doc):
            image_list = page.get_images(full=True)
            
            for image_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Filter small images (likely icons or logos)
                if len(image_bytes) < 2048: # Skip < 2KB
                    continue
                    
                image_filename = f"page{page_index+1}_img{image_index+1}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                image_paths.append(image_path)
    except Exception as e:
        print(f"Error extracting images: {e}")
        
    return image_paths

def parse_pdf_with_vlm(pdf_path: str) -> str:
    """
    Parse PDF using VLM (Visual Language Model) page by page.
    This provides the highest quality for formulas and tables.
    """
    print(f"Starting VLM parsing for {pdf_path}...")
    full_text = []
    
    try:
        doc = fitz.open(pdf_path)
        # Use a configured LLM (must support vision, e.g., GPT-4o, Claude 3.5)
        # We assume get_llm() returns a vision-capable model if VLM mode is enabled.
        # Ideally, we should check or enforce a vision model here.
        llm = get_vlm_llm() 
        chain = VLM_PARSING_PROMPT | llm | StrOutputParser()
        
        total_pages = len(doc)
        print(f"PDF has {total_pages} pages. Processing...")
        
        for page_index, page in enumerate(doc):
            print(f"Parsing page {page_index + 1}/{total_pages} with VLM...")
            
            # Convert page to image (pixmap)
            # Zoom=2 for better resolution (important for formulas)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            
            # Encode to base64
            img_base64 = base64.b64encode(img_data).decode("utf-8")
            
            # Call VLM
            try:
                page_markdown = chain.invoke({"image_data": img_base64})
                full_text.append(f"## Page {page_index + 1}\n\n{page_markdown}")
            except Exception as e:
                print(f"Error parsing page {page_index + 1}: {e}")
                full_text.append(f"## Page {page_index + 1}\n\n[VLM Parsing Error]")
                
    except Exception as e:
        print(f"VLM parsing failed: {e}")
        return ""
        
    return "\n\n---\n\n".join(full_text)

def load_paper(source: str, use_vlm: bool = False) -> Tuple[str, dict, List[str]]:
    """
    Load paper content from Arxiv URL or local PDF path.
    
    Args:
        source: Arxiv URL (e.g., https://arxiv.org/abs/2310.00000) or local file path.
        use_vlm: If True, use VLM for high-quality visual parsing.
        
    Returns:
        Tuple containing (full_text_content, metadata_dict, image_paths_list)
    """
    docs: List[Document] = []
    local_pdf_path = None
    full_text = ""
    metadata = {}
    
    if "arxiv.org" in source:
        # Extract arxiv ID if it's a URL
        # Handle cases like:
        # https://arxiv.org/abs/2310.00000
        # https://arxiv.org/pdf/2310.00000.pdf
        # https://arxiv.org/abs/2310.00000v1
        query = source.split("/")[-1]
        if query.endswith(".pdf"):
            query = query.replace(".pdf", "")
            
        print(f"Loading from Arxiv: {query}...")
        try:
            # Step 1: Always download PDF first to ensure we have the file for both PyMuPDF4LLM and image extraction
            local_pdf_path = download_arxiv_pdf(query)
            
            # Step 2: Use PyMuPDF4LLM or VLM for content extraction
            if use_vlm:
                print(f"Mode: VLM Visual Parsing enabled. Processing {local_pdf_path}...")
                full_text = parse_pdf_with_vlm(local_pdf_path)
                if not full_text:
                    print("VLM parsing returned empty. Falling back to PyMuPDF4LLM...")
                    full_text = pymupdf4llm.to_markdown(local_pdf_path)
            else:
                print(f"Extracting content using PyMuPDF4LLM from {local_pdf_path}...")
                full_text = pymupdf4llm.to_markdown(local_pdf_path)
            
            # Step 3: Get metadata using ArxivLoader (just for metadata)
            try:
                # Use a light load just to get metadata if possible, or use the API wrapper directly
                # For simplicity, we can use ArxivLoader but ignore its text content if we have better one
                # Or we can just use the metadata from the first page if ArxivLoader fails
                loader = ArxivLoader(query=query, load_max_docs=1, load_all_available_meta=True)
                # We load but only take metadata
                meta_docs = loader.load()
                if meta_docs:
                    metadata = meta_docs[0].metadata
                else:
                    metadata = {"source": source, "title": f"Arxiv:{query}"}
            except Exception as meta_e:
                print(f"Warning: Failed to fetch Arxiv metadata: {meta_e}")
                metadata = {"source": source, "title": f"Arxiv:{query}"}

        except Exception as e:
            # Fallback mechanism if download or PyMuPDF4LLM fails
            print(f"Primary loading failed: {e}. Trying legacy fallback...")
            try:
                if not local_pdf_path:
                    local_pdf_path = download_arxiv_pdf(query)
                loader = PyPDFLoader(local_pdf_path)
                docs = loader.load()
                full_text = "\n\n".join([doc.page_content for doc in docs])
                metadata = docs[0].metadata if docs else {}
            except Exception as e2:
                raise ValueError(f"Failed to load Arxiv paper via both Loader and Fallback: {e} | {e2}")
    else:
        # Assume local file path
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")
            
        print(f"Loading local PDF: {source}...")
        local_pdf_path = source
        
        # Use PyMuPDF4LLM or VLM for local files too
        try:
            if use_vlm:
                print(f"Mode: VLM Visual Parsing enabled. Processing {local_pdf_path}...")
                full_text = parse_pdf_with_vlm(local_pdf_path)
                if not full_text:
                    print("VLM parsing returned empty. Falling back to PyMuPDF4LLM...")
                    full_text = pymupdf4llm.to_markdown(local_pdf_path)
            else:
                print(f"Extracting content using PyMuPDF4LLM from {local_pdf_path}...")
                full_text = pymupdf4llm.to_markdown(local_pdf_path)
            
            # Create basic metadata
            metadata = {"source": source, "title": os.path.basename(source)}
            
            # Try to enrich metadata using PyPDFLoader just for consistency if needed, 
            # but PyMuPDF4LLM doesn't return structured metadata object directly same as LangChain
            # We can use fitz to get metadata
            with fitz.open(local_pdf_path) as doc:
                metadata.update(doc.metadata)
                
        except Exception as e:
            print(f"PyMuPDF4LLM failed for local file: {e}. Falling back to PyPDFLoader.")
            loader = PyPDFLoader(source)
            docs = loader.load()
            full_text = "\n\n".join([doc.page_content for doc in docs])
            metadata = docs[0].metadata if docs else {}
        
    if not full_text:
        raise ValueError("No content loaded from source.")
        
    # Extract Images if we have a local PDF path
    image_paths = []
    if local_pdf_path:
        # Create a unique directory for this session/paper to avoid conflicts?
        # For simplicity, we use a temp dir structure.
        # Ideally, pass a unique ID.
        import uuid
        session_id = str(uuid.uuid4())[:8]
        output_dir = os.path.join("temp", "figures", session_id)
        image_paths = extract_images_from_pdf(local_pdf_path, output_dir)
    
    return full_text, metadata, image_paths
