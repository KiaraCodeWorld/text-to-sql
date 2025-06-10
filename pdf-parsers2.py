import tabula
import pdfplumber
import pandas as pd
import re
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import warnings

# Suppress Tabula warnings
warnings.filterwarnings("ignore", message=".*Got unexpected.*")

# Configuration
PDF_PATH = "your_document.pdf"
DB_CONNECTION = "postgresql://user:password@localhost/dbname"
MODEL_NAME = "all-MiniLM-L6-v2"

def extract_pdf_content(pdf_path):
    """
    Extracts text and tables from PDF using:
    - Tabula for table extraction (with position detection)
    - pdfplumber for text extraction (excluding table areas)
    Returns list of elements in reading order
    """
    document = []
    
    # First extract tables with Tabula (preserve positions)
    tables = tabula.read_pdf(
        pdf_path, 
        pages='all',
        multiple_tables=True,
        output_format='json',
        lattice=True,      # For well-defined tables
        stream=True,       # For less structured tables
        guess=False        # Disable guessing for more precision
    )
    
    # Process with pdfplumber to integrate text and tables
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Get tables for this page
            page_tables = [t for t in tables if t['page'] == page_num + 1]
            table_bboxes = []
            
            # Process tables and get their positions
            for table in page_tables:
                # Get table bounding box
                x0 = min([cell['x0'] for row in table['data'] for cell in row])
                y0 = min([cell['y0'] for row in table['data'] for cell in row])
                x1 = max([cell['x1'] for row in table['data'] for cell in row])
                y1 = max([cell['y1'] for row in table['data'] for cell in row])
                bbox = (x0, y0, x1, y1)
                table_bboxes.append(bbox)
                
                # Extract table content
                table_data = []
                for row in table['data']:
                    row_data = [cell['text'] for cell in row]
                    table_data.append(row_data)
                
                document.append({
                    "type": "table",
                    "content": table_data,
                    "bbox": bbox,
                    "page": page_num
                })
            
            # Extract text outside table areas
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            text_areas = []
            
            for word in words:
                word_bbox = (word['x0'], word['top'], word['x1'], word['bottom'])
                if not any(bbox_contains(table_bbox, word_bbox) for table_bbox in table_bboxes):
                    text_areas.append({
                        "text": word["text"],
                        "top": word["top"],
                        "x0": word["x0"]
                    })
            
            # Group text areas into lines
            lines = {}
            for word in text_areas:
                line_key = round(word["top"], 1)
                if line_key not in lines:
                    lines[line_key] = []
                lines[line_key].append(word)
            
            # Sort lines and words within lines
            for line_key in sorted(lines.keys()):
                words_sorted = sorted(lines[line_key], key=lambda w: w["x0"])
                text_line = " ".join(w["text"] for w in words_sorted)
                document.append({
                    "type": "text",
                    "content": text_line,
                    "page": page_num
                })
    
    return document

def bbox_contains(outer, inner):
    """Check if inner bbox is contained within outer bbox"""
    return (outer[0] <= inner[0] and 
            outer[1] <= inner[1] and 
            outer[2] >= inner[2] and 
            outer[3] >= inner[3])

def convert_table_to_markdown(table_data):
    """Convert 2D table data to markdown format"""
    if not table_data or not table_data[0]:
        return ""
    
    # Create DataFrame and handle empty cells
    df = pd.DataFrame(table_data[1:], columns=table_data[0]).fillna('')
    return df.to_markdown(index=False)

def chunk_document(document):
    """Split content into semantically meaningful chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    current_text = ""
    
    for element in document:
        if element["type"] == "text":
            current_text += element["content"] + "\n"
        else:
            # Process text chunk before table
            if current_text:
                text_chunks = text_splitter.split_text(current_text)
                chunks.extend(text_chunks)
                current_text = ""
            
            # Convert table to markdown
            try:
                table_md = convert_table_to_markdown(element["content"])
                chunks.append(table_md)
            except Exception as e:
                print(f"Table conversion error: {e}")
                # Fallback to raw table text
                table_text = "\n".join([", ".join(row) for row in element["content"]])
                chunks.append(f"TABLE:\n{table_text}")
    
    # Process remaining text
    if current_text:
        text_chunks = text_splitter.split_text(current_text)
        chunks.extend(text_chunks)
    
    return chunks

def create_pg_table(conn):
    """Create table with vector column if not exists"""
    with conn.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT,
                page INTEGER,
                embedding VECTOR(384)
            )
        """)
        conn.commit()

def store_embeddings(chunks, model, conn):
    """Generate embeddings and store in PostgreSQL"""
    embeddings = model.encode(chunks)
    
    with conn.cursor() as cursor:
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Try to extract page number from document metadata
            page_num = 0  # Default if not found
            if re.search(r'page:\s*\d+', chunk, re.IGNORECASE):
                page_match = re.search(r'page:\s*(\d+)', chunk, re.IGNORECASE)
                page_num = int(page_match.group(1)) if page_match else 0
            
            cursor.execute(
                "INSERT INTO documents (content, page, embedding) VALUES (%s, %s, %s)",
                (chunk, page_num, embedding.tolist())
            )
        conn.commit()

def main():
    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)
    
    # Process PDF
    print("Extracting PDF content with Tabula...")
    document = extract_pdf_content(PDF_PATH)
    
    # Chunk document
    print("Chunking document...")
    chunks = chunk_document(document)
    
    # Print chunks for verification
    print("\nGenerated Chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} (Page {document[i]['page'] if i < len(document) else 'N/A'}):\n{'-'*40}")
        print(chunk[:500] + "..." if len(chunk) > 500 else chunk)
        print(f"\n{'-'*40}\n")
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(DB_CONNECTION)
    register_vector(conn)
    create_pg_table(conn)
    
    # Store embeddings
    print("Storing embeddings in pgvector...")
    store_embeddings(chunks, model, conn)
    print(f"Inserted {len(chunks)} chunks into database")
    print("Process completed successfully!")
    
    conn.close()

if __name__ == "__main__":
    main()
