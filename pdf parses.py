import os
import asyncio
from typing import List, Dict, Any, Tuple
import logging
from dataclasses import dataclass
from pathlib import Path

# Core libraries
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor
import uuid

# PDF processing libraries
import pdfplumber
import camelot
import tabula

# Text processing
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessedContent:
    """Structure to hold processed PDF content"""
    text_chunks: List[str]
    table_chunks: List[str]
    metadata: Dict[str, Any]
    page_number: int
    content_type: str

class PDFProcessor:
    """Enhanced PDF processor with table extraction and text processing"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def extract_tables_camelot(self, pdf_path: str) -> List[Tuple[pd.DataFrame, int]]:
        """Extract tables using Camelot (more accurate for well-structured tables)"""
        try:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            result = []
            for table in tables:
                df = table.df
                page_num = table.page
                if not df.empty and df.shape[0] > 1:  # Skip empty or single-row tables
                    result.append((df, page_num))
            return result
        except Exception as e:
            logger.warning(f"Camelot extraction failed: {e}")
            return []
    
    def extract_tables_tabula(self, pdf_path: str) -> List[Tuple[pd.DataFrame, int]]:
        """Extract tables using Tabula (fallback method)"""
        try:
            tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            result = []
            for i, df in enumerate(tables):
                if not df.empty and df.shape[0] > 1:
                    result.append((df, i + 1))  # Approximate page number
            return result
        except Exception as e:
            logger.warning(f"Tabula extraction failed: {e}")
            return []
    
    def dataframe_to_markdown(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to clean markdown format"""
        # Clean the dataframe
        df = df.copy()
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Fill NaN values with empty strings
        df = df.fillna('')
        
        # Clean column names
        df.columns = [str(col).strip() if str(col).strip() else f'Col_{i}' 
                     for i, col in enumerate(df.columns)]
        
        # Convert to markdown
        markdown = df.to_markdown(index=False, tablefmt='grid')
        return markdown
    
    def extract_text_with_pdfplumber(self, pdf_path: str) -> List[Tuple[str, int]]:
        """Extract text content using pdfplumber"""
        text_content = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        # Clean text
                        text = re.sub(r'\n+', '\n', text)
                        text = re.sub(r'\s+', ' ', text)
                        text_content.append((text.strip(), page_num))
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num}: {e}")
                    
        return text_content
    
    def remove_table_text_overlap(self, text: str, tables_md: List[str]) -> str:
        """Remove table content that appears in text to avoid duplication"""
        cleaned_text = text
        
        for table_md in tables_md:
            # Extract table content for comparison
            lines = table_md.split('\n')
            table_content_lines = [line.strip() for line in lines if line.strip() and not line.startswith('|--')]
            
            for line in table_content_lines[2:]:  # Skip header lines
                if '|' in line:
                    # Extract cell content
                    cells = [cell.strip() for cell in line.split('|')[1:-1]]
                    for cell in cells:
                        if cell and len(cell) > 3:  # Only remove substantial content
                            cleaned_text = cleaned_text.replace(cell, '')
        
        # Clean up extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text.strip()
    
    def process_pdf(self, pdf_path: str) -> List[ProcessedContent]:
        """Main method to process PDF and extract all content"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract tables using multiple methods
        tables_camelot = self.extract_tables_camelot(pdf_path)
        if not tables_camelot:
            tables_camelot = self.extract_tables_tabula(pdf_path)
        
        # Extract text
        text_content = self.extract_text_with_pdfplumber(pdf_path)
        
        processed_content = []
        
        # Process tables
        logger.info(f"Found {len(tables_camelot)} tables")
        for i, (df, page_num) in enumerate(tables_camelot):
            try:
                markdown_table = self.dataframe_to_markdown(df)
                
                # Add context to the table
                table_with_context = f"Table {i+1} from page {page_num}:\n\n{markdown_table}"
                
                # Chunk large tables
                table_chunks = self.text_splitter.split_text(table_with_context)
                
                for chunk in table_chunks:
                    processed_content.append(ProcessedContent(
                        text_chunks=[],
                        table_chunks=[chunk],
                        metadata={
                            'source': pdf_path,
                            'page': page_num,
                            'table_id': i,
                            'table_shape': f"{df.shape[0]}x{df.shape[1]}"
                        },
                        page_number=page_num,
                        content_type='table'
                    ))
                    
            except Exception as e:
                logger.error(f"Error processing table {i}: {e}")
        
        # Process text content
        logger.info(f"Processing text from {len(text_content)} pages")
        all_tables_md = [self.dataframe_to_markdown(df) for df, _ in tables_camelot]
        
        for text, page_num in text_content:
            try:
                # Remove table content from text to avoid duplication
                cleaned_text = self.remove_table_text_overlap(text, all_tables_md)
                
                if cleaned_text and len(cleaned_text.strip()) > 50:  # Only process substantial text
                    # Chunk the text
                    text_chunks = self.text_splitter.split_text(cleaned_text)
                    
                    for chunk in text_chunks:
                        processed_content.append(ProcessedContent(
                            text_chunks=[chunk],
                            table_chunks=[],
                            metadata={
                                'source': pdf_path,
                                'page': page_num,
                                'char_count': len(chunk)
                            },
                            page_number=page_num,
                            content_type='text'
                        ))
                        
            except Exception as e:
                logger.error(f"Error processing text from page {page_num}: {e}")
        
        logger.info(f"Generated {len(processed_content)} content chunks")
        return processed_content

class VectorEmbedder:
    """Handle vector embedding using sentence transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def embed_content(self, content_list: List[ProcessedContent]) -> List[Dict[str, Any]]:
        """Embed processed content and return structured data"""
        embeddings_data = []
        
        for content in content_list:
            # Determine text to embed
            if content.table_chunks:
                text_to_embed = content.table_chunks[0]
            else:
                text_to_embed = content.text_chunks[0]
            
            # Generate embedding
            embedding = self.model.encode(text_to_embed)
            
            embeddings_data.append({
                'id': str(uuid.uuid4()),
                'content': text_to_embed,
                'embedding': embedding.tolist(),
                'metadata': content.metadata,
                'content_type': content.content_type,
                'page_number': content.page_number
            })
        
        logger.info(f"Generated embeddings for {len(embeddings_data)} chunks")
        return embeddings_data

class PgVectorStore:
    """Handle pgvector database operations"""
    
    def __init__(self, connection_string: str, table_name: str = 'pdf_embeddings'):
        self.connection_string = connection_string
        self.table_name = table_name
        
    def create_table(self, embedding_dim: int):
        """Create table with pgvector extension"""
        create_table_sql = f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        
        DROP TABLE IF EXISTS {self.table_name};
        
        CREATE TABLE {self.table_name} (
            id UUID PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector({embedding_dim}) NOT NULL,
            metadata JSONB,
            content_type VARCHAR(50),
            page_number INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX ON {self.table_name} USING ivfflat (embedding vector_cosine_ops);
        """
        
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
            conn.commit()
        
        logger.info(f"Created table {self.table_name}")
    
    def insert_embeddings(self, embeddings_data: List[Dict[str, Any]]):
        """Insert embeddings into pgvector"""
        insert_sql = f"""
        INSERT INTO {self.table_name} 
        (id, content, embedding, metadata, content_type, page_number)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                for data in embeddings_data:
                    cur.execute(insert_sql, (
                        data['id'],
                        data['content'],
                        data['embedding'],
                        data['metadata'],
                        data['content_type'],
                        data['page_number']
                    ))
            conn.commit()
        
        logger.info(f"Inserted {len(embeddings_data)} embeddings")
    
    def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        """Search for similar content"""
        search_sql = f"""
        SELECT id, content, metadata, content_type, page_number,
               embedding <=> %s as distance
        FROM {self.table_name}
        ORDER BY embedding <=> %s
        LIMIT %s;
        """
        
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(search_sql, (query_embedding, query_embedding, limit))
                return cur.fetchall()

class PDFToPgVectorPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, 
                 connection_string: str,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        
        self.pdf_processor = PDFProcessor(chunk_size, chunk_overlap)
        self.embedder = VectorEmbedder(embedding_model)
        self.vector_store = PgVectorStore(connection_string)
    
    def process_pdf_to_vector(self, pdf_path: str, print_output: bool = True) -> Dict[str, Any]:
        """Complete pipeline: PDF -> Chunks -> Embeddings -> PgVector"""
        
        # Step 1: Process PDF
        logger.info("Step 1: Processing PDF content")
        processed_content = self.pdf_processor.process_pdf(pdf_path)
        
        if not processed_content:
            raise ValueError("No content extracted from PDF")
        
        # Step 2: Generate embeddings
        logger.info("Step 2: Generating embeddings")
        embeddings_data = self.embedder.embed_content(processed_content)
        
        # Step 3: Setup database
        logger.info("Step 3: Setting up database")
        self.vector_store.create_table(self.embedder.embedding_dim)
        
        # Step 4: Insert into pgvector
        logger.info("Step 4: Inserting into pgvector")
        self.vector_store.insert_embeddings(embeddings_data)
        
        # Step 5: Print output for checking
        if print_output:
            self.print_processing_results(processed_content, embeddings_data)
        
        return {
            'processed_chunks': len(processed_content),
            'embeddings_created': len(embeddings_data),
            'embedding_dimension': self.embedder.embedding_dim,
            'table_name': self.vector_store.table_name
        }
    
    def print_processing_results(self, processed_content: List[ProcessedContent], 
                               embeddings_data: List[Dict[str, Any]]):
        """Print detailed results for verification"""
        
        print("\n" + "="*80)
        print("PDF PROCESSING RESULTS")
        print("="*80)
        
        # Summary statistics
        total_chunks = len(processed_content)
        table_chunks = sum(1 for c in processed_content if c.content_type == 'table')
        text_chunks = sum(1 for c in processed_content if c.content_type == 'text')
        
        print(f"\nSUMMARY:")
        print(f"Total chunks created: {total_chunks}")
        print(f"Table chunks: {table_chunks}")
        print(f"Text chunks: {text_chunks}")
        print(f"Embedding dimension: {self.embedder.embedding_dim}")
        
        # Sample content preview
        print(f"\nSAMPLE CONTENT PREVIEW:")
        print("-" * 50)
        
        for i, content in enumerate(processed_content[:5]):  # Show first 5 chunks
            content_text = content.table_chunks[0] if content.table_chunks else content.text_chunks[0]
            preview = content_text[:200] + "..." if len(content_text) > 200 else content_text
            
            print(f"\nChunk {i+1} ({content.content_type.upper()}) - Page {content.page_number}:")
            print(f"Preview: {preview}")
            print(f"Full length: {len(content_text)} characters")
            if content.metadata:
                print(f"Metadata: {content.metadata}")
        
        # Table-specific information
        table_contents = [c for c in processed_content if c.content_type == 'table']
        if table_contents:
            print(f"\nTABLE INFORMATION:")
            print("-" * 30)
            for i, content in enumerate(table_contents[:3]):  # Show first 3 tables
                print(f"\nTable {i+1}:")
                print(content.table_chunks[0][:500] + "..." if len(content.table_chunks[0]) > 500 else content.table_chunks[0])
        
        print("\n" + "="*80)

# Example usage and test function
def main():
    """Example usage of the PDF to PgVector pipeline"""
    
    # Configuration
    PDF_PATH = "sample_document.pdf"  # Replace with your PDF path
    CONNECTION_STRING = "postgresql://username:password@localhost:5432/your_database"
    
    # Initialize pipeline
    pipeline = PDFToPgVectorPipeline(
        connection_string=CONNECTION_STRING,
        chunk_size=800,
        chunk_overlap=100,
        embedding_model='all-MiniLM-L6-v2'
    )
    
    try:
        # Process PDF
        results = pipeline.process_pdf_to_vector(PDF_PATH, print_output=True)
        
        print(f"\nPipeline completed successfully!")
        print(f"Results: {results}")
        
        # Test similarity search
        print("\nTesting similarity search...")
        query = "financial data analysis"
        query_embedding = pipeline.embedder.model.encode(query).tolist()
        
        similar_docs = pipeline.vector_store.search_similar(query_embedding, limit=3)
        
        print(f"\nTop 3 similar documents for query: '{query}'")
        for i, doc in enumerate(similar_docs, 1):
            print(f"\n{i}. Distance: {doc['distance']:.4f}")
            print(f"   Content type: {doc['content_type']}")
            print(f"   Page: {doc['page_number']}")
            print(f"   Preview: {doc['content'][:150]}...")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()

=====================

Galaxy.ai : 

import os
import psycopg2
from pdfminer.high_level import extract_text
import camelot
import pandas as pd
from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_values

# Step 1: Extract text from the PDF
pdf_path = 'path_to_your_pdf.pdf'  # Replace with your PDF file path
text = extract_text(pdf_path)

# Extract tables from the PDF using Camelot
tables = camelot.read_pdf(pdf_path, pages='all')

# Step 2: Convert tables to Markdown
markdown_tables = []
for i, table in enumerate(tables):
    markdown = table.df.to_markdown(index=False)
    markdown_tables.append(markdown)

# Combine the extracted text and Markdown tables
combined_text = text + '\n\n' + '\n\n'.join(markdown_tables)

# Step 3: Apply chunking
def chunk_text(text, max_length=500):
    words = text.split()
    chunks = []
    chunk = []
    length = 0
    
    for word in words:
        chunk.append(word)
        length += 1
        if length >= max_length:
            chunks.append(' '.join(chunk))
            chunk = []
            length = 0
    if chunk:
        chunks.append(' '.join(chunk))
    return chunks

chunks = chunk_text(combined_text)

# Step 4: Generate vector embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose a different model if preferred
embeddings = model.encode(chunks)

# Step 5: Print the output for checking
for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    print(f"Chunk {idx+1}:\n{chunk}\n")
    print(f"Embedding {idx+1}:\n{embedding}\n{'-'*50}\n")

# Step 6: Embed the documents into pgvector
# Database connection details
conn = psycopg2.connect(
    host="your_host",       # Replace with your host
    database="your_db",     # Replace with your database name
    user="your_user",       # Replace with your username
    password="your_pass",   # Replace with your password
    port="your_port"        # Replace with your port number, usually 5432
)

cursor = conn.cursor()

# Create table if it doesn't exist
create_table_query = """
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(384)  -- Adjust the dimension according to your model's output
);
"""
cursor.execute(create_table_query)
conn.commit()

# Insert data into the table
insert_query = "INSERT INTO documents (content, embedding) VALUES %s"
records = [(chunk, embedding.tolist()) for chunk, embedding in zip(chunks, embeddings)]
execute_values(cursor, insert_query, records)
conn.commit()

# Close the connection
cursor.close()
conn.close()

====================

deepseek: 

import pdfplumber
import pandas as pd
import re
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration
PDF_PATH = "your_document.pdf"
DB_CONNECTION = "postgresql://user:password@localhost/dbname"
MODEL_NAME = "all-MiniLM-L6-v2"

def extract_pdf_content(pdf_path):
    """
    Extracts text and tables from PDF with accurate spatial positioning
    Returns list of elements in reading order with metadata
    """
    document = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extract tables with their bounding boxes
            tables = []
            for table in page.find_tables():
                table_data = table.extract()
                bbox = table.bbox
                tables.append({
                    "type": "table",
                    "content": table_data,
                    "bbox": bbox,
                    "page": page_num
                })
            
            # Extract text outside tables
            text_areas = []
            table_bboxes = [t["bbox"] for t in tables]
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            
            for word in words:
                if not any(contains_bbox(word, bbox) for bbox in table_bboxes):
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
            
            # Add tables to document
            for table in tables:
                document.append(table)
    
    return document

def contains_bbox(word, bbox):
    """Check if word is contained within a bounding box"""
    return (word["x0"] >= bbox[0] and word["x1"] <= bbox[2] and
            word["top"] >= bbox[1] and word["bottom"] <= bbox[3])

def convert_table_to_markdown(table_data):
    """Convert 2D table data to markdown format"""
    df = pd.DataFrame(table_data[1:], columns=table_data[0])
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
            table_md = convert_table_to_markdown(element["content"])
            chunks.append(table_md)
    
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
            cursor.execute(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                (chunk, embedding.tolist())
            )
        conn.commit()

def main():
    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)
    
    # Process PDF
    print("Extracting PDF content...")
    document = extract_pdf_content(PDF_PATH)
    
    # Chunk document
    print("Chunking document...")
    chunks = chunk_document(document)
    
    # Print chunks for verification
    print("\nGenerated Chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:\n{'-'*40}")
        print(chunk[:500] + "..." if len(chunk) > 500 else chunk)
        print(f"\n{'-'*40}\n")
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(DB_CONNECTION)
    register_vector(conn)
    create_pg_table(conn)
    
    # Store embeddings
    print("Storing embeddings in pgvector...")
    store_embeddings(chunks, model, conn)
    print("Process completed successfully!")
    
    conn.close()

if __name__ == "__main__":
    main()
