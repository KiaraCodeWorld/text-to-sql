from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import psycopg2
import os

# PDF file path
pdf_path = 'sample.pdf'  # Replace with your PDF file path

# Load the PDF file
loader = PyPDFLoader(pdf_path)
documents = loader.load()  # Loads the PDF page by page

# Initialize the embeddings model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose a different model if you prefer

# Connect to the Postgres database
conn = psycopg2.connect(
    host="localhost",        # Replace with your host
    database="your_database",  # Replace with your database name
    user="your_user",          # Replace with your username
    password="your_password"   # Replace with your password
)
cursor = conn.cursor()

# Ensure the pgvector extension and table exist
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id serial PRIMARY KEY,
        content text,
        embedding vector(384)  -- Adjust dimension based on your embeddings model
    );
""")

# Process each page/document
for idx, doc in enumerate(documents):
    text = doc.page_content  # Extract text from the page
    # Generate embeddings for the text
    embedding = model.encode(text)
    embedding_list = embedding.tolist()
    embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'

    # Insert the content and embedding into the database
    cursor.execute(
        "INSERT INTO documents (content, embedding) VALUES (%s, %s::vector)",
        (text, embedding_str)
    )
    print(f"Inserted page {idx + 1}")

# Commit the transaction and close the connection
conn.commit()
cursor.close()
conn.close()

--------------

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import psycopg2
import os

# PDF file path
pdf_path = 'sample.pdf'  # Replace with your PDF file path

# Load the PDF file
loader = PyPDFLoader(pdf_path)
documents = loader.load()  # Loads the PDF page by page

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

# Initialize the embeddings model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose a different model if you prefer

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    host="localhost",          # Replace with your host
    database="your_database",  # Replace with your database name
    user="your_user",          # Replace with your username
    password="your_password"   # Replace with your password
)
cursor = conn.cursor()

# Ensure the pgvector extension and table exist
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id serial PRIMARY KEY,
        content text,
        embedding vector(384)  -- Adjust dimension based on your embeddings model
    );
""")

# Process each page/document
for idx, doc in enumerate(documents):
    text = doc.page_content  # Extract text from the page

    # Split the text into chunks
    chunks = text_splitter.split_text(text)

    # Process each chunk
    for chunk in chunks:
        # Generate embeddings for the chunk
        embedding = model.encode(chunk)
        embedding_list = embedding.tolist()

        # Insert the content and embedding into the database
        cursor.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (chunk, embedding_list)
        )

    print(f"Processed page {idx + 1}")

# Commit the transaction and close the connection
conn.commit()
cursor.close()
conn.close()
=======

deepseek: 
# Install required packages
# pip install pypdf langchain langchain-postgres langchain-community sentence-transformers pgvector psycopg2-binary

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_postgres.embeddings import HuggingFaceEmbeddings

# Configuration - Update these values
PDF_PATH = "your_document.pdf"
COLLECTION_NAME = "pdf_documents"
CONNECTION_STRING = "postgresql://user:password@localhost:5432/dbname"

def process_and_store_pdf():
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # Efficient open-source model
    )
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Initialize vector store
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        use_jsonb=True
    )
    
    # Load PDF and get total pages
    loader = PyPDFLoader(PDF_PATH)
    total_pages = len(loader.pages)
    print(f"Processing {total_pages} pages...")
    
    # Process each page individually
    for page_num in range(total_pages):
        try:
            # Load single page
            page_docs = loader.load([page_num])
            
            # Split page into chunks
            chunks = text_splitter.split_documents(page_docs)
            
            # Add page-specific metadata
            for chunk in chunks:
                chunk.metadata.update({
                    "page_number": page_num + 1,
                    "source": PDF_PATH
                })
            
            # Add chunks to vector store
            vector_store.add_documents(chunks)
            print(f"Processed page {page_num+1}/{total_pages} → {len(chunks)} chunks")
            
        except Exception as e:
            print(f"Error processing page {page_num+1}: {str(e)}")
    
    print(f"Completed! All pages stored in '{COLLECTION_NAME}' collection")

if __name__ == "__main__":
    process_and_store_pdf()

============

claude - 

import os
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import uuid
from datetime import datetime

class PDFVectorLoader:
    def __init__(self, db_config: Dict[str, str], embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the PDF Vector Loader
        
        Args:
            db_config: Dictionary with database connection parameters
                      {'host': 'localhost', 'database': 'vectordb', 'user': 'user', 'password': 'pass', 'port': '5432'}
            embedding_model: Name of the sentence transformer model to use
        """
        self.db_config = db_config
        self.embedding_model = SentenceTransformer(embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def setup_database(self):
        """Create the necessary table and extension if they don't exist"""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        try:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create table for storing document chunks
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    content TEXT NOT NULL,
                    embedding VECTOR(384),  -- Adjust dimension based on your model
                    metadata JSONB,
                    page_number INTEGER,
                    chunk_index INTEGER,
                    source_file TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create index for vector similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
                ON document_chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            conn.commit()
            print("Database setup completed successfully")
            
        except Exception as e:
            print(f"Error setting up database: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()
    
    def load_and_chunk_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Load PDF and create chunks page by page
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        all_chunks = []
        
        for page_num, page in enumerate(pages, 1):
            print(f"Processing page {page_num}...")
            
            # Split the page content into chunks
            page_chunks = self.text_splitter.split_text(page.page_content)
            
            for chunk_idx, chunk_content in enumerate(page_chunks):
                if chunk_content.strip():  # Skip empty chunks
                    chunk_data = {
                        'content': chunk_content.strip(),
                        'page_number': page_num,
                        'chunk_index': chunk_idx,
                        'source_file': os.path.basename(pdf_path),
                        'metadata': {
                            'page_number': page_num,
                            'chunk_index': chunk_idx,
                            'source_file': os.path.basename(pdf_path),
                            'total_pages': len(pages)
                        }
                    }
                    all_chunks.append(chunk_data)
        
        print(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for all chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of chunks with embeddings added
        """
        print("Generating embeddings...")
        
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
        
        return chunks
    
    def insert_chunks_to_db(self, chunks: List[Dict[str, Any]]):
        """
        Insert chunks with embeddings into the database
        
        Args:
            chunks: List of chunk dictionaries with embeddings
        """
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        try:
            print(f"Inserting {len(chunks)} chunks into database...")
            
            # Prepare data for batch insert
            insert_data = []
            for chunk in chunks:
                insert_data.append((
                    chunk['content'],
                    chunk['embedding'],
                    chunk['metadata'],
                    chunk['page_number'],
                    chunk['chunk_index'],
                    chunk['source_file']
                ))
            
            # Batch insert
            insert_query = """
                INSERT INTO document_chunks 
                (content, embedding, metadata, page_number, chunk_index, source_file)
                VALUES %s
            """
            
            execute_values(
                cur, insert_query, insert_data,
                template=None, page_size=100
            )
            
            conn.commit()
            print(f"Successfully inserted {len(chunks)} chunks into database")
            
        except Exception as e:
            print(f"Error inserting chunks: {e}")
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()
    
    def process_pdf(self, pdf_path: str):
        """
        Complete pipeline to process PDF and load into vector database
        
        Args:
            pdf_path: Path to the PDF file
        """
        print(f"Starting to process PDF: {pdf_path}")
        
        # Load and chunk the PDF
        chunks = self.load_and_chunk_pdf(pdf_path)
        
        if not chunks:
            print("No chunks created from PDF")
            return
        
        # Generate embeddings
        chunks_with_embeddings = self.generate_embeddings(chunks)
        
        # Insert into database
        self.insert_chunks_to_db(chunks_with_embeddings)
        
        print("PDF processing completed successfully!")
    
    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks based on query
        
        Args:
            query: Search query
            limit: Number of results to return
            
        Returns:
            List of similar chunks with similarity scores
        """
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        try:
            search_query = """
                SELECT content, metadata, page_number, source_file,
                       1 - (embedding <=> %s::vector) as similarity
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            
            cur.execute(search_query, (query_embedding, query_embedding, limit))
            results = cur.fetchall()
            
            formatted_results = []
            for row in results:
                formatted_results.append({
                    'content': row[0],
                    'metadata': row[1],
                    'page_number': row[2],
                    'source_file': row[3],
                    'similarity': row[4]
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching: {e}")
            return []
        finally:
            cur.close()
            conn.close()

# Example usage
def main():
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'vectordb',
        'user': 'your_user',
        'password': 'your_password',
        'port': '5432'
    }
    
    # Initialize the loader
    pdf_loader = PDFVectorLoader(db_config)
    
    # Setup database (run once)
    pdf_loader.setup_database()
    
    # Process a PDF file
    pdf_path = "path/to/your/document.pdf"
    pdf_loader.process_pdf(pdf_path)
    
    # Example search
    results = pdf_loader.search_similar("your search query", limit=3)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (Page {result['page_number']}, Similarity: {result['similarity']:.3f}):")
        print(f"Content: {result['content'][:200]}...")

if __name__ == "__main__":
    main()
