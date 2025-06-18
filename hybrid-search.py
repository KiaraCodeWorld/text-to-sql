import sqlalchemy
from sqlalchemy import text
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector

# Setup the connection and embeddings
connection = "postgresql+psycopg2://postgres:postgres@localhost:5432/test"
collection_name = "test"

# Initialize the embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    # Replace 'YOUR_API_KEY' with your actual OpenAI API key
    openai_api_key="YOUR_API_KEY",
    timeout=30
)

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)

# Initialize the vector store
vector_store = PGVector(
    connection_string=connection,
    embedding_function=embeddings.embed_query,
    collection_name=collection_name,
    use_jsonb=True,
    distance_strategy="cosine"
)

# Assume you have a list of documents to index
docs = [
    {"text": "This is a sample document about artificial intelligence.", "metadata": {"source": "Doc1"}},
    {"text": "Another document discussing machine learning techniques.", "metadata": {"source": "Doc2"}},
    # Add more documents as needed
]

# Split documents into chunks and collect metadata
texts = []
metadatas = []
for doc in docs:
    splits = text_splitter.split_text(doc['text'])
    texts.extend(splits)
    metadatas.extend([doc['metadata']] * len(splits))

# Add texts and metadata to the vector store
vector_store.add_texts(texts=texts, metadatas=metadatas)

# Function to perform hybrid search
def hybrid_search(query, top_k=5):
    # Generate the embedding for the query
    query_embedding = embeddings.embed_query(query)
    # Convert the embedding to a PostgreSQL vector literal
    query_embedding_str = '[' + ','.join([str(x) for x in query_embedding]) + ']'

    # Prepare the SQL query
    sql = f"""
    SELECT
        metadata,
        text,
        1 - (embedding <#> '{query_embedding_str}') AS similarity
    FROM {collection_name}
    WHERE text ILIKE :keyword
    ORDER BY embedding <#> '{query_embedding_str}' ASC
    LIMIT :top_k
    """
    # Use the query text for keyword matching
    keyword = f"%{query}%"

    # Execute the SQL query
    engine = sqlalchemy.create_engine(connection)
    with engine.connect() as conn:
        result = conn.execute(
            text(sql),
            {"keyword": keyword, "top_k": top_k}
        )
        records = result.fetchall()
        # Process and return the results
        docs = []
        for record in records:
            metadata = record['metadata']
            text_content = record['text']
            similarity = record['similarity']
            docs.append({
                "text": text_content,
                "metadata": metadata,
                "similarity": similarity
            })
    return docs

# Use the hybrid search function
query = "machine learning"
results = hybrid_search(query)
for result in results:
    print(f"Text: {result['text']}")
    print(f"Metadata: {result['metadata']}")
    print(f"Similarity: {result['similarity']}\n")

==========


from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores.pgvector import PGVector, EmbeddingStore
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

class PGVectorHybrid(PGVector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._enable_full_text_support()

    def _enable_full_text_support(self) -> None:
        """Create TSVector column and triggers for full-text search"""
        with self.Session() as session:
            # Add TSVector column if not exists
            session.execute(text("""
                ALTER TABLE langchain_pg_embedding
                ADD COLUMN IF NOT EXISTS document_tsvector TSVECTOR;
            """))
            
            # Create trigger function for updating TSVector
            session.execute(text("""
                CREATE OR REPLACE FUNCTION update_document_tsvector()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.document_tsvector := to_tsvector('english', NEW.document);
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """))
            
            # Create trigger
            session.execute(text("""
                CREATE TRIGGER document_tsvector_update
                BEFORE INSERT OR UPDATE ON langchain_pg_embedding
                FOR EACH ROW EXECUTE FUNCTION update_document_tsvector();
            """))
            
            # Initialize existing records
            session.execute(text("""
                UPDATE langchain_pg_embedding
                SET document_tsvector = to_tsvector('english', document)
                WHERE document_tsvector IS NULL;
            """))
            session.commit()

    def hybrid_search(
        self,
        query: str,
        k: int = 4,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform hybrid search combining vector and full-text search"""
        # Generate query embedding
        embedding = self.embedding_function.embed_query(query)
        
        with self.Session() as session:
            # Get collection UUID
            collection = self.get_collection(session)
            if not collection:
                raise ValueError("Collection not found")
            
            # Build base query
            embedding_store = EmbeddingStore
            base_query = select(
                embedding_store,
                (
                    (1 - embedding_store.embedding.cosine_distance(embedding)) * alpha +
                    func.ts_rank(embedding_store.document_tsvector, 
                                func.plainto_tsquery('english', query)) * (1 - alpha)
                ).label("score")
            ).where(
                embedding_store.collection_id == collection.uuid
            )
            
            # Apply metadata filter if provided
            if filter:
                filter_clause = self._create_filter_clause(filter)
                base_query = base_query.where(filter_clause)
            
            # Execute query
            results = (
                session.execute(
                    base_query.order_by(text("score DESC")).limit(k)
                )
                .fetchall()
            )
            
            return [
                Document(
                    page_content=result.EmbeddingStore.document,
                    metadata=result.EmbeddingStore.cmetadata
                ) for result in results
            ]

# Usage example
vector_store = PGVectorHybrid(
    embeddings=embeddings,
    collection_name=collection_name,
    use_jsonb=True,
    connection=connection,
    distance_strategy="cosine"
)

# Perform hybrid search
results = vector_store.hybrid_search(
    query="your search query",
    k=5,
    alpha=0.6  # Weight for vector search (0.6 = 60% vector, 40% full-text)
)

=============

import psycopg2
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from langchain.schema import Document
from langchain_community.vectorstores import PGVector

def setup_fulltext_search(connection_string: str, collection_name: str):
    """
    Setup full-text search capabilities for the existing PGVector table
    """
    # Convert connection string format
    conn_str = connection_string.replace("postgressql+psycopg2://", "postgresql://")
    
    conn = psycopg2.connect(conn_str)
    conn.autocommit = True
    
    try:
        with conn.cursor() as cur:
            # Add tsvector column if it doesn't exist
            cur.execute("""
                ALTER TABLE langchain_pg_embedding 
                ADD COLUMN IF NOT EXISTS document_tsvector tsvector;
            """)
            
            # Update existing records with tsvector
            cur.execute("""
                UPDATE langchain_pg_embedding 
                SET document_tsvector = to_tsvector('english', document)
                WHERE collection_id = (
                    SELECT uuid FROM langchain_pg_collection 
                    WHERE name = %s
                ) AND document_tsvector IS NULL;
            """, (collection_name,))
            
            # Create GIN index for full-text search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_langchain_pg_embedding_tsvector 
                ON langchain_pg_embedding USING gin(document_tsvector);
            """)
            
            # Create function to update tsvector
            cur.execute("""
                CREATE OR REPLACE FUNCTION update_langchain_tsvector() 
                RETURNS trigger AS $$
                BEGIN
                    NEW.document_tsvector := to_tsvector('english', NEW.document);
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # Create trigger for automatic tsvector updates
            cur.execute("""
                DROP TRIGGER IF EXISTS langchain_tsvector_trigger ON langchain_pg_embedding;
                CREATE TRIGGER langchain_tsvector_trigger 
                BEFORE INSERT OR UPDATE ON langchain_pg_embedding
                FOR EACH ROW EXECUTE FUNCTION update_langchain_tsvector();
            """)
            
            print("Full-text search setup completed successfully!")
            
    except Exception as e:
        print(f"Error setting up full-text search: {e}")
    finally:
        conn.close()

def hybrid_search(
    query: str,
    vector_store: PGVector,
    k: int = 5,
    vector_weight: float = 0.5,
    text_weight: float = 0.5,
    vector_k: int = 20,
    text_k: int = 20,
    score_threshold: float = 0.0
) -> List[Document]:
    """
    Perform hybrid search combining vector similarity and full-text search
    
    Args:
        query: Search query string
        vector_store: PGVector instance
        k: Number of final results to return
        vector_weight: Weight for vector similarity score (0-1)
        text_weight: Weight for full-text search score (0-1)
        vector_k: Number of candidates from vector search
        text_k: Number of candidates from text search
        score_threshold: Minimum hybrid score threshold
    
    Returns:
        List of Document objects with hybrid scores in metadata
    """
    
    # Get connection details from vector_store
    connection_string = vector_store._connection_string
    collection_name = vector_store.collection_name
    
    # Convert connection string
    conn_str = connection_string.replace("postgressql+psycopg2://", "postgresql://")
    
    conn = psycopg2.connect(conn_str)
    
    try:
        with conn.cursor() as cur:
            # Get collection ID
            cur.execute(
                "SELECT uuid FROM langchain_pg_collection WHERE name = %s",
                (collection_name,)
            )
            result = cur.fetchone()
            if not result:
                raise ValueError(f"Collection '{collection_name}' not found")
            
            collection_id = result[0]
            
            # Generate query embedding
            query_embedding = vector_store.embeddings.embed_query(query)
            
            # Perform vector similarity search
            vector_results = _get_vector_results(cur, collection_id, query_embedding, vector_k)
            
            # Perform full-text search
            text_results = _get_text_results(cur, collection_id, query, text_k)
            
            # Combine and score results
            hybrid_results = _combine_and_score(
                vector_results, text_results, vector_weight, text_weight, score_threshold
            )
            
            # Convert to Document objects and return top k
            documents = []
            for result in hybrid_results[:k]:
                doc = Document(
                    page_content=result['content'],
                    metadata={
                        **result['metadata'],
                        'vector_score': result['vector_score'],
                        'text_score': result['text_score'],
                        'hybrid_score': result['hybrid_score'],
                        'search_type': 'hybrid'
                    }
                )
                documents.append(doc)
            
            return documents
            
    except Exception as e:
        print(f"Error in hybrid search: {e}")
        return []
    finally:
        conn.close()

def _get_vector_results(cur, collection_id: str, query_embedding: List[float], k: int) -> Dict[str, Dict]:
    """Get vector similarity search results"""
    cur.execute("""
        SELECT uuid, document, cmetadata, 
               1 - (embedding <=> %s) as similarity
        FROM langchain_pg_embedding
        WHERE collection_id = %s
        ORDER BY embedding <=> %s
        LIMIT %s
    """, (query_embedding, collection_id, query_embedding, k))
    
    results = {}
    for uuid, document, metadata, similarity in cur.fetchall():
        results[uuid] = {
            'content': document,
            'metadata': metadata or {},
            'vector_score': float(similarity),
            'text_score': 0.0
        }
    
    return results

def _get_text_results(cur, collection_id: str, query: str, k: int) -> Dict[str, Dict]:
    """Get full-text search results"""
    cur.execute("""
        SELECT uuid, document, cmetadata,
               ts_rank_cd(document_tsvector, plainto_tsquery('english', %s)) as rank
        FROM langchain_pg_embedding
        WHERE collection_id = %s
        AND document_tsvector @@ plainto_tsquery('english', %s)
        AND document_tsvector IS NOT NULL
        ORDER BY rank DESC
        LIMIT %s
    """, (query, collection_id, query, k))
    
    results = {}
    for uuid, document, metadata, rank in cur.fetchall():
        results[uuid] = {
            'content': document,
            'metadata': metadata or {},
            'vector_score': 0.0,
            'text_score': float(rank)
        }
    
    return results

def _normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to 0-1 range"""
    if not scores or max(scores) == 0:
        return [0.0] * len(scores)
    
    max_score = max(scores)
    min_score = min(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    return [(score - min_score) / (max_score - min_score) for score in scores]

def _combine_and_score(
    vector_results: Dict[str, Dict],
    text_results: Dict[str, Dict],
    vector_weight: float,
    text_weight: float,
    score_threshold: float
) -> List[Dict]:
    """Combine vector and text results with hybrid scoring"""
    
    # Combine all unique documents
    all_results = {}
    
    # Add vector results
    for uuid, data in vector_results.items():
        all_results[uuid] = data.copy()
    
    # Merge text results
    for uuid, data in text_results.items():
        if uuid in all_results:
            all_results[uuid]['text_score'] = data['text_score']
        else:
            all_results[uuid] = data.copy()
    
    # Normalize scores
    vector_scores = [data['vector_score'] for data in all_results.values()]
    text_scores = [data['text_score'] for data in all_results.values()]
    
    normalized_vector = _normalize_scores(vector_scores)
    normalized_text = _normalize_scores(text_scores)
    
    # Calculate hybrid scores
    final_results = []
    for i, (uuid, data) in enumerate(all_results.items()):
        hybrid_score = (
            vector_weight * normalized_vector[i] + 
            text_weight * normalized_text[i]
        )
        
        if hybrid_score >= score_threshold:
            data['vector_score'] = normalized_vector[i]
            data['text_score'] = normalized_text[i]
            data['hybrid_score'] = hybrid_score
            data['uuid'] = uuid
            final_results.append(data)
    
    # Sort by hybrid score
    final_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return final_results

# Usage with your existing code
if __name__ == "__main__":
    # Your existing setup
    connection = "postgressql+psycopg2://postgres:postgres@localhost:5432/test"
    collection_name = "test"

    embeddings = CustomEmbedder(
        api_url="https://api.openai.com/v1/embeddings",
        model_name="text-embedding-ada-002",
        timeout=30
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        use_jsonb=True,
        connection=connection,
        distance_strategy="cosine"
    )

    # Setup full-text search (run this once)
    setup_fulltext_search(connection, collection_name)

    # Perform hybrid search
    query = "machine learning"
    docs = hybrid_search(
        query, 
        vector_store, 
        k=3,
        vector_weight=0.6,  # Favor semantic similarity
        text_weight=0.4,    # Some weight on keyword matching
        vector_k=10,        # Get top 10 from vector search
        text_k=10          # Get top 10 from text search
    )

    # Display results
    for i, doc in enumerate(docs, 1):
        print(f"Result {i}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Vector Score: {doc.metadata.get('vector_score', 0):.3f}")
        print(f"Text Score: {doc.metadata.get('text_score', 0):.3f}")
        print(f"Hybrid Score: {doc.metadata.get('hybrid_score', 0):.3f}")
        print(f"Other Metadata: {doc.metadata}")
        print("-" * 50)
