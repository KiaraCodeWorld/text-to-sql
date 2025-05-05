# Text-to-SQL Application with Langchain and PG Vector
# app.py

import os
import json
import psycopg2
import gradio as gr
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import PGVector
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY", OPENAI_API_KEY)  # Using same key by default

# PG Vector Connection String
CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
COLLECTION_NAME = "table_metadata"

class TextToSQLApp:
    def __init__(self):
        # Initialize components
        self.embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.vector_store = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
            embedding_function=self.embedding
        )
        self.llm = ChatOpenAI(api_key=LLM_API_KEY, temperature=0, model="gpt-4")
        self.db_conn = None
        self.tables_metadata = {}
        
    def connect_db(self):
        """Establish connection to PostgreSQL database"""
        try:
            self.db_conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            return True
        except Exception as e:
            print(f"Database connection error: {e}")
            return False
    
    def extract_table_metadata(self):
        """Extract schema information from PostgreSQL tables"""
        if not self.db_conn:
            if not self.connect_db():
                return "Failed to connect to database"
        
        cursor = self.db_conn.cursor()
        
        # Get all tables in the database
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [table[0] for table in cursor.fetchall()]
        
        # Extract metadata for each table
        for table in tables:
            # Get column information
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = '{table}'
            """)
            columns = cursor.fetchall()
            
            # Get primary key information
            cursor.execute(f"""
                SELECT c.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage AS ccu USING (constraint_schema, constraint_name)
                JOIN information_schema.columns AS c ON c.table_schema = tc.constraint_schema
                    AND tc.table_name = c.table_name AND ccu.column_name = c.column_name
                WHERE constraint_type = 'PRIMARY KEY' AND tc.table_name = '{table}'
            """)
            primary_keys = [pk[0] for pk in cursor.fetchall()]
            
            # Get foreign key information
            cursor.execute(f"""
                SELECT
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE constraint_type = 'FOREIGN KEY' AND tc.table_name = '{table}'
            """)
            foreign_keys = cursor.fetchall()
            
            # Create metadata object
            table_metadata = {
                "table_name": table,
                "columns": [{
                    "name": col[0],
                    "data_type": col[1],
                    "is_nullable": col[2],
                    "default": col[3]
                } for col in columns],
                "primary_keys": primary_keys,
                "foreign_keys": [{
                    "column": fk[0],
                    "references_table": fk[1],
                    "references_column": fk[2]
                } for fk in foreign_keys]
            }
            
            self.tables_metadata[table] = table_metadata
        
        cursor.close()
        return f"Successfully extracted metadata for {len(tables)} tables"
    
    def store_metadata_in_vector_db(self):
        """Store table metadata in PG Vector database"""
        if not self.tables_metadata:
            return "No metadata to store. Please extract metadata first."
        
        documents = []
        
        # Convert metadata to documents
        for table, metadata in self.tables_metadata.items():
            # Create a detailed description for the table
            column_descriptions = []
            for col in metadata["columns"]:
                col_desc = f"{col['name']} ({col['data_type']})"
                if col["name"] in metadata["primary_keys"]:
                    col_desc += " [PRIMARY KEY]"
                for fk in metadata["foreign_keys"]:
                    if fk["column"] == col["name"]:
                        col_desc += f" [FOREIGN KEY to {fk['references_table']}.{fk['references_column']}]"
                column_descriptions.append(col_desc)
            
            # Create document content
            content = f"Table: {table}\n"
            content += f"Columns: {', '.join(column_descriptions)}\n"
            content += f"Primary Keys: {', '.join(metadata['primary_keys'])}\n"
            
            fk_descriptions = []
            for fk in metadata["foreign_keys"]:
                fk_descriptions.append(f"{fk['column']} → {fk['references_table']}.{fk['references_column']}")
            
            content += f"Foreign Keys: {', '.join(fk_descriptions) if fk_descriptions else 'None'}\n"
            
            # Convert metadata to JSON for storage in metadata field
            metadata_json = json.dumps(metadata)
            
            # Create document
            doc = Document(
                page_content=content,
                metadata={"table_name": table, "raw_metadata": metadata_json}
            )
            documents.append(doc)
        
        # Store documents in vector database
        try:
            self.vector_store.add_documents(documents)
            return f"Successfully stored metadata for {len(documents)} tables in vector database"
        except Exception as e:
            return f"Error storing metadata in vector database: {e}"
    
    def generate_sql(self, query):
        """Generate SQL from natural language query"""
        if not self.tables_metadata:
            return "No metadata available. Please extract and store table metadata first."
        
        # Search for relevant tables
        relevant_docs = self.vector_store.similarity_search(query, k=3)
        
        # Prepare context for LLM
        context = ""
        for doc in relevant_docs:
            context += doc.page_content + "\n\n"
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
        Your task is to convert a natural language query into a valid SQL query.
        
        Here is the database schema information:
        {context}
        
        User query: {query}
        
        Generate a SQL query that answers the user's question.
        Only return the SQL query without any explanations or additional text.
        Make sure the SQL is valid for PostgreSQL.
        """)
        
        # Create chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Run chain
        try:
            result = chain.run(context=context, query=query)
            return result.strip()
        except Exception as e:
            return f"Error generating SQL: {e}"
    
    def execute_sql(self, sql):
        """Execute SQL query and return results"""
        if not self.db_conn:
            if not self.connect_db():
                return "Failed to connect to database", None
        
        try:
            df = pd.read_sql_query(sql, self.db_conn)
            return "Query executed successfully", df
        except Exception as e:
            return f"Error executing SQL: {e}", None
    
    def process_query(self, query):
        """Process natural language query to SQL and execute"""
        # Generate SQL from query
        sql = self.generate_sql(query)
        
        # Check if SQL generation was successful
        if sql.startswith("Error"):
            return sql, None, sql
        
        # Execute SQL
        message, results = self.execute_sql(sql)
        
        # Return results
        if results is not None:
            return message, results.to_dict('records'), sql
        else:
            return message, None, sql

# Create Gradio interface
def create_gradio_interface():
    app = TextToSQLApp()
    
    with gr.Blocks(title="Text to SQL Converter") as interface:
        gr.Markdown("# Text to SQL Converter")
        gr.Markdown("Convert natural language queries to SQL and execute them against your PostgreSQL database.")
        
        with gr.Tab("Setup"):
            gr.Markdown("## Setup Database Connection")
            setup_button = gr.Button("Connect to Database")
            extract_button = gr.Button("Extract Table Metadata")
            store_button = gr.Button("Store Metadata in Vector DB")
            setup_output = gr.Textbox(label="Setup Status")
            
            setup_button.click(app.connect_db, inputs=[], outputs=[setup_output])
            extract_button.click(app.extract_table_metadata, inputs=[], outputs=[setup_output])
            store_button.click(app.store_metadata_in_vector_db, inputs=[], outputs=[setup_output])
        
        with gr.Tab("Query"):
            gr.Markdown("## Convert Text to SQL")
            query_input = gr.Textbox(label="Enter your question", placeholder="e.g., Show me the top 5 customers by order value")
            query_button = gr.Button("Generate and Execute SQL")
            
            with gr.Row():
                with gr.Column():
                    sql_output = gr.Textbox(label="Generated SQL")
                    message_output = gr.Textbox(label="Execution Status")
                
            results_output = gr.DataFrame(label="Query Results")
            
            query_button.click(
                app.process_query, 
                inputs=[query_input], 
                outputs=[message_output, results_output, sql_output]
            )
    
    return interface

# Main function
def main():
    interface = create_gradio_interface()
    interface.launch()

if __name__ == "__main__":
    main()
