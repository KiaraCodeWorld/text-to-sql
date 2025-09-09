# deepseek : 

import os
import gradio as gr
import psycopg2
import pandas as pd
from typing import List, Dict, Any, Optional
import requests
import json
import re
from datetime import datetime, timedelta

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv("DB_HOST", "localhost"),
    'database': os.getenv("DB_NAME", "your_database"),
    'user': os.getenv("DB_USER", "your_username"),
    'password': os.getenv("DB_PASSWORD", "your_password"),
    'port': os.getenv("DB_PORT", "5432")
}

# LLAMA API configuration
LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://localhost:11434/v1/chat/completions")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "your-llama-api-key")

class CustomLLM:
    """Custom LLM wrapper for LLAMA API"""
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else ""
        }
    
    def query_llama(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.1) -> str:
        """Query the LLAMA API with a prompt"""
        payload = {
            "model": "llama3",  # Adjust based on your available model
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["\n\n"]
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Error querying LLAMA API: {e}")

class DatabaseManager:
    """Manage database connections and queries"""
    
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.connection = self._create_connection()
    
    def _create_connection(self):
        """Create database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            raise ValueError(f"Database connection failed: {str(e)}")
    
    def get_table_schema(self, table_name: str = "CX.member_call_info") -> str:
        """Get schema description for a table"""
        query = """
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position;
        """
        
        # Split schema and table name
        schema, table = table_name.split('.') if '.' in table_name else ('public', table_name)
        
        with self.connection.cursor() as cursor:
            cursor.execute(query, (schema, table))
            columns = cursor.fetchall()
        
        schema_description = f"Table {table_name} has the following columns:\n"
        for col_name, data_type in columns:
            schema_description += f"- {col_name} ({data_type})\n"
        
        return schema_description
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            return pd.read_sql_query(sql_query, self.connection)
        except Exception as e:
            raise ValueError(f"Query execution failed: {str(e)}\nQuery: {sql_query}")

class TableAugmentedGenerator:
    """Implement Table-Augmented Generation for database querying"""
    
    def __init__(self, db_manager: DatabaseManager, llm: CustomLLM):
        self.db_manager = db_manager
        self.llm = llm
    
    def generate_sql_query(self, question: str, schema_info: str) -> str:
        """Generate SQL query from natural language question"""
        prompt = f"""
        You are an expert SQL developer. Based on the following table schema:
        
        {schema_info}
        
        Generate a PostgreSQL query to answer this question: {question}
        
        Important guidelines:
        1. Use only the columns mentioned in the schema
        2. Use appropriate WHERE clauses for filtering
        3. Use appropriate date functions for the calldate column
        4. Return the query only, without any explanation
        5. Use proper SQL syntax for PostgreSQL
        6. For date filters, use current date if needed: {datetime.now().date()}
        7. Always include a LIMIT clause if returning many rows to prevent excessive data
        
        SQL Query:
        """
        
        response = self.llm.query_llama(prompt)
        # Extract SQL query from response (in case LLM adds explanation)
        sql_match = re.search(r"(SELECT.*?;)", response, re.IGNORECASE | re.DOTALL)
        if sql_match:
            return sql_match.group(1)
        return response
    
    def generate_natural_language_response(self, question: str, data: pd.DataFrame) -> str:
        """Generate natural language response from query results"""
        if data.empty:
            return "I couldn't find any data matching your query."
        
        # Format data for LLM consumption
        data_str = data.to_string(index=False)
        
        prompt = f"""
        Based on the following data retrieved from the database:
        
        {data_str}
        
        Please provide a concise, natural language answer to the question: {question}
        
        Guidelines:
        1. Be specific and reference actual numbers from the data
        2. Keep the response professional but conversational
        3. If the data contains multiple records, summarize the key insights
        4. Do not make up information not present in the data
        5. Format the response nicely with line breaks and bullet points if needed
        
        Response:
        """
        
        return self.llm.query_llama(prompt)
    
    def process_query(self, question: str, table_name: str = "CX.member_call_info") -> str:
        """Complete TAG process for a question"""
        try:
            # Step 1: Get schema information
            schema_info = self.db_manager.get_table_schema(table_name)
            
            # Step 2: Generate SQL query
            sql_query = self.generate_sql_query(question, schema_info)
            
            # Step 3: Execute query
            result_data = self.db_manager.execute_query(sql_query)
            
            # Step 4: Generate natural language response
            response = self.generate_natural_language_response(question, result_data)
            
            return response
        
        except Exception as e:
            return f"I encountered an error processing your request: {str(e)}"

# Initialize components
db_manager = DatabaseManager(DB_CONFIG)
llm = CustomLLM(api_url=LLAMA_API_URL, api_key=LLAMA_API_KEY)
tag_engine = TableAugmentedGenerator(db_manager, llm)

def chat_function(message, history):
    """
    Process user message and generate response using TAG approach
    """
    # Use Table-Augmented Generation to process query
    response = tag_engine.process_query(
        question=message, 
        table_name="CX.member_call_info"
    )
    
    return response

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_function,
    type="messages",
    title="Call Data Analytics Assistant",
    description="Ask questions about your call data in natural language. Example: 'How many calls did enterprise X receive last month?'",
    examples=[
        "Show me all calls from enterprise XYZ last week",
        "Which agent handled the most calls in March?",
        "What are the most common caller intents for division ABC?",
        "How many calls occurred by day last week?",
        "Show calls where the caller intent was 'complaint'"
    ],
    chatbot=gr.Chatbot(height=500, label="Call Data Chatbot"),
    textbox=gr.Textbox(placeholder="Type your question about call data here...", container=False, scale=7),
    theme="soft",
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear Chat"
)

# Launch application
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)


===============
# gmeini

import os
import gradio as gr
import psycopg2
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database credentials
DB_NAME = os.getenv("PG_DB_NAME")
DB_USER = os.getenv("PG_DB_USER")
DB_PASS = os.getenv("PG_DB_PASS")
DB_HOST = os.getenv("PG_DB_HOST")
DB_PORT = os.getenv("PG_DB_PORT")

# Llama LLM API endpoint
LLAMA_API_URL = os.getenv("LLAMA_API_URL") # Example: "http://localhost:8000/v1/chat/completions"
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY") # This might not be needed for a local server

# Database schema for the LLM
DB_SCHEMA = """
CREATE TABLE CX.member_call_info (
    caller VARCHAR,
    agent VARCHAR,
    pty_id VARCHAR,
    calldate TIMESTAMP,
    enterprise VARCHAR,
    division VARCHAR,
    business_name VARCHAR,
    caller_intent TEXT
);
"""

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to database: {e}")
        return None

def query_llama_for_sql(question):
    """
    Queries a Llama LLM API to generate a SQL query based on a user's question.
    """
    prompt = f"""
    You are a SQL expert. Your task is to generate a valid SQL query for a PostgreSQL database
    based on the user's question.

    Database Schema:
    {DB_SCHEMA}

    Only provide the SQL query. Do not include any explanations, code blocks, or markdown.

    User's Question: {question}

    SQL Query:
    """
    
    headers = {
        "Content-Type": "application/json",
        # "Authorization": f"Bearer {LLAMA_API_KEY}" # Uncomment if your API requires a key
    }
    
    data = {
        "model": "llama-model-name", # Replace with your Llama model name
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }

    try:
        response = requests.post(LLAMA_API_URL, headers=headers, json=data)
        response.raise_for_status() # Raise an exception for bad status codes
        
        # Adjust based on the actual API response structure
        sql_query = response.json()['choices'][0]['message']['content'].strip()
        
        # Simple cleanup to remove potential code blocks or unwanted text
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").strip()
        if sql_query.endswith("```"):
            sql_query = sql_query.replace("```", "").strip()
            
        return sql_query

    except requests.exceptions.RequestException as e:
        print(f"Error querying Llama API: {e}")
        return None
    except KeyError:
        print("Unexpected response format from Llama API.")
        return None

def query_llama_for_response(question, sql_results):
    """
    Queries a Llama LLM API to generate a conversational response based on SQL results.
    """
    results_str = ""
    if sql_results:
        # Convert results to a readable string format
        results_str = "\n".join([str(row) for row in sql_results])

    prompt = f"""
    You are an expert assistant for analyzing call data.
    Based on the following user question and the corresponding database query results,
    provide a clear, concise, and conversational answer.

    User's Question: {question}

    Database Query Results:
    {results_str}

    If the results are empty, inform the user that no data was found for their query.
    If there are results, summarize the key findings in a human-readable format.
    """
    
    headers = {
        "Content-Type": "application/json",
        # "Authorization": f"Bearer {LLAMA_API_KEY}"
    }

    data = {
        "model": "llama-model-name", # Replace with your Llama model name
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(LLAMA_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error generating conversational response: {e}")
        return "Sorry, I couldn't generate a response. There was an issue with the LLM."
    except KeyError:
        print("Unexpected response format from Llama API.")
        return "Sorry, I couldn't generate a response. There was an issue with the LLM."

def chat_logic(user_message, history):
    """
    The main function for the chatbot's logic.
    """
    conn = get_db_connection()
    if not conn:
        return "Sorry, I couldn't connect to the database. Please check the credentials."

    with conn:
        with conn.cursor() as cur:
            # Step 1: Generate SQL query from user message using LLM
            sql_query = query_llama_for_sql(user_message)
            if not sql_query:
                return "I couldn't generate a valid SQL query from your request."

            try:
                # Step 2: Execute the generated SQL query
                cur.execute(sql_query)
                results = cur.fetchall()
                
                # Step 3: Use the results to generate a conversational response
                conversational_response = query_llama_for_response(user_message, results)
                return conversational_response

            except psycopg2.Error as e:
                print(f"SQL execution error: {e}")
                return f"An error occurred while executing the query. The generated query was:\n```sql\n{sql_query}\n```\nError: {e}"

### 3. Gradio UI Code

This section sets up the Gradio interface to use the `chat_logic` function.

```python
# The `chat_logic` function is defined above
# ...

# Gradio Interface
gr.ChatInterface(
    fn=chat_logic,
    chatbot=gr.Chatbot(height=500),
    title="Call Data Analyst Chatbot",
    description="Ask me questions about the call data in the `CX.member_call_info` table.",
    examples=[
        "Show me all calls handled by agent 'johndoe'.",
        "What was the most common caller intent in the last week?",
        "Find the calls from 'ACME Corporation' and their duration."
    ],
    theme="soft"
).launch()


  ==============

  Perplexity : 

  import os
import gradio as gr
import psycopg2
import pandas as pd
import requests

# Load environment variables
LLAMA_API = os.getenv('LLAMA_API_URL', 'http://localhost:11434/api/generate')
PG_CONN = os.getenv('PG_CONN', 'postgresql://user:pass@localhost:5432/db')

def get_table_df(query="SELECT * FROM CX.member_call_info LIMIT 100"):
    with psycopg2.connect(PG_CONN) as conn:
        return pd.read_sql(query, conn)

def tab_aug_prompt(df, user_query):
    preview = df.head(5).to_markdown(index=False)
    schema = ", ".join(df.columns)
    prompt = (
        f"The table schema is: {schema}\n"
        "Sample records are below:\n"
        f"{preview}\n\n"
        f"Answer this user query based on all the data: {user_query}\n"
        "Give relevant results, statistics, or lists using the real call data. "
        "Cite 'caller', 'calldate', 'intent', 'division', 'agent' as needed."
    )
    return prompt

def call_llama(prompt, system=None):
    body = {"prompt": prompt}
    if system:
        body["system"] = system
    resp = requests.post(LLAMA_API, json=body, timeout=60)
    return resp.json()["response"]

def chat_fn(message, history):
    # Fetch table subset for context (you may want to use semantic or keyword search here)
    df = get_table_df()
    prompt = tab_aug_prompt(df, message)
    response = call_llama(prompt)
    return response

with gr.Blocks() as demo:
    gr.ChatInterface(
        chat_fn,
        title="Call Data Chatbot",
        description="Ask questions about the call records stored in the CX.member_call_info table."
    )

if __name__ == "__main__":
    demo.launch()


=========
Claude :

import os
import pandas as pd
import gradio as gr
import psycopg2
from sqlalchemy import create_engine
import requests
import json
from dotenv import load_dotenv
from typing import List, Dict, Any
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CallDataAnalyzer:
    def __init__(self):
        self.db_engine = None
        self.llama_api_url = os.getenv("LLAMA_API_URL", "http://localhost:8000/v1/chat/completions")
        self.llama_api_key = os.getenv("LLAMA_API_KEY", "")
        self.setup_database()
        
    def setup_database(self):
        """Setup PostgreSQL database connection"""
        try:
            db_url = f"postgresql://{os.getenv('DB_USER', 'postgres')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME', 'calldata')}"
            self.db_engine = create_engine(db_url)
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def query_llama_api(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Query LLAMA API for chat completion
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llama_api_key}"
        }
        
        payload = {
            "model": "llama-2-7b-chat",  # Adjust model name as needed
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(self.llama_api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logger.error(f"LLAMA API request failed: {e}")
            return f"Error querying LLAMA API: {e}"
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing LLAMA API response: {e}")
            return f"Error parsing API response: {e}"

    def get_table_schema(self) -> str:
        """Get the schema of the member_call_info table"""
        schema_query = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'cx' AND table_name = 'member_call_info'
        ORDER BY ordinal_position;
        """
        
        try:
            df = pd.read_sql(schema_query, self.db_engine)
            schema_info = []
            for _, row in df.iterrows():
                nullable = "NULL" if row['is_nullable'] == 'YES' else "NOT NULL"
                schema_info.append(f"- {row['column_name']}: {row['data_type']} ({nullable})")
            
            return "\n".join(schema_info)
        except Exception as e:
            logger.error(f"Error getting table schema: {e}")
            return "Error retrieving table schema"

    def get_sample_data(self, limit: int = 5) -> str:
        """Get sample data from the table"""
        try:
            query = f"SELECT * FROM cx.member_call_info LIMIT {limit}"
            df = pd.read_sql(query, self.db_engine)
            return df.to_string(index=False)
        except Exception as e:
            logger.error(f"Error getting sample data: {e}")
            return "Error retrieving sample data"

    def generate_sql_query(self, user_question: str) -> str:
        """Generate SQL query based on user question using LLAMA"""
        schema = self.get_table_schema()
        sample_data = self.get_sample_data()
        
        system_prompt = f"""
        You are an expert SQL analyst working with a PostgreSQL database containing call center data.
        
        Table: cx.member_call_info
        Schema:
        {schema}
        
        Sample data:
        {sample_data}
        
        Column descriptions:
        - caller: The phone number or identifier of the person making the call
        - agent: The agent who handled the call
        - pty_id: Party ID or unique identifier for the call
        - calldate: Date and time when the call occurred
        - enterprise: The enterprise or company division
        - division: Specific division within the enterprise
        - business_name: Name of the business unit
        - caller_intent: The intent or purpose of the caller (e.g., complaint, inquiry, support)
        
        Generate a PostgreSQL SQL query to answer the user's question. 
        Return ONLY the SQL query, no explanations or additional text.
        Use proper PostgreSQL syntax and make sure to reference the full table name 'cx.member_call_info'.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {user_question}"}
        ]
        
        return self.query_llama_api(messages, temperature=0.3)

    def execute_sql_query(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            # Clean up the SQL query (remove markdown formatting if present)
            sql_query = sql_query.strip()
            if sql_query.startswith("```sql"):
                sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            df = pd.read_sql(sql_query, self.db_engine)
            return df
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            raise

    def generate_conversational_response(self, user_question: str, sql_query: str, query_results: pd.DataFrame) -> str:
        """Generate conversational response based on query results"""
        results_summary = self.summarize_results(query_results)
        
        system_prompt = """
        You are a helpful call center data analyst assistant. Your role is to provide clear, 
        insightful analysis of call center data in a conversational manner.
        
        When analyzing call data, consider:
        - Call volume patterns and trends
        - Agent performance metrics
        - Customer intent analysis
        - Business unit performance
        - Time-based patterns
        - Operational insights
        
        Provide actionable insights and highlight important findings.
        Use emojis and formatting to make responses engaging and easy to read.
        """
        
        user_prompt = f"""
        User Question: {user_question}
        
        SQL Query Used: {sql_query}
        
        Query Results Summary: {results_summary}
        
        Please provide a conversational analysis of these call center results, 
        including any insights, trends, or recommendations based on the data.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.query_llama_api(messages, temperature=0.7)

    def summarize_results(self, df: pd.DataFrame) -> str:
        """Create a summary of query results for the LLM"""
        if df.empty:
            return "No results found."
        
        summary = f"Found {len(df)} records.\n"
        
        # Add column info
        summary += f"Columns: {', '.join(df.columns.tolist())}\n"
        
        # Add sample of data (first few rows)
        if len(df) > 0:
            summary += f"\nSample data:\n{df.head(3).to_string(index=False)}\n"
        
        # Add basic statistics for numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
            summary += f"\nNumeric column statistics:\n{df[numeric_columns].describe().to_string()}\n"
        
        return summary

    def process_user_query(self, user_question: str, chat_history: List[List[str]]) -> tuple:
        """Main function to process user queries using Table Augmented Generation (TAG)"""
        try:
            # Step 1: Generate SQL query using LLAMA
            sql_query = self.generate_sql_query(user_question)
            
            # Step 2: Execute the query
            query_results = self.execute_sql_query(sql_query)
            
            # Step 3: Generate conversational response
            conversational_response = self.generate_conversational_response(
                user_question, sql_query, query_results
            )
            
            # Format the complete response
            complete_response = f"""
## 🔍 Analysis Results

{conversational_response}

### 📊 Data Summary
- **Records Found**: {len(query_results)}
- **SQL Query**: 
```sql
{sql_query}
```

### 📋 Raw Data
{query_results.to_markdown(index=False) if not query_results.empty else "No data found"}
            """
            
            # Update chat history
            chat_history.append([user_question, complete_response])
            
            return "", chat_history, query_results
            
        except Exception as e:
            error_response = f"❌ **Error**: {str(e)}\n\nPlease try rephrasing your question or check if the data exists."
            chat_history.append([user_question, error_response])
            return "", chat_history, pd.DataFrame()

# Initialize the analyzer
analyzer = CallDataAnalyzer()

# Define Gradio interface
def create_gradio_interface():
    with gr.Blocks(title="Call Data Analysis Chat", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 📞 Call Data Analysis Chat
        
        Ask questions about your call center data and get intelligent insights!
        
        **Example questions:**
        - "Show me calls from last week"
        - "Which agents handled the most complaints?"
        - "What are the top caller intents?"
        - "Show call volume by business unit"
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    value=[],
                    label="Chat History",
                    height=500,
                    show_label=True,
                    container=True
                )
                
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Ask a question about your call data...",
                        label="Your Question",
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                clear_btn = gr.Button("Clear Chat", variant="secondary")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📈 Query Results")
                results_df = gr.Dataframe(
                    value=pd.DataFrame(),
                    label="Data Results",
                    interactive=False,
                    wrap=True
                )
        
        # Event handlers
        submit_btn.click(
            fn=analyzer.process_user_query,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot, results_df]
        )
        
        user_input.submit(
            fn=analyzer.process_user_query,
            inputs=[user_input, chatbot],
            outputs=[user_input, chatbot, results_df]
        )
        
        clear_btn.click(
            fn=lambda: ([], pd.DataFrame()),
            outputs=[chatbot, results_df]
        )
    
    return interface

# Launch the application
if __name__ == "__main__":
    try:
        # Create and launch Gradio interface
        interface = create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True
        )
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        print(f"Error: {e}")

==
# requirements.txt
gradio==4.8.0
pandas==2.0.3
psycopg2-binary==2.9.7
sqlalchemy==2.0.21
requests==2.31.0
python-dotenv==1.0.0

# .env file template
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=calldata
DB_USER=postgres
DB_PASSWORD=your_password_here

# LLAMA API Configuration
LLAMA_API_URL=http://localhost:8000/v1/chat/completions
LLAMA_API_KEY=your_llama_api_key_here

# Optional: For local LLAMA deployment
LLAMA_MODEL_PATH=/path/to/your/llama/model

# setup_database.py - Script to create the sample table structure
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def create_sample_table():
    """Create the sample member_call_info table with sample data"""
    
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    
    cur = conn.cursor()
    
    try:
        # Create schema if it doesn't exist
        cur.execute("CREATE SCHEMA IF NOT EXISTS cx;")
        
        # Create the table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS cx.member_call_info (
            id SERIAL PRIMARY KEY,
            caller VARCHAR(50),
            agent VARCHAR(100),
            pty_id VARCHAR(50),
            calldate TIMESTAMP,
            enterprise VARCHAR(100),
            division VARCHAR(100),
            business_name VARCHAR(200),
            caller_intent VARCHAR(100)
        );
        """
        cur.execute(create_table_query)
        
        # Insert sample data
        sample_data = [
            ('555-1234', 'John Smith', 'PTY001', '2024-01-15 10:30:00', 'TechCorp', 'Customer Service', 'Technical Support', 'Technical Issue'),
            ('555-5678', 'Sarah Jones', 'PTY002', '2024-01-15 11:45:00', 'TechCorp', 'Sales', 'Product Sales', 'Product Inquiry'),
            ('555-9012', 'Mike Wilson', 'PTY003', '2024-01-15 14:20:00', 'TechCorp', 'Customer Service', 'Billing Support', 'Billing Question'),
            ('555-3456', 'Lisa Brown', 'PTY004', '2024-01-15 15:10:00', 'TechCorp', 'Customer Service', 'Technical Support', 'Complaint'),
            ('555-7890', 'David Chen', 'PTY005', '2024-01-16 09:15:00', 'TechCorp', 'Sales', 'Product Sales', 'Product Inquiry'),
            ('555-2468', 'Emma Davis', 'PTY006', '2024-01-16 10:30:00', 'TechCorp', 'Customer Service', 'General Support', 'General Inquiry'),
            ('555-1357', 'Tom Miller', 'PTY007', '2024-01-16 13:45:00', 'TechCorp', 'Customer Service', 'Technical Support', 'Technical Issue'),
            ('555-8642', 'Amy Johnson', 'PTY008', '2024-01-16 16:20:00', 'TechCorp', 'Sales', 'Product Sales', 'Price Inquiry'),
        ]
        
        insert_query = """
        INSERT INTO cx.member_call_info 
        (caller, agent, pty_id, calldate, enterprise, division, business_name, caller_intent)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING;
        """
        
        cur.executemany(insert_query, sample_data)
        
        conn.commit()
        print("Sample table and data created successfully!")
        
    except Exception as e:
        print(f"Error creating table: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    create_sample_table()

# docker-compose.yml - For easy PostgreSQL setup
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: calldata
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:

# llama_server.py - Simple LLAMA API server setup using Ollama or similar
import subprocess
import os

def setup_llama_server():
    """
    Setup script for running LLAMA locally using Ollama
    This assumes you have Ollama installed
    """
    
    print("Setting up LLAMA server...")
    
    # Pull the LLAMA model (adjust model name as needed)
    try:
        subprocess.run(["ollama", "pull", "llama2:7b-chat"], check=True)
        print("LLAMA model pulled successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error pulling LLAMA model: {e}")
        return False
    
    # Start the server
    try:
        print("Starting LLAMA server on port 11434...")
        print("Server will be available at: http://localhost:11434/v1/chat/completions")
        subprocess.run(["ollama", "serve"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting LLAMA server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    setup_llama_server()






