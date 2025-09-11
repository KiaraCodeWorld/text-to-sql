# Call Data AI Chat Application with Lotus.ai TAG - Setup Guide

## 🚀 Quick Start

### 1. Installation

```bash
# Install required packages
pip install lotus-ai gradio pandas psycopg2-binary python-dotenv

# For GPU support (optional but recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Environment Setup

Create a `.env` file in your project directory:

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password

# LLM API Keys (choose one or more)
GROQ_API_KEY=gsk_your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=sk-your_openai_key_here

# Optional: Embedding model cache directory
LOTUS_CACHE_DIR=./lotus_cache
```

### 3. Database Schema Verification

Ensure your PostgreSQL table exists with the correct structure:

```sql
-- Verify table structure
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_schema = 'cx' AND table_name = 'member_call_info';

-- Expected columns:
-- caller (varchar)
-- agent (varchar) 
-- pty_id (varchar)
-- calldate (timestamp/date)
-- enterprise (varchar)
-- division (varchar)
-- business_name (varchar)
-- caller_intent (text)
```

### 4. Test Database Connection

```python
import psycopg2
import pandas as pd

# Test connection
conn = psycopg2.connect(
    host="localhost",
    port="5432", 
    database="your_db",
    user="your_user",
    password="your_password"
)

# Test query
df = pd.read_sql("SELECT COUNT(*) FROM cx.member_call_info", conn)
print(f"Total records: {df.iloc[0,0]}")
```

## 🔧 Configuration Options

### LLM Model Options

The application supports multiple LLM providers through LiteLLM:

```python
# Groq (Fast, good for production)
lm = LM(model="groq/llama-3.1-70b-versatile")

# OpenAI (High quality)  
lm = LM(model="gpt-4o-mini")

# Google (Good balance)
lm = LM(model="gemini/gemini-1.5-flash")

# Local models (if you have them)
lm = LM(model="ollama/llama3.1")
```

### Embedding Models

```python
# Default (good performance)
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")

# Higher quality (slower)
rm = SentenceTransformersRM(model="intfloat/e5-large-v2")

# Multilingual support
rm = SentenceTransformersRM(model="intfloat/multilingual-e5-base")
```

## 🧪 Testing Semantic Operations

### Sample Test Script

```python
import pandas as pd
import lotus
from lotus.models import LM, SentenceTransformersRM

# Configure Lotus
lm = LM(model="groq/llama-3.1-70b-versatile") 
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
lotus.settings.configure(lm=lm, rm=rm)

# Create sample call data
sample_data = {
    "caller": ["555-0101", "555-0102", "555-0103"],
    "agent": ["John Smith", "Jane Doe

              ========
claude : 

              import pandas as pd
import gradio as gr
import psycopg2
import os
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import json

# Lotus.ai imports
import lotus
from lotus.models import LM, SentenceTransformersRM, CrossEncoderReranker
import litellm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL database connections and queries"""
    
    def __init__(self, host: str, port: int, database: str, username: str, password: str):
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': username,
            'password': password
        }
        self.connection = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def get_call_data(self, limit: int = None, where_clause: str = None) -> pd.DataFrame:
        """Fetch call data from the database"""
        if not self.connection:
            self.connect()
        
        query = "SELECT caller, agent, pty_id, calldate, enterprise, division, business_name, caller_intent FROM cx.member_call_info"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            df = pd.read_sql_query(query, self.connection)
            logger.info(f"Retrieved {len(df)} records from database")
            return df
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise
    
    def execute_custom_query(self, query: str) -> pd.DataFrame:
        """Execute a custom SQL query"""
        if not self.connection:
            self.connect()
        
        try:
            df = pd.read_sql_query(query, self.connection)
            return df
        except Exception as e:
            logger.error(f"Custom query execution failed: {e}")
            raise

class LotusSemanticProcessor:
    """Handles semantic processing using Lotus.ai library"""
    
    def __init__(self, llm_model: str = "groq/llama-3.1-70b-versatile", 
                 embedding_model: str = "intfloat/e5-base-v2",
                 reranker_model: str = "mixedbread-ai/mxbai-rerank-large-v1"):
        
        # Configure Lotus models
        self.lm = LM(model=llm_model)
        self.rm = SentenceTransformersRM(model=embedding_model)
        self.reranker = CrossEncoderReranker(model=reranker_model)
        
        # Configure Lotus settings
        lotus.settings.configure(lm=self.lm, rm=self.rm, reranker=self.reranker)
        logger.info("Lotus semantic processor initialized")
    
    def semantic_filter(self, df: pd.DataFrame, condition: str) -> pd.DataFrame:
        """Apply semantic filtering to the DataFrame"""
        try:
            filtered_df = df.sem_filter(condition)
            logger.info(f"Semantic filter applied. Rows: {len(df)} -> {len(filtered_df)}")
            return filtered_df
        except Exception as e:
            logger.error(f"Semantic filter failed: {e}")
            return df
    
    def semantic_map(self, df: pd.DataFrame, instruction: str, column_name: str = None) -> pd.DataFrame:
        """Apply semantic mapping to generate new insights"""
        try:
            if column_name:
                instruction = instruction.replace("{column}", f"{{{column_name}}}")
            mapped_df = df.sem_map(instruction)
            logger.info("Semantic mapping applied successfully")
            return mapped_df
        except Exception as e:
            logger.error(f"Semantic mapping failed: {e}")
            return df
    
    def semantic_topk(self, df: pd.DataFrame, ranking_criteria: str, k: int = 10, 
                      method: str = "quick") -> Tuple[pd.DataFrame, Dict]:
        """Get top-k results based on semantic ranking"""
        try:
            topk_df, stats = df.sem_topk(ranking_criteria, K=k, method=method, return_stats=True)
            logger.info(f"Semantic top-k applied. Returned {len(topk_df)} rows")
            return topk_df, stats
        except Exception as e:
            logger.error(f"Semantic top-k failed: {e}")
            return df.head(k), {}
    
    def semantic_aggregate(self, df: pd.DataFrame, agg_instruction: str, group_by: str = None) -> pd.DataFrame:
        """Apply semantic aggregation"""
        try:
            if group_by:
                # Group by column and then apply semantic aggregation
                grouped = df.groupby(group_by)
                results = []
                for name, group in grouped:
                    agg_result = group.sem_agg(agg_instruction)
                    agg_result[group_by] = name
                    results.append(agg_result)
                return pd.concat(results, ignore_index=True)
            else:
                agg_df = df.sem_agg(agg_instruction)
                return agg_df
        except Exception as e:
            logger.error(f"Semantic aggregation failed: {e}")
            return df
    
    def semantic_join(self, df1: pd.DataFrame, df2: pd.DataFrame, join_condition: str) -> pd.DataFrame:
        """Perform semantic join between two DataFrames"""
        try:
            joined_df = df1.sem_join(df2, join_condition)
            logger.info(f"Semantic join completed. Result has {len(joined_df)} rows")
            return joined_df
        except Exception as e:
            logger.error(f"Semantic join failed: {e}")
            return df1

class CallDataChatProcessor:
    """Main processor for handling chat queries about call data"""
    
    def __init__(self, db_manager: DatabaseManager, lotus_processor: LotusSemanticProcessor):
        self.db_manager = db_manager
        self.lotus_processor = lotus_processor
        self.conversation_history = []
    
    def process_natural_language_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query and return results with insights"""
        
        # Determine query type and appropriate data retrieval strategy
        query_analysis = self._analyze_query(query)
        
        # Get base data from database
        base_df = self._get_relevant_data(query_analysis)
        
        if base_df.empty:
            return {
                "success": False,
                "message": "No data found for your query.",
                "data": pd.DataFrame(),
                "insights": []
            }
        
        # Apply semantic operations based on query type
        result_df, insights = self._apply_semantic_operations(base_df, query, query_analysis)
        
        # Generate natural language summary
        summary = self._generate_summary(result_df, query, insights)
        
        return {
            "success": True,
            "message": summary,
            "data": result_df,
            "insights": insights,
            "query_analysis": query_analysis
        }
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the natural language query to determine processing strategy"""
        query_lower = query.lower()
        
        analysis = {
            "intent": "general",
            "operations": [],
            "entities": [],
            "time_filter": None,
            "limit": None
        }
        
        # Determine query intent
        if any(word in query_lower for word in ["filter", "show", "find", "get"]):
            analysis["intent"] = "filter"
            analysis["operations"].append("filter")
        
        if any(word in query_lower for word in ["top", "best", "worst", "rank", "highest", "lowest"]):
            analysis["intent"] = "ranking"
            analysis["operations"].append("topk")
        
        if any(word in query_lower for word in ["similar", "like", "related", "compare"]):
            analysis["intent"] = "similarity"
            analysis["operations"].append("map")
        
        if any(word in query_lower for word in ["group", "by", "aggregate", "count", "sum"]):
            analysis["intent"] = "aggregation"
            analysis["operations"].append("aggregate")
        
        # Extract entities
        entities = []
        if "agent" in query_lower:
            entities.append("agent")
        if any(word in query_lower for word in ["enterprise", "company", "business"]):
            entities.append("enterprise")
        if "division" in query_lower:
            entities.append("division")
        if any(word in query_lower for word in ["intent", "purpose", "reason"]):
            entities.append("caller_intent")
        
        analysis["entities"] = entities
        
        # Extract time filters
        if any(word in query_lower for word in ["today", "yesterday", "week", "month", "recent"]):
            analysis["time_filter"] = self._parse_time_filter(query_lower)
        
        # Extract limits
        if any(word in query_lower for word in ["top 5", "first 10", "limit"]):
            analysis["limit"] = self._extract_limit(query_lower)
        
        return analysis
    
    def _parse_time_filter(self, query: str) -> str:
        """Parse time-related filters from query"""
        if "today" in query:
            return "calldate >= CURRENT_DATE"
        elif "yesterday" in query:
            return "calldate >= CURRENT_DATE - INTERVAL '1 day' AND calldate < CURRENT_DATE"
        elif "week" in query:
            return "calldate >= CURRENT_DATE - INTERVAL '7 days'"
        elif "month" in query:
            return "calldate >= CURRENT_DATE - INTERVAL '30 days'"
        return None
    
    def _extract_limit(self, query: str) -> int:
        """Extract limit from query"""
        import re
        numbers = re.findall(r'\d+', query)
        if numbers:
            return int(numbers[0])
        return 100  # default limit
    
    def _get_relevant_data(self, query_analysis: Dict) -> pd.DataFrame:
        """Get relevant data from database based on query analysis"""
        where_clause = query_analysis.get("time_filter")
        limit = query_analysis.get("limit", 1000)  # Default limit to prevent large queries
        
        return self.db_manager.get_call_data(limit=limit, where_clause=where_clause)
    
    def _apply_semantic_operations(self, df: pd.DataFrame, query: str, 
                                   query_analysis: Dict) -> Tuple[pd.DataFrame, List[str]]:
        """Apply appropriate semantic operations based on query analysis"""
        result_df = df.copy()
        insights = []
        
        operations = query_analysis.get("operations", [])
        
        try:
            # Apply semantic filter if needed
            if "filter" in operations:
                filter_condition = self._create_filter_condition(query, query_analysis)
                if filter_condition:
                    result_df = self.lotus_processor.semantic_filter(result_df, filter_condition)
                    insights.append(f"Applied semantic filter: {filter_condition}")
            
            # Apply semantic ranking if needed
            if "topk" in operations:
                ranking_criteria = self._create_ranking_criteria(query, query_analysis)
                k = query_analysis.get("limit", 10)
                result_df, stats = self.lotus_processor.semantic_topk(result_df, ranking_criteria, k)
                insights.append(f"Applied semantic ranking: {ranking_criteria}")
                if stats:
                    insights.append(f"Ranking statistics: {stats}")
            
            # Apply semantic mapping for similarity/insights
            if "map" in operations:
                map_instruction = self._create_map_instruction(query, query_analysis)
                result_df = self.lotus_processor.semantic_map(result_df, map_instruction)
                insights.append(f"Applied semantic mapping: {map_instruction}")
            
            # Apply semantic aggregation
            if "aggregate" in operations:
                agg_instruction = self._create_aggregation_instruction(query, query_analysis)
                group_by = self._determine_group_by_column(query_analysis)
                result_df = self.lotus_processor.semantic_aggregate(result_df, agg_instruction, group_by)
                insights.append(f"Applied semantic aggregation: {agg_instruction}")
        
        except Exception as e:
            logger.error(f"Semantic operations failed: {e}")
            insights.append(f"Some semantic operations failed: {str(e)}")
        
        return result_df, insights
    
    def _create_filter_condition(self, query: str, analysis: Dict) -> str:
        """Create semantic filter condition"""
        # Example conditions based on query content
        if "billing" in query.lower():
            return "caller_intent is related to billing or payment"
        elif "technical" in query.lower():
            return "caller_intent is related to technical support or issues"
        elif "complaint" in query.lower():
            return "caller_intent indicates a complaint or dissatisfaction"
        else:
            # Generic filter based on query
            return f"The record is relevant to: {query}"
    
    def _create_ranking_criteria(self, query: str, analysis: Dict) -> str:
        """Create semantic ranking criteria"""
        if "important" in query.lower():
            return "Which call is most important or urgent?"
        elif "complex" in query.lower():
            return "Which call represents the most complex issue?"
        elif "satisfaction" in query.lower():
            return "Which call indicates highest customer satisfaction?"
        else:
            return f"Rank calls by relevance to: {query}"
    
    def _create_map_instruction(self, query: str, analysis: Dict) -> str:
        """Create semantic map instruction"""
        if "sentiment" in query.lower():
            return "What is the sentiment of this {caller_intent}?"
        elif "category" in query.lower():
            return "What category does this {caller_intent} belong to?"
        elif "priority" in query.lower():
            return "What is the priority level of this {caller_intent}?"
        else:
            return f"Provide insights about {{caller_intent}} related to: {query}"
    
    def _create_aggregation_instruction(self, query: str, analysis: Dict) -> str:
        """Create semantic aggregation instruction"""
        if "summary" in query.lower():
            return "Summarize the main themes in these calls"
        elif "trends" in query.lower():
            return "What are the key trends in these calls?"
        else:
            return f"Aggregate insights about: {query}"
    
    def _determine_group_by_column(self, analysis: Dict) -> str:
        """Determine which column to group by"""
        entities = analysis.get("entities", [])
        if "division" in entities:
            return "division"
        elif "enterprise" in entities:
            return "enterprise"
        elif "agent" in entities:
            return "agent"
        return None
    
    def _generate_summary(self, df: pd.DataFrame, query: str, insights: List[str]) -> str:
        """Generate natural language summary of results"""
        summary_parts = []
        
        # Basic statistics
        summary_parts.append(f"Found {len(df)} relevant records for your query: '{query}'")
        
        # Key insights
        if insights:
            summary_parts.append("Key insights:")
            for insight in insights[:3]:  # Limit to top 3 insights
                summary_parts.append(f"• {insight}")
        
        # Data highlights
        if not df.empty:
            if 'caller_intent' in df.columns:
                top_intents = df['caller_intent'].value_counts().head(3)
                summary_parts.append(f"Top caller intents: {', '.join(top_intents.index.tolist())}")
            
            if 'division' in df.columns:
                top_divisions = df['division'].value_counts().head(3)
                summary_parts.append(f"Top divisions: {', '.join(top_divisions.index.tolist())}")
        
        return "\n".join(summary_parts)

class CallDataChatApp:
    """Main Gradio chat application"""
    
    def __init__(self):
        # Initialize components
        self.setup_environment()
        self.db_manager = self.create_db_manager()
        self.lotus_processor = self.create_lotus_processor()
        self.chat_processor = CallDataChatProcessor(self.db_manager, self.lotus_processor)
        
    def setup_environment(self):
        """Setup environment variables and API keys"""
        # Set API keys (user should replace with actual keys)
        required_keys = {
            'GROQ_API_KEY': 'your-groq-api-key-here',
            'GOOGLE_API_KEY': 'your-google-api-key-here',
            'DB_HOST': 'localhost',
            'DB_PORT': '5432',
            'DB_NAME': 'your_database',
            'DB_USER': 'your_username',
            'DB_PASSWORD': 'your_password'
        }
        
        for key, default in required_keys.items():
            if key not in os.environ:
                os.environ[key] = default
                logger.warning(f"Using default value for {key}. Please set actual value.")
    
    def create_db_manager(self) -> DatabaseManager:
        """Create database manager instance"""
        return DatabaseManager(
            host=os.environ.get('DB_HOST', 'localhost'),
            port=int(os.environ.get('DB_PORT', '5432')),
            database=os.environ.get('DB_NAME', 'your_database'),
            username=os.environ.get('DB_USER', 'your_username'),
            password=os.environ.get('DB_PASSWORD', 'your_password')
        )
    
    def create_lotus_processor(self) -> LotusSemanticProcessor:
        """Create Lotus semantic processor"""
        return LotusSemanticProcessor()
    
    def process_chat_message(self, message: str, history: List) -> Tuple[List, str]:
        """Process chat message and return updated history"""
        if not message.strip():
            return history, ""
        
        try:
            # Process the query
            result = self.chat_processor.process_natural_language_query(message)
            
            if result["success"]:
                response = self.format_response(result)
            else:
                response = result["message"]
            
            # Update history
            history.append([message, response])
            
        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            error_response = f"I encountered an error processing your request: {str(e)}"
            history.append([message, error_response])
        
        return history, ""
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """Format the processing result for display"""
        response_parts = []
        
        # Main message
        response_parts.append(result["message"])
        
        # Data preview if available
        if not result["data"].empty:
            response_parts.append(f"\n📊 **Data Preview** ({len(result['data'])} rows):")
            
            # Show first few rows
            preview_df = result["data"].head(5)
            response_parts.append("```")
            response_parts.append(preview_df.to_string(index=False, max_cols=6))
            response_parts.append("```")
        
        # Additional insights
        if result.get("insights"):
            response_parts.append(f"\n🔍 **Technical Details:**")
            for insight in result["insights"]:
                response_parts.append(f"• {insight}")
        
        return "\n".join(response_parts)
    
    def create_gradio_interface(self):
        """Create Gradio interface for the chat application"""
        
        def clear_chat():
            return [], ""
        
        def get_sample_queries():
            return [
                "Show me calls from the billing department from last week",
                "Find the top 10 most important calls today",
                "What are calls similar to technical support issues?",
                "Group calls by division and show common themes",
                "Show me complaint-related calls from enterprise customers",
                "Find calls that indicate customer satisfaction issues",
                "What are the trending call intents this month?",
                "Show me calls handled by the best performing agents"
            ]
        
        # Create interface
        with gr.Blocks(title="Call Data AI Assistant with Lotus TAG", theme=gr.themes.Default()) as app:
            gr.Markdown(
                """
                # 📞 Call Data AI Assistant with Lotus.ai TAG
                
                This application uses **Table Augmented Generation (TAG)** with Lotus.ai semantic operators to analyze your call data intelligently.
                
                **Semantic Capabilities:**
                - 🔍 **Semantic Filtering**: Find records based on meaning, not just keywords  
                - 📊 **Semantic Ranking**: Rank calls by importance, urgency, or complexity
                - 🗺️ **Semantic Mapping**: Generate insights and categorizations
                - 📈 **Semantic Aggregation**: Summarize and analyze trends
                
                Ask questions in natural language about your call data!
                """
            )
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Main chat interface
                    chatbot = gr.Chatbot(
                        label="Chat with your Call Data using Lotus.ai",
                        height=500,
                        show_label=True,
                        avatar_images=["🧑‍💼", "🤖"]
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Ask about your call data",
                            placeholder="e.g., Show me important calls from last week...",
                            scale=5
                        )
                        submit_btn = gr.Button("Send", scale=1, variant="primary")
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat", scale=1)
                        
                with gr.Column(scale=1):
                    gr.Markdown("### 💡 Sample Queries")
                    
                    # Sample query buttons
                    sample_queries = get_sample_queries()
                    for i, query in enumerate(sample_queries):
                        btn = gr.Button(f"{i+1}. {query[:40]}...", size="sm")
                        btn.click(
                            lambda q=query: (self.process_chat_message(q, []), "")[0],
                            outputs=[chatbot]
                        )
            
            # Setup event handlers
            submit_btn.click(
                self.process_chat_message,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input]
            )
            
            msg_input.submit(
                self.process_chat_message,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input]
            )
            
            clear_btn.click(
                clear_chat,
                outputs=[chatbot, msg_input]
            )
            
            # Add information about setup
            with gr.Accordion("⚙️ Setup Information", open=False):
                gr.Markdown(
                    """
                    ### Required Setup:
                    
                    1. **API Keys**: Set environment variables:
                       ```bash
                       export GROQ_API_KEY="your-groq-api-key"
                       # OR
                       export GOOGLE_API_KEY="your-google-api-key"
                       ```
                    
                    2. **Database**: Configure PostgreSQL connection:
                       ```bash
                       export DB_HOST="your-db-host"
                       export DB_PORT="5432"
                       export DB_NAME="your-database"
                       export DB_USER="your-username"
                       export DB_PASSWORD="your-password"
                       ```
                    
                    3. **Install Dependencies**:
                       ```bash
                       pip install lotus-ai gradio pandas psycopg2-binary
                       ```
                    
                    ### Semantic Operations Available:
                    - **sem_filter**: Filter records based on semantic meaning
                    - **sem_topk**: Rank and get top-k results  
                    - **sem_map**: Generate insights and new columns
                    - **sem_agg**: Aggregate data semantically
                    """
                )
        
        return app
    
    def launch(self, **kwargs):
        """Launch the Gradio application"""
        app = self.create_gradio_interface()
        app.launch(**kwargs)

def main():
    """Main function to run the application"""
    try:
        # Create and launch the application
        chat_app = CallDataChatApp()
        
        # Launch with custom settings
        chat_app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True
        )
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        print(f"Error: {e}")
        print("\nPlease ensure:")
        print("1. All required environment variables are set")
        print("2. Database is accessible")
        print("3. API keys are valid")
        print("4. All dependencies are installed")

if __name__ == "__main__":
    main()


  ====================================


import pandas as pd
import lotus
from lotus.models import LM
from sentence_transformers import SentenceTransformer
from lotus.retrieval import SentenceTransformersRM
from lotus.reranking import CrossEncoderReranker
import psycopg2
from sqlalchemy import create_engine, text
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API keys (use either Groq or Google)
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
# Alternatively: os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

# Initialize Lotus-ai components
lm = LM(model='groq/llama-3.1-70b-versatile')
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
reranker = CrossEncoderReranker(model="mixedbread-ai/mxbai-rerank-large-v1")
lotus.settings.configure(lm=lm, rm=rm, reranker=reranker)

# Database connection
def get_db_connection():
    """Establish connection to PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "your_database"),
            user=os.getenv("DB_USER", "your_username"),
            password=os.getenv("DB_PASSWORD", "your_password"),
            port=os.getenv("DB_PORT", "5432")
        )
        return conn
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return None

def execute_sql_query(query):
    """Execute SQL query and return results as DataFrame"""
    conn = get_db_connection()
    if conn:
        try:
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            st.error(f"Query execution error: {str(e)}")
            conn.close()
            return pd.DataFrame()
    return pd.DataFrame()

def natural_language_to_sql(natural_language_query):
    """
    Convert natural language query to SQL using Lotus-ai
    """
    # Define table schema context for the LLM
    table_context = """
    Table: CX.member_call_info
    Columns: 
    - caller (text): Phone number of the caller
    - agent (text): Agent identifier
    - pty_id (text): Party ID of the call
    - calldate (timestamp): Date and time of the call
    - enterprise (text): Enterprise name
    - division (text): Division within enterprise
    - business_name (text): Specific business name
    - caller_intent (text): Classified intent of the call
    """
    
    # Create a prompt for SQL generation
    prompt = f"""
    Based on the following table schema:
    {table_context}
    
    Convert this natural language query to PostgreSQL SQL:
    "{natural_language_query}"
    
    Return only the SQL query without any additional explanation.
    Make sure to use proper SQL syntax and include the schema (CX) in the table name.
    """
    
    # Use Lotus to generate the SQL query
    sql_query = lm.generate(prompt)
    
    # Clean the SQL response (remove markdown code blocks if present)
    if "```sql" in sql_query:
        sql_query = sql_query.split("```sql")[1].split("```")[0]
    elif "```" in sql_query:
        sql_query = sql_query.split("```")[1].split("```")[0]
    
    return sql_query.strip()

def analyze_results_with_semantic_ops(original_query, result_df):
    """
    Apply Lotus-ai semantic operations to analyze query results
    """
    if result_df.empty:
        return "No results found for your query."
    
    # Apply different semantic operations based on query type
    if "trend" in original_query.lower() or "over time" in original_query.lower():
        # Use sem_map to generate insights about trends
        analysis_df = result_df.sem_map(
            "Based on this call data, what trends do you notice about {calldate} and {caller_intent}?"
        )
        analysis = "\n".join(analysis_df["_map"].tolist())
    
    elif "intent" in original_query.lower():
        # Use sem_filter to focus on specific intents
        analysis_df = result_df.sem_filter(
            "Identify calls where the caller intent indicates a complaint or urgent issue"
        )
        analysis = f"Filtered results: {len(analysis_df)} calls match your criteria.\n\n"
        analysis += analysis_df.to_string()
    
    elif "summary" in original_query.lower() or "overview" in original_query.lower():
        # Use sem_map to generate a summary
        analysis_df = result_df.sem_map(
            "Provide a concise summary of these call records: {caller} spoke to {agent} about {caller_intent}"
        )
        analysis = "\n".join(analysis_df["_map"].tolist())
    
    elif "most" in original_query.lower() or "least" in original_query.lower():
        # Use sem_topk to find top/bottom results
        sort_order = "least" if "least" in original_query.lower() else "most"
        analysis_df, stats = result_df.sem_topk(
            f"Which calls had the {sort_order} issues?",
            K=5,
            method="quick",
            return_stats=True
        )
        analysis = f"Top {len(analysis_df)} results:\n\n"
        analysis += analysis_df.to_string()
    
    else:
        # Default analysis - use sem_map for general insights
        analysis_df = result_df.sem_map(
            "Generate insights from this call data: {business_name} in {division} had a call about {caller_intent}"
        )
        analysis = "\n".join(analysis_df["_map"].tolist())
    
    return analysis

# Streamlit chat application
def main():
    st.set_page_config(page_title="Call Analytics AI", page_icon="📞", layout="wide")
    
    st.title("📞 Call Center Analytics Assistant")
    st.write("Ask natural language questions about your call data")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # React to user input
    if prompt := st.chat_input("What do you want to know about your call data?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question..."):
                try:
                    # Stage 1: Convert natural language to SQL and execute
                    sql_query = natural_language_to_sql(prompt)
                    st.code(f"Generated SQL:\n{sql_query}", language="sql")
                    
                    result_df = execute_sql_query(sql_query)
                    
                    # Stage 2: Apply semantic operations to analyze results
                    if not result_df.empty:
                        analysis = analyze_results_with_semantic_ops(prompt, result_df)
                        
                        st.write("### Results")
                        st.dataframe(result_df)
                        
                        st.write("### Analysis")
                        st.write(analysis)
                    else:
                        st.write("No results found for your query.")
                        
                except Exception as e:
                    st.error(f"Error processing your query: {str(e)}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"Query results and analysis for: {prompt}"
        })

if __name__ == "__main__":
    main()

  ============

              
