# Call Data AI Chat Application with Lotus.ai TAG - Complete Setup Guide

## 🚀 Installation & Environment Setup

### 1. Install Dependencies

```bash
# Core packages
pip install lotus-ai gradio pandas psycopg2-binary python-dotenv

# For different LLM providers
pip install litellm groq openai google-generativeai

# Optional: GPU acceleration for embeddings
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Environment Configuration

Create a `.env` file:

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password

# API Keys (you need at least one)
GROQ_API_KEY=gsk_your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=sk_your_openai_key_here

# Optional settings
LOTUS_CACHE_DIR=./lotus_cache
LOTUS_LOG_LEVEL=INFO
```

### 3. Get API Keys

#### Groq API (Recommended - Fast & Free Tier)
1. Go to https://console.groq.com/
2. Sign up and create an API key
3. Free tier includes generous limits

#### Google Gemini API
1. Go to https://ai.google.dev/
2. Get API key from Google AI Studio
3. Free tier available

## 🧪 Testing Lotus.ai with Sample Data

### Step 1: Basic Lotus Setup Test

Create `test_lotus_basic.py`:

```python
import pandas as pd
import lotus
from lotus.models import LM, SentenceTransformersRM, CrossEncoderReranker
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Lotus
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY', 'your-key-here')

try:
    # Initialize models
    lm = LM(model='groq/llama-3.1-70b-versatile')
    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
    reranker = CrossEncoderReranker(model="mixedbread-ai/mxbai-rerank-large-v1")
    
    # Configure Lotus
    lotus.settings.configure(lm=lm, rm=rm, reranker=reranker)
    
    print("✅ Lotus.ai configured successfully!")
    
except Exception as e:
    print(f"❌ Error configuring Lotus: {e}")
    exit(1)

# Create sample call data
sample_data = {
    "caller": ["+1-555-0101", "+1-555-0102", "+1-555-0103", "+1-555-0104", "+1-555-0105"],
    "agent": ["John Smith", "Jane Doe", "Bob Wilson", "Alice Brown", "Charlie Davis"],
    "pty_id": ["PTY001", "PTY002", "PTY003", "PTY004", "PTY005"],
    "calldate": ["2024-01-15", "2024-01-15", "2024-01-16", "2024-01-16", "2024-01-17"],
    "enterprise": ["Acme Corp", "TechStart Inc", "Global Systems", "Acme Corp", "TechStart Inc"],
    "division": ["Billing", "Technical Support", "Sales", "Customer Service", "Technical Support"],
    "business_name": ["Acme Corporation", "TechStart Innovations", "Global Systems Ltd", "Acme Corporation", "TechStart Innovations"],
    "caller_intent": [
        "billing inquiry about overcharge",
        "technical issue with software installation",
        "interested in purchasing new product",
        "complaint about service quality",
        "need help with system configuration"
    ]
}

df = pd.DataFrame(sample_data)
print("📊 Sample data created:")
print(df.head())
print("\n" + "="*50)

# Test 1: Semantic Filter
print("🔍 Test 1: Semantic Filter")
print("Query: Find calls related to technical problems")

try:
    filtered_df = df.sem_filter("caller_intent is related to technical problems or issues")
    print(f"Original rows: {len(df)}, Filtered rows: {len(filtered_df)}")
    print("Filtered results:")
    print(filtered_df[['caller', 'agent', 'division', 'caller_intent']])
    print("✅ Semantic filter working!")
except Exception as e:
    print(f"❌ Semantic filter error: {e}")

print("\n" + "="*50)

# Test 2: Semantic Map
print("🗺️ Test 2: Semantic Map")
print("Query: Categorize the sentiment of each call")

try:
    mapped_df = df.sem_map("What is the sentiment (positive/negative/neutral) of this {caller_intent}?")
    print("Mapped results with sentiment:")
    print(mapped_df[['caller_intent', 'sem_map']].head())
    print("✅ Semantic mapping working!")
except Exception as e:
    print(f"❌ Semantic mapping error: {e}")

print("\n" + "="*50)

# Test 3: Semantic Top-K
print("📈 Test 3: Semantic Top-K")
print("Query: Rank calls by urgency/importance")

try:
    topk_df, stats = df.sem_topk("Which {caller_intent} represents the most urgent or important issue?", K=3, return_stats=True)
    print(f"Top 3 most important calls:")
    print(topk_df[['caller', 'division', 'caller_intent']])
    print(f"Stats: {stats}")
    print("✅ Semantic top-k working!")
except Exception as e:
    print(f"❌ Semantic top-k error: {e}")

print("\n" + "="*50)

# Test 4: Semantic Aggregate
print("📊 Test 4: Semantic Aggregation")
print("Query: Summarize common themes by division")

try:
    # Group by division and summarize
    divisions = df['division'].unique()
    summaries = []
    
    for division in divisions:
        div_data = df[df['division'] == division]
        if len(div_data) > 0:
            summary = div_data.sem_agg("What are the common themes in these {caller_intent}?")
            summaries.append({
                'division': division,
                'summary': summary,
                'call_count': len(div_data)
            })
    
    summary_df = pd.DataFrame(summaries)
    print("Division summaries:")
    for _, row in summary_df.iterrows():
        print(f"- {row['division']} ({row['call_count']} calls): {row['summary']}")
    print("✅ Semantic aggregation working!")
except Exception as e:
    print(f"❌ Semantic aggregation error: {e}")

print("\n🎉 All Lotus.ai tests completed!")
```

### Step 2: Database Integration Test

Create `test_database_integration.py`:

```python
import pandas as pd
import psycopg2
import lotus
from lotus.models import LM, SentenceTransformersRM
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Lotus
lm = LM(model='groq/llama-3.1-70b-versatile')
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
lotus.settings.configure(lm=lm, rm=rm)

# Test database connection
def test_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        
        # Test query
        test_query = "SELECT COUNT(*) as total_records FROM cx.member_call_info LIMIT 1"
        df = pd.read_sql(test_query, conn)
        print(f"✅ Database connected! Total records: {df['total_records'].iloc[0]}")
        
        # Get sample data
        sample_query = """
        SELECT caller, agent, pty_id, calldate, enterprise, division, business_name, caller_intent 
        FROM cx.member_call_info 
        LIMIT 10
        """
        
        sample_df = pd.read_sql(sample_query, conn)
        print("📊 Sample data from database:")
        print(sample_df.head())
        
        conn.close()
        return sample_df
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("Using mock data instead...")
        
        # Return mock data if DB connection fails
        mock_data = {
            "caller": ["+1-555-0101", "+1-555-0102", "+1-555-0103"],
            "agent": ["John Smith", "Jane Doe", "Bob Wilson"],
            "pty_id": ["PTY001", "PTY002", "PTY003"],
            "calldate": ["2024-01-15", "2024-01-15", "2024-01-16"],
            "enterprise": ["Acme Corp", "TechStart Inc", "Global Systems"],
            "division": ["Billing", "Technical Support", "Sales"],
            "business_name": ["Acme Corporation", "TechStart Innovations", "Global Systems Ltd"],
            "caller_intent": [
                "billing inquiry about monthly charges",
                "software installation problem",
                "interested in enterprise package"
            ]
        }
        return pd.DataFrame(mock_data)

# Test with real/mock data
df = test_db_connection()

# Apply semantic operations
print("\n🧪 Testing semantic operations on real data:")

# Semantic search for billing-related calls
billing_calls = df.sem_filter("caller_intent is related to billing, payments, or charges")
print(f"Found {len(billing_calls)} billing-related calls")

# Categorize call complexity
complexity_df = df.sem_map("Rate the complexity of this {caller_intent} as Simple/Medium/Complex")
print("\nCall complexity analysis:")
print(complexity_df[['caller_intent', 'sem_map']])

print("\n✅ Database integration test completed!")
```

### Step 3: Full Application Test

Create `test_full_app.py`:

```python
import gradio as gr
import pandas as pd
import lotus
from lotus.models import LM, SentenceTransformersRM
import os
from dotenv import load_dotenv

load_dotenv()

# Simple test application
class TestChatApp:
    def __init__(self):
        # Configure Lotus
        lm = LM(model='groq/llama-3.1-70b-versatile')
        rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
        lotus.settings.configure(lm=lm, rm=rm)
        
        # Sample data
        self.sample_data = pd.DataFrame({
            "caller": ["+1-555-0101", "+1-555-0102", "+1-555-0103", "+1-555-0104"],
            "agent": ["John Smith", "Jane Doe", "Bob Wilson", "Alice Brown"],
            "division": ["Billing", "Technical Support", "Sales", "Customer Service"],
            "caller_intent": [
                "billing dispute about overcharges",
                "cannot install software properly",
                "wants to upgrade service plan",
                "complaining about slow response time"
            ]
        })
    
    def process_query(self, query, history):
        try:
            response_parts = [f"🔍 Processing: {query}"]
            
            # Apply semantic operations based on query
            if "billing" in query.lower():
                result = self.sample_data.sem_filter("caller_intent is related to billing or payments")
                response_parts.append(f"Found {len(result)} billing-related calls")
                
            elif "urgent" in query.lower() or "important" in query.lower():
                result, _ = self.sample_data.sem_topk("Which {caller_intent} is most urgent?", K=2, return_stats=True)
                response_parts.append("Most urgent calls:")
                
            elif "sentiment" in query.lower():
                result = self.sample_data.sem_map("What is the sentiment of {caller_intent}?")
                response_parts.append("Sentiment analysis completed")
                
            else:
                result = self.sample_data.head(3)
                response_parts.append("Showing sample data")
            
            # Format result
            if not result.empty:
                response_parts.append("\n📊 Results:")
                response_parts.append(result.to_string(index=False))
            
            response = "\n".join(response_parts)
            history.append([query, response])
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            history.append([query, error_msg])
        
        return history, ""

# Create and launch test app
app = TestChatApp()

def create_interface():
    with gr.Blocks(title="Lotus.ai Test App") as interface:
        gr.Markdown("# 🧪 Lotus.ai Semantic Operations Test")
        
        chatbot = gr.Chatbot(label="Test Chat", height=400)
        
        msg = gr.Textbox(label="Test Query", 
                        placeholder="Try: 'show billing calls', 'find urgent calls', 'analyze sentiment'")
        
        submit = gr.Button("Test", variant="primary")
        clear = gr.Button("Clear")
        
        submit.click(app.process_query, [msg, chatbot], [chatbot, msg])
        msg.submit(app.process_query, [msg, chatbot], [chatbot, msg])
        clear.click(lambda: ([], ""), outputs=[chatbot, msg])
        
        gr.Markdown("""
        ### Test Commands:
        - "show billing calls" - Test semantic filter
        - "find urgent calls" - Test semantic ranking  
        - "analyze sentiment" - Test semantic mapping
        - "show sample data" - View raw data
        """)
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(debug=True, share=False)
```

## 🔧 Troubleshooting Guide

### Common Issues & Solutions

#### 1. Import Errors
```bash
# If lotus-ai import fails
pip install --upgrade lotus-ai

# If specific model imports fail
pip install sentence-transformers torch
```

#### 2. API Key Issues
```python
# Test API key
import litellm
response = litellm.completion(
    model="groq/llama-3.1-70b-versatile",
    messages=[{"role": "user", "content": "Hello"}]
)
print("API key working!" if response else "API key issue")
```

#### 3. Memory Issues
```python
# Use lighter models for testing
rm = SentenceTransformersRM(model="all-MiniLM-L6-v2")  # Smaller model
```

#### 4. Database Connection Issues
```bash
# Test PostgreSQL connection
psql -h localhost -p 5432 -U your_user -d your_db -c "SELECT 1;"
```

## 🚀 Running the Tests

Execute tests in order:

```bash
# 1. Basic Lotus functionality
python test_lotus_basic.py

# 2. Database integration  
python test_database_integration.py

# 3. Full application test
python test_full_app.py

# 4. Run main application
python call_data_chat_app.py
```

## 📊 Expected Output

### Successful Test Results:
```
✅ Lotus.ai configured successfully!
📊 Sample data created: 5 rows
🔍 Test 1: Semantic Filter - ✅ Working!
🗺️ Test 2: Semantic Map - ✅ Working!  
📈 Test 3: Semantic Top-K - ✅ Working!
📊 Test 4: Semantic Aggregation - ✅ Working!
🎉 All Lotus.ai tests completed!
```

## 🎯 Performance Tips

1. **Model Selection**: Start with smaller models for testing
2. **Caching**: Enable embedding caching for better performance
3. **Batch Processing**: Process multiple records together when possible
4. **Memory Management**: Monitor RAM usage with large datasets

## 📚 Next Steps

After successful testing:
1. Configure your actual database connection
2. Customize semantic operations for your use cases  
3. Add more sophisticated query parsing
4. Implement user authentication and access controls
5. Deploy using Docker or cloud services

## 🆘 Support

If you encounter issues:
1. Check the [Lotus.ai GitHub](https://github.com/stanford-futuredata/lotus) for latest updates
2. Verify all API keys are correctly set
3. Test with minimal sample data first
4. Check Python and package versions
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

              
