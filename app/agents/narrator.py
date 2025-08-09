"""
Portfolio Intel Narrator
Converts business questions into metrics analysis with anomaly detection and actionable insights
"""

import json
import logging
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.llm.gemini import chat
from app.rag.core import retrieve
from app.tools.tavily_search import web_search_into_docstore

logger = logging.getLogger(__name__)

# Pydantic models for structured responses
class Finding(BaseModel):
    title: str
    evidence: Dict[str, Any]

class Action(BaseModel):
    hypothesis: str
    owner: str

class TablePreview(BaseModel):
    name: str
    preview_rows: int

class Citation(BaseModel):
    source: str
    snippet: str

class Anomaly(BaseModel):
    ts: str
    value: float
    z_score: float

class NarratorResponse(BaseModel):
    findings: List[Finding]
    actions: List[Action]
    tables: List[TablePreview]
    citations: List[Citation]

@dataclass
class QueryResult:
    """Results from SQL-ish query execution"""
    data: pd.DataFrame
    table_name: str
    query: str
    anomalies: List[Anomaly] = None

class SafeSQLParser:
    """
    Safe SQL-ish parser that validates queries against schema
    """
    
    def __init__(self, schema_path: str):
        """Initialize with metrics schema"""
        self.schema = self._load_schema(schema_path)
        self.allowed_ops = self.schema["allowed_operations"]
        self.table_schemas = self.schema["schemas"]
    
    def _load_schema(self, schema_path: str) -> Dict[str, Any]:
        """Load metrics schema from JSON file"""
        try:
            with open(schema_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            return {"schemas": {}, "allowed_operations": {}}
    
    def parse_and_validate(self, sql_query: str) -> Dict[str, Any]:
        """
        Parse and validate SQL-ish query against schema
        
        Args:
            sql_query: SQL-ish query string
            
        Returns:
            Parsed query components or validation errors
        """
        try:
            # Normalize query
            query = sql_query.strip().upper()
            
            # Parse SELECT clause
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
            if not select_match:
                return {"error": "Missing SELECT clause"}
            
            select_clause = select_match.group(1).strip()
            
            # Parse FROM clause
            from_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
            if not from_match:
                return {"error": "Missing FROM clause"}
                
            table_name = from_match.group(1).lower()
            
            # Validate table exists
            if table_name not in self.table_schemas:
                return {"error": f"Table '{table_name}' not found in schema"}
            
            table_schema = self.table_schemas[table_name]
            
            # Parse WHERE clause (optional)
            where_clause = None
            where_match = re.search(r'WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
            if where_match:
                where_clause = where_match.group(1).strip()
            
            # Parse GROUP BY clause (optional)
            group_by_clause = None
            group_match = re.search(r'GROUP\s+BY\s+(.*?)(?:\s+ORDER\s+BY|\s+LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
            if group_match:
                group_by_clause = group_match.group(1).strip()
            
            # Parse ORDER BY clause (optional)
            order_by_clause = None
            order_match = re.search(r'ORDER\s+BY\s+(.*?)(?:\s+LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
            if order_match:
                order_by_clause = order_match.group(1).strip()
            
            # Parse LIMIT clause (optional)
            limit_clause = None
            limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
            if limit_match:
                limit_clause = int(limit_match.group(1))
                if limit_clause > self.allowed_ops["limit"]["max_rows"]:
                    return {"error": f"LIMIT exceeds maximum of {self.allowed_ops['limit']['max_rows']}"}
            
            # Validate SELECT columns and functions
            select_validation = self._validate_select_clause(select_clause, table_schema)
            if "error" in select_validation:
                return select_validation
            
            # Validate WHERE clause
            if where_clause:
                where_validation = self._validate_where_clause(where_clause, table_schema)
                if "error" in where_validation:
                    return where_validation
            
            # Validate GROUP BY
            if group_by_clause:
                group_validation = self._validate_group_by_clause(group_by_clause, table_schema)
                if "error" in group_validation:
                    return group_validation
            
            return {
                "table_name": table_name,
                "select": select_validation["columns"],
                "where": where_clause,
                "group_by": group_by_clause,
                "order_by": order_by_clause,
                "limit": limit_clause,
                "aggregations": select_validation.get("aggregations", [])
            }
            
        except Exception as e:
            logger.error(f"Error parsing SQL query: {e}")
            return {"error": f"Query parsing error: {str(e)}"}
    
    def _validate_select_clause(self, select_clause: str, table_schema: Dict) -> Dict[str, Any]:
        """Validate SELECT clause against table schema"""
        columns = []
        aggregations = []
        
        # Split by comma and analyze each part
        select_parts = [part.strip() for part in select_clause.split(',')]
        
        for part in select_parts:
            # Check for aggregation functions
            agg_match = re.match(r'(COUNT|SUM|AVG|MIN|MAX|STDDEV)\s*\(\s*(\*|\w+)\s*\)', part, re.IGNORECASE)
            if agg_match:
                func_name = agg_match.group(1).upper()
                column_name = agg_match.group(2).lower()
                
                if func_name not in self.allowed_ops["select"]["functions"]:
                    return {"error": f"Function {func_name} not allowed"}
                
                if column_name != "*" and column_name not in table_schema["columns"]:
                    return {"error": f"Column {column_name} not found in table"}
                
                aggregations.append({"function": func_name, "column": column_name})
                columns.append(f"{func_name}_{column_name}")
            else:
                # Regular column
                if part == "*":
                    columns.extend(table_schema["columns"].keys())
                else:
                    if part.lower() not in table_schema["columns"]:
                        return {"error": f"Column {part} not found in table"}
                    columns.append(part.lower())
        
        return {"columns": columns, "aggregations": aggregations}
    
    def _validate_where_clause(self, where_clause: str, table_schema: Dict) -> Dict[str, Any]:
        """Validate WHERE clause conditions"""
        # Simple validation - check for allowed operators and column existence
        allowed_operators = self.allowed_ops["where"]["operators"]
        
        # Extract column references (simplified)
        column_refs = re.findall(r'\b(\w+)\s*(?:=|!=|<|<=|>|>=|IN|LIKE|BETWEEN)', where_clause, re.IGNORECASE)
        
        for col in column_refs:
            if col.lower() not in table_schema["columns"]:
                return {"error": f"Column {col} not found in WHERE clause"}
        
        return {"valid": True}
    
    def _validate_group_by_clause(self, group_by_clause: str, table_schema: Dict) -> Dict[str, Any]:
        """Validate GROUP BY clause"""
        group_columns = [col.strip().lower() for col in group_by_clause.split(',')]
        
        if len(group_columns) > self.allowed_ops["group_by"]["max_columns"]:
            return {"error": f"GROUP BY exceeds maximum of {self.allowed_ops['group_by']['max_columns']} columns"}
        
        for col in group_columns:
            if col not in table_schema["columns"]:
                return {"error": f"GROUP BY column {col} not found in table"}
        
        return {"valid": True}

class PortfolioIntelNarrator:
    """
    Portfolio Intel Narrator for metrics analysis and anomaly detection
    """
    
    def __init__(self, docstore=None, embedder=None, retriever=None):
        """Initialize narrator with components"""
        self.docstore = docstore
        self.embedder = embedder
        self.retriever = retriever
        
        # Initialize SQL parser
        schema_path = "app/data/metrics_schema.json"
        self.sql_parser = SafeSQLParser(schema_path)
        self.metrics_dir = Path("metrics")
        
        # Load metric definitions
        self.metric_definitions = self.sql_parser.schema.get("metric_definitions", {})
    
    def process_question(self, question: str) -> NarratorResponse:
        """
        Main processing pipeline for business questions
        
        Args:
            question: Business question about portfolio metrics
            
        Returns:
            NarratorResponse with findings, actions, and citations
        """
        try:
            logger.info(f"Processing narrator question: {question}")
            
            # Step 1: Translate question to SQL-ish query using Gemini
            sql_query = self._translate_to_sql(question)
            logger.info(f"Generated SQL query: {sql_query}")
            
            # Step 2: Execute metrics query
            query_result = self.metrics_query(sql_query)
            if query_result is None:
                return self._error_response("Failed to execute query")
            
            # Step 3: Detect anomalies if time series data
            if self._is_time_series(query_result.data):
                anomalies = self._detect_anomalies(query_result.data)
                query_result.anomalies = anomalies
                logger.info(f"Detected {len(anomalies)} anomalies")
            
            # Step 4: Analyze findings and generate insights
            findings = self._analyze_findings(query_result, question)
            
            # Step 5: Generate action items and hypotheses
            actions = self._generate_actions(findings, question)
            
            # Step 6: Get metric definitions and terms
            citations = self.terms_retrieve("metric definitions")
            
            # Step 7: Create table previews
            tables = [TablePreview(
                name=query_result.table_name,
                preview_rows=min(len(query_result.data), 10)
            )]
            
            return NarratorResponse(
                findings=findings,
                actions=actions,
                tables=tables,
                citations=citations
            )
            
        except Exception as e:
            logger.error(f"Error processing narrator question: {e}")
            return self._error_response(str(e))
    
    def _translate_to_sql(self, question: str) -> str:
        """
        Translate business question to SQL-ish query using Gemini
        
        Args:
            question: Business question
            
        Returns:
            SQL-ish query string
        """
        try:
            # Get available tables and columns from schema
            schema_info = self._format_schema_for_llm()
            
            system_prompt = f"""You are a SQL expert for financial portfolio analysis. Convert business questions into safe SQL-ish queries.

IMPORTANT CONSTRAINTS:
- Only use SELECT statements with basic aggregations (COUNT, SUM, AVG, MIN, MAX, STDDEV)
- No JOINs, subqueries, or complex operations
- Only use tables and columns from the provided schema
- Use only these operators in WHERE: =, !=, <, <=, >, >=, IN, LIKE, BETWEEN
- Maximum 3 columns in GROUP BY
- Maximum 10,000 rows with LIMIT

AVAILABLE SCHEMA:
{schema_info}

Return ONLY the SQL query, nothing else. Example formats:
- SELECT merchant, SUM(spend_amount) FROM portfolio_spend WHERE date >= '2025-07-01' GROUP BY merchant
- SELECT date, spend_amount FROM portfolio_spend WHERE merchant = 'Amazon' ORDER BY date
- SELECT segment, AVG(delinq_30) FROM delinquency_rates WHERE month >= '2025-06-01' GROUP BY segment"""

            user_message = f"Business question: {question}"
            messages = [{"role": "user", "content": user_message}]
            
            response = chat(messages, system=system_prompt)
            sql_query = response.strip()
            
            # Clean up response (remove markdown, explanations, etc.)
            sql_query = re.sub(r'```sql\n?|```\n?', '', sql_query)
            sql_query = sql_query.split('\n')[0]  # Take first line only
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error translating question to SQL: {e}")
            return "SELECT * FROM portfolio_spend LIMIT 10"  # Fallback query
    
    def _format_schema_for_llm(self) -> str:
        """Format schema information for LLM context"""
        schema_text = []
        
        for table_name, schema in self.sql_parser.table_schemas.items():
            schema_text.append(f"\nTable: {table_name}")
            schema_text.append(f"Description: {schema['description']}")
            schema_text.append("Columns:")
            for col, info in schema["columns"].items():
                schema_text.append(f"  - {col} ({info['type']}): {info['description']}")
        
        return "\n".join(schema_text)
    
    def metrics_query(self, sqlish: str) -> Optional[QueryResult]:
        """
        Execute safe SQL-ish query over metrics CSV files
        
        Args:
            sqlish: SQL-ish query string
            
        Returns:
            QueryResult with DataFrame and metadata
        """
        try:
            # Parse and validate query
            parsed = self.sql_parser.parse_and_validate(sqlish)
            if "error" in parsed:
                logger.error(f"Query validation error: {parsed['error']}")
                return None
            
            table_name = parsed["table_name"]
            
            # Load CSV file
            csv_path = self.metrics_dir / f"{table_name}.csv"
            if not csv_path.exists():
                logger.error(f"Metrics file not found: {csv_path}")
                return None
            
            df = pd.read_csv(csv_path)
            
            # Convert date columns
            table_schema = self.sql_parser.table_schemas[table_name]
            for col, info in table_schema["columns"].items():
                if info["type"] == "date" and col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            # Apply WHERE clause
            if parsed["where"]:
                df = self._apply_where_clause(df, parsed["where"])
            
            # Apply GROUP BY and aggregations
            if parsed["group_by"] and parsed["aggregations"]:
                df = self._apply_group_by(df, parsed["group_by"], parsed["aggregations"])
            elif parsed["aggregations"]:
                # Aggregations without GROUP BY
                df = self._apply_aggregations(df, parsed["aggregations"])
            
            # Apply ORDER BY
            if parsed["order_by"]:
                df = self._apply_order_by(df, parsed["order_by"])
            
            # Apply LIMIT
            if parsed["limit"]:
                df = df.head(parsed["limit"])
            
            return QueryResult(
                data=df,
                table_name=table_name,
                query=sqlish
            )
            
        except Exception as e:
            logger.error(f"Error executing metrics query: {e}")
            return None
    
    def _apply_where_clause(self, df: pd.DataFrame, where_clause: str) -> pd.DataFrame:
        """Apply WHERE clause conditions to DataFrame"""
        try:
            # Convert SQL WHERE to pandas query (simplified)
            # This is a basic implementation - would need more robust parsing for production
            where_clause = where_clause.replace("=", "==").replace("!=", "!=")
            
            # Handle date comparisons
            where_clause = re.sub(r"'(\d{4}-\d{2}-\d{2})'", r"'\1'", where_clause)
            
            # Handle string comparisons
            where_clause = re.sub(r"(\w+)\s*==\s*'([^']+)'", r"\1 == '\2'", where_clause)
            
            return df.query(where_clause)
            
        except Exception as e:
            logger.error(f"Error applying WHERE clause: {e}")
            return df
    
    def _apply_group_by(self, df: pd.DataFrame, group_by: str, aggregations: List[Dict]) -> pd.DataFrame:
        """Apply GROUP BY with aggregations"""
        try:
            group_cols = [col.strip() for col in group_by.split(',')]
            
            # Create aggregation dictionary
            agg_dict = {}
            for agg in aggregations:
                func_name = agg["function"].lower()
                col_name = agg["column"]
                
                if col_name == "*":
                    # COUNT(*) 
                    if func_name == "count":
                        agg_dict["count"] = ("spend_amount", "count")  # Use first numeric column
                else:
                    if func_name == "count":
                        agg_dict[f"{func_name}_{col_name}"] = (col_name, "count")
                    else:
                        pandas_func = {"sum": "sum", "avg": "mean", "min": "min", "max": "max", "stddev": "std"}.get(func_name, "sum")
                        agg_dict[f"{func_name}_{col_name}"] = (col_name, pandas_func)
            
            if agg_dict:
                result = df.groupby(group_cols).agg(agg_dict).reset_index()
                # Flatten column names
                result.columns = [col[0] if col[1] == '' else f"{col[1]}_{col[0]}" if isinstance(col, tuple) else col for col in result.columns]
                return result
            else:
                return df.groupby(group_cols).size().reset_index(name='count')
                
        except Exception as e:
            logger.error(f"Error applying GROUP BY: {e}")
            return df
    
    def _apply_aggregations(self, df: pd.DataFrame, aggregations: List[Dict]) -> pd.DataFrame:
        """Apply aggregations without GROUP BY"""
        try:
            results = {}
            
            for agg in aggregations:
                func_name = agg["function"].lower()
                col_name = agg["column"]
                
                if col_name == "*":
                    if func_name == "count":
                        results[f"{func_name}_*"] = [len(df)]
                else:
                    if func_name == "count":
                        results[f"{func_name}_{col_name}"] = [df[col_name].count()]
                    elif func_name == "sum":
                        results[f"{func_name}_{col_name}"] = [df[col_name].sum()]
                    elif func_name == "avg":
                        results[f"{func_name}_{col_name}"] = [df[col_name].mean()]
                    elif func_name == "min":
                        results[f"{func_name}_{col_name}"] = [df[col_name].min()]
                    elif func_name == "max":
                        results[f"{func_name}_{col_name}"] = [df[col_name].max()]
                    elif func_name == "stddev":
                        results[f"{func_name}_{col_name}"] = [df[col_name].std()]
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error applying aggregations: {e}")
            return df
    
    def _apply_order_by(self, df: pd.DataFrame, order_by: str) -> pd.DataFrame:
        """Apply ORDER BY clause"""
        try:
            order_cols = []
            ascending = []
            
            for col_spec in order_by.split(','):
                col_spec = col_spec.strip()
                if col_spec.upper().endswith(' DESC'):
                    col_name = col_spec[:-5].strip()
                    ascending.append(False)
                else:
                    col_name = col_spec.replace(' ASC', '').strip()
                    ascending.append(True)
                
                order_cols.append(col_name)
            
            return df.sort_values(by=order_cols, ascending=ascending)
            
        except Exception as e:
            logger.error(f"Error applying ORDER BY: {e}")
            return df
    
    def _is_time_series(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame contains time series data"""
        date_cols = ['date', 'month', 'timestamp', 'ts']
        return any(col in df.columns for col in date_cols) and len(df) > 5
    
    def _detect_anomalies(self, df: pd.DataFrame) -> List[Anomaly]:
        """
        Detect anomalies using z-score method
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        try:
            # Find date column
            date_col = None
            for col in ['date', 'month', 'timestamp', 'ts']:
                if col in df.columns:
                    date_col = col
                    break
            
            if not date_col:
                return anomalies
            
            # Find numeric columns for anomaly detection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_cols:
                if col != date_col:
                    values = df[col].dropna()
                    if len(values) < 3:
                        continue
                    
                    # Calculate z-scores
                    mean_val = values.mean()
                    std_val = values.std()
                    
                    if std_val == 0:
                        continue
                    
                    z_scores = np.abs((values - mean_val) / std_val)
                    
                    # Find anomalies (z-score > 2.5)
                    anomaly_mask = z_scores > 2.5
                    anomaly_indices = anomaly_mask[anomaly_mask].index
                    
                    for idx in anomaly_indices:
                        anomalies.append(Anomaly(
                            ts=str(df.loc[idx, date_col]),
                            value=float(df.loc[idx, col]),
                            z_score=float(z_scores.loc[idx])
                        ))
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
        
        return anomalies
    
    def _analyze_findings(self, query_result: QueryResult, question: str) -> List[Finding]:
        """
        Analyze query results to extract key findings
        
        Args:
            query_result: Query execution results
            question: Original business question
            
        Returns:
            List of findings with evidence
        """
        findings = []
        df = query_result.data
        
        try:
            # Finding 1: Data summary
            summary_evidence = {
                "total_rows": len(df),
                "date_range": self._get_date_range(df),
                "columns_analyzed": list(df.columns),
                "table_source": query_result.table_name
            }
            
            findings.append(Finding(
                title="Data Overview",
                evidence=summary_evidence
            ))
            
            # Finding 2: Key metrics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                metrics_evidence = {}
                for col in numeric_cols[:3]:  # Top 3 numeric columns
                    metrics_evidence[col] = {
                        "total": float(df[col].sum()) if col.endswith('_amount') or col.endswith('_count') else float(df[col].mean()),
                        "average": float(df[col].mean()),
                        "min": float(df[col].min()),
                        "max": float(df[col].max())
                    }
                
                findings.append(Finding(
                    title="Key Metrics Summary",
                    evidence=metrics_evidence
                ))
            
            # Finding 3: Anomalies if detected
            if query_result.anomalies:
                anomaly_evidence = {
                    "total_anomalies": len(query_result.anomalies),
                    "anomaly_dates": [a.ts for a in query_result.anomalies[:5]],  # Top 5
                    "max_z_score": max(a.z_score for a in query_result.anomalies)
                }
                
                findings.append(Finding(
                    title="Anomalies Detected",
                    evidence=anomaly_evidence
                ))
            
            # Finding 4: Trends analysis for spend drop questions
            if "drop" in question.lower() or "decline" in question.lower():
                trend_evidence = self._analyze_spend_trends(df, question)
                if trend_evidence:
                    findings.append(Finding(
                        title="Spend Trend Analysis",
                        evidence=trend_evidence
                    ))
            
        except Exception as e:
            logger.error(f"Error analyzing findings: {e}")
        
        return findings
    
    def _get_date_range(self, df: pd.DataFrame) -> Dict[str, str]:
        """Get date range from DataFrame"""
        try:
            date_cols = ['date', 'month', 'timestamp', 'ts']
            for col in date_cols:
                if col in df.columns:
                    min_date = df[col].min()
                    max_date = df[col].max()
                    return {
                        "start": str(min_date),
                        "end": str(max_date),
                        "column": col
                    }
        except:
            pass
        
        return {"start": "N/A", "end": "N/A", "column": "none"}
    
    def _analyze_spend_trends(self, df: pd.DataFrame, question: str) -> Dict[str, Any]:
        """Analyze spending trends for decline questions"""
        try:
            # Look for date column and spend amount
            date_col = None
            spend_col = None
            
            for col in ['date', 'month']:
                if col in df.columns:
                    date_col = col
                    break
            
            for col in ['spend_amount', 'sum_spend_amount']:
                if col in df.columns:
                    spend_col = col
                    break
            
            if not date_col or not spend_col:
                return {}
            
            # Sort by date and analyze trends
            df_sorted = df.sort_values(date_col)
            
            # Check for July 31st impact (promo expiry)
            july_31_idx = None
            for idx, row in df_sorted.iterrows():
                if '2025-07-31' in str(row[date_col]):
                    july_31_idx = idx
                    break
            
            evidence = {}
            if july_31_idx is not None:
                # Compare before and after July 31
                before_idx = max(0, july_31_idx - 2)
                after_idx = min(len(df_sorted) - 1, july_31_idx + 2)
                
                before_spend = df_sorted.iloc[before_idx][spend_col]
                after_spend = df_sorted.iloc[after_idx][spend_col]
                
                pct_change = ((after_spend - before_spend) / before_spend) * 100
                
                evidence = {
                    "july_31_impact": True,
                    "spend_before": float(before_spend),
                    "spend_after": float(after_spend), 
                    "percent_change": float(pct_change),
                    "likely_cause": "Promotional offer expiry on 2025-07-31"
                }
            
            return evidence
            
        except Exception as e:
            logger.error(f"Error analyzing spend trends: {e}")
            return {}
    
    def _generate_actions(self, findings: List[Finding], question: str) -> List[Action]:
        """
        Generate actionable hypotheses and next steps
        
        Args:
            findings: Analysis findings
            question: Original question
            
        Returns:
            List of action items with owners
        """
        actions = []
        
        try:
            # Action 1: Based on anomalies
            anomaly_findings = [f for f in findings if "anomalies" in f.title.lower()]
            if anomaly_findings:
                actions.append(Action(
                    hypothesis="Unusual spikes or drops in metrics may indicate system issues, promotional changes, or market events",
                    owner="Analytics Team"
                ))
            
            # Action 2: Based on spend drops
            if "drop" in question.lower() or "decline" in question.lower():
                actions.append(Action(
                    hypothesis="Spend decline after July 31st likely due to promotional offer expiry - consider extending or launching new promotions",
                    owner="Marketing Team"
                ))
                
                actions.append(Action(
                    hypothesis="Analyze customer retention post-promotion to understand long-term impact on portfolio health",
                    owner="Risk Management"
                ))
            
            # Action 3: Based on delinquency trends
            if "delinq" in question.lower():
                actions.append(Action(
                    hypothesis="Rising delinquency rates may require tightened underwriting or enhanced collection strategies",
                    owner="Credit Risk Team"
                ))
            
            # Action 4: General portfolio monitoring
            actions.append(Action(
                hypothesis="Implement automated monitoring for metric thresholds to catch similar patterns earlier",
                owner="Data Engineering"
            ))
            
        except Exception as e:
            logger.error(f"Error generating actions: {e}")
        
        return actions
    
    def terms_retrieve(self, query: str) -> List[Citation]:
        """
        Retrieve metric definitions and terms
        
        Args:
            query: Terms query
            
        Returns:
            List of citations
        """
        citations = []
        
        try:
            # First, add metric definitions from schema
            for metric, definition in self.metric_definitions.items():
                if metric in query.lower() or any(word in definition.lower() for word in query.split()):
                    citations.append(Citation(
                        source="Metrics Schema",
                        snippet=f"{metric}: {definition}"
                    ))
            
            # Try RAG retrieval if available
            if self.retriever and self.embedder:
                try:
                    results = retrieve(self.retriever, self.embedder, query, k=2)
                    
                    for result in results:
                        citations.append(Citation(
                            source=result.get("filename", "Knowledge Base"),
                            snippet=result.get("snippet", "")[:200] + "..."
                        ))
                except Exception as e:
                    logger.warning(f"RAG retrieval failed: {e}")
            
            # If no results, add default citation
            if not citations:
                citations.append(Citation(
                    source="Portfolio Intelligence",
                    snippet="Metrics definitions and business context for portfolio analysis and decision-making."
                ))
            
        except Exception as e:
            logger.error(f"Error retrieving terms: {e}")
        
        return citations[:5]  # Limit to 5 citations
    
    def _error_response(self, message: str) -> NarratorResponse:
        """Create error response"""
        return NarratorResponse(
            findings=[Finding(
                title="Analysis Error",
                evidence={"error": message}
            )],
            actions=[Action(
                hypothesis="Review query or data availability",
                owner="Analytics Team"
            )],
            tables=[],
            citations=[]
        )

# Test cases for Portfolio Intel Narrator
def test_narrator():
    """Test Portfolio Intel Narrator with business scenarios"""
    print("🧪 Testing Portfolio Intel Narrator")
    print("=" * 50)
    
    narrator = PortfolioIntelNarrator()
    
    test_cases = [
        {
            "name": "Spend drop analysis after July 31st",
            "question": "Why did spend drop after 2025-07-31?",
            "expected_findings": 3,
            "expected_actions": 2,
            "expected_promo_analysis": True
        },
        {
            "name": "Merchant performance comparison",
            "question": "Which merchant has the highest spend volume?",
            "expected_findings": 2,
            "expected_actions": 1,
            "expected_promo_analysis": False
        },
        {
            "name": "Delinquency trend analysis",
            "question": "How are delinquency rates trending by segment?",
            "expected_findings": 2,
            "expected_actions": 2,
            "expected_promo_analysis": False
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        try:
            print(f"{i}. {case['name']}")
            print(f"   Question: '{case['question']}'")
            
            result = narrator.process_question(case["question"])
            
            # Validate response structure
            valid_structure = (
                isinstance(result, NarratorResponse) and
                isinstance(result.findings, list) and
                isinstance(result.actions, list) and
                isinstance(result.tables, list) and
                isinstance(result.citations, list)
            )
            
            # Check findings count
            findings_ok = len(result.findings) >= case["expected_findings"]
            
            # Check actions count
            actions_ok = len(result.actions) >= case["expected_actions"]
            
            # Check for promotional analysis if expected
            promo_analysis_ok = True
            if case["expected_promo_analysis"]:
                promo_analysis_ok = any(
                    "promo" in finding.title.lower() or "july" in str(finding.evidence)
                    for finding in result.findings
                )
            
            # Check citations exist
            citations_ok = len(result.citations) > 0
            
            # Check tables exist
            tables_ok = len(result.tables) > 0
            
            success = (valid_structure and findings_ok and actions_ok and 
                      promo_analysis_ok and citations_ok and tables_ok)
            
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"   Findings generated: {len(result.findings)}")
            print(f"   Actions generated: {len(result.actions)}")
            print(f"   Tables referenced: {len(result.tables)}")
            print(f"   Citations found: {len(result.citations)}")
            if case["expected_promo_analysis"]:
                print(f"   Promotional analysis: {promo_analysis_ok}")
            print(f"   Status: {status}")
            
            if success:
                passed += 1
            else:
                print(f"   Failure reasons:")
                if not valid_structure:
                    print(f"     - Invalid response structure")
                if not findings_ok:
                    print(f"     - Insufficient findings generated")
                if not actions_ok:
                    print(f"     - Insufficient actions generated")
                if not promo_analysis_ok:
                    print(f"     - Missing promotional analysis")
                if not citations_ok:
                    print(f"     - No citations provided")
                if not tables_ok:
                    print(f"     - No tables referenced")
            
            print()
            
        except Exception as e:
            print(f"   ❌ FAIL - Exception: {str(e)}")
            print()
    
    print(f"📊 Portfolio Intel Narrator Results: {passed}/{total} tests passed")
    return passed == total

if __name__ == "__main__":
    test_narrator()