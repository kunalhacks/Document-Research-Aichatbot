import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re
import json
import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.models.document import Document, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class Theme:
    """Represents a theme extracted from documents."""
    name: str
    description: str
    keywords: List[str]
    confidence: float
    supporting_docs: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "keywords": self.keywords,
            "confidence": self.confidence,
            "supporting_docs": self.supporting_docs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Theme':
        return cls(
            name=data["name"],
            description=data["description"],
            keywords=data["keywords"],
            confidence=data["confidence"],
            supporting_docs=data["supporting_docs"]
        )


class ThemeExtractor:
    """Extracts themes from documents using LLM and clustering techniques."""
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        max_themes: int = 5,
        min_theme_confidence: float = 0.7
    ):
        """Initialize the theme extractor.
        
        Args:
            openai_api_key: OpenAI API key. If None, will use OPENAI_API_KEY env var.
            model: Name of the OpenAI model to use.
            max_themes: Maximum number of themes to extract.
            min_theme_confidence: Minimum confidence threshold for including a theme.
        """
        self.model = model
        self.max_themes = max_themes
        self.min_theme_confidence = min_theme_confidence
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def extract_themes(
        self,
        search_results: List[SearchResult],
        query: Optional[str] = None
    ) -> List[Theme]:
        """Extract themes from search results.
        
        Args:
            search_results: List of search results to analyze
            query: Optional query to guide theme extraction
            
        Returns:
            List of extracted themes
        """
        try:
            if not search_results:
                return []
                
            # Group results by document
            doc_groups = {}
            for result in search_results:
                doc_id = result.chunk.doc_id
                if doc_id not in doc_groups:
                    doc_groups[doc_id] = {
                        'chunks': [],
                        'metadata': {
                            'doc_id': doc_id,
                            'title': result.document.title,
                            'source': result.chunk.metadata.get('source', ''),
                            'doc_type': result.document.doc_type,
                            'upload_date': result.document.upload_date.isoformat() if result.document.upload_date else ''
                        }
                    }
                doc_groups[doc_id]['chunks'].append({
                    'content': result.chunk.content,
                    'page_number': result.chunk.page_number + 1,  # 1-based for display
                    'score': result.score
                })
            
            # Prepare documents for analysis
            documents = []
            for doc_id, data in doc_groups.items():
                # Combine chunks with highest scores first
                sorted_chunks = sorted(data['chunks'], key=lambda x: x['score'], reverse=True)
                full_text = '\n\n'.join([c['content'] for c in sorted_chunks])
                
                documents.append({
                    'id': doc_id,
                    'text': full_text[:10000],  # Limit text length
                    'metadata': data['metadata']
                })
            
            # Extract themes using LLM
            themes = self._extract_themes_with_llm(documents, query)
            
            # Filter by confidence and limit number of themes
            filtered_themes = [
                theme for theme in themes 
                if theme.confidence >= self.min_theme_confidence
            ][:self.max_themes]

            # Additional filtering: keep only data technology related themes, remove general AI/ML
            data_keywords = [
                "data", "database", "sql", "nosql", "analytics", "pipeline", "storage",
                "etl", "warehouse", "stream", "kafka", "spark", "hadoop", "postgres",
                "mysql", "mongodb", "redis", "elasticsearch", "cassandra", "airflow"
            ]
            ai_terms = [
                "machine learning", "deep learning", "artificial intelligence", "ai", "nlp",
                "model", "training", "inference", "computer vision"
            ]

            final_themes = []
            for theme in filtered_themes:
                name_lower = theme.name.lower()
                # Exclude if it is clearly an AI/ML topic
                if any(term in name_lower for term in ai_terms):
                    continue
                # Keep only if at least one data keyword is present
                if any(kw in name_lower for kw in data_keywords):
                    final_themes.append(theme)

            return final_themes
            
        except Exception as e:
            logger.error(f"Error extracting themes: {e}", exc_info=True)
            return []
    
    def _extract_themes_with_llm(
        self, 
        documents: List[Dict[str, Any]],
        query: Optional[str] = None
    ) -> List[Theme]:
        """Extract themes from documents using LLM."""
        try:
            # Prepare the prompt
            prompt = self._build_theme_extraction_prompt(documents, query)
            
            # Call the LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes documents and identifies key themes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Parse the response
            content = response.choices[0].message.content
            
            # Try to extract JSON from the response
            try:
                # Look for JSON in the response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    themes_data = json.loads(json_str)
                    
                    # Handle both single theme and list of themes
                    if isinstance(themes_data, dict):
                        themes_data = [themes_data]
                    elif not isinstance(themes_data, list):
                        raise ValueError("Expected a list or object in the response")
                    
                    # Convert to Theme objects
                    themes = []
                    for theme_data in themes_data:
                        try:
                            theme = Theme(
                                name=theme_data.get("name", ""),
                                description=theme_data.get("description", ""),
                                keywords=theme_data.get("keywords", []),
                                confidence=float(theme_data.get("confidence", 0.0)),
                                supporting_docs=theme_data.get("supporting_docs", [])
                            )
                            themes.append(theme)
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Skipping invalid theme data: {e}")
                    
                    return themes
                
                logger.warning("Could not find valid JSON in LLM response")
                return []
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing theme extraction response: {e}")
                logger.debug(f"Response content: {content}")
                return []
                
        except Exception as e:
            logger.error(f"Error in LLM theme extraction: {e}", exc_info=True)
            return []
    
    def _build_theme_extraction_prompt(
        self, 
        documents: List[Dict[str, Any]],
        query: Optional[str] = None
    ) -> str:
        """Build the prompt for theme extraction."""
        # Prepare document summaries
        doc_summaries = []
        for doc in documents:
            doc_id = doc['id']
            doc_text = doc['text']
            metadata = doc.get('metadata', {})
            
            # Truncate long documents
            if len(doc_text) > 2000:
                doc_text = doc_text[:2000] + "... [truncated]"
            
            summary = f"""
            Document ID: {doc_id}
            Title: {metadata.get('title', 'Untitled')}
            Source: {metadata.get('source', 'Unknown')}
            Type: {metadata.get('doc_type', 'Unknown')}
            Upload Date: {metadata.get('upload_date', 'Unknown')}
            
            Content:
            {doc_text}
            """
            doc_summaries.append(summary)
        
        # Build the prompt
        prompt = f"""
        Analyze the following documents and identify the key themes. For each theme, provide:
        1. A clear, concise name
        2. A brief description
        3. 3-5 keywords or phrases
        4. A confidence score between 0.0 and 1.0
        5. A list of supporting document IDs
        """
        
        if query:
            prompt += f"\n\nFocus on themes related to: {query}\n"
        
        prompt += """
        Format your response as a JSON array of theme objects with these fields:
        - name: str
        - description: str
        - keywords: List[str]
        - confidence: float (0.0-1.0)
        - supporting_docs: List[{"doc_id": str, "relevance": float}]
        
        Documents:
        """
        
        # Add document summaries
        prompt += "\n".join(doc_summaries)
        
        # Add instructions
        prompt += f"""
        
        Instructions:
        - Focus on recurring topics, concepts, and patterns
        - Be specific and avoid generic themes
        - Include only themes with strong supporting evidence
        - Maximum {self.max_themes} themes
        - Minimum confidence: {self.min_theme_confidence}
        - Return only valid JSON, no other text
        """
        
        return prompt
    
    def format_theme_for_display(self, theme: Theme) -> str:
        """Format a theme for display in the UI."""
        # Format supporting docs
        supporting_docs = []
        for doc in theme.supporting_docs[:5]:  # Limit to top 5
            doc_info = f"- {doc.get('doc_id', 'Unknown')}"
            if 'relevance' in doc:
                doc_info += f" (relevance: {doc['relevance']:.2f})"
            supporting_docs.append(doc_info)
        
        return f"""
        ### {theme.name}
        *Confidence: {theme.confidence:.1%}*
        
        {theme.description}
        
        **Keywords:** {', '.join(theme.keywords)}
        
        **Supporting Documents:**
        {chr(10).join(supporting_docs) if supporting_docs else 'No supporting documents'}
        """
    
    def _format_supporting_docs(self, supporting_docs: List[Dict[str, Any]]) -> str:
        """Format the list of supporting documents."""
        if not supporting_docs:
            return "No supporting documents"
        
        formatted = []
        for doc in supporting_docs:
            doc_info = f"- {doc.get('title', doc.get('doc_id', 'Untitled'))}"
            
            # Add source if available
            if 'source' in doc and doc['source']:
                doc_info += f" (Source: {doc['source']})"
            
            # Add page number if available
            if 'page_number' in doc and doc['page_number']:
                doc_info += f", Page {doc['page_number']}"
            
            # Add relevance score if available
            if 'relevance' in doc and doc['relevance'] is not None:
                doc_info += f" [Relevance: {doc['relevance']:.2f}]"
            
            formatted.append(doc_info)
        
        return '\n'.join(formatted)
