import logging
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Union, TYPE_CHECKING

import chromadb
import numpy as np
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from app.models.document import Document, DocumentChunk, SearchResult, DocumentMetadata

if TYPE_CHECKING:
    from .vector_store import VectorStore

logger = logging.getLogger(__name__)


def create_vector_store(persist_directory: str = "./data/vector_db") -> 'VectorStore':
    """
    Create and return a VectorStore instance with default settings.
    
    Args:
        persist_directory: Directory to persist the vector store
        
    Returns:
        VectorStore: Initialized vector store instance
    """
    return VectorStore(persist_directory=persist_directory)


class VectorStore:
    """Handles vector storage and similarity search for document chunks."""
    
    def __init__(
        self, 
        persist_directory: str = "./data/vector_db",
        collection_name: str = "document_chunks",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_function = None
    ):
        """Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the ChromaDB
            collection_name: Name of the collection to use
            embedding_model: Name of the sentence transformer model to use
            embedding_function: Custom embedding function to use (overrides embedding_model if provided)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            )
        )
        
        # Configure embedding function
        if embedding_function is None:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        else:
            self.embedding_function = embedding_function
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def add_document(self, document: Document) -> bool:
        """Add a document to the vector store."""
        try:
            # Prepare data for batch insertion
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            for chunk in document.chunks:
                # Generate a unique ID for the chunk
                chunk_id = str(uuid.uuid4())
                
                # Create metadata
                metadata = {
                    "doc_id": document.metadata.doc_id,
                    "title": document.metadata.title,
                    "doc_type": document.metadata.doc_type.value,
                    "page_number": str(chunk.page_number + 1),  # 1-based for display
                    "chunk_index": str(chunk.chunk_index),
                    "source": document.metadata.title,
                    "upload_date": document.metadata.upload_date.isoformat() if document.metadata.upload_date else "",
                    **chunk.metadata
                }
                
                # Add to batch
                ids.append(chunk_id)
                documents.append(chunk.content)
                metadatas.append(metadata)
                
                # If we have a lot of chunks, we might want to process in batches
                if len(ids) >= 100:  # Adjust batch size as needed
                    self.collection.upsert(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas
                    )
                    ids, documents, metadatas = [], [], []
            
            # Insert any remaining chunks
            if ids:
                self.collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            
            # Persist the database
            self.client.persist()
            return True
            
        except Exception as e:
            logger.error(f"Error adding document to vector store: {e}", exc_info=True)
            return False
    
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        filter_conditions: Optional[Dict[str, str]] = None,
        include: List[str] = ["documents", "metadatas", "distances"]
    ) -> List[SearchResult]:
        """Search for similar document chunks.
        
        Args:
            query: The search query
            n_results: Number of results to return
            filter_conditions: Optional filters to apply (e.g., {"doc_id": "123"})
            include: What to include in the results
            
        Returns:
            List of SearchResult objects
        """
        try:
            if filter_conditions is None:
                filter_conditions = {}
            
            # Convert filter conditions to Chroma format
            chroma_filters = {}
            for k, v in filter_conditions.items():
                if k in ["doc_id", "title", "doc_type", "page_number"]:
                    chroma_filters[k] = v
            
            # Perform the search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=chroma_filters if chroma_filters else None,
                include=include
            )
            
            # Convert to SearchResult objects
            search_results = []
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                content = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i] if "distances" in results else 1.0
                
                # Convert metadata back to DocumentMetadata
                doc_metadata = DocumentMetadata(
                    doc_id=metadata.get("doc_id", ""),
                    title=metadata.get("title", ""),
                    doc_type=metadata.get("doc_type", "text"),
                    upload_date=datetime.fromisoformat(metadata.get("upload_date", "1970-01-01")),
                    custom_metadata={
                        k: v for k, v in metadata.items() 
                        if k not in ["doc_id", "title", "doc_type", "upload_date"]
                    }
                )
                
                # Create DocumentChunk
                chunk = DocumentChunk(
                    chunk_id=doc_id,
                    doc_id=metadata.get("doc_id", ""),
                    content=content,
                    page_number=int(metadata.get("page_number", 1)) - 1,  # Convert back to 0-based
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    metadata=metadata
                )
                
                # Calculate similarity score (convert distance to similarity)
                similarity = 1.0 / (1.0 + distance) if distance > 0 else 1.0
                
                search_results.append(SearchResult(
                    chunk=chunk,
                    score=similarity,
                    document=doc_metadata
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}", exc_info=True)
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            self.collection.delete(where={"doc_id": doc_id})
            self.client.persist()
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def get_document_ids(self) -> List[str]:
        """Get a list of all document IDs in the vector store."""
        try:
            # Get all document IDs
            results = self.collection.get(include=[])
            doc_ids = list(set(metadata.get("doc_id") for metadata in results["metadatas"]))
            return [doc_id for doc_id in doc_ids if doc_id]  # Filter out None values
        except Exception as e:
            logger.error(f"Error getting document IDs: {e}")
            return []
    
    def clear(self) -> bool:
        """Clear the entire vector store."""
        try:
            self.client.reset()
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False
