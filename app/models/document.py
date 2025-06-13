from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    DOCX = "docx"
    PPTX = "pptx"

class DocumentMetadata(BaseModel):
    doc_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    doc_type: DocumentType = Field(..., description="Type of document")
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    author: Optional[str] = None
    source: Optional[str] = None
    page_count: int = 0
    word_count: int = 0
    language: str = "en"
    custom_metadata: Dict[str, str] = Field(default_factory=dict)

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    page_number: int
    chunk_index: int
    metadata: Dict[str, str] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class Document(BaseModel):
    metadata: DocumentMetadata
    chunks: List[DocumentChunk] = []
    raw_text: Optional[str] = None
    
    def add_chunk(self, chunk: DocumentChunk):
        self.chunks.append(chunk)
        
    def get_full_text(self) -> str:
        if self.raw_text:
            return self.raw_text
        return "\n\n".join([chunk.content for chunk in sorted(self.chunks, key=lambda x: (x.page_number, x.chunk_index))])

class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float
    document: DocumentMetadata
