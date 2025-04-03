"""
Knowledge Base Module for CSM Voice Chat Assistant

This module provides knowledge base functionality to store and retrieve
information, supporting both SQLite and Web API-based storage.
"""

import os
import json
import logging
import time
import sqlite3
import uuid
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class KnowledgeBase:
    """Base class for knowledge base implementations"""
    
    def __init__(self):
        """Initialize the knowledge base"""
        pass
    
    def add_document(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Optional[str]]:
        """
        Add a document to the knowledge base
        
        Args:
            title: Title of the document
            content: Content of the document
            metadata: Optional metadata for the document
            
        Returns:
            Tuple of (success, message, document_id)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document from the knowledge base
        
        Args:
            document_id: ID of the document to get
            
        Returns:
            Document data or None if document not found
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching documents
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the knowledge base
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if document was deleted successfully, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the knowledge base
        
        Returns:
            Dictionary with summary information
        """
        raise NotImplementedError("Subclasses must implement this method")


class SQLiteKnowledgeBase(KnowledgeBase):
    """SQLite-based implementation of knowledge base"""
    
    def __init__(self, db_path: str = "knowledge.db"):
        """
        Initialize the SQLite knowledge base
        
        Args:
            db_path: Path to the SQLite database file
        """
        super().__init__()
        self.db_path = db_path
        self._init_db()
        
        # Try to import sentence-transformers for embeddings
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.has_embeddings = True
            logger.info("Sentence transformers loaded for embeddings")
        except ImportError:
            self.embedding_model = None
            self.has_embeddings = False
            logger.warning("Sentence transformers not available. Semantic search will be limited.")
        
        logger.info(f"Initialized SQLite knowledge base with db_path={db_path}")
    
    def _init_db(self) -> None:
        """Initialize the database and create tables if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                metadata TEXT,
                embedding BLOB,
                created_at REAL,
                updated_at REAL
            )
            ''')
            
            # Create full-text search virtual table
            cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                title, content, document_id UNINDEXED
            )
            ''')
            
            # Create index on title
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_title ON documents (title)
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Initialized SQLite knowledge base database")
        except Exception as e:
            logger.error(f"Error initializing SQLite knowledge base database: {str(e)}")
            raise
    
    def _generate_embedding(self, text: str) -> Optional[bytes]:
        """
        Generate an embedding for text
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding as bytes or None if embedding model not available
        """
        if not self.has_embeddings or not self.embedding_model:
            return None
        
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(text)
            
            # Convert to bytes for storage
            embedding_bytes = embedding.tobytes()
            
            return embedding_bytes
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def _bytes_to_embedding(self, embedding_bytes: bytes) -> Optional[np.ndarray]:
        """
        Convert embedding bytes to numpy array
        
        Args:
            embedding_bytes: Embedding as bytes
            
        Returns:
            Embedding as numpy array or None if conversion fails
        """
        if embedding_bytes is None:
            return None
        
        try:
            # Get embedding dimension from model
            if self.has_embeddings and self.embedding_model:
                embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            else:
                # Default to 384 for all-MiniLM-L6-v2
                embedding_dim = 384
            
            # Convert bytes to numpy array
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(-1)
            
            # Ensure correct shape
            if embedding.shape[0] != embedding_dim:
                logger.warning(f"Embedding dimension mismatch: {embedding.shape[0]} != {embedding_dim}")
                return None
            
            return embedding
        except Exception as e:
            logger.error(f"Error converting embedding bytes: {str(e)}")
            return None
    
    def add_document(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Optional[str]]:
        """
        Add a document to the knowledge base
        
        Args:
            title: Title of the document
            content: Content of the document
            metadata: Optional metadata for the document
            
        Returns:
            Tuple of (success, message, document_id)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Generate document ID
            document_id = str(uuid.uuid4())
            
            # Generate embedding
            embedding_bytes = self._generate_embedding(title + " " + content)
            
            # Get current timestamp
            timestamp = time.time()
            
            # Insert document
            cursor.execute('''
            INSERT INTO documents (document_id, title, content, metadata, embedding, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                document_id,
                title,
                content,
                json.dumps(metadata or {}),
                embedding_bytes,
                timestamp,
                timestamp
            ))
            
            # Insert into FTS table
            cursor.execute('''
            INSERT INTO documents_fts (document_id, title, content)
            VALUES (?, ?, ?)
            ''', (
                document_id,
                title,
                content
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added document: {title} ({document_id})")
            return True, "Document added successfully", document_id
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            return False, f"Error adding document: {str(e)}", None
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document from the knowledge base
        
        Args:
            document_id: ID of the document to get
            
        Returns:
            Document data or None if document not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get document
            cursor.execute('''
            SELECT title, content, metadata, created_at, updated_at
            FROM documents
            WHERE document_id = ?
            ''', (document_id,))
            
            document = cursor.fetchone()
            
            if not document:
                conn.close()
                return None
            
            title, content, metadata_str, created_at, updated_at = document
            
            # Parse metadata
            try:
                metadata = json.loads(metadata_str)
            except json.JSONDecodeError:
                metadata = {}
            
            conn.close()
            
            # Return document data
            return {
                "document_id": document_id,
                "title": title,
                "content": content,
                "metadata": metadata,
                "created_at": created_at,
                "updated_at": updated_at
            }
        except Exception as e:
            logger.error(f"Error getting document: {str(e)}")
            return None
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            results = []
            
            # Try semantic search if embeddings are available
            if self.has_embeddings and self.embedding_model:
                # Generate query embedding
                query_embedding = self.embedding_model.encode(query)
                
                # Get all documents with embeddings
                cursor.execute('''
                SELECT document_id, title, content, metadata, embedding
                FROM documents
                WHERE embedding IS NOT NULL
                ''')
                
                documents = cursor.fetchall()
                
                # Calculate similarity scores
                similarities = []
                for doc in documents:
                    document_id, title, content, metadata_str, embedding_bytes = doc
                    
                    # Convert embedding bytes to numpy array
                    doc_embedding = self._bytes_to_embedding(embedding_bytes)
                    
                    if doc_embedding is not None:
                        # Calculate cosine similarity
                        similarity = np.dot(query_embedding, doc_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                        )
                        
                        similarities.append((document_id, title, content, metadata_str, similarity))
                
                # Sort by similarity (descending)
                similarities.sort(key=lambda x: x[4], reverse=True)
                
                # Get top results
                for doc_id, title, content, metadata_str, similarity in similarities[:limit]:
                    # Parse metadata
                    try:
                        metadata = json.loads(metadata_str)
                    except json.JSONDecodeError:
                        metadata = {}
                    
                    results.append({
                        "document_id": doc_id,
                        "title": title,
                        "content": content,
                        "metadata": metadata,
                        "score": float(similarity)
                    })
            
            # If no results from semantic search, try full-text search
            if not results:
                # Search using FTS
                cursor.execute('''
                SELECT document_id, title, content, rank
                FROM documents_fts
                WHERE documents_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                ''', (query, limit))
                
                for doc_id, title, content, rank in cursor.fetchall():
                    # Get metadata
                    cursor.execute('''
                    SELECT metadata
                    FROM documents
                    WHERE document_id = ?
                    ''', (doc_id,))
                    
                    metadata_str = cursor.fetchone()[0]
                    
                    # Parse metadata
                    try:
                        metadata = json.loads(metadata_str)
                    except json.JSONDecodeError:
                        metadata = {}
                    
                    results.append({
                        "document_id": doc_id,
                        "title": title,
                        "content": content,
                        "metadata": metadata,
                        "score": 1.0 / (rank + 1)  # Convert rank to score
                    })
            
            conn.close()
            
            return results
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the knowledge base
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if document was deleted successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if document exists
            cursor.execute('''
            SELECT document_id FROM documents WHERE document_id = ?
            ''', (document_id,))
            
            if not cursor.fetchone():
                conn.close()
                return False
            
            # Delete from documents table
            cursor.execute('''
            DELETE FROM documents WHERE document_id = ?
            ''', (document_id,))
            
            # Delete from FTS table
            cursor.execute('''
            DELETE FROM documents_fts WHERE document_id = ?
            ''', (document_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the knowledge base
        
        Returns:
            Dictionary with summary information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total documents
            cursor.execute('''
            SELECT COUNT(*) FROM documents
            ''')
            total_documents = cursor.fetchone()[0]
            
            # Get total documents with embeddings
            cursor.execute('''
            SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL
            ''')
            total_with_embeddings = cursor.fetchone()[0]
            
            # Get average document length
            cursor.execute('''
            SELECT AVG(LENGTH(content)) FROM documents
            ''')
            avg_length = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "type": "sqlite",
                "total_documents": total_documents,
                "total_with_embeddings": total_with_embeddings,
                "average_document_length": avg_length,
                "has_embeddings": self.has_embeddings,
                "db_path": self.db_path
            }
        except Exception as e:
            logger.error(f"Error getting knowledge base summary: {str(e)}")
            return {
                "type": "sqlite",
                "error": str(e)
            }


class WebAPIKnowledgeBase(KnowledgeBase):
    """Web API-based implementation of knowledge base"""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        """
        Initialize the Web API knowledge base
        
        Args:
            api_url: URL of the knowledge base API
            api_key: Optional API key for authentication
        """
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
        logger.info(f"Initialized Web API knowledge base with URL: {api_url}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    def add_document(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Optional[str]]:
        """
        Add a document to the knowledge base
        
        Args:
            title: Title of the document
            content: Content of the document
            metadata: Optional metadata for the document
            
        Returns:
            Tuple of (success, message, document_id)
        """
        try:
            url = f"{self.api_url}/documents"
            headers = self._get_headers()
            
            payload = {
                "title": title,
                "content": content,
                "metadata": metadata or {}
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code in (200, 201):
                data = response.json()
                document_id = data.get("document_id")
                
                logger.info(f"Added document via API: {title} ({document_id})")
                return True, "Document added successfully", document_id
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return False, f"API error: {response.status_code}", None
        except Exception as e:
            logger.error(f"Error adding document via API: {str(e)}")
            return False, f"Error adding document: {str(e)}", None
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document from the knowledge base
        
        Args:
            document_id: ID of the document to get
            
        Returns:
            Document data or None if document not found
        """
        try:
            url = f"{self.api_url}/documents/{document_id}"
            headers = self._get_headers()
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting document via API: {str(e)}")
            return None
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge base
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            url = f"{self.api_url}/search"
            headers = self._get_headers()
            
            payload = {
                "query": query,
                "limit": limit
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("results", [])
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error searching knowledge base via API: {str(e)}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the knowledge base
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if document was deleted successfully, False otherwise
        """
        try:
            url = f"{self.api_url}/documents/{document_id}"
            headers = self._get_headers()
            
            response = requests.delete(url, headers=headers, timeout=30)
            
            if response.status_code in (200, 204):
                logger.info(f"Deleted document via API: {document_id}")
                return True
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error deleting document via API: {str(e)}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the knowledge base
        
        Returns:
            Dictionary with summary information
        """
        try:
            url = f"{self.api_url}/summary"
            headers = self._get_headers()
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                data["type"] = "web_api"
                data["api_url"] = self.api_url
                return data
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {
                    "type": "web_api",
                    "api_url": self.api_url,
                    "error": f"API error: {response.status_code}"
                }
        except Exception as e:
            logger.error(f"Error getting knowledge base summary via API: {str(e)}")
            return {
                "type": "web_api",
                "api_url": self.api_url,
                "error": str(e)
            }


def get_knowledge_base() -> KnowledgeBase:
    """
    Get the appropriate knowledge base implementation based on configuration
    
    Returns:
        An instance of KnowledgeBase
    """
    kb_type = os.getenv("KNOWLEDGE_BASE_TYPE", "sqlite")
    
    if kb_type == "web_api":
        api_url = os.getenv("KNOWLEDGE_BASE_API_URL")
        api_key = os.getenv("KNOWLEDGE_BASE_API_KEY")
        
        if api_url:
            return WebAPIKnowledgeBase(api_url=api_url, api_key=api_key)
        else:
            logger.warning("Knowledge base API URL not configured. Falling back to SQLite.")
            kb_type = "sqlite"
    
    if kb_type == "sqlite":
        db_path = os.getenv("KNOWLEDGE_BASE_DB_PATH", "knowledge.db")
        return SQLiteKnowledgeBase(db_path=db_path)
    
    # Default to SQLite
    logger.warning(f"Unknown knowledge base type: {kb_type}. Using SQLite.")
    return SQLiteKnowledgeBase()