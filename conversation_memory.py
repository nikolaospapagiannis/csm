"""
Conversation Memory Module for CSM Voice Chat Assistant

This module provides conversation memory functionality to store and retrieve
conversation history, supporting both in-memory and SQLite-based storage.
"""

import os
import json
import logging
import time
import sqlite3
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ConversationMemory:
    """Base class for conversation memory implementations"""
    
    def __init__(self):
        """Initialize the conversation memory"""
        pass
    
    def add_message(self, role: str, content: str, user_id: Optional[str] = None) -> bool:
        """
        Add a message to the conversation history
        
        Args:
            role: Role of the message sender (user or assistant)
            content: Content of the message
            user_id: Optional user ID
            
        Returns:
            True if message was added successfully, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_history(self, user_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Args:
            user_id: Optional user ID
            limit: Maximum number of messages to return
            
        Returns:
            List of messages
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def clear_history(self, user_id: Optional[str] = None) -> bool:
        """
        Clear conversation history
        
        Args:
            user_id: Optional user ID
            
        Returns:
            True if history was cleared successfully, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation memory
        
        Returns:
            Dictionary with summary information
        """
        raise NotImplementedError("Subclasses must implement this method")


class InMemoryConversationMemory(ConversationMemory):
    """In-memory implementation of conversation memory"""
    
    def __init__(self, max_history: int = 50):
        """
        Initialize the in-memory conversation memory
        
        Args:
            max_history: Maximum number of messages to store per user
        """
        super().__init__()
        self.max_history = max_history
        self.conversations = {}  # user_id -> list of messages
        logger.info(f"Initialized in-memory conversation memory with max_history={max_history}")
    
    def add_message(self, role: str, content: str, user_id: Optional[str] = None) -> bool:
        """
        Add a message to the conversation history
        
        Args:
            role: Role of the message sender (user or assistant)
            content: Content of the message
            user_id: Optional user ID
            
        Returns:
            True if message was added successfully, False otherwise
        """
        try:
            # Use default user ID if not provided
            user_id = user_id or "default"
            
            # Create conversation for user if it doesn't exist
            if user_id not in self.conversations:
                self.conversations[user_id] = []
            
            # Add message to conversation
            self.conversations[user_id].append({
                "role": role,
                "content": content,
                "timestamp": time.time(),
                "message_id": str(uuid.uuid4())
            })
            
            # Trim conversation if it exceeds max_history
            if len(self.conversations[user_id]) > self.max_history:
                self.conversations[user_id] = self.conversations[user_id][-self.max_history:]
            
            logger.debug(f"Added message for user {user_id}: {role} - {content[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            return False
    
    def get_history(self, user_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Args:
            user_id: Optional user ID
            limit: Maximum number of messages to return
            
        Returns:
            List of messages
        """
        try:
            # Use default user ID if not provided
            user_id = user_id or "default"
            
            # Return empty list if user has no conversation
            if user_id not in self.conversations:
                return []
            
            # Return the most recent messages up to the limit
            return self.conversations[user_id][-limit:]
        except Exception as e:
            logger.error(f"Error getting history: {str(e)}")
            return []
    
    def clear_history(self, user_id: Optional[str] = None) -> bool:
        """
        Clear conversation history
        
        Args:
            user_id: Optional user ID
            
        Returns:
            True if history was cleared successfully, False otherwise
        """
        try:
            # Use default user ID if not provided
            user_id = user_id or "default"
            
            # Clear conversation for user
            if user_id in self.conversations:
                self.conversations[user_id] = []
            
            logger.info(f"Cleared conversation history for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing history: {str(e)}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation memory
        
        Returns:
            Dictionary with summary information
        """
        try:
            total_messages = sum(len(messages) for messages in self.conversations.values())
            total_users = len(self.conversations)
            
            return {
                "type": "in_memory",
                "total_messages": total_messages,
                "total_users": total_users,
                "max_history": self.max_history
            }
        except Exception as e:
            logger.error(f"Error getting summary: {str(e)}")
            return {
                "type": "in_memory",
                "error": str(e)
            }


class SQLiteConversationMemory(ConversationMemory):
    """SQLite-based implementation of conversation memory"""
    
    def __init__(self, db_path: str = "conversations.db", max_history: int = 50):
        """
        Initialize the SQLite conversation memory
        
        Args:
            db_path: Path to the SQLite database file
            max_history: Maximum number of messages to return per user
        """
        super().__init__()
        self.db_path = db_path
        self.max_history = max_history
        self._init_db()
        logger.info(f"Initialized SQLite conversation memory with db_path={db_path}, max_history={max_history}")
    
    def _init_db(self) -> None:
        """Initialize the database and create tables if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create messages table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                user_id TEXT,
                role TEXT,
                content TEXT,
                timestamp REAL
            )
            ''')
            
            # Create index on user_id
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_user_id ON messages (user_id)
            ''')
            
            # Create index on timestamp
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON messages (timestamp)
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Initialized SQLite conversation database")
        except Exception as e:
            logger.error(f"Error initializing SQLite conversation database: {str(e)}")
            raise
    
    def add_message(self, role: str, content: str, user_id: Optional[str] = None) -> bool:
        """
        Add a message to the conversation history
        
        Args:
            role: Role of the message sender (user or assistant)
            content: Content of the message
            user_id: Optional user ID
            
        Returns:
            True if message was added successfully, False otherwise
        """
        try:
            # Use default user ID if not provided
            user_id = user_id or "default"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Add message to database
            message_id = str(uuid.uuid4())
            timestamp = time.time()
            
            cursor.execute('''
            INSERT INTO messages (message_id, user_id, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                message_id,
                user_id,
                role,
                content,
                timestamp
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Added message for user {user_id}: {role} - {content[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            return False
    
    def get_history(self, user_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Args:
            user_id: Optional user ID
            limit: Maximum number of messages to return
            
        Returns:
            List of messages
        """
        try:
            # Use default user ID if not provided
            user_id = user_id or "default"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get messages for user
            cursor.execute('''
            SELECT message_id, role, content, timestamp
            FROM messages
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (user_id, min(limit, self.max_history)))
            
            # Convert to list of dictionaries
            messages = []
            for row in cursor.fetchall():
                message_id, role, content, timestamp = row
                messages.append({
                    "message_id": message_id,
                    "role": role,
                    "content": content,
                    "timestamp": timestamp
                })
            
            conn.close()
            
            # Return messages in chronological order
            return list(reversed(messages))
        except Exception as e:
            logger.error(f"Error getting history: {str(e)}")
            return []
    
    def clear_history(self, user_id: Optional[str] = None) -> bool:
        """
        Clear conversation history
        
        Args:
            user_id: Optional user ID
            
        Returns:
            True if history was cleared successfully, False otherwise
        """
        try:
            # Use default user ID if not provided
            user_id = user_id or "default"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete messages for user
            cursor.execute('''
            DELETE FROM messages
            WHERE user_id = ?
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleared conversation history for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing history: {str(e)}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation memory
        
        Returns:
            Dictionary with summary information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total messages
            cursor.execute('''
            SELECT COUNT(*) FROM messages
            ''')
            total_messages = cursor.fetchone()[0]
            
            # Get total users
            cursor.execute('''
            SELECT COUNT(DISTINCT user_id) FROM messages
            ''')
            total_users = cursor.fetchone()[0]
            
            # Get oldest message timestamp
            cursor.execute('''
            SELECT MIN(timestamp) FROM messages
            ''')
            oldest_timestamp = cursor.fetchone()[0]
            
            # Get newest message timestamp
            cursor.execute('''
            SELECT MAX(timestamp) FROM messages
            ''')
            newest_timestamp = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "type": "sqlite",
                "total_messages": total_messages,
                "total_users": total_users,
                "max_history": self.max_history,
                "db_path": self.db_path,
                "oldest_message": datetime.fromtimestamp(oldest_timestamp).isoformat() if oldest_timestamp else None,
                "newest_message": datetime.fromtimestamp(newest_timestamp).isoformat() if newest_timestamp else None
            }
        except Exception as e:
            logger.error(f"Error getting summary: {str(e)}")
            return {
                "type": "sqlite",
                "error": str(e)
            }


def get_conversation_memory() -> ConversationMemory:
    """
    Get the appropriate conversation memory implementation based on configuration
    
    Returns:
        An instance of ConversationMemory
    """
    memory_type = os.getenv("CONVERSATION_MEMORY_TYPE", "in_memory")
    max_history = int(os.getenv("CONVERSATION_MAX_HISTORY", 50))
    
    if memory_type == "sqlite":
        db_path = os.getenv("CONVERSATION_DB_PATH", "conversations.db")
        return SQLiteConversationMemory(db_path=db_path, max_history=max_history)
    else:
        return InMemoryConversationMemory(max_history=max_history)