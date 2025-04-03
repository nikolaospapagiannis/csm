"""
User Authentication Module for CSM Voice Chat Assistant

This module provides user authentication functionality, including user registration,
login, and session management using JWT tokens.
"""

import os
import json
import logging
import time
import sqlite3
import uuid
import hashlib
import secrets
import jwt
from typing import Dict, Any, List, Optional, Tuple, Union
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

# Get JWT configuration from environment
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_EXPIRATION = int(os.getenv("JWT_EXPIRATION", 86400))  # 24 hours in seconds
AUTH_DB_PATH = os.getenv("AUTH_DB_PATH", "users.db")

class UserAuth:
    """User authentication manager"""
    
    def __init__(self, db_path: str = AUTH_DB_PATH):
        """
        Initialize the user authentication manager
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_db()
        logger.info(f"Initialized user authentication with db_path={db_path}")
    
    def _init_db(self) -> None:
        """Initialize the database and create tables if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE,
                email TEXT UNIQUE,
                password_hash TEXT,
                salt TEXT,
                created_at REAL,
                last_login REAL,
                is_active INTEGER,
                role TEXT
            )
            ''')
            
            # Create sessions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                token TEXT,
                created_at REAL,
                expires_at REAL,
                is_active INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
            ''')
            
            # Create index on username
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_username ON users (username)
            ''')
            
            # Create index on email
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_email ON users (email)
            ''')
            
            # Create index on user_id in sessions
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_session_user_id ON sessions (user_id)
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Initialized user authentication database")
        except Exception as e:
            logger.error(f"Error initializing user authentication database: {str(e)}")
            raise
    
    def _hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        Hash a password with a salt
        
        Args:
            password: The password to hash
            salt: Optional salt to use (generates a new one if not provided)
            
        Returns:
            Tuple of (password_hash, salt)
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Hash the password with the salt
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # Number of iterations
        ).hex()
        
        return password_hash, salt
    
    def register_user(self, username: str, email: str, password: str, role: str = "user") -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Register a new user
        
        Args:
            username: Username for the new user
            email: Email address for the new user
            password: Password for the new user
            role: Role for the new user (default: "user")
            
        Returns:
            Tuple of (success, message, user_data)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if username already exists
            cursor.execute('''
            SELECT user_id FROM users WHERE username = ?
            ''', (username,))
            
            if cursor.fetchone():
                conn.close()
                return False, "Username already exists", None
            
            # Check if email already exists
            cursor.execute('''
            SELECT user_id FROM users WHERE email = ?
            ''', (email,))
            
            if cursor.fetchone():
                conn.close()
                return False, "Email already exists", None
            
            # Generate user ID
            user_id = str(uuid.uuid4())
            
            # Hash the password
            password_hash, salt = self._hash_password(password)
            
            # Get current timestamp
            timestamp = time.time()
            
            # Insert the new user
            cursor.execute('''
            INSERT INTO users (user_id, username, email, password_hash, salt, created_at, last_login, is_active, role)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                username,
                email,
                password_hash,
                salt,
                timestamp,
                timestamp,
                1,  # is_active
                role
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Registered new user: {username} ({user_id})")
            
            # Return user data (without sensitive information)
            user_data = {
                "user_id": user_id,
                "username": username,
                "email": email,
                "created_at": timestamp,
                "role": role
            }
            
            return True, "User registered successfully", user_data
        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
            return False, f"Error registering user: {str(e)}", None
    
    def login(self, username_or_email: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Login a user
        
        Args:
            username_or_email: Username or email of the user
            password: Password of the user
            
        Returns:
            Tuple of (success, message, session_data)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if username or email exists
            cursor.execute('''
            SELECT user_id, username, email, password_hash, salt, role, is_active
            FROM users
            WHERE username = ? OR email = ?
            ''', (username_or_email, username_or_email))
            
            user = cursor.fetchone()
            
            if not user:
                conn.close()
                return False, "Invalid username or email", None
            
            user_id, username, email, password_hash, salt, role, is_active = user
            
            # Check if user is active
            if not is_active:
                conn.close()
                return False, "User account is inactive", None
            
            # Verify the password
            hashed_password, _ = self._hash_password(password, salt)
            
            if hashed_password != password_hash:
                conn.close()
                return False, "Invalid password", None
            
            # Update last login timestamp
            timestamp = time.time()
            
            cursor.execute('''
            UPDATE users
            SET last_login = ?
            WHERE user_id = ?
            ''', (timestamp, user_id))
            
            # Generate a JWT token
            token_data = {
                "user_id": user_id,
                "username": username,
                "email": email,
                "role": role,
                "exp": int(timestamp + JWT_EXPIRATION)
            }
            
            token = jwt.encode(token_data, JWT_SECRET, algorithm="HS256")
            
            # Create a new session
            session_id = str(uuid.uuid4())
            expires_at = timestamp + JWT_EXPIRATION
            
            cursor.execute('''
            INSERT INTO sessions (session_id, user_id, token, created_at, expires_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                user_id,
                token,
                timestamp,
                expires_at,
                1  # is_active
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"User logged in: {username} ({user_id})")
            
            # Return session data
            session_data = {
                "user_id": user_id,
                "username": username,
                "email": email,
                "role": role,
                "token": token,
                "expires_at": expires_at
            }
            
            return True, "Login successful", session_data
        except Exception as e:
            logger.error(f"Error logging in: {str(e)}")
            return False, f"Error logging in: {str(e)}", None
    
    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify a JWT token
        
        Args:
            token: JWT token to verify
            
        Returns:
            Tuple of (is_valid, user_data)
        """
        try:
            # Decode the token
            token_data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            
            # Check if token is expired
            if token_data.get("exp", 0) < time.time():
                return False, None
            
            # Get user data
            user_id = token_data.get("user_id")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user exists and is active
            cursor.execute('''
            SELECT username, email, role, is_active
            FROM users
            WHERE user_id = ?
            ''', (user_id,))
            
            user = cursor.fetchone()
            
            if not user:
                conn.close()
                return False, None
            
            username, email, role, is_active = user
            
            # Check if user is active
            if not is_active:
                conn.close()
                return False, None
            
            # Check if session is active
            cursor.execute('''
            SELECT session_id
            FROM sessions
            WHERE user_id = ? AND token = ? AND is_active = 1 AND expires_at > ?
            ''', (user_id, token, time.time()))
            
            session = cursor.fetchone()
            
            if not session:
                conn.close()
                return False, None
            
            conn.close()
            
            # Return user data
            user_data = {
                "user_id": user_id,
                "username": username,
                "email": email,
                "role": role
            }
            
            return True, user_data
        except jwt.InvalidTokenError:
            logger.warning(f"Invalid token: {token[:20]}...")
            return False, None
        except Exception as e:
            logger.error(f"Error verifying token: {str(e)}")
            return False, None
    
    def logout(self, token: str) -> bool:
        """
        Logout a user by invalidating their session
        
        Args:
            token: JWT token to invalidate
            
        Returns:
            True if logout was successful, False otherwise
        """
        try:
            # Decode the token
            token_data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], options={"verify_exp": False})
            
            # Get user ID
            user_id = token_data.get("user_id")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Invalidate the session
            cursor.execute('''
            UPDATE sessions
            SET is_active = 0
            WHERE user_id = ? AND token = ?
            ''', (user_id, token))
            
            conn.commit()
            conn.close()
            
            logger.info(f"User logged out: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error logging out: {str(e)}")
            return False
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user data by user ID
        
        Args:
            user_id: ID of the user to get
            
        Returns:
            User data or None if user not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user data
            cursor.execute('''
            SELECT username, email, created_at, last_login, is_active, role
            FROM users
            WHERE user_id = ?
            ''', (user_id,))
            
            user = cursor.fetchone()
            
            if not user:
                conn.close()
                return None
            
            username, email, created_at, last_login, is_active, role = user
            
            conn.close()
            
            # Return user data
            return {
                "user_id": user_id,
                "username": username,
                "email": email,
                "created_at": created_at,
                "last_login": last_login,
                "is_active": bool(is_active),
                "role": role
            }
        except Exception as e:
            logger.error(f"Error getting user: {str(e)}")
            return None
    
    def update_user(self, user_id: str, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Update user data
        
        Args:
            user_id: ID of the user to update
            data: Dictionary of user data to update
            
        Returns:
            Tuple of (success, message)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute('''
            SELECT user_id FROM users WHERE user_id = ?
            ''', (user_id,))
            
            if not cursor.fetchone():
                conn.close()
                return False, "User not found"
            
            # Build update query
            update_fields = []
            update_values = []
            
            if "username" in data:
                # Check if username is already taken
                cursor.execute('''
                SELECT user_id FROM users WHERE username = ? AND user_id != ?
                ''', (data["username"], user_id))
                
                if cursor.fetchone():
                    conn.close()
                    return False, "Username already exists"
                
                update_fields.append("username = ?")
                update_values.append(data["username"])
            
            if "email" in data:
                # Check if email is already taken
                cursor.execute('''
                SELECT user_id FROM users WHERE email = ? AND user_id != ?
                ''', (data["email"], user_id))
                
                if cursor.fetchone():
                    conn.close()
                    return False, "Email already exists"
                
                update_fields.append("email = ?")
                update_values.append(data["email"])
            
            if "password" in data:
                # Hash the new password
                password_hash, salt = self._hash_password(data["password"])
                
                update_fields.append("password_hash = ?")
                update_values.append(password_hash)
                
                update_fields.append("salt = ?")
                update_values.append(salt)
            
            if "is_active" in data:
                update_fields.append("is_active = ?")
                update_values.append(1 if data["is_active"] else 0)
            
            if "role" in data:
                update_fields.append("role = ?")
                update_values.append(data["role"])
            
            if not update_fields:
                conn.close()
                return False, "No fields to update"
            
            # Execute update query
            query = f'''
            UPDATE users
            SET {", ".join(update_fields)}
            WHERE user_id = ?
            '''
            
            update_values.append(user_id)
            
            cursor.execute(query, update_values)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated user: {user_id}")
            return True, "User updated successfully"
        except Exception as e:
            logger.error(f"Error updating user: {str(e)}")
            return False, f"Error updating user: {str(e)}"
    
    def delete_user(self, user_id: str) -> Tuple[bool, str]:
        """
        Delete a user
        
        Args:
            user_id: ID of the user to delete
            
        Returns:
            Tuple of (success, message)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute('''
            SELECT user_id FROM users WHERE user_id = ?
            ''', (user_id,))
            
            if not cursor.fetchone():
                conn.close()
                return False, "User not found"
            
            # Delete user's sessions
            cursor.execute('''
            DELETE FROM sessions WHERE user_id = ?
            ''', (user_id,))
            
            # Delete user
            cursor.execute('''
            DELETE FROM users WHERE user_id = ?
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted user: {user_id}")
            return True, "User deleted successfully"
        except Exception as e:
            logger.error(f"Error deleting user: {str(e)}")
            return False, f"Error deleting user: {str(e)}"
    
    def get_active_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get active sessions
        
        Args:
            user_id: Optional user ID to filter sessions
            
        Returns:
            List of active sessions
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
            SELECT s.session_id, s.user_id, u.username, s.created_at, s.expires_at
            FROM sessions s
            JOIN users u ON s.user_id = u.user_id
            WHERE s.is_active = 1 AND s.expires_at > ?
            '''
            
            params = [time.time()]
            
            if user_id:
                query += " AND s.user_id = ?"
                params.append(user_id)
            
            cursor.execute(query, params)
            
            sessions = []
            for row in cursor.fetchall():
                session_id, user_id, username, created_at, expires_at = row
                
                sessions.append({
                    "session_id": session_id,
                    "user_id": user_id,
                    "username": username,
                    "created_at": created_at,
                    "expires_at": expires_at
                })
            
            conn.close()
            
            return sessions
        except Exception as e:
            logger.error(f"Error getting active sessions: {str(e)}")
            return []
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current timestamp
            timestamp = time.time()
            
            # Update expired sessions
            cursor.execute('''
            UPDATE sessions
            SET is_active = 0
            WHERE is_active = 1 AND expires_at <= ?
            ''', (timestamp,))
            
            count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {count} expired sessions")
            return count
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {str(e)}")
            return 0


# Create a global instance
auth = UserAuth()