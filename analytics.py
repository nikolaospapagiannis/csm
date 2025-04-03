"""
Analytics Module for CSM Voice Chat Assistant

This module provides analytics and usage tracking functionality to monitor
system usage, performance, and user interactions.
"""

import os
import json
import logging
import time
import sqlite3
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import threading
import queue
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

class AnalyticsEvent:
    """Class representing an analytics event"""
    
    def __init__(self, event_type: str, data: Dict[str, Any], user_id: Optional[str] = None):
        """
        Initialize an analytics event
        
        Args:
            event_type: Type of the event
            data: Event data
            user_id: Optional user ID
        """
        self.event_id = str(uuid.uuid4())
        self.event_type = event_type
        self.data = data
        self.user_id = user_id or "anonymous"
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "user_id": self.user_id,
            "timestamp": self.timestamp
        }


class AnalyticsTracker:
    """Base class for analytics trackers"""
    
    def __init__(self):
        """Initialize the analytics tracker"""
        pass
    
    def track_event(self, event: AnalyticsEvent) -> bool:
        """
        Track an analytics event
        
        Args:
            event: The event to track
            
        Returns:
            True if event was tracked successfully, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_events(self, event_type: Optional[str] = None, 
                  user_id: Optional[str] = None, 
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get analytics events
        
        Args:
            event_type: Optional event type filter
            user_id: Optional user ID filter
            start_time: Optional start time filter (timestamp)
            end_time: Optional end time filter (timestamp)
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_metrics(self, metric_name: str, 
                   user_id: Optional[str] = None,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Get analytics metrics
        
        Args:
            metric_name: Name of the metric to get
            user_id: Optional user ID filter
            start_time: Optional start time filter (timestamp)
            end_time: Optional end time filter (timestamp)
            
        Returns:
            Dictionary with metric data
        """
        raise NotImplementedError("Subclasses must implement this method")


class SQLiteAnalyticsTracker(AnalyticsTracker):
    """SQLite-based implementation of analytics tracker"""
    
    def __init__(self, db_path: str = "analytics.db"):
        """
        Initialize the SQLite analytics tracker
        
        Args:
            db_path: Path to the SQLite database file
        """
        super().__init__()
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the database and create tables if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create events table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT,
                user_id TEXT,
                timestamp REAL,
                data TEXT
            )
            ''')
            
            # Create index on event_type
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_event_type ON events (event_type)
            ''')
            
            # Create index on user_id
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_user_id ON events (user_id)
            ''')
            
            # Create index on timestamp
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON events (timestamp)
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Initialized SQLite analytics database")
        except Exception as e:
            logger.error(f"Error initializing SQLite analytics database: {str(e)}")
            raise
    
    def track_event(self, event: AnalyticsEvent) -> bool:
        """
        Track an analytics event
        
        Args:
            event: The event to track
            
        Returns:
            True if event was tracked successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO events (event_id, event_type, user_id, timestamp, data)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.event_type,
                event.user_id,
                event.timestamp,
                json.dumps(event.data)
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Tracked event: {event.event_type}")
            return True
        except Exception as e:
            logger.error(f"Error tracking event: {str(e)}")
            return False
    
    def get_events(self, event_type: Optional[str] = None, 
                  user_id: Optional[str] = None, 
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get analytics events
        
        Args:
            event_type: Optional event type filter
            user_id: Optional user ID filter
            start_time: Optional start time filter (timestamp)
            end_time: Optional end time filter (timestamp)
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT event_id, event_type, user_id, timestamp, data FROM events WHERE 1=1"
            params = []
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            events = []
            for row in cursor.fetchall():
                event_id, event_type, user_id, timestamp, data_str = row
                
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    data = {}
                
                events.append({
                    "event_id": event_id,
                    "event_type": event_type,
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "data": data
                })
            
            conn.close()
            return events
        except Exception as e:
            logger.error(f"Error getting events: {str(e)}")
            return []
    
    def get_metrics(self, metric_name: str, 
                   user_id: Optional[str] = None,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Get analytics metrics
        
        Args:
            metric_name: Name of the metric to get
            user_id: Optional user ID filter
            start_time: Optional start time filter (timestamp)
            end_time: Optional end time filter (timestamp)
            
        Returns:
            Dictionary with metric data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if metric_name == "event_count":
                query = "SELECT event_type, COUNT(*) FROM events WHERE 1=1"
                params = []
                
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                query += " GROUP BY event_type"
                
                cursor.execute(query, params)
                
                result = {}
                for row in cursor.fetchall():
                    event_type, count = row
                    result[event_type] = count
                
                conn.close()
                return {
                    "metric": "event_count",
                    "data": result
                }
            
            elif metric_name == "user_activity":
                query = "SELECT user_id, COUNT(*) FROM events WHERE 1=1"
                params = []
                
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                query += " GROUP BY user_id ORDER BY COUNT(*) DESC LIMIT 100"
                
                cursor.execute(query, params)
                
                result = {}
                for row in cursor.fetchall():
                    user_id, count = row
                    result[user_id] = count
                
                conn.close()
                return {
                    "metric": "user_activity",
                    "data": result
                }
            
            elif metric_name == "time_series":
                # Get hourly event counts
                query = """
                SELECT 
                    CAST((timestamp / 3600) AS INTEGER) * 3600 AS hour,
                    COUNT(*) 
                FROM events 
                WHERE 1=1
                """
                params = []
                
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                query += " GROUP BY hour ORDER BY hour"
                
                cursor.execute(query, params)
                
                result = {}
                for row in cursor.fetchall():
                    hour, count = row
                    result[hour] = count
                
                conn.close()
                return {
                    "metric": "time_series",
                    "data": result
                }
            
            else:
                conn.close()
                logger.warning(f"Unknown metric: {metric_name}")
                return {
                    "metric": metric_name,
                    "error": "Unknown metric"
                }
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return {
                "metric": metric_name,
                "error": str(e)
            }


class WebAPIAnalyticsTracker(AnalyticsTracker):
    """Web API-based implementation of analytics tracker"""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        """
        Initialize the Web API analytics tracker
        
        Args:
            api_url: URL of the analytics API
            api_key: Optional API key for authentication
        """
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
        self.event_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    def _worker(self) -> None:
        """Worker thread for sending events to the API"""
        while True:
            try:
                # Get events from queue (up to 10 at a time)
                events = []
                for _ in range(10):
                    try:
                        event = self.event_queue.get(block=True, timeout=1)
                        events.append(event)
                        self.event_queue.task_done()
                    except queue.Empty:
                        break
                
                if not events:
                    time.sleep(1)
                    continue
                
                # Send events to API
                url = f"{self.api_url}/events/batch"
                headers = self._get_headers()
                
                payload = {
                    "events": [event.to_dict() for event in events]
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                
                if response.status_code not in (200, 201, 202):
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    
                    # Put events back in queue
                    for event in events:
                        self.event_queue.put(event)
            except Exception as e:
                logger.error(f"Error in analytics worker thread: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def track_event(self, event: AnalyticsEvent) -> bool:
        """
        Track an analytics event
        
        Args:
            event: The event to track
            
        Returns:
            True if event was queued successfully, False otherwise
        """
        try:
            self.event_queue.put(event)
            logger.debug(f"Queued event: {event.event_type}")
            return True
        except Exception as e:
            logger.error(f"Error queuing event: {str(e)}")
            return False
    
    def get_events(self, event_type: Optional[str] = None, 
                  user_id: Optional[str] = None, 
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get analytics events from the API
        
        Args:
            event_type: Optional event type filter
            user_id: Optional user ID filter
            start_time: Optional start time filter (timestamp)
            end_time: Optional end time filter (timestamp)
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        try:
            url = f"{self.api_url}/events"
            headers = self._get_headers()
            
            params = {}
            if event_type:
                params["event_type"] = event_type
            if user_id:
                params["user_id"] = user_id
            if start_time:
                params["start_time"] = start_time
            if end_time:
                params["end_time"] = end_time
            params["limit"] = limit
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("events", [])
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error getting events from API: {str(e)}")
            return []
    
    def get_metrics(self, metric_name: str, 
                   user_id: Optional[str] = None,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Get analytics metrics from the API
        
        Args:
            metric_name: Name of the metric to get
            user_id: Optional user ID filter
            start_time: Optional start time filter (timestamp)
            end_time: Optional end time filter (timestamp)
            
        Returns:
            Dictionary with metric data
        """
        try:
            url = f"{self.api_url}/metrics/{metric_name}"
            headers = self._get_headers()
            
            params = {}
            if user_id:
                params["user_id"] = user_id
            if start_time:
                params["start_time"] = start_time
            if end_time:
                params["end_time"] = end_time
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {
                    "metric": metric_name,
                    "error": f"API error: {response.status_code}"
                }
        except Exception as e:
            logger.error(f"Error getting metrics from API: {str(e)}")
            return {
                "metric": metric_name,
                "error": str(e)
            }


class AnalyticsFactory:
    """Factory for creating analytics tracker instances"""
    
    @staticmethod
    def create_tracker(tracker_type: str, **kwargs) -> AnalyticsTracker:
        """
        Create an analytics tracker instance
        
        Args:
            tracker_type: Type of tracker to create (sqlite or web_api)
            **kwargs: Additional arguments to pass to the tracker constructor
            
        Returns:
            An analytics tracker instance
        """
        if tracker_type == "sqlite":
            return SQLiteAnalyticsTracker(**kwargs)
        elif tracker_type == "web_api":
            return WebAPIAnalyticsTracker(**kwargs)
        else:
            raise ValueError(f"Unknown tracker type: {tracker_type}")


# Create a global instance
tracker_type = os.getenv("ANALYTICS_TRACKER_TYPE", "sqlite")
db_path = os.getenv("ANALYTICS_DB_PATH", "analytics.db")
api_url = os.getenv("ANALYTICS_API_URL")
api_key = os.getenv("ANALYTICS_API_KEY")

if tracker_type == "web_api" and api_url:
    analytics = WebAPIAnalyticsTracker(api_url=api_url, api_key=api_key)
    logger.info(f"Using Web API analytics tracker with URL: {api_url}")
else:
    analytics = SQLiteAnalyticsTracker(db_path=db_path)
    logger.info(f"Using SQLite analytics tracker with database: {db_path}")


# Helper functions for common analytics events

def track_user_message(message: str, user_id: Optional[str] = None) -> None:
    """
    Track a user message event
    
    Args:
        message: The user's message
        user_id: Optional user ID
    """
    event = AnalyticsEvent(
        event_type="user_message",
        data={
            "message": message,
            "length": len(message)
        },
        user_id=user_id
    )
    analytics.track_event(event)


def track_assistant_response(response: str, processing_time: float, user_id: Optional[str] = None) -> None:
    """
    Track an assistant response event
    
    Args:
        response: The assistant's response
        processing_time: Time taken to generate the response (in seconds)
        user_id: Optional user ID
    """
    event = AnalyticsEvent(
        event_type="assistant_response",
        data={
            "response": response,
            "length": len(response),
            "processing_time": processing_time
        },
        user_id=user_id
    )
    analytics.track_event(event)


def track_error(error_type: str, error_message: str, user_id: Optional[str] = None) -> None:
    """
    Track an error event
    
    Args:
        error_type: Type of error
        error_message: Error message
        user_id: Optional user ID
    """
    event = AnalyticsEvent(
        event_type="error",
        data={
            "error_type": error_type,
            "error_message": error_message
        },
        user_id=user_id
    )
    analytics.track_event(event)


def track_session_start(user_id: Optional[str] = None) -> None:
    """
    Track a session start event
    
    Args:
        user_id: Optional user ID
    """
    event = AnalyticsEvent(
        event_type="session_start",
        data={},
        user_id=user_id
    )
    analytics.track_event(event)


def track_session_end(duration: float, user_id: Optional[str] = None) -> None:
    """
    Track a session end event
    
    Args:
        duration: Session duration in seconds
        user_id: Optional user ID
    """
    event = AnalyticsEvent(
        event_type="session_end",
        data={
            "duration": duration
        },
        user_id=user_id
    )
    analytics.track_event(event)


def track_feature_usage(feature_name: str, user_id: Optional[str] = None) -> None:
    """
    Track a feature usage event
    
    Args:
        feature_name: Name of the feature
        user_id: Optional user ID
    """
    event = AnalyticsEvent(
        event_type="feature_usage",
        data={
            "feature_name": feature_name
        },
        user_id=user_id
    )
    analytics.track_event(event)