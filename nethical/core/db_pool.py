"""Database Connection Pooling for Performance Optimization.

This module provides connection pooling to reduce database connection overhead
and improve performance for high-throughput scenarios.
"""

import sqlite3
from queue import Queue, Empty
from contextlib import contextmanager
from typing import Optional
from pathlib import Path


class SQLiteConnectionPool:
    """Connection pool for SQLite databases.
    
    This pool maintains a fixed number of database connections that can be
    reused across requests, significantly reducing connection overhead.
    
    Usage:
        pool = SQLiteConnectionPool("database.db", pool_size=10)
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM table")
            results = cursor.fetchall()
    """
    
    def __init__(self, db_path: str, pool_size: int = 10, timeout: float = 30.0):
        """Initialize connection pool.
        
        Args:
            db_path: Path to SQLite database file
            pool_size: Number of connections to maintain in pool
            timeout: Timeout in seconds when waiting for a connection
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self.timeout = timeout
        self.pool: Queue = Queue(maxsize=pool_size)
        self._closed = False
        
        # Ensure database directory exists
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize pool with connections
        for _ in range(pool_size):
            conn = self._create_connection()
            self.pool.put(conn)
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection.
        
        Returns:
            SQLite connection configured for pooling
        """
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,  # Allow connection sharing across threads
            timeout=self.timeout
        )
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        # Set reasonable timeout
        conn.execute(f"PRAGMA busy_timeout={int(self.timeout * 1000)}")
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool (context manager).
        
        Yields:
            SQLite connection from the pool
            
        Raises:
            RuntimeError: If pool is closed or connection timeout occurs
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        conn = None
        try:
            # Get connection from pool with timeout
            conn = self.pool.get(timeout=self.timeout)
            yield conn
        except Empty:
            raise RuntimeError(f"Connection pool timeout after {self.timeout} seconds")
        finally:
            # Return connection to pool
            if conn is not None and not self._closed:
                self.pool.put(conn)
    
    def close(self):
        """Close all connections in the pool.
        
        This should be called when shutting down the application.
        """
        self._closed = True
        
        # Close all connections
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                conn.close()
            except (Empty, Exception):
                # Ignore errors during shutdown
                pass
    
    def __del__(self):
        """Cleanup on garbage collection."""
        if not self._closed:
            self.close()
