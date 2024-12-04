"""
Database connection management for EthoPy.

This module provides centralized database connection handling and schema access.
It maintains a single connection instance while providing convenient access to schemas
across the entire package.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import datajoint as dj
from contextlib import contextmanager

@dataclass
class SchemaManager:
    """Manages database schemas and connections for EthoPy."""
    
    def __init__(self):
        self._experiment = None
        self._stimulus = None
        self._behavior = None
        self._recording = None
        self._mice = None
        self._public_conn = None
        self._private_conn = None
        self._initialized = False

    def initialize(self, schemata: Dict[str, str]) -> None:
        """Initialize database connections and create virtual modules.
        
        This method should be called once at application startup, typically
        by the Logger class initialization.
        """
        if self._initialized:
            return
            
        # Create connections
        self._public_conn = dj.Connection()
        self._private_conn = dj.Connection()
        
        # Create virtual modules
        self._experiment = dj.create_virtual_module('experiment', schemata['experiment'])
        self._stimulus = dj.create_virtual_module('stimulus', schemata['stimulus'])
        self._behavior = dj.create_virtual_module('behavior', schemata['behavior'])
        self._recording = dj.create_virtual_module('recording', schemata['recording'])
        self._mice = dj.create_virtual_module('mice', schemata['mice'])
        
        self._initialized = True

    @property
    def experiment(self):
        """Access to experiment schema"""
        self._ensure_initialized()
        return self._experiment

    @property
    def stimulus(self):
        """Access to stimulus schema"""
        self._ensure_initialized()
        return self._stimulus

    @property
    def behavior(self):
        """Access to behavior schema"""
        self._ensure_initialized()
        return self._behavior

    @property
    def recording(self):
        """Access to recording schema"""
        self._ensure_initialized()
        return self._recording

    @property
    def mice(self):
        """Access to mice schema"""
        self._ensure_initialized()
        return self._mice

    @property
    def public_conn(self):
        """Access to public connection"""
        self._ensure_initialized()
        return self._public_conn

    @property
    def private_conn(self):
        """Access to private connection"""
        self._ensure_initialized()
        return self._private_conn

    def _ensure_initialized(self):
        """Ensure the manager has been initialized"""
        if not self._initialized:
            raise RuntimeError(
                "SchemaManager not initialized. "
                "Make sure Logger is instantiated before accessing schemas."
            )

    @property
    def is_connected(self) -> bool:
        """Check if database connections are active"""
        return (
            self._initialized
            and self._public_conn is not None 
            and self._public_conn.is_connected
            and self._private_conn is not None
            and self._private_conn.is_connected
        )

    def reconnect(self) -> None:
        """Reconnect if connection is lost"""
        if not self.is_connected and self._initialized:
            self._public_conn = dj.Connection()
            self._private_conn = dj.Connection()

    @contextmanager
    def connection_check(self):
        """Context manager for ensuring database connection"""
        if not self.is_connected:
            self.reconnect()
        yield