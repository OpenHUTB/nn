import logging
import os
import datetime

class Logger:
    """
    Logger class for PilotNet project.
    Provides logging functionality for different log levels.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._log_dir = 'logs'
        self._ensure_log_dir()
        
        # Get current timestamp for log file
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f'pilotnet_{timestamp}.log'
        self._log_path = os.path.join(self._log_dir, log_filename)
        
        # Configure logging
        self._logger = logging.getLogger('PilotNet')
        self._logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplication
        self._logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler - detailed logs
        file_handler = logging.FileHandler(self._log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self._logger.addHandler(file_handler)
        
        # Console handler - simpler output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self._logger.addHandler(console_handler)
    
    def _ensure_log_dir(self):
        """Ensure the log directory exists."""
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
    
    def debug(self, message, *args, **kwargs):
        """Log a debug message."""
        self._logger.debug(message, *args, **kwargs)
    
    def info(self, message, *args, **kwargs):
        """Log an info message."""
        self._logger.info(message, *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        """Log a warning message."""
        self._logger.warning(message, *args, **kwargs)
    
    def error(self, message, *args, **kwargs):
        """Log an error message."""
        self._logger.error(message, *args, **kwargs)
    
    def critical(self, message, *args, **kwargs):
        """Log a critical message."""
        self._logger.critical(message, *args, **kwargs)
    
    def get_log_path(self):
        """Get the path to the current log file."""
        return self._log_path

# Create a singleton instance
logger = Logger()
