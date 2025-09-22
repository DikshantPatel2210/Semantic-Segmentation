import os
import sys
import logging
import io

# ----------------------------
# Logging format
# ----------------------------
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# ----------------------------
# Create logs directory if it doesn't exist
# ----------------------------
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

# ----------------------------
# Ensure stdout uses UTF-8 encoding (needed for some environments)
# ----------------------------
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ----------------------------
# Configure logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,          # Set logging level to INFO
    format=logging_str,          # Set the log message format
    handlers=[
        logging.FileHandler(log_filepath, encoding='utf-8'),  # Log to file
        logging.StreamHandler(sys.stdout)                     # Log to console
    ]
)

# ----------------------------
# Create a logger instance for the project
# ----------------------------
logger = logging.getLogger("SegmentationLogger")
