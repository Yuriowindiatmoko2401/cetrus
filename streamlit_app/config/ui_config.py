"""UI configuration for DATSR Streamlit app"""

import os

# App metadata
APP_TITLE = "DATSR Super-Resolution"
APP_DESCRIPTION = """
**Reference-based Image Super-Resolution with Deformable Attention Transformer**

Upload a low-resolution image and a high-quality reference image to generate enhanced super-resolution results.
The model uses the reference image to guide the super-resolution process for better quality results.
"""

# Supported image formats
SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp']
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Default processing settings
DEFAULT_SCALE_FACTOR = 4
DEFAULT_MODEL_TYPE = "mse"

# Paths and assets
EXAMPLE_IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'example_images')
ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')

# UI layout settings
SIDEBAR_WIDTH = 300
IMAGE_MAX_HEIGHT = 400
COMPARISON_VIEWER_HEIGHT = 600

# Processing settings
PROGRESS_BAR_FORMAT = "%(percent)s completed [%(elapsed)s elapsed, %(remaining)s remaining]"
STATUS_REFRESH_RATE = 1  # seconds

# Color scheme
PRIMARY_COLOR = "#FF6B6B"
SECONDARY_COLOR = "#4ECDC4"
BACKGROUND_COLOR = "#F8F9FA"
TEXT_COLOR = "#212529"