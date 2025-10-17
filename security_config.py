"""
Konfigurasi Keamanan untuk Aplikasi Forecasting
"""

import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# =============================================================================
# SECURITY CONFIGURATION - Railway Optimized
# =============================================================================
# Konfigurasi keamanan yang dioptimalkan untuk deployment Railway

import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Railway-specific security configuration
RAILWAY_SECURITY_CONFIG = {
    # File handling
    "MAX_FILE_SIZE": 100 * 1024 * 1024,  # 100MB max file size
    "ALLOWED_EXTENSIONS": [".xlsx", ".xls"],
    "MAX_FILE_AGE_HOURS": 2,  # Delete files after 2 hours
    
    # Rate limiting - Railway optimized
    "RATE_LIMIT_REQUESTS": 50,  # Max 50 requests per minute (ditingkatkan untuk Railway)
    "RATE_LIMIT_WINDOW": 60,  # 60 seconds
    
    # User management - Railway optimized
    "MAX_CONCURRENT_USERS": 200,  # Max 200 users (ditingkatkan untuk Railway)
    "USER_SESSION_TIMEOUT": 3600,  # 1 jam timeout (3600 detik)
    
    # Data validation
    "MIN_DATA_ROWS": 5,
    # Maksimal baris data untuk validasi upload - Railway optimized
    "MAX_DATA_ROWS": int(os.environ.get("MAX_UPLOAD_ROWS", "1000000")),  # 1M rows max
    "MAX_MONTHS_DATA": 12,  # Maksimal 12 bulan data
    "REQUIRED_COLUMNS": ['PART_NO', 'MONTH', 'ORIGINAL_SHIPPING_QTY'],
    "VALIDATE_EXCEL_STRUCTURE": True,  # Enable Excel structure validation
    
    # Security features
    "ENCRYPT_TEMP_FILES": True,
    "AUTO_DELETE_AFTER_DOWNLOAD": True,
    "MEMORY_WIPE_AFTER_PROCESSING": True,
    "SECURE_FILE_DELETION": True,
    
    # Network security
    "ENFORCE_HTTPS": True,
    "SECURE_COOKIES": True,
    
    # Railway specific optimizations
    "ENABLE_LAZY_LOADING": True,
    "MEMORY_OPTIMIZATION": True,
    "TIMEOUT_OPTIMIZATION": True,
    
    # Railway resource limits untuk 40 parts, 12 months
    "MAX_PARTS_RAILWAY": 40,
    "MAX_MONTHS_RAILWAY": 12,
    "LSTM_EPOCHS_RAILWAY": 8,
    "LSTM_BATCH_SIZE_RAILWAY": 4,
    "LSTM_LOOKBACK_RAILWAY": 8,
    "XGB_N_ESTIMATORS_RAILWAY": 50,
    "RF_N_ESTIMATORS_RAILWAY": 50,
    "RF_MAX_DEPTH_RAILWAY": 6,
    "PROCESSING_TIMEOUT_RAILWAY": 900  # 15 minutes max
}

# Enhanced Security Headers untuk Railway
RAILWAY_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self';",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=(), payment=(), usb=(), magnetometer=(), gyroscope=()",
    "X-Permitted-Cross-Domain-Policies": "none",
    "X-Download-Options": "noopen",
    "X-DNS-Prefetch-Control": "off",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin"
}

def get_railway_security_config():
    """Get Railway-specific security configuration"""
    return RAILWAY_SECURITY_CONFIG

def get_railway_security_headers():
    """Get Railway-specific security headers"""
    return RAILWAY_SECURITY_HEADERS

def validate_railway_environment():
    """Validate Railway environment variables"""
    required_env_vars = [
        'PORT',
        'RAILWAY_ENVIRONMENT'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing Railway environment variables: {missing_vars}")
        return False
    
    logger.info("✅ Railway environment validation passed")
    return True

def setup_railway_optimizations():
    """Setup Railway-specific optimizations"""
    # Set environment variables for optimization
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('JOBLIB_TEMP_FOLDER', '/tmp')
    
    logger.info("✅ Railway optimizations applied")

def get_railway_memory_limit():
    """Get Railway memory limit"""
    memory_limit = os.environ.get('MEMORY_LIMIT', '512M')
    logger.info(f"Railway memory limit: {memory_limit}")
    return memory_limit

def check_railway_deployment():
    """Check if running on Railway"""
    is_railway = bool(os.environ.get('RAILWAY_ENVIRONMENT'))
    if is_railway:
        logger.info("✅ Running on Railway platform")
    else:
        logger.info("⚠️ Not running on Railway platform")
    return is_railway 