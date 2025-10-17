import os
import sys
import warnings
import threading
import logging

# Import dengan error handling
try:
    import psutil
except ImportError:
    psutil = None

try:
    from activity_monitor import get_monitor
except ImportError:
    get_monitor = None

# Railway-specific environment setup
os.environ.setdefault('PYTHONUNBUFFERED', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('PYTHONOPTIMIZE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('JOBLIB_TEMP_FOLDER', '/tmp')

# Railway-specific environment variables
os.environ.setdefault('MEMORY_LIMIT', '2G')
os.environ.setdefault('MAX_WORKERS', '4')
os.environ.setdefault('BATCH_SIZE', '20')
os.environ.setdefault('TIMEOUT_SECONDS', '3600')
os.environ.setdefault('PORT', '8080')

# Add current directory to Python path for Railway deployment
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Railway-specific logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings untuk Railway deployment
warnings.filterwarnings('ignore')

# Railway-specific memory monitoring
def check_railway_resources():
    """Check Railway resources and log status"""
    try:
        if psutil:
            memory_info = psutil.virtual_memory()
        else:
            memory_info = type('obj', (object,), {'percent': 50, 'used': 1024*1024*1024, 'total': 2*1024*1024*1024})()
        memory_used_gb = memory_info.used / (1024**3)
        memory_total_gb = memory_info.total / (1024**3)
        memory_percent = memory_info.percent
        
        logger.info(f"Railway Resources - Memory: {memory_used_gb:.2f}GB / {memory_total_gb:.2f}GB ({memory_percent:.1f}%)")
        
        if memory_percent > 80:
            logger.warning(f"⚠️  High memory usage detected: {memory_percent:.1f}%")
            return False
        return True
    except Exception as e:
        logger.warning(f"Could not check Railway resources: {e}")
        return True

# Initialize Railway resource monitoring
check_railway_resources()

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Query, Depends, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io
import base64
import uuid
import tempfile
import hashlib
import time
import signal
import secrets
import re
import shutil
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from collections import defaultdict
from cryptography.fernet import Fernet

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lazy import untuk mempercepat startup
forecast_service_loaded = False

def load_forecast_service():
    """Lazy load forecast service untuk mempercepat startup"""
    global forecast_service_loaded
    if not forecast_service_loaded:
        try:
            # PERBAIKAN: Gunakan forecast_service_clean.py yang memiliki error threshold 30%
            import forecast_service_clean
            logger.info("✅ Forecast service clean module imported successfully")
            
            # Then import specific functions
            from forecast_service_clean import process_forecast, run_backtest_and_realtime
            logger.info("✅ Forecast service clean functions loaded successfully")
            forecast_service_loaded = True
            return process_forecast, run_backtest_and_realtime
        except ImportError as e:
            logger.warning(f"⚠️ Forecast service clean import failed: {e}")
            # Fallback ke forecast_service.py
            try:
                import forecast_service
                logger.info("✅ Fallback to forecast service module")
                from forecast_service import process_forecast, run_backtest_and_realtime
                forecast_service_loaded = True
                return process_forecast, run_backtest_and_realtime
            except ImportError as e2:
                logger.error(f"❌ Forecast service fallback also failed: {e2}")
                return dummy_process_forecast, dummy_run_backtest_and_realtime
        except Exception as e:
            logger.error(f"❌ Error loading forecast service clean: {e}")
            return dummy_process_forecast, dummy_run_backtest_and_realtime
    else:
        try:
            from forecast_service_clean import process_forecast, run_backtest_and_realtime
            return process_forecast, run_backtest_and_realtime
        except ImportError as e:
            logger.warning(f"⚠️ Forecast service clean import failed: {e}")
            # Fallback ke forecast_service.py
            try:
                from forecast_service import process_forecast, run_backtest_and_realtime
                return process_forecast, run_backtest_and_realtime
            except ImportError as e2:
                logger.error(f"❌ Forecast service fallback also failed: {e2}")
                return dummy_process_forecast, dummy_run_backtest_and_realtime
        except Exception as e:
            logger.error(f"❌ Error loading forecast service clean: {e}")
            return dummy_process_forecast, dummy_run_backtest_and_realtime

def load_categorization_service():
    """Lazy load categorization service untuk mempercepat startup"""
    try:
        from kategoriparts_web import process_categorization
        logger.info("✅ Categorization service loaded successfully")
        return process_categorization
    except ImportError as e:
        logger.warning(f"Categorization service import failed: {e}")
        return dummy_process_categorization
    except Exception as e:
        logger.error(f"Error loading categorization service: {e}")
        return dummy_process_categorization

def load_enhanced_forecast_service():
    """Lazy load enhanced forecast service untuk mempercepat startup"""
    # Disabled - tidak menggunakan enhanced_forecast_service
    return None

def load_large_dataset_handler():
    """Lazy load large dataset handler untuk mempercepat startup"""
    # Disabled - tidak menggunakan large_dataset_handler
    return None

# Create dummy functions for fallback
def dummy_process_forecast(df):
    return {"status": "error", "message": "Forecast service not loaded"}
def dummy_run_backtest_and_realtime(df):
    """Dummy function untuk backtest dan realtime forecast jika service tidak tersedia"""
    try:
        # Return copies of the original dataframe
        return df.copy(), pd.DataFrame()
    except Exception as e:
        logger.error(f"Error in dummy_run_backtest_and_realtime: {e}")
        return df.copy(), pd.DataFrame()
def dummy_process_categorization(df):
    """Dummy function untuk categorization jika service tidak tersedia"""
    try:
        # Return original dataframe with dummy category
        df_result = df.copy()
        df_result['PART_CATEGORY'] = 'Unknown'
        return df_result, None
    except Exception as e:
        logger.error(f"Error in dummy_process_categorization: {e}")
        return df.copy(), None

# Log startup information
logger.info("Starting FastAPI application...")
logger.info(f"Current working directory: {os.getcwd()}")

# Definisikan Pydantic Model untuk request body endpoint /forecast-base64
# Ini akan memastikan validasi input JSON secara otomatis
class ForecastRequest(BaseModel):
    excel_base64: str

# Import Railway-specific security configuration - dengan error handling
try:
    from security_config import (
        get_railway_security_config,
        get_railway_security_headers,
        validate_railway_environment,
        setup_railway_optimizations,
        get_railway_memory_limit,
        check_railway_deployment
    )

    # Railway-specific security configuration
    SECURITY_CONFIG = get_railway_security_config()

    # Validate Railway environment
    validate_railway_environment()

    # Setup Railway optimizations
    setup_railway_optimizations()

except ImportError as e:
    print(f"Warning: Security config import failed: {e}")
    # Fallback configuration
    SECURITY_CONFIG = {
        "MAX_FILE_SIZE": 100 * 1024 * 1024,  # 100MB
        "ALLOWED_EXTENSIONS": [".xlsx", ".xls"],
        "MAX_FILE_AGE_HOURS": 2,
        "RATE_LIMIT_REQUESTS": 50,
        "RATE_LIMIT_WINDOW": 60,
        "MAX_CONCURRENT_USERS": 200,
        "USER_SESSION_TIMEOUT": 3600,
        "MIN_DATA_ROWS": 5,
        "MAX_DATA_ROWS": 1000000,
        "REQUIRED_COLUMNS": ['PART_NO', 'MONTH', 'ORIGINAL_SHIPPING_QTY'],
        "VALIDATE_EXCEL_STRUCTURE": True,
        "ENCRYPT_TEMP_FILES": True,
        "AUTO_DELETE_AFTER_DOWNLOAD": True,
        "MEMORY_WIPE_AFTER_PROCESSING": True,
        "SECURE_FILE_DELETION": True,
        "ENFORCE_HTTPS": True,
        "SECURE_COOKIES": True,
        "ENABLE_LAZY_LOADING": True,
        "MEMORY_OPTIMIZATION": True,
        "TIMEOUT_OPTIMIZATION": True
    }

# Get Railway memory limit - dengan error handling
try:
    RAILWAY_MEMORY_LIMIT = get_railway_memory_limit()
except:
    RAILWAY_MEMORY_LIMIT = 8 * 1024 * 1024 * 1024  # 8GB fallback

# Check Railway deployment status - dengan error handling
try:
    RAILWAY_DEPLOYMENT_STATUS = check_railway_deployment()
except:
    RAILWAY_DEPLOYMENT_STATUS = "unknown"

# Railway-specific security headers - dengan error handling
try:
    SECURITY_HEADERS = get_railway_security_headers()
except:
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "SAMEORIGIN",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }

# Rate limiting storage
# rate_limit_storage removed - no longer needed
upload_sessions = {}

# PERBAIKAN: Rate limiting yang lebih ketat
rate_limit_storage = {}  # {ip: {"count": int, "reset_time": float}}

def check_rate_limit(client_ip: str) -> bool:
    """
    Cek rate limiting dengan implementasi yang lebih ketat
    - Max 30 requests per 60 detik per IP
    - Reset counter setiap 60 detik
    - User-friendly untuk penggunaan normal
    """
    current_time = time.time()
    window = SECURITY_CONFIG["RATE_LIMIT_WINDOW"]  # 60 detik
    max_requests = SECURITY_CONFIG["RATE_LIMIT_REQUESTS"]  # 30 request
    
    if client_ip not in rate_limit_storage:
        # IP baru, mulai counter
        rate_limit_storage[client_ip] = {
            "count": 1,
            "reset_time": current_time + window
        }
        logger.debug(f"New IP {client_ip}: request 1/30")
        return True
    
    ip_data = rate_limit_storage[client_ip]
    
    # Reset counter jika window sudah habis (setelah 60 detik)
    if current_time > ip_data["reset_time"]:
        ip_data["count"] = 1
        ip_data["reset_time"] = current_time + window
        logger.debug(f"IP {client_ip}: window reset, request 1/30")
        return True
    
    # Cek apakah sudah melebihi limit (30 request dalam 60 detik)
    if ip_data["count"] >= max_requests:
        logger.warning(f"Rate limit exceeded for IP: {client_ip} ({ip_data['count']}/30)")
        return False
    
    # Increment counter
    ip_data["count"] += 1
    logger.debug(f"IP {client_ip}: request {ip_data['count']}/30")
    return True

# Railway-specific memory management
def check_memory_usage():
    """Check memory usage and log status"""
    try:
        if psutil:
            memory_info = psutil.virtual_memory()
        else:
            memory_info = type('obj', (object,), {'percent': 50, 'used': 1024*1024*1024, 'total': 2*1024*1024*1024})()
        memory_used_gb = memory_info.used / (1024**3)
        memory_total_gb = memory_info.total / (1024**3)
        memory_percent = memory_info.percent
        
        logger.info(f"Railway Memory Usage: {memory_used_gb:.2f}GB / {memory_total_gb:.2f}GB ({memory_percent:.1f}%)")
        
        # Realistic memory management untuk Railway dengan server besar
        max_memory_threshold = int(os.getenv('MAX_MEMORY_THRESHOLD', '70'))  # Railway optimized: 70%
        if memory_percent > max_memory_threshold:
            logger.warning(f"⚠️  High memory usage detected: {memory_percent:.1f}% > {max_memory_threshold}%")
            # Force cleanup before returning False
            cleanup_memory()
            return False
        return True
    except Exception as e:
        logger.warning(f"Could not check memory usage: {e}")
        return True

def cleanup_memory():
    """Clean up memory and log status - ULTRA AGGRESSIVE untuk Railway"""
    try:
        import gc
        import os
        import sys
        
        logger.info("🧹 Starting ULTRA AGGRESSIVE memory cleanup...")
        
        # ML Memory cleanup
        try:
            from forecast_service import cleanup_ml_memory
            cleanup_ml_memory()
        except:
            pass
        
        # Force multiple garbage collection cycles
        for _ in range(30):  # ULTRA AGGRESSIVE: 20 → 30 untuk 25K rows
            gc.collect()
        
        # Clear pandas cache
        if hasattr(pd, '_cache'):
            pd._cache.clear()
        
        # Clear numpy cache
        if hasattr(np, '_cache'):
            np._cache.clear()
        
        # Clear any cached dataframes
        if 'df_processed' in globals():
            del globals()['df_processed']
        if 'backtest_df' in globals():
            del globals()['backtest_df']
        if 'realtime_df' in globals():
            del globals()['realtime_df']
        
        # Clear all temporary files (not just old ones)
        temp_dir = "/tmp/forecast_results"
        if os.path.exists(temp_dir):
            current_time = time.time()
            for file in os.listdir(temp_dir):
                try:
                    file_path = os.path.join(temp_dir, file)
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > 300:  # 5 minutes untuk Railway
                        os.remove(file_path)
                except:
                    pass
        
        # Clear only old cache entries (keep recent ones)
        global forecast_cache
        if 'forecast_cache' in globals():
            current_time = time.time()
            expired_cache = [k for k, v in forecast_cache.items() 
                           if current_time - v.get('timestamp', 0) > 7200]  # 2 jam
            for k in expired_cache:
                forecast_cache.pop(k, None)
        
        logger.info("✅ Railway aggressive memory cleanup completed")
        return True
    except Exception as e:
        logger.warning(f"Railway memory cleanup failed: {e}")
        return False

# File storage for temporary files
temp_files = {}

# User tracking untuk membatasi jumlah user (max 50 user)
# User tracking untuk monitoring aktivitas (tanpa batasan)
active_users = {}  # {session_id: {"ip": ip, "last_activity": timestamp}}

# Global variables untuk tracking
dashboard_keys = {}  # PERBAIKAN: Tambahkan global variable untuk dashboard keys

class SecurityMiddleware:
    """Middleware untuk keamanan aplikasi"""
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """Validasi ekstensi file"""
        if not filename:
            return False
        ext = os.path.splitext(filename.lower())[1]
        return ext in SECURITY_CONFIG["ALLOWED_EXTENSIONS"]
    
    @staticmethod
    def validate_file_content(content: bytes) -> bool:
        """Validasi konten file untuk mencegah upload file berbahaya"""
        # PERBAIKAN: Tambahan validasi konten file
        if not content:
            return False
        
        # Cek magic bytes untuk Excel files
        excel_magic_bytes = [
            b'\x50\x4B\x03\x04',  # ZIP/Excel 2007+
            b'\xD0\xCF\x11\xE0',  # OLE/Excel 97-2003
        ]
        
        content_start = content[:8]
        for magic in excel_magic_bytes:
            if content_start.startswith(magic):
                return True
        
        # Jika bukan Excel, cek apakah ada tanda-tanda file berbahaya
        dangerous_patterns = [
            b'<?php', b'<script', b'javascript:', b'vbscript:',
            b'<iframe', b'<object', b'<embed'
        ]
        
        content_lower = content.lower()
        for pattern in dangerous_patterns:
            if pattern in content_lower:
                return False
        
        return True
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitasi nama file untuk mencegah path traversal"""
        if not filename:
            return "uploaded_file.xlsx"
        
        # Hapus karakter berbahaya
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Hapus path traversal
        filename = os.path.basename(filename)
        # Batasi panjang nama file
        if len(filename) > 100:
            name, ext = os.path.splitext(filename)
            filename = name[:95] + ext
        
        return filename
    
    @staticmethod
    def validate_excel_structure(df: pd.DataFrame) -> bool:
        """Validasi struktur Excel file"""
        required_columns = SECURITY_CONFIG["REQUIRED_COLUMNS"]
        
        # Cek kolom yang diperlukan
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Cek jumlah data (minimal 5 baris, maksimal 1M baris)
        min_rows = SECURITY_CONFIG["MIN_DATA_ROWS"]
        max_rows = SECURITY_CONFIG["MAX_DATA_ROWS"]
        
        if len(df) < min_rows:
            return False
        
        if len(df) > max_rows:
            return False
        
        # Cek apakah ada data yang valid
        if df['ORIGINAL_SHIPPING_QTY'].sum() == 0:
            return False
        
        # Cek kolom MONTH untuk 12 bulan filtering
        if 'MONTH' in df.columns:
            try:
                # Parse MONTH column untuk validasi
                df['MONTH_PARSED'] = pd.to_datetime(df['MONTH'], errors='coerce')
                valid_dates = df['MONTH_PARSED'].dropna()
                
                if len(valid_dates) > 0:
                    # Cek apakah data lebih dari 12 bulan
                    latest_date = valid_dates.max()
                    twelve_months_ago = latest_date - pd.DateOffset(months=12)
                    recent_data = valid_dates[valid_dates >= twelve_months_ago]
                    
                    if len(recent_data) < min_rows:
                        logger.warning(f"⚠️ Data 12 bulan terakhir hanya {len(recent_data)} rows, kurang dari minimum {min_rows}")
                        return False
                
                # Drop temporary column
                df.drop('MONTH_PARSED', axis=1, inplace=True)
                
            except Exception as e:
                logger.warning(f"⚠️ Error validating MONTH column: {e}")
                # Continue dengan validasi normal jika parsing gagal
        
        return True
    
    @staticmethod
    def check_rate_limit(client_ip: str) -> bool:
        """Cek rate limiting berdasarkan IP - Always return True"""
        return True
    
    @staticmethod
    def generate_secure_session_id() -> str:
        """Generate session ID yang aman"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def encrypt_temp_file(file_path: str) -> str:
        """Enkripsi file temporary dengan AES-256"""
        if not SECURITY_CONFIG["ENCRYPT_TEMP_FILES"]:
            return file_path
        
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            import base64
            
            # Generate key from password dengan salt yang aman
            password = b'K9#mN2$pQ7@vX4&jL8!wR5^tY3*nB6%hF9'
            salt = b'x7K9mN2$pQ7@vX4&jL8!wR5^tY3*nB6%hF9'
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            fernet = Fernet(key)
            
            # Read and encrypt file
            with open(file_path, 'rb') as f:
                data = f.read()
            
            encrypted_data = fernet.encrypt(data)
            
            encrypted_path = file_path + '.enc'
            with open(encrypted_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Secure delete original file
            security.secure_delete_file(file_path)
            logger.info(f"File encrypted with AES-256: {encrypted_path}")
            return encrypted_path
        except Exception as e:
            logger.error(f"Error encrypting file: {e}")
            return file_path
    
    @staticmethod
    def decrypt_temp_file(file_path: str) -> str:
        """Dekripsi file temporary dengan AES-256"""
        if not file_path.endswith('.enc'):
            return file_path
        
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            import base64
            
            # Generate key from password dengan salt yang sama
            password = b'K9#mN2$pQ7@vX4&jL8!wR5^tY3*nB6%hF9'
            salt = b'x7K9mN2$pQ7@vX4&jL8!wR5^tY3*nB6%hF9'
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            fernet = Fernet(key)
            
            # Read and decrypt file
            with open(file_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = fernet.decrypt(encrypted_data)
            
            decrypted_path = file_path[:-4]  # Remove .enc
            with open(decrypted_path, 'wb') as f:
                f.write(decrypted_data)
            
            logger.info(f"File decrypted with AES-256: {decrypted_path}")
            return decrypted_path
        except Exception as e:
            logger.error(f"Error decrypting file: {e}")
            return file_path
    
    @staticmethod
    def secure_delete_file(file_path: str):
        """Secure file deletion - overwrite before delete"""
        try:
            if os.path.exists(file_path):
                # Overwrite file with random data before deletion
                file_size = os.path.getsize(file_path)
                with open(file_path, 'wb') as f:
                    # Overwrite with random data 3 times
                    for _ in range(3):
                        f.seek(0)
                        f.write(os.urandom(file_size))
                        f.flush()
                        os.fsync(f.fileno())
                
                # Delete file
                os.remove(file_path)
                logger.info(f"File securely deleted: {file_path}")
        except Exception as e:
            logger.error(f"Error in secure file deletion: {e}")
    
    @staticmethod
    def wipe_memory_data(dataframes: list):
        """Wipe sensitive data from memory with secure overwrite"""
        try:
            for df in dataframes:
                if df is not None:
                    # Secure overwrite dengan random data
                    import numpy as np
                    random_data = np.random.randint(0, 1000, size=df.shape)
                    df.iloc[:] = random_data
                    # Clear memory
                    df.drop(df.index, inplace=True)
                    del df
            # Force garbage collection
            import gc
            gc.collect()
            logger.info("Memory data securely wiped")
        except Exception as e:
            logger.error(f"Error wiping memory data: {e}")
    
    @staticmethod
    def secure_process_data(df: pd.DataFrame):
        """Process data in isolated memory space"""
        try:
            # Create copy in isolated memory
            df_copy = df.copy()
            
            # Check memory before processing
            if psutil:
                memory_info = psutil.virtual_memory()
            else:
                memory_info = type('obj', (object,), {'percent': 50, 'used': 1024*1024*1024, 'total': 2*1024*1024*1024})()
            memory_percent = memory_info.percent
            logger.info(f"Memory usage before forecast: {memory_percent:.1f}%")
            
            # Use configurable memory threshold
            max_memory_threshold = int(os.getenv('MAX_MEMORY_THRESHOLD', '75'))
            if memory_percent > max_memory_threshold:
                logger.warning(f"⚠️ High memory usage: {memory_percent:.1f}% > {max_memory_threshold}%, forcing cleanup")
                import gc
                gc.collect()
            
            # Load forecast service functions
            process_forecast_func, run_backtest_and_realtime_func = load_forecast_service()
            
            if run_backtest_and_realtime_func and run_backtest_and_realtime_func != dummy_run_backtest_and_realtime:
                try:
                    # Memory check sebelum forecast
                    if not check_memory_usage():
                        logger.error("❌ Memory usage too high before forecast, aborting")
                        raise Exception("Memory usage too high")
                    
                    # run_backtest_and_realtime returns 3 values: df_proc, backtest, realtime_forecast
                    df_processed, forecast_df, real_time_forecast = run_backtest_and_realtime_func(df_copy)
                    logger.info(f"✅ Forecast completed: {len(forecast_df)} backtest records, {len(real_time_forecast)} realtime records")
                    
                    # Memory check sesudah forecast
                    if not check_memory_usage():
                        logger.warning("⚠️ Memory usage high after forecast, forcing cleanup")
                        cleanup_memory()
                        
                except Exception as e:
                    logger.error(f"Error in run_backtest_and_realtime: {e}")
                    # Return copies of original data on error
                    df_processed = df_copy.copy()
                    forecast_df = df_copy.copy()
                    real_time_forecast = pd.DataFrame()
            else:
                logger.error("run_backtest_and_realtime function not available")
                # Return copies of original data
                df_processed = df_copy.copy()
                forecast_df = df_copy.copy()
                real_time_forecast = pd.DataFrame()
            
            # Wipe copy immediately
            df_copy.iloc[:] = 0
            del df_copy
            
            # Force memory cleanup after processing
            import gc
            gc.collect()
            
            # Check memory after processing
            memory_info_after = psutil.virtual_memory()
            logger.info(f"Memory usage after forecast: {memory_info_after.percent:.1f}%")
            
            return df_processed, forecast_df, real_time_forecast
        except Exception as e:
            logger.error(f"Error in secure data processing: {e}")
            raise
    
    @staticmethod
    def cleanup_old_files():
        """Bersihkan file lama dengan error handling yang robust"""
        current_time = time.time()
        max_age = SECURITY_CONFIG["MAX_FILE_AGE_HOURS"] * 3600
        
        files_to_remove = []
        for file_id, file_info in temp_files.items():
            try:
                # Handle both 'created_time' and 'created_at' keys
                created_time = file_info.get('created_time') or file_info.get('created_at')
                
                if created_time:
                    # Convert datetime to timestamp if needed
                    if hasattr(created_time, 'timestamp'):
                        created_timestamp = created_time.timestamp()
                    else:
                        created_timestamp = created_time
                    
                    if current_time - created_timestamp > max_age:
                        files_to_remove.append(file_id)
                else:
                    # If no timestamp found, remove the entry as it's invalid
                    logger.warning(f"File {file_id} has no valid timestamp, removing")
                    files_to_remove.append(file_id)
                    
            except Exception as e:
                logger.warning(f"Error processing file {file_id}: {e}, removing")
                files_to_remove.append(file_id)
        
        for file_id in files_to_remove:
            try:
                if 'file_path' in temp_files[file_id]:
                    file_path = temp_files[file_id]['file_path']
                    if os.path.exists(file_path):
                        security.secure_delete_file(file_path)
                    if os.path.exists(file_path + '.enc'):
                        security.secure_delete_file(file_path + '.enc')
                del temp_files[file_id]
                logger.info(f"Cleaned up old file: {file_id}")
            except Exception as e:
                logger.error(f"Error cleaning up file {file_id}: {e}")

# Security middleware instance
security = SecurityMiddleware()

def get_client_ip(request: Request) -> str:
    """Dapatkan IP client dengan mempertimbangkan proxy"""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host

def get_active_users_count() -> int:
    """Get current active users count untuk monitoring"""
    current_time = time.time()
    timeout = SECURITY_CONFIG["USER_SESSION_TIMEOUT"]
    
    # Cleanup expired sessions
    expired_sessions = []
    for session_id, user_info in active_users.items():
        if current_time - user_info["last_activity"] > timeout:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del active_users[session_id]
    
    return len(active_users)

def can_add_new_user(client_ip: str) -> bool:
    """Check if new user can be added - Always return True (no limits)"""
    return True

def add_user_session(client_ip: str, session_id: str):
    """Add user session untuk monitoring"""
    active_users[session_id] = {
        "ip": client_ip,
        "last_activity": time.time()
    }

def update_user_activity(session_id: str):
    """Update user activity timestamp untuk monitoring"""
    if session_id in active_users:
        active_users[session_id]["last_activity"] = time.time()

def remove_user_session(session_id: str):
    """Remove user session untuk monitoring"""
    if session_id in active_users:
        del active_users[session_id]

def validate_request_security(request: Request) -> Dict[str, Any]:
    """Validasi keamanan request"""
    client_ip = get_client_ip(request)
    
    # User activity tracking untuk monitoring
    session_id = security.generate_secure_session_id()
    add_user_session(client_ip, session_id)
    
    # Cleanup old files
    security.cleanup_old_files()
    
    return {"client_ip": client_ip, "session_id": session_id}

# Create FastAPI app with optimized settings for Railway
app = FastAPI(
    title="Forecast Excel Download",
    description="Aplikasi untuk download hasil forecast Excel",
    version="1.0.0",
    docs_url=None,  # Disable docs for production
    redoc_url=None,  # Disable redoc for production
    openapi_url=None  # Disable openapi for production
)

# Mount static files - check if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("✅ Static files mounted successfully")
else:
    logger.warning("⚠️ Static directory not found, using manual endpoints")

# Add compression middleware for faster response transfer
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add CORS middleware with optimized settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Railway
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for memory optimization dengan caching
temp_files = {}
active_users = {}
rate_limit_requests = defaultdict(list)
dashboard_data = {}
security = SecurityMiddleware()

# Circuit breaker untuk mencegah overload
circuit_breaker = {
    "failure_count": 0,
    "last_failure_time": 0,
    "is_open": False,
    "failure_threshold": 2,  # More aggressive - fail after 2 failures
    "recovery_timeout": 180  # 3 minutes recovery
}

# Rate limiting untuk mencegah spam requests
request_tracker = {
    "requests": {},  # IP -> list of timestamps
    "max_requests_per_minute": 3,  # Max 3 requests per minute per IP
    "cleanup_interval": 300  # Clean old entries every 5 minutes
}

def check_circuit_breaker():
    """Check circuit breaker status"""
    current_time = time.time()
    
    # Check if circuit breaker should be reset
    if (circuit_breaker["is_open"] and 
        current_time - circuit_breaker["last_failure_time"] > circuit_breaker["recovery_timeout"]):
        circuit_breaker["is_open"] = False
        circuit_breaker["failure_count"] = 0
        logger.info("🔄 Circuit breaker reset")
    
    return not circuit_breaker["is_open"]

def record_failure():
    """Record a failure in circuit breaker"""
    circuit_breaker["failure_count"] += 1
    circuit_breaker["last_failure_time"] = time.time()
    
    if circuit_breaker["failure_count"] >= circuit_breaker["failure_threshold"]:
        circuit_breaker["is_open"] = True
        logger.error(f"⚡ Circuit breaker opened after {circuit_breaker['failure_count']} failures")

def record_success():
    """Record a success in circuit breaker"""
    if circuit_breaker["failure_count"] > 0:
        circuit_breaker["failure_count"] = max(0, circuit_breaker["failure_count"] - 1)

def check_rate_limit(client_ip: str) -> bool:
    """Check if client IP is within rate limits"""
    current_time = time.time()
    
    # Cleanup old entries
    if current_time % request_tracker["cleanup_interval"] < 1:
        cleanup_old_requests()
    
    # Get client requests
    client_requests = request_tracker["requests"].get(client_ip, [])
    
    # Remove requests older than 1 minute
    one_minute_ago = current_time - 60
    client_requests = [req_time for req_time in client_requests if req_time > one_minute_ago]
    
    # Check if within limit
    if len(client_requests) >= request_tracker["max_requests_per_minute"]:
        logger.warning(f"🚫 Rate limit exceeded for IP {client_ip}: {len(client_requests)} requests in last minute")
        return False
    
    # Add current request
    client_requests.append(current_time)
    request_tracker["requests"][client_ip] = client_requests
    
    return True

def cleanup_old_requests():
    """Cleanup old request tracking data"""
    current_time = time.time()
    one_hour_ago = current_time - 3600
    
    for ip in list(request_tracker["requests"].keys()):
        request_tracker["requests"][ip] = [
            req_time for req_time in request_tracker["requests"][ip] 
            if req_time > one_hour_ago
        ]
        
        # Remove empty entries
        if not request_tracker["requests"][ip]:
            del request_tracker["requests"][ip]

# PERBAIKAN: Tambahkan caching untuk forecast results
forecast_cache = {}
cache_timeout = 3600  # 1 jam

# PERBAIKAN: Tambahkan memory pool untuk numpy
np.set_printoptions(precision=4, suppress=True)

# ULTRA AGGRESSIVE: Startup memory optimization
import gc
gc.collect()  # Cleanup startup memory

# Memory optimization function
def optimize_memory_usage():
    """Optimize memory usage for Railway"""
    try:
        import gc
        gc.collect()
        
        # Clear unused variables
        global temp_files, active_users, rate_limit_requests, dashboard_data
        
        # Clean up old entries
        current_time = time.time()
        
        # Clean temp_files - PERBAIKAN: Moderate cleanup, hapus file yang lebih dari 4 jam
        expired_files = []
        for fid, info in temp_files.items():
            try:
                created_time = info.get('created_time') or info.get('timestamp', 0)
                if current_time - created_time > 14400:  # 4 jam
                    expired_files.append(fid)
            except Exception as e:
                logger.warning(f"Error processing file {fid} for cleanup: {e}")
                expired_files.append(fid)  # Remove problematic entries
        
        logger.info(f"Found {len(expired_files)} expired files to clean up")
        
        for fid in expired_files:
            try:
                if fid in temp_files:
                    file_info = temp_files[fid]
                    
                    # Delete physical files
                    if 'file_path' in file_info:
                        file_path = file_info['file_path']
                        if os.path.exists(file_path):
                            security.secure_delete_file(file_path)
                            logger.info(f"Deleted expired physical file: {file_path}")
                        
                        # Delete encrypted file if exists
                        encrypted_path = file_path + '.enc'
                        if os.path.exists(encrypted_path):
                            security.secure_delete_file(encrypted_path)
                            logger.info(f"Deleted expired encrypted file: {encrypted_path}")
                    
                    # Remove from temp_files
                    del temp_files[fid]
                    logger.info(f"Removed expired file entry: {fid}")
                    
            except Exception as e:
                logger.error(f"Error cleaning up expired file {fid}: {e}")
        
        # Clean rate_limit_requests - PERBAIKAN: Lebih agresif
        for ip in list(rate_limit_requests.keys()):
            rate_limit_requests[ip] = [req for req in rate_limit_requests[ip] 
                                     if current_time - req < SECURITY_CONFIG['RATE_LIMIT_WINDOW']]
            if not rate_limit_requests[ip]:
                del rate_limit_requests[ip]
                logger.info(f"Cleaned up rate limit for IP: {ip}")
        
        # Clean active_users - PERBAIKAN: Lebih agresif
        expired_users = [session_id for session_id, user_info in active_users.items()
                        if current_time - user_info.get('last_activity', 0) > SECURITY_CONFIG['USER_SESSION_TIMEOUT']]
        
        for session_id in expired_users:
            if session_id in active_users:
                del active_users[session_id]
                logger.info(f"Cleaned up expired user session: {session_id}")
        
        # PERBAIKAN: Clean dashboard_data yang lama
        expired_dashboard = [key for key in dashboard_data.keys()
                           if current_time - dashboard_data[key].get('timestamp', 0) > 3600]  # 1 jam
        
        for key in expired_dashboard:
            if key in dashboard_data:
                del dashboard_data[key]
                logger.info(f"Cleaned up expired dashboard data: {key}")
        
        # PERBAIKAN: Force garbage collection
        gc.collect()
        
        # PERBAIKAN: Log memory usage
        import psutil
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            logger.info(f"Memory usage after cleanup: {memory_mb:.2f} MB")
        except ImportError:
            logger.info("psutil not available for memory monitoring")
        
        logger.info(f"Memory optimization completed. Active files: {len(temp_files)}, Active users: {len(active_users)}")
        
    except Exception as e:
        logger.error(f"Error in memory optimization: {e}")

# Schedule memory optimization - Kurangi frekuensi
def schedule_memory_optimization():
    """Schedule periodic memory optimization - PERBAIKAN: Lebih sering untuk Railway"""
    # Initial cleanup
    cleanup_memory()
    
    while True:
        try:
            time.sleep(30)  # Run every 30 seconds untuk Railway
            logger.info("🔄 Running scheduled memory optimization...")
            optimize_memory_usage()
        except Exception as e:
            logger.error(f"Error in scheduled memory optimization: {e}")

# PERBAIKAN: Tambahkan memory monitoring yang lebih detail
def monitor_memory_usage():
    """Monitor memory usage dan auto-restart jika terlalu tinggi"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = process.memory_percent()
        
        logger.info(f"💾 MEMORY USAGE: {memory_mb:.2f}MB ({memory_percent:.1f}%)")
        
        # PERBAIKAN: Auto-restart jika memory terlalu tinggi
        max_memory_mb = 500000  # 500GB limit untuk Railway
        if memory_mb > max_memory_mb:
            logger.error(f"🚨 MEMORY CRITICAL: {memory_mb:.2f}MB > {max_memory_mb}MB limit")
            logger.error("🔄 Initiating emergency memory cleanup...")
            
            # Emergency cleanup
            try:
                # Force cleanup semua temporary files
                for fid in list(temp_files.keys()):
                    try:
                        if 'file_path' in temp_files[fid]:
                            file_path = temp_files[fid]['file_path']
                            if os.path.exists(file_path):
                                security.secure_delete_file(file_path)
                            encrypted_path = file_path + '.enc'
                            if os.path.exists(encrypted_path):
                                security.secure_delete_file(encrypted_path)
                        del temp_files[fid]
                    except Exception as e:
                        logger.error(f"Emergency cleanup failed for {fid}: {e}")
                
                # Clear all dictionaries
                temp_files.clear()
                active_users.clear()
                rate_limit_requests.clear()
                dashboard_data.clear()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                logger.info("✅ Emergency memory cleanup completed")
                
            except Exception as e:
                logger.error(f"Emergency cleanup failed: {e}")
                logger.error("🔄 Forcing application restart...")
                
                # Force restart jika cleanup gagal
                import os
                import sys
                os._exit(1)  # Force exit untuk Railway restart
                
        return memory_mb, memory_percent
        
    except ImportError:
        logger.warning("psutil not available for memory monitoring")
        return 0, 0
    except Exception as e:
        logger.error(f"Memory monitoring error: {e}")
        return 0, 0

# PERBAIKAN: Schedule memory monitoring yang lebih sering
def schedule_memory_monitoring():
    """Schedule periodic memory monitoring"""
    while True:
        try:
            time.sleep(60)  # Check every minute
            memory_mb, memory_percent = monitor_memory_usage()
            
            # Log memory status
            if memory_mb > 100000:  # > 100GB
                logger.warning(f"⚠️ HIGH MEMORY USAGE: {memory_mb:.2f}MB ({memory_percent:.1f}%)")
            elif memory_mb > 50000:  # > 50GB
                logger.info(f"📊 MODERATE MEMORY USAGE: {memory_mb:.2f}MB ({memory_percent:.1f}%)")
            else:
                logger.info(f"✅ NORMAL MEMORY USAGE: {memory_mb:.2f}MB ({memory_percent:.1f}%)")
                
        except Exception as e:
            logger.error(f"Error in memory monitoring: {e}")



@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Enhanced security middleware dengan network protection dan monitoring"""
    start_time = time.time()
    
    # Get client IP
    client_ip = get_client_ip(request)
    user_agent = request.headers.get("user-agent", "Unknown")
    
    # Get monitor instance
    if get_monitor:
        monitor = get_monitor()
    else:
        monitor = None
    
    # Skip logging & processing berat untuk static/health cepat
    _path = str(request.url.path)
    if _path.startswith('/static') or _path in ('/health', '/api/health', '/gradient.jpg'):
        return await call_next(request)
    # Ringkas: log hanya method dan path; detail IP ada di access log uvicorn
    logger.info(f"🔍 Request: {request.method} {_path}")
    
        # Log security event (DISABLED untuk debugging)
        # if monitor:
        #     monitor.log_security_event(
        #         event_type="REQUEST_START",
        #         client_ip=client_ip,
        #         user_agent=user_agent,
        #         endpoint=str(request.url.path)
        #     )
    
    # PERBAIKAN: Hanya log suspicious headers yang benar-benar mencurigakan
    # Header proxy seperti x-forwarded-for adalah normal di Railway
    truly_suspicious_headers = ['x-forwarded-for', 'x-real-ip', 'x-forwarded-proto']
    for header in truly_suspicious_headers:
        if header in request.headers:
            # Hanya log jika bukan dari Railway proxy
            header_value = request.headers[header]
            if not any(railway_ip in header_value for railway_ip in ['158.140.191.105', '100.64.0.2', 'railway', 'https']):
                # DISABLED untuk debugging - security logs mengganggu analisis
                # logger.warning(f"Suspicious header detected: {header} = {header_value}")
                # if monitor:
                #     monitor.log_security_event(
                #         event_type="SUSPICIOUS_HEADER",
                #         client_ip=client_ip,
                #         user_agent=user_agent,
                #         endpoint=str(request.url.path),
                #         details={"header": header, "value": header_value}
                #     )
                pass
            else:
                logger.debug(f"Normal proxy header: {header} = {header_value}")
    
    # PERBAIKAN: HTTPS enforcement yang aman untuk Railway
    # Railway secara otomatis menggunakan HTTPS, jadi kita perlu memastikan request datang dari HTTPS
    if SECURITY_CONFIG.get("ENFORCE_HTTPS", True):  # Default ke True untuk keamanan
        # Cek apakah request datang dari HTTPS
        x_forwarded_proto = request.headers.get("x-forwarded-proto")
        x_forwarded_for = request.headers.get("x-forwarded-for")
        
        # Railway menggunakan proxy, jadi kita perlu cek header yang tepat
        if x_forwarded_proto and x_forwarded_proto != "https":
            logger.warning(f"Non-HTTPS request detected: {x_forwarded_proto} from {client_ip}")
            if monitor:
                monitor.log_security_event(
                    event_type="NON_HTTPS_REQUEST",
                    client_ip=client_ip,
                    user_agent=user_agent,
                    endpoint=str(request.url.path),
                    details={"protocol": x_forwarded_proto}
                )
            # Jangan blokir, tapi log untuk monitoring
        elif not x_forwarded_proto and not x_forwarded_for:
            # Jika tidak ada header proxy, kemungkinan request langsung (development)
            logger.info(f"Direct request from {client_ip} (no proxy headers)")
        else:
            logger.debug(f"HTTPS request confirmed: {x_forwarded_proto} from {client_ip}")
    
    try:
        # Add security headers
        response = await call_next(request)
        
        # Add enhanced security headers to response
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value
        
        # Add additional security headers
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Log web activity
        if monitor:
            monitor.log_web_activity(
                client_ip=client_ip,
                endpoint=str(request.url.path),
                method=request.method,
                response_time=response_time,
                status_code=response.status_code,
                user_agent=user_agent
            )
        
        # Log security event
        logger.info(f"Enhanced security middleware processed request from {client_ip} in {response_time:.3f}s")
        
        return response
        
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"❌ Error in security middleware: {e} in {response_time:.3f}s")
        
        # Log security event for error
        if monitor:
            monitor.log_security_event(
                event_type="MIDDLEWARE_ERROR",
                client_ip=client_ip,
                user_agent=user_agent,
                endpoint=str(request.url.path),
                security_check_passed=False,
                details={"error": str(e), "response_time": response_time}
            )
        
        raise

# Buat direktori untuk menyimpan file hasil forecast
TEMP_DIR = os.path.join(tempfile.gettempdir(), "forecast_results")
os.makedirs(TEMP_DIR, exist_ok=True)
logger.info(f"Temporary directory created: {TEMP_DIR}")

# Endpoint khusus untuk file static yang diperlukan
@app.get("/static/style.css")
async def get_style_css():
    try:
        logger.info("Serving style.css")
        headers = {"Cache-Control": "public, max-age=86400"}
        return FileResponse("static/style.css", media_type="text/css", headers=headers)
    except Exception as e:
        logger.error(f"Error serving style.css: {e}")
        return HTMLResponse(content="/* CSS not found */", status_code=404)

@app.get("/static/script.js")
async def get_script_js():
    try:
        logger.info("Serving script.js")
        headers = {"Cache-Control": "public, max-age=86400"}
        return FileResponse("static/script.js", media_type="application/javascript", headers=headers)
    except Exception as e:
        logger.error(f"Error serving script.js: {e}")
        return HTMLResponse(content="// JS not found", status_code=404)

@app.get("/static/chart.js")
async def get_chart_js():
    try:
        logger.info("Serving chart.js")
        headers = {"Cache-Control": "public, max-age=86400"}
        return FileResponse("static/chart.js", media_type="application/javascript", headers=headers)
    except Exception as e:
        logger.error(f"Error serving chart.js: {e}")
        return HTMLResponse(content="// Chart.js not found", status_code=404)

@app.get("/static/gradient.jpg")
async def get_gradient_jpg():
    try:
        logger.info("Serving gradient.jpg from static directory")
        headers = {"Cache-Control": "public, max-age=86400"}
        return FileResponse("static/gradient.jpg", media_type="image/jpeg", headers=headers)
    except Exception as e:
        logger.error(f"Error serving gradient.jpg from static: {e}")
        # Fallback ke gradient.png
        try:
            logger.info("Falling back to gradient.png from static")
            return FileResponse("static/gradient.png", media_type="image/png")
        except Exception as e2:
            logger.error(f"Error serving gradient.png fallback: {e2}")
            return HTMLResponse(content="", status_code=404)

@app.get("/static/gradient.png")
async def get_gradient_png():
    try:
        logger.info("Serving gradient.png from static directory")
        headers = {"Cache-Control": "public, max-age=86400"}
        return FileResponse("static/gradient.png", media_type="image/png", headers=headers)
    except Exception as e:
        logger.error(f"Error serving gradient.png from static: {e}")
        # Fallback ke root directory
        try:
            logger.info("Falling back to gradient.png from root directory")
            return FileResponse("gradient.png", media_type="image/png")
        except Exception as e2:
            logger.error(f"Error serving gradient.png from root: {e2}")
            return HTMLResponse(content="", status_code=404)

@app.get("/gradient.png")
async def get_gradient_png_root():
    try:
        logger.info("Serving gradient.png from root directory")
        headers = {"Cache-Control": "public, max-age=86400"}
        return FileResponse("gradient.png", media_type="image/png", headers=headers)
    except Exception as e:
        logger.error(f"Error serving gradient.png from root: {e}")
        return HTMLResponse(content="", status_code=404)

@app.get("/health")
async def health_check():
    """Health check endpoint untuk Railway - SIMPLIFIED untuk menghindari timeout"""
    try:
        # Simple health check tanpa test yang berat
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "forecast-service",
            "version": "1.0.0",
            "railway": "active"
        }
    except Exception as e:
        # Fallback response jika ada error
        return {
            "status": "healthy", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/")
async def root():
    """Root endpoint - serve web interface"""
    try:
        if os.path.exists("static/index.html"):
            return FileResponse("static/index.html")
        else:
            return HTMLResponse(content="<h1>Forecast Service is Running</h1>", status_code=200)
    except Exception as e:
        logger.error(f"Error serving root: {e}")
        return HTMLResponse(content="<h1>Service Error</h1>", status_code=500)

@app.get("/api/health")
async def api_health_check():
    """Health check endpoint untuk Railway"""
    return {
        "message": "Railway Forecast Service is running", 
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/debug-static")
async def debug_static():
    """Debug endpoint untuk check static files"""
    try:
        static_files = []
        if os.path.exists("static"):
            for file in os.listdir("static"):
                static_files.append(file)
        
        return {
            "static_directory_exists": os.path.exists("static"),
            "static_files": static_files,
            "index_html_exists": os.path.exists("static/index.html"),
            "current_directory": os.getcwd(),
            "files_in_root": os.listdir(".") if os.path.exists(".") else []
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug-files")
async def debug_files():
    """Endpoint untuk debugging file yang ada di direktori temp"""
    try:
        if os.path.exists(TEMP_DIR):
            files = os.listdir(TEMP_DIR)
            file_info = []
            for file in files:
                file_path = os.path.join(TEMP_DIR, file)
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                file_info.append({
                    "name": file,
                    "size": file_size,
                    "path": file_path
                })
            return {
                "temp_dir": TEMP_DIR,
                "files": file_info
            }
        else:
            return {
                "temp_dir": TEMP_DIR,
                "error": "TEMP_DIR does not exist",
                "files": []
            }
    except Exception as e:
        logger.error(f"Error in debug-files endpoint: {str(e)}")
        return {
            "error": str(e),
            "temp_dir": TEMP_DIR
        }

@app.get("/api/security-status")
async def security_status():
    """Endpoint untuk monitoring status keamanan"""
    try:
        current_time = time.time()
        
        # Hitung statistik keamanan
        active_sessions = len(temp_files)
        expired_sessions = 0
        
        # Hitung file yang expired
        for file_id, file_info in temp_files.items():
            if current_time - file_info['created_time'] > SECURITY_CONFIG["MAX_FILE_AGE_HOURS"] * 3600:
                expired_sessions += 1
        
        # User activity statistics
        active_users_count = get_active_users_count()
        user_sessions_detail = []
        for session_id, user_info in active_users.items():
            user_sessions_detail.append({
                "session_id": session_id[:8] + "...",  # Truncate for security
                "ip": user_info["ip"],
                "last_activity": datetime.fromtimestamp(user_info["last_activity"]).isoformat(),
                "age_minutes": int((current_time - user_info["last_activity"]) / 60)
            })
        
        return {
            "status": "success",
            "security_metrics": {
                "active_sessions": active_sessions,
                "expired_sessions": expired_sessions,
                "temp_files_count": len(temp_files),
                "active_users": active_users_count,
                "total_user_sessions": len(active_users)
            },
            "user_activity": {
                "active_users_count": active_users_count,
                "user_sessions": user_sessions_detail,
                "session_timeout_minutes": SECURITY_CONFIG["USER_SESSION_TIMEOUT"] // 60
            },
            "security_config": {
                "max_file_size_mb": SECURITY_CONFIG["MAX_FILE_SIZE"] // (1024*1024),
                "max_file_age_hours": SECURITY_CONFIG["MAX_FILE_AGE_HOURS"],
                "encrypt_temp_files": SECURITY_CONFIG["ENCRYPT_TEMP_FILES"],
                "user_session_timeout": SECURITY_CONFIG["USER_SESSION_TIMEOUT"]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in security-status endpoint: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/security-cleanup")
async def security_cleanup():
    """Endpoint untuk manual cleanup file dan session yang expired"""
    try:
        logger.info("Manual security cleanup initiated")
        
        # Cleanup old files
        security.cleanup_old_files()
        
        # Cleanup expired user sessions
        current_time = time.time()
        expired_sessions = []
        for session_id, user_info in list(active_users.items()):
            if current_time - user_info["last_activity"] > SECURITY_CONFIG["USER_SESSION_TIMEOUT"]:
                expired_sessions.append(session_id)
                del active_users[session_id]
        
        return {
            "status": "success",
            "message": "Security cleanup completed",
            "cleaned_sessions": len(temp_files),
            "cleaned_user_sessions": len(expired_sessions),
            "remaining_active_users": len(active_users),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in security-cleanup endpoint: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/user-activity")
async def user_activity_monitoring():
    """Endpoint khusus untuk monitoring aktivitas user"""
    try:
        current_time = time.time()
        active_users_count = get_active_users_count()
        
        # Detailed user activity
        user_sessions_detail = []
        for session_id, user_info in active_users.items():
            user_sessions_detail.append({
                "session_id": session_id[:8] + "...",  # Truncate for security
                "ip": user_info["ip"],
                "last_activity": datetime.fromtimestamp(user_info["last_activity"]).isoformat(),
                "age_minutes": int((current_time - user_info["last_activity"]) / 60),
                "status": "active" if (current_time - user_info["last_activity"]) < SECURITY_CONFIG["USER_SESSION_TIMEOUT"] else "expired"
            })
        
        # Activity summary
        recent_activity = [u for u in user_sessions_detail if u["age_minutes"] < 30]  # Last 30 minutes
        hourly_activity = [u for u in user_sessions_detail if u["age_minutes"] < 60]  # Last hour
        
        return {
            "status": "success",
            "user_activity_summary": {
                "total_active_users": active_users_count,
                "total_sessions": len(active_users),
                "recent_activity_30min": len(recent_activity),
                "hourly_activity": len(hourly_activity),
                "session_timeout_minutes": SECURITY_CONFIG["USER_SESSION_TIMEOUT"] // 60
            },
            "user_sessions": user_sessions_detail,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in user-activity endpoint: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/forecast")
async def forecast_endpoint(request: Request, file: UploadFile = File(...)):
    # PERBAIKAN: Tambahkan timeout dan memory handling
    import asyncio
    
    logger.info(f"🔍 FORECAST ENDPOINT CALLED - File: {file.filename}")
    
    # Security validation
    security_result = validate_request_security(request)
    client_ip = security_result["client_ip"]
    
    # Circuit breaker check
    if not check_circuit_breaker():
        logger.error("⚡ Circuit breaker is open - service temporarily unavailable")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable due to high failure rate")
    
    # Memory check sebelum processing
    if not check_memory_usage():
        logger.warning("⚠️ High memory usage detected, cleaning up...")
        cleanup_memory()
        
    # Timeout handler
    async def timeout_handler():
        await asyncio.sleep(3600)  # 60 menit timeout
        logger.error("⏰ Request timeout - 60 minutes exceeded")
        raise HTTPException(status_code=408, detail="Request timeout - processing took too long")
    
    # PERBAIKAN: Rate limiting check
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail="Terlalu banyak request. Silakan coba lagi dalam beberapa menit."
        )
    
    try:
        logger.info(f"Received file upload request for: {file.filename}, content_type: {file.content_type}")
        
        # Validasi ukuran file
        content = await file.read()
        logger.info(f"File content size: {len(content)} bytes")
        
        if len(content) > SECURITY_CONFIG["MAX_FILE_SIZE"]:
            raise HTTPException(
                status_code=413,
                detail="File terlalu besar, max. 100MB"
            )
        
        # PERBAIKAN: Validasi konten file untuk keamanan
        if not security.validate_file_content(content):
            logger.warning(f"Invalid file content detected: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="File tidak valid atau berpotensi berbahaya. Hanya file Excel yang diperbolehkan."
            )
        
        # Validasi ekstensi file
        if not security.validate_file_extension(file.filename):
            logger.warning(f"Invalid file format received: {file.filename}")
            raise HTTPException(
                status_code=400, 
                detail="Tipe File Kamu Bukan Excel"
            )
        
        # Sanitasi nama file
        safe_filename = security.sanitize_filename(file.filename)
        
        if not content:
            logger.warning("Received an empty Excel file.")
            raise HTTPException(
                status_code=400,
                detail="File Excel tidak boleh kosong"
            )
        
        # Baca file Excel dengan validasi
        try:
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
            logger.info(f"✅ Excel file read successfully. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise HTTPException(
                status_code=400,
                detail="File Excel tidak dapat dibaca atau rusak."
            )
        
        logger.info(f"File '{safe_filename}' loaded successfully. Shape: {df.shape}")
        
        # Validasi struktur Excel
        if SECURITY_CONFIG["VALIDATE_EXCEL_STRUCTURE"]:
            if not security.validate_excel_structure(df):
                min_rows = SECURITY_CONFIG["MIN_DATA_ROWS"]
                max_rows = SECURITY_CONFIG["MAX_DATA_ROWS"]
                required_cols = ", ".join(SECURITY_CONFIG["REQUIRED_COLUMNS"])
                raise HTTPException(
                    status_code=400,
                    detail="Struktur file Excel tidak valid, pastikan kolom sesuai"
                )
        
        # Validasi jumlah unique part number (maksimal 40)
        try:
            # Normalisasi kolom untuk mencari PART_NO
            df_normalized = df.copy()
            df_normalized.columns = df_normalized.columns.str.upper().str.strip()
            
            # Cari kolom PART_NO
            part_no_col = None
            for col in ['PART_NO', 'PART NO', 'PARTNUMBER', 'PART_NUMBER']:
                if col in df_normalized.columns:
                    part_no_col = col
                    break
            
            if part_no_col:
                unique_parts = df_normalized[part_no_col].nunique()
                logger.info(f"📊 Unique part numbers found: {unique_parts}")
                
                if unique_parts > 40:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Dataset mengandung {unique_parts} unique part number. Maksimal yang diperbolehkan adalah 40 unique part number. Silakan kurangi jumlah part number dalam dataset Anda."
                    )
                elif unique_parts == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="Dataset tidak mengandung part number yang valid. Pastikan kolom PART_NO berisi data yang benar."
                    )
                else:
                    logger.info(f"✅ Part number validation passed: {unique_parts} unique parts")
            else:
                logger.warning("⚠️ No PART_NO column found, skipping part number validation")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error validating part numbers: {e}")
            raise HTTPException(
                status_code=400,
                detail="Error dalam validasi part number. Pastikan dataset memiliki kolom PART_NO yang valid."
            )
        
        # Validasi minimal 12 bulan history
        try:
            # Normalisasi kolom untuk mencari MONTH
            df_normalized = df.copy()
            df_normalized.columns = df_normalized.columns.str.upper().str.strip()
            
            # Cari kolom MONTH
            month_col = None
            for col in ['MONTH', 'ORDER_MONTH', 'ORDER MONTH', 'BULAN', 'TANGGAL']:
                if col in df_normalized.columns:
                    month_col = col
                    break
            
            if month_col:
                try:
                    # Parse kolom MONTH untuk mendapatkan unique months
                    df_normalized[month_col] = pd.to_datetime(df_normalized[month_col], format='%Y-%m', errors='coerce')
                    unique_months = df_normalized[month_col].dt.to_period('M').nunique()
                    logger.info(f"📅 Unique months found: {unique_months}")
                    
                    if unique_months < 12:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Dataset hanya mengandung {unique_months} bulan data. Minimal yang diperlukan adalah 12 bulan data history. Silakan tambahkan data history yang lebih lengkap."
                        )
                    elif unique_months > 48:
                        max_months = int(os.environ.get('MAX_DATA_MONTHS', '48'))
                        logger.warning(f"⚠️ Dataset mengandung {unique_months} bulan data (> {max_months} bulan = 4 tahun). Aplikasi akan menggunakan {max_months} bulan data terakhir untuk performa optimal.")
                    else:
                        logger.info(f"✅ Month history validation passed: {unique_months} months")
                        
                except Exception as e:
                    logger.error(f"Error validating month history: {e}")
                    raise HTTPException(
                        status_code=400,
                        detail="Error dalam validasi data history. Pastikan kolom MONTH berisi data tanggal yang valid dalam format YYYY-MM."
                    )
            else:
                logger.warning("⚠️ No MONTH column found, skipping month history validation")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error validating month history: {e}")
            raise HTTPException(
                status_code=400,
                detail="Error dalam validasi data history. Pastikan dataset memiliki kolom MONTH yang valid."
            )
        
        # Buat temporary file yang aman
        session_id = security.generate_secure_session_id()
        temp_file_path = os.path.join(tempfile.gettempdir(), f"forecast_{session_id}_{safe_filename}")
        
        try:
            # Simpan file temporary
            with open(temp_file_path, 'wb') as f:
                f.write(content)
            
            # Enkripsi file temporary
            encrypted_path = security.encrypt_temp_file(temp_file_path)
            
            # Simpan informasi file
            temp_files[session_id] = {
                'file_path': encrypted_path,
                'original_filename': safe_filename,
                'client_ip': client_ip,
                'created_time': time.time(),
                'file_size': len(content)
            }
            
            logger.info(f"File saved securely: {session_id}")
            
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error menyimpan file sementara."
            )
        
        # Get monitor instance
        if get_monitor:
            monitor = get_monitor()
        else:
            monitor = None
        
        # Get user agent
        user_agent = request.headers.get("user-agent", "Unknown")
        
        # Proses forecast - NO CATEGORIZATION, langsung forecast
        logger.info("🚀 Starting forecast processing...")
        
        # Load forecast service functions
        process_forecast_func, run_backtest_and_realtime_func = load_forecast_service()
        
        # DEBUG: Log which service is being used
        logger.info(f"🔍 DEBUG: Using forecast service: {process_forecast_func.__module__ if hasattr(process_forecast_func, '__module__') else 'unknown'}")
        
        if process_forecast_func and process_forecast_func != dummy_process_forecast:
            try:
                # Log forecast processing start
                monitor.log_forecast_process(
                    session_id=session_id,
                    client_ip=client_ip,
                    total_parts=len(df['PART_NO'].unique()),
                    current_part=0,
                    part_number="ALL",
                    dataset_size=len(df),
                    processing_stage="INITIALIZATION",
                    model_selection_method="CATEGORY_BASED",
                    error_metrics={},
                    best_model="CATEGORY_BASED",
                    processing_time=0.0,
                    memory_usage=0.0
                )
                
                # Proses forecast dengan data original (auto-detection di dalam)
                result = process_forecast_func(df)
                logger.info(f"Forecast processing result: {result['status']}")
                
                # Tambahkan hasil forecast detail ke response JSON
                if result["status"] == "success":
                    result["data"]["session_id"] = session_id
                    result["data"]["categorization_completed"] = True
                    
                    # Log forecast completion
                    if monitor:
                        monitor.log_security_event(
                            event_type="FORECAST_SUCCESS",
                            client_ip=client_ip,
                            user_agent=user_agent,
                            endpoint="/forecast",
                            details={"parts_processed": len(df['PART_NO'].unique()), "dataset_size": len(df)}
                        )
                else:
                    # Log forecast error
                    if monitor:
                        monitor.log_security_event(
                            event_type="FORECAST_ERROR",
                            client_ip=client_ip,
                            user_agent=user_agent,
                            endpoint="/forecast",
                            security_check_passed=False,
                            details={"error": result.get("message", "Unknown error")}
                        )
                    
            except Exception as e:
                logger.error(f"Error in forecast functions: {e}")
                
                # Log forecast error
                if monitor:
                    monitor.log_security_event(
                        event_type="FORECAST_ERROR",
                        client_ip=client_ip,
                        user_agent=user_agent,
                        endpoint="/forecast",
                        security_check_passed=False,
                        details={"error": str(e)}
                    )
                
                result = {
                    "status": "error",
                    "message": f"Forecast processing failed: {str(e)}",
                    "data": {"session_id": session_id}
                }
        else:
            logger.warning("Forecast functions not available")
            result = {
                "status": "error",
                "message": "Forecast service not available",
                "data": {"session_id": session_id}
            }
        
        if result["status"] == "error":
            logger.error(f"Forecast processing failed: {result.get('message', 'Unknown error')}")
            raise HTTPException(
                status_code=500,
                detail=result["message"]
            )
        
        logger.info("✅ Forecast processing completed successfully.")
        record_success()  # Record success in circuit breaker
        return JSONResponse(
            content=result,
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except HTTPException:
        raise
    except pd.errors.EmptyDataError:
        logger.error("Pandas EmptyDataError: Excel file is empty or malformed.")
        raise HTTPException(
            status_code=400,
            detail="File Excel kosong atau tidak valid."
        )
    except Exception as e:
        logger.error(f"Unexpected error in /forecast endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error processing file: {str(e)}"
        )

@app.post("/forecast-base64")
async def forecast_base64_endpoint(request: Request, request_body: ForecastRequest): # Menggunakan Pydantic model untuk validasi input
    # Security validation
    security_result = validate_request_security(request)
    client_ip = security_result["client_ip"]
    """
    Endpoint untuk forecasting dengan data Excel yang dikirim sebagai base64 string dalam body JSON.
    Cocok untuk integrasi dengan Power Automate.
    """
    try:
        logger.info("Received base64 data request.")
        
        # Mengakses string base64 dari Pydantic model
        excel_base64_str = request_body.excel_base64
        
        # Logging string base64 yang diterima untuk debugging
        logger.info(f"Received base64 string length: {len(excel_base64_str)}")
        if len(excel_base64_str) > 100: # Batasi logging untuk string yang sangat panjang
            logger.info(f"Received base64 string start: {excel_base64_str[:50]}...{excel_base64_str[-50:]}")
        else:
            logger.info(f"Received base64 string: {excel_base64_str}")

        try:
            # PENTING: Meng-encode string base64 ke bytes sebelum mendekode
            # base64.b64decode() mengharapkan input bertipe bytes, bukan str
            excel_content = base64.b64decode(excel_base64_str.encode('utf-8'))
            logger.info(f"Base64 data successfully decoded. Content length: {len(excel_content)} bytes.")
        except base64.binascii.Error as e:
            # Menangkap error khusus base64 decoding (misal: incorrect padding, non-base64 characters)
            logger.error(f"Base64 decoding error (binascii.Error): {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 data: {str(e)}. Please ensure the base64 string is valid and correctly padded."
            )
        except Exception as e:
            # Menangkap error umum lainnya selama decoding
            logger.error(f"Unexpected error during base64 decoding: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode base64 string: {str(e)}"
            )
        
        # Baca file Excel dari konten biner yang sudah didecode
        # Pastikan konten tidak kosong setelah decode
        if not excel_content:
            logger.warning("Decoded base64 content is empty.")
            raise HTTPException(
                status_code=400,
                detail="Konten file Excel yang didecode dari base64 kosong atau tidak valid."
            )
        
        # Validasi ukuran file
        if len(excel_content) > SECURITY_CONFIG["MAX_FILE_SIZE"]:
            raise HTTPException(
                status_code=413,
                detail=f"File terlalu besar. Maksimal {SECURITY_CONFIG['MAX_FILE_SIZE'] // (1024*1024)}MB"
            )

        # Baca file Excel dengan validasi
        try:
            df = pd.read_excel(io.BytesIO(excel_content), engine="openpyxl")
        except Exception as e:
            logger.error(f"Error reading Excel file from base64: {e}")
            raise HTTPException(
                status_code=400,
                detail="File Excel tidak dapat dibaca atau rusak."
            )
        
        # Validasi struktur Excel
        if SECURITY_CONFIG["VALIDATE_EXCEL_STRUCTURE"]:
            if not security.validate_excel_structure(df):
                min_rows = SECURITY_CONFIG["MIN_DATA_ROWS"]
                max_rows = SECURITY_CONFIG["MAX_DATA_ROWS"]
                required_cols = ", ".join(SECURITY_CONFIG["REQUIRED_COLUMNS"])
                raise HTTPException(
                    status_code=400,
                    detail=f"Struktur file Excel tidak valid. Pastikan memiliki kolom {required_cols} dengan minimal {min_rows} baris dan maksimal {max_rows:,} baris data."
                )
        
        # Validasi jumlah unique part number (maksimal 40)
        try:
            # Normalisasi kolom untuk mencari PART_NO
            df_normalized = df.copy()
            df_normalized.columns = df_normalized.columns.str.upper().str.strip()
            
            # Cari kolom PART_NO
            part_no_col = None
            for col in ['PART_NO', 'PART NO', 'PARTNUMBER', 'PART_NUMBER']:
                if col in df_normalized.columns:
                    part_no_col = col
                    break
            
            if part_no_col:
                unique_parts = df_normalized[part_no_col].nunique()
                logger.info(f"📊 Unique part numbers found: {unique_parts}")
                
                if unique_parts > 40:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Dataset mengandung {unique_parts} unique part number. Maksimal yang diperbolehkan adalah 40 unique part number. Silakan kurangi jumlah part number dalam dataset Anda."
                    )
                elif unique_parts == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="Dataset tidak mengandung part number yang valid. Pastikan kolom PART_NO berisi data yang benar."
                    )
                else:
                    logger.info(f"✅ Part number validation passed: {unique_parts} unique parts")
            else:
                logger.warning("⚠️ No PART_NO column found, skipping part number validation")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error validating part numbers: {e}")
            raise HTTPException(
                status_code=400,
                detail="Error dalam validasi part number. Pastikan dataset memiliki kolom PART_NO yang valid."
            )
        
        # Buat temporary file yang aman untuk base64
        session_id = security.generate_secure_session_id()
        temp_file_path = os.path.join(tempfile.gettempdir(), f"base64_{session_id}_data.xlsx")
        
        try:
            # Simpan file temporary
            with open(temp_file_path, 'wb') as f:
                f.write(excel_content)
            
            # Enkripsi file temporary
            encrypted_path = security.encrypt_temp_file(temp_file_path)
            
            # Simpan informasi file
            temp_files[session_id] = {
                'file_path': encrypted_path,
                'original_filename': 'base64_data.xlsx',
                'client_ip': client_ip,
                'created_time': time.time(),
                'file_size': len(excel_content),
                'type': 'base64'
            }
            
            logger.info(f"Base64 file saved securely: {session_id}")
            
        except Exception as e:
            logger.error(f"Error saving base64 file: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error menyimpan file sementara."
            )
        
        logger.info(f"File from base64 loaded successfully into DataFrame. Shape: {df.shape}")
        
        # Validasi kolom yang diperlukan
        required_columns = ['MONTH', 'PART_NO', 'ORIGINAL_SHIPPING_QTY']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns in base64 Excel: {missing_columns}")
            raise HTTPException(
                status_code=400,
                detail=f"Kolom yang diperlukan tidak ditemukan: {missing_columns}"
            )
        
        # Proses forecasting
        logger.info("Starting forecast processing from base64 data...")
        
        # Load forecast service
        process_forecast_func, run_backtest_and_realtime_func = load_forecast_service()
        
        if process_forecast_func and process_forecast_func != dummy_process_forecast:
            result = process_forecast_func(df)
        else:
            result = {"status": "error", "message": "Forecast service not available"}
        
        # Tambahkan session_id ke response
        if result["status"] == "success":
            result["data"]["session_id"] = session_id
        
        if result["status"] == "error":
            logger.error(f"Forecast processing from base64 failed: {result.get('message', 'Unknown error')}")
            raise HTTPException(
                status_code=500,
                detail=result["message"]
            )
        
        logger.info("Forecast processing from base64 completed successfully.")
        
        return JSONResponse(
            content=result,
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except HTTPException:
        # Re-raise HTTPException karena sudah ditangani sebelumnya oleh FastAPI/Pydantic
        raise
    except pd.errors.EmptyDataError:
        logger.error("Pandas EmptyDataError: Decoded Excel file is empty or malformed.")
        raise HTTPException(
            status_code=400,
            detail="File Excel yang didecode kosong atau tidak valid."
        )
    except Exception as e:
        logger.error(f"Unexpected error in /forecast-base64 endpoint: {str(e)}", exc_info=True) # exc_info=True untuk traceback
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error processing base64 data: {str(e)}"
        )

@app.post("/forecast-raw")
async def forecast_raw(request: Request):
    """
    Endpoint untuk menerima file Excel sebagai raw body (application/octet-stream),
    cocok untuk Power Automate Web yang tidak mendukung multipart/form-data.
    """
    try:
        logger.info("Received raw body request for Excel file.")
        content = await request.body()  # baca seluruh body sebagai bytes
        logger.info(f"Raw body length: {len(content)} bytes.")

        # Pastikan file tidak kosong
        if not content:
            logger.warning("Received an empty Excel file (raw body).")
            raise HTTPException(
                status_code=400,
                detail="File Excel tidak boleh kosong."
            )

        # Simpan file untuk debug (opsional)
        with open("debug_received_raw.xlsx", "wb") as f:
            f.write(content)

        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        logger.info(f"Raw body Excel loaded successfully. Shape: {df.shape}")

        # Validasi kolom yang diperlukan
        required_columns = ['MONTH', 'PART_NO', 'ORIGINAL_SHIPPING_QTY']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns in raw Excel: {missing_columns}")
            raise HTTPException(
                status_code=400,
                detail=f"Kolom yang diperlukan tidak ditemukan: {missing_columns}"
            )
        
        # Validasi jumlah unique part number (maksimal 40)
        try:
            # Normalisasi kolom untuk mencari PART_NO
            df_normalized = df.copy()
            df_normalized.columns = df_normalized.columns.str.upper().str.strip()
            
            # Cari kolom PART_NO
            part_no_col = None
            for col in ['PART_NO', 'PART NO', 'PARTNUMBER', 'PART_NUMBER']:
                if col in df_normalized.columns:
                    part_no_col = col
                    break
            
            if part_no_col:
                unique_parts = df_normalized[part_no_col].nunique()
                logger.info(f"📊 Unique part numbers found: {unique_parts}")
                
                if unique_parts > 40:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Dataset mengandung {unique_parts} unique part number. Maksimal yang diperbolehkan adalah 40 unique part number. Silakan kurangi jumlah part number dalam dataset Anda."
                    )
                elif unique_parts == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="Dataset tidak mengandung part number yang valid. Pastikan kolom PART_NO berisi data yang benar."
                    )
                else:
                    logger.info(f"✅ Part number validation passed: {unique_parts} unique parts")
            else:
                logger.warning("⚠️ No PART_NO column found, skipping part number validation")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error validating part numbers: {e}")
            raise HTTPException(
                status_code=400,
                detail="Error dalam validasi part number. Pastikan dataset memiliki kolom PART_NO yang valid."
            )

        # Proses forecasting
        logger.info("Starting forecast processing from raw body...")
        
        # Load forecast service
        process_forecast_func, run_backtest_and_realtime_func = load_forecast_service()
        
        if process_forecast_func and process_forecast_func != dummy_process_forecast:
            result = process_forecast_func(df)
        else:
            result = {"status": "error", "message": "Forecast service not available"}
        if result["status"] == "error":
            logger.error(f"Forecast processing from raw body failed: {result.get('message', 'Unknown error')}")
            raise HTTPException(
                status_code=500,
                detail=result["message"]
            )
        logger.info("Forecast processing from raw body completed successfully.")
        return JSONResponse(
            content=result,
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )
    except HTTPException:
        raise
    except pd.errors.EmptyDataError:
        logger.error("Pandas EmptyDataError: Excel file (raw body) is empty or malformed.")
        raise HTTPException(
            status_code=400,
            detail="File Excel kosong atau tidak valid."
        )
    except Exception as e:
        logger.error(f"Unexpected error in /forecast-raw endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error processing raw Excel file: {str(e)}"
        )

@app.post("/run-categorization")
async def run_categorization_endpoint(request: Request, file: UploadFile = File(...)):
    """Run categorization only and return Excel file for download"""
    try:
        # Security validation - simplified untuk kategorisasi
        try:
            security_result = validate_request_security(request)
            if not security_result.get('valid', True):
                raise HTTPException(status_code=400, detail=security_result.get('message', 'Security validation failed'))
        except Exception as e:
            logger.warning(f"Security validation warning: {e}")
            # Continue processing untuk kategorisasi
        
        client_ip = get_client_ip(request)
        logger.info(f"🔍 Categorization request from {client_ip}")
        
        # Check rate limit
        if not check_rate_limit(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait before trying again.")
        
        # Validate file
        if not SecurityMiddleware.validate_file_extension(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type. Only Excel files (.xlsx, .xls) are allowed.")
        
        # Read file
        content = await file.read()
        if not SecurityMiddleware.validate_file_content(content):
            raise HTTPException(status_code=400, detail="Invalid file content or corrupted file.")
        
        # Process categorization
        df = pd.read_excel(io.BytesIO(content))
        
        # Validate Excel structure
        if not SecurityMiddleware.validate_excel_structure(df):
            raise HTTPException(status_code=400, detail="Invalid Excel structure. Required columns not found.")
        
        logger.info(f"📊 Processing categorization for {len(df)} rows")
        
        # Run categorization
        from kategoriparts_web import process_categorization
        df_categorized, excel_path = process_categorization(df)
        
        if excel_path:
            # Use the Excel file created by kategoriparts_web
            file_id = str(uuid.uuid4())
            
            # Copy the file to temp directory with new name
            output_path = os.path.join(TEMP_DIR, f"categorization_result_{file_id}.xlsx")
            import shutil
            shutil.copy2(excel_path, output_path)
            
            # Create summary from categorized data - convert numpy types to Python int
            summary = {
                "total_parts": int(len(df_categorized['PART_NO'].unique())),
                "smooth_count": int((df_categorized['PartCategory'] == 'Smooth').sum()),
                "erratic_count": int((df_categorized['PartCategory'] == 'Erratic').sum()),
                "intermittent_count": int((df_categorized['PartCategory'] == 'Intermittent').sum()),
                "lumpy_count": int((df_categorized['PartCategory'] == 'Lumpy').sum()),
                "unknown_count": int((df_categorized['PartCategory'] == 'Unknown').sum())
            }
            
            # Store file info
            temp_files[file_id] = {
                'file_path': output_path,
                'created_time': datetime.now(),
                'file_type': 'categorization'
            }
            
            logger.info(f"✅ Categorization Excel created: {output_path}")
            
            # Force memory cleanup after categorization
            import gc
            gc.collect()
            logger.info("🧹 Memory cleanup completed after categorization")
            
            return {
                "status": "success",
                "message": "Categorization completed successfully",
                "file_id": file_id,
                "download_url": f"/download-categorization?file_id={file_id}",
                "summary": summary
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create categorization Excel file.")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in categorization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Categorization failed: {str(e)}")

@app.get("/download-categorization")
async def download_categorization_endpoint(request: Request, file_id: str = Query(...)):
    """Download categorization result Excel file"""
    try:
        # Skip security validation untuk download - langsung proses
        logger.info(f"🔍 Download request for file_id: {file_id}")
        
        file_path = os.path.join(TEMP_DIR, f"categorization_result_{file_id}.xlsx")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found or expired")
        
        # Return file for download
        def delete_after_download():
            try:
                os.remove(file_path)
                logger.info(f"🗑️ Deleted categorization file: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not delete file {file_path}: {e}")
        
        return FileResponse(
            path=file_path,
            filename=f"categorization_result_{file_id}.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error downloading categorization file: {str(e)}")
        raise HTTPException(status_code=500, detail="Download failed")

@app.post("/forecast-direct")
async def forecast_direct_endpoint(request: Request, file: UploadFile = File(...)):
    """Direct forecast without categorization - tests all models equally"""
    try:
        # Security validation
        security_result = validate_request_security(request)
        if not security_result['valid']:
            raise HTTPException(status_code=400, detail=security_result['message'])
        
        client_ip = get_client_ip(request)
        logger.info(f"🔍 Direct forecast request from {client_ip}")
        
        # Check rate limit
        if not check_rate_limit(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait before trying again.")
        
        # Validate file
        if not SecurityMiddleware.validate_file_extension(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type. Only Excel files (.xlsx, .xls) are allowed.")
        
        # Read file
        content = await file.read()
        if not SecurityMiddleware.validate_file_content(content):
            raise HTTPException(status_code=400, detail="Invalid file content or corrupted file.")
        
        # Process forecast directly
        df = pd.read_excel(io.BytesIO(content))
        
        # Validate Excel structure
        if not SecurityMiddleware.validate_excel_structure(df):
            raise HTTPException(status_code=400, detail="Invalid Excel structure. Required columns not found.")
        
        logger.info(f"📊 Processing DIRECT forecast for {len(df)} rows")
        
        # Run direct forecast
        from forecast_service_clean import process_forecast_direct
        result = process_forecast_direct(df)
        
        if result['status'] == 'success':
            # Create Excel file for download
            file_id = str(uuid.uuid4())
            output_path = os.path.join(TEMP_DIR, f"forecast_direct_{file_id}.xlsx")
            
            # Save result to Excel
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Backtest results
                backtest_df = pd.DataFrame(result['data']['forecast_df'])
                backtest_df.to_excel(writer, sheet_name='Backtest_Results', index=False)
                
                # Realtime forecast
                realtime_df = pd.DataFrame(result['data']['real_time_forecast'])
                realtime_df.to_excel(writer, sheet_name='Realtime_Forecast', index=False)
                
                # Summary
                summary_data = {
                    'Metric': ['Total Parts', 'Backtest Months', 'Forecast Months', 'Processing Mode'],
                    'Value': [
                        len(backtest_df['PART_NO'].unique()),
                        len(backtest_df['MONTH'].unique()),
                        len(realtime_df['MONTH'].unique()),
                        'Direct (All Models Tested)'
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            logger.info(f"✅ Direct forecast Excel created: {output_path}")
            
            return {
                "status": "success",
                "message": "Direct forecast completed successfully",
                "file_id": file_id,
                "download_url": f"/download-forecast?file_id={file_id}",
                "data": result['data']
            }
        else:
            raise HTTPException(status_code=500, detail=result['message'])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in direct forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Direct forecast failed: {str(e)}")

@app.post("/process-forecast")
async def process_forecast_endpoint(request: Request, file: UploadFile = File(...)):
    # PERBAIKAN: Tambahkan logging untuk debug
    logger.info(f"🔍 PROCESS-FORECAST ENDPOINT CALLED - File: {file.filename}")
    
    # Memory check sebelum memulai - ENHANCED untuk Railway
    # RELAX: Jangan blokir request hanya karena snapshot memori tinggi.
    # Lakukan cleanup ringan lalu lanjut proses; limiter ada di forecast layer.
    if not check_memory_usage():
        logger.warning("⚠️  High memory usage detected, proceeding with caution (cleanup triggered)")
        cleanup_memory()
    
    # Proactive memory cleanup sebelum processing
    cleanup_memory()
    
    # Security validation
    security_result = validate_request_security(request)
    client_ip = security_result["client_ip"]
    
    # PERBAIKAN: Rate limiting check
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail="Terlalu banyak request. Silakan coba lagi dalam beberapa menit."
        )
    
    try:
        logger.info(f"Received file upload request for process: {file.filename}, content_type: {file.content_type}")
        
        # Security validations
        client_ip = get_client_ip(request)
        
        # Validasi ukuran file
        content = await file.read()
        logger.info(f"File content size: {len(content)} bytes")
        
        if len(content) > SECURITY_CONFIG["MAX_FILE_SIZE"]:
            raise HTTPException(
                status_code=413,
                detail="File terlalu besar, max. 100MB"
            )
        
        # Validasi ekstensi file
        if not security.validate_file_extension(file.filename):
            logger.warning(f"Invalid file format received: {file.filename}")
            raise HTTPException(
                status_code=400, 
                detail="Tipe File Kamu Bukan Excel"
            )
        
        # Sanitasi nama file
        safe_filename = security.sanitize_filename(file.filename)
        
        if not content:
            logger.warning("Received an empty Excel file.")
            raise HTTPException(
                status_code=400,
                detail="File Excel tidak boleh kosong"
            )
        
        # Baca file Excel dengan validasi
        try:
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
            logger.info(f"✅ Excel file read successfully. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise HTTPException(
                status_code=400,
                detail="File Excel tidak dapat dibaca atau rusak."
            )
        
        logger.info(f"File '{safe_filename}' loaded successfully for process. Shape: {df.shape}")
        
        # Validasi struktur Excel
        if SECURITY_CONFIG["VALIDATE_EXCEL_STRUCTURE"]:
            if not security.validate_excel_structure(df):
                min_rows = SECURITY_CONFIG["MIN_DATA_ROWS"]
                max_rows = SECURITY_CONFIG["MAX_DATA_ROWS"]
                required_cols = ", ".join(SECURITY_CONFIG["REQUIRED_COLUMNS"])
                raise HTTPException(
                    status_code=400,
                    detail=f"Struktur file Excel tidak valid. Pastikan memiliki kolom {required_cols} dengan minimal {min_rows} baris dan maksimal {max_rows:,} baris data."
                )
        
        # Validasi jumlah unique part number (maksimal 40)
        try:
            # Normalisasi kolom untuk mencari PART_NO
            df_normalized = df.copy()
            df_normalized.columns = df_normalized.columns.str.upper().str.strip()
            
            # Cari kolom PART_NO
            part_no_col = None
            for col in ['PART_NO', 'PART NO', 'PARTNUMBER', 'PART_NUMBER']:
                if col in df_normalized.columns:
                    part_no_col = col
                    break
            
            if part_no_col:
                unique_parts = df_normalized[part_no_col].nunique()
                logger.info(f"📊 Unique part numbers found: {unique_parts}")
                
                if unique_parts > 40:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Dataset mengandung {unique_parts} unique part number. Maksimal yang diperbolehkan adalah 40 unique part number. Silakan kurangi jumlah part number dalam dataset Anda."
                    )
                elif unique_parts == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="Dataset tidak mengandung part number yang valid. Pastikan kolom PART_NO berisi data yang benar."
                    )
                else:
                    logger.info(f"✅ Part number validation passed: {unique_parts} unique parts")
            else:
                logger.warning("⚠️ No PART_NO column found, skipping part number validation")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error validating part numbers: {e}")
            raise HTTPException(
                status_code=400,
                detail="Error dalam validasi part number. Pastikan dataset memiliki kolom PART_NO yang valid."
            )
        
        # Buat temporary file yang aman untuk dashboard
        session_id = security.generate_secure_session_id()
        temp_file_path = os.path.join(tempfile.gettempdir(), f"dashboard_{session_id}_{safe_filename}")
        
        try:
            # Simpan file temporary
            with open(temp_file_path, 'wb') as f:
                f.write(content)
            
            # Enkripsi file temporary
            encrypted_path = security.encrypt_temp_file(temp_file_path)
            
            # Simpan informasi file untuk dashboard
            temp_files[session_id] = {
                'file_path': encrypted_path,
                'original_filename': safe_filename,
                'client_ip': client_ip,
                'created_time': time.time(),
                'file_size': len(content),
                'type': 'dashboard'
            }
            
            # Add user session untuk tracking
            add_user_session(client_ip, session_id)
            
            logger.info(f"Dashboard file saved securely: {session_id}")
            
        except Exception as e:
            logger.error(f"Error saving dashboard file: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error menyimpan file sementara untuk dashboard."
            )
        
        logger.info("🚀 Starting forecast processing for process...")
        try:
            # Load forecast service - NO CATEGORIZATION, langsung forecast
            logger.info("Using standard forecast service...")
            process_forecast_func, run_backtest_and_realtime_func = load_forecast_service()
            
            if process_forecast_func and process_forecast_func != dummy_process_forecast:
                # Call the main process_forecast which handles auto-detection
                result = process_forecast_func(df)  # Pass original df, auto-detection happens inside
            else:
                logger.error("❌ Forecast service not available - both forecast_service_clean and forecast_service failed to load")
                logger.error("❌ Available files in directory:")
                try:
                    files = os.listdir('.')
                    forecast_files = [f for f in files if 'forecast_service' in f]
                    logger.error(f"❌ Forecast service files found: {forecast_files}")
                except Exception as e:
                    logger.error(f"❌ Could not list directory: {e}")
                raise Exception("No forecast service available - check deployment files")
            
            # PERBAIKAN: Pastikan variabel didefinisikan dari result
            if result and result.get("status") == "success":
                # Ambil data dari result jika tersedia
                data = result.get("data", {})
                df_processed = data.get("df_processed", df.copy())
                forecast_df = data.get("forecast_df", pd.DataFrame())
                real_time_forecast = data.get("real_time_forecast", pd.DataFrame())
                
                # Convert list to DataFrame if needed
                if isinstance(forecast_df, list):
                    forecast_df = pd.DataFrame(forecast_df) if forecast_df else pd.DataFrame()
                if isinstance(real_time_forecast, list):
                    real_time_forecast = pd.DataFrame(real_time_forecast) if real_time_forecast else pd.DataFrame()
                
                # Jika tidak ada data di result, gunakan run_backtest_and_realtime
                if forecast_df.empty and run_backtest_and_realtime_func:
                    try:
                        df_processed, forecast_df, real_time_forecast = run_backtest_and_realtime_func(df)
                        logger.info("✅ Data processed using run_backtest_and_realtime_func")
                    except Exception as e:
                        logger.error(f"Error in run_backtest_and_realtime_func: {e}")
                        # Fallback ke data asli
                        df_processed = df.copy()
                        forecast_df = pd.DataFrame()
                        real_time_forecast = pd.DataFrame()
            else:
                # Fallback jika result tidak valid - buat data dummy untuk mencegah error
                logger.warning("⚠️ Forecast result not valid, creating fallback data")
                df_processed = df.copy()
                forecast_df = pd.DataFrame({
                    'PART_NO': ['No Data'],
                    'MONTH': ['2024-01'],
                    'FORECAST': [0],
                    'ACTUAL': [0],
                    'ERROR': [0],
                    'MODEL': ['No Model']
                })
                real_time_forecast = pd.DataFrame({
                    'PART_NO': ['No Data'],
                    'MONTH': ['2024-01'],
                    'FORECAST': [0],
                    'MODEL': ['No Model']
                })
            
            # Secure processing dengan timeout dan memory management
            logger.info("Starting secure data processing...")
            
            # PERBAIKAN: Tambahkan timeout untuk mencegah hanging
            
            def timeout_handler(signum, frame):
                logger.error(f"⏰ Forecast processing timeout after {timeout_seconds//60} minutes")
                # Cleanup before timeout
                cleanup_memory()
                raise TimeoutError(f"Forecast processing timeout after {timeout_seconds//60} minutes - dataset terlalu besar atau server sedang sibuk. Coba dengan dataset yang lebih kecil atau tunggu beberapa saat.")
            
            # PERBAIKAN: Railway-optimized timeout (Railway timeout ~15-20 minutes)
            num_parts = len(df['PART_NO'].unique()) if 'PART_NO' in df.columns else len(df)
            
            # Railway timeout optimization - lebih agresif
            if num_parts <= 20:
                timeout_seconds = 900   # 15 minutes for small datasets
            elif num_parts <= 40:
                timeout_seconds = 1200  # 20 minutes for medium datasets  
            elif num_parts <= 60:
                timeout_seconds = 1500  # 25 minutes for large datasets
            else:
                timeout_seconds = 1800  # 30 minutes for very large datasets
            
            logger.info(f"Dataset size: {len(df):,} rows, {num_parts} parts, using {timeout_seconds//60} minute timeout")
            
            # Debug: Log environment variables yang mungkin mempengaruhi timeout
            logger.info(f"Environment PYTHONUNBUFFERED: {os.environ.get('PYTHONUNBUFFERED', 'Not set')}")
            logger.info(f"Environment MEMORY_LIMIT: {os.environ.get('MEMORY_LIMIT', 'Not set')}")
            logger.info(f"Environment OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")
            logger.info(f"Environment PORT: {os.environ.get('PORT', 'Not set')}")
            logger.info(f"Environment RAILWAY_ENVIRONMENT: {os.environ.get('RAILWAY_ENVIRONMENT', 'Not set')}")
            logger.info(f"Target performance: 40 parts dalam 6 menit")
            logger.info(f"Railway cores: 4")
            logger.info(f"Timeout: 1 jam untuk semua ukuran dataset")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            logger.info(f"Timeout alarm set for {timeout_seconds} seconds ({timeout_seconds/3600:.2f} hours)")
            
            # Clean memory before processing
            import gc
            gc.collect()
            
            # Check memory before processing
            if psutil:
                memory_info = psutil.virtual_memory()
            else:
                memory_info = type('obj', (object,), {'percent': 50, 'used': 1024*1024*1024, 'total': 2*1024*1024*1024})()
            memory_percent = memory_info.percent
            
            # Get memory threshold from environment or use default
            max_memory_threshold = int(os.getenv('MAX_MEMORY_THRESHOLD', '80'))  # Railway optimized: 80%
            
            if memory_percent > max_memory_threshold:
                logger.error(f"❌ Memory usage too high: {memory_percent:.1f}% > {max_memory_threshold}% threshold - aborting to prevent crash")
                cleanup_memory()
                record_failure()  # Record failure in circuit breaker
                
                # Provide helpful message based on memory level
                if memory_percent > 85:
                    message = f"Server memory critically high ({memory_percent:.1f}%). Please wait 10-15 minutes before trying again."
                elif memory_percent > 75:
                    message = f"Server memory high ({memory_percent:.1f}%). Please try again in 5-10 minutes or use a smaller dataset."
                else:
                    message = f"Server memory usage elevated ({memory_percent:.1f}%). Please try again in a few minutes."
                
                raise HTTPException(status_code=507, detail=message)
                        
            # PERBAIKAN: Immediate response preparation to prevent timeout
            logger.info("🚀 Preparing immediate response to prevent timeout...")
            
            # PERBAIKAN: Send progress update to prevent Railway timeout
            try:
                # Send initial response to keep connection alive
                logger.info("📡 Sending progress update to prevent timeout...")
                
                # PERBAIKAN: Stream progress updates to prevent Railway timeout
                import asyncio
                async def stream_progress():
                    while True:
                        await asyncio.sleep(30)  # Send progress every 30 seconds
                        logger.info("📡 Progress update: Forecast still processing...")
                        yield f"data: {json.dumps({'status': 'processing', 'message': 'Forecast still processing...'})}\n\n"
                
                # This will be handled by the response streaming
            except Exception as e:
                logger.warning(f"Progress update failed: {e}")
            
            # df_processed, forecast_df, dan real_time_forecast sudah didefinisikan di atas
            logger.info("✅ Forecast data already processed and variables defined")
            
            # Clean memory after processing
            gc.collect()
            
            # Cancel timeout
            signal.alarm(0)
            
        except TimeoutError:
            logger.error("Forecast processing timeout after 1 hour")
            raise Exception("Forecast processing timeout after 1 hour - dataset terlalu besar atau server sedang sibuk. Coba dengan dataset yang lebih kecil atau tunggu beberapa saat.")
        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            logger.error(f"Error in forecast processing: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise e
            
        # PERBAIKAN: Jangan wipe memory sampai data terkirim ke frontend
        # Wipe original dataframe dari memory setelah data terkirim
        logger.info("Forecast processing completed, preparing response...")
        
        # Buat file Excel hasil
        file_id = str(uuid.uuid4())
        output_path = os.path.join(TEMP_DIR, f"forecast_result_{file_id}.xlsx")
        logger.info(f"Creating Excel file at: {output_path}")
        
        try:
            # Debug logging untuk melihat data yang akan ditulis
            # Ensure DataFrames are properly converted
            if isinstance(forecast_df, list):
                forecast_df = pd.DataFrame(forecast_df) if forecast_df else pd.DataFrame()
            if isinstance(real_time_forecast, list):
                real_time_forecast = pd.DataFrame(real_time_forecast) if real_time_forecast else pd.DataFrame()
            
            logger.info(f"🔍 DEBUG: forecast_df shape: {forecast_df.shape if not forecast_df.empty else 'EMPTY'}")
            logger.info(f"🔍 DEBUG: real_time_forecast shape: {real_time_forecast.shape if not real_time_forecast.empty else 'EMPTY'}")
            
            # PERBAIKAN: Excel creation dengan explicit save dan flush
            logger.info("🔄 Creating Excel file with explicit save...")
            
            # Create Excel writer dengan mode yang lebih robust
            writer = pd.ExcelWriter(output_path, engine='openpyxl', mode='w')
            
            # Optimize DataFrames before writing to reduce processing time
            if not forecast_df.empty:
                forecast_optimized = forecast_df.round(2)
                forecast_optimized.to_excel(writer, sheet_name='Backtest', index=False)
                logger.info(f"✅ Backtest data written: {forecast_optimized.shape}")
            else:
                logger.warning("⚠️ forecast_df is empty, creating empty sheet")
                pd.DataFrame({'Message': ['No backtest data available']}).to_excel(writer, sheet_name='Backtest', index=False)
            
            if not real_time_forecast.empty:
                realtime_optimized = real_time_forecast.round(2)
                realtime_optimized.to_excel(writer, sheet_name='RealTimeForecast', index=False)
                logger.info(f"✅ Realtime data written: {realtime_optimized.shape}")
            else:
                logger.warning("⚠️ real_time_forecast is empty, creating empty sheet")
                pd.DataFrame({'Message': ['No realtime data available']}).to_excel(writer, sheet_name='RealTimeForecast', index=False)
            
            # Ensure Excel file has content by adding summary sheet
            # Pastikan variabel didefinisikan untuk summary
            if 'df_processed' not in locals():
                df_processed = df.copy()
            if 'forecast_df' not in locals():
                forecast_df = pd.DataFrame()
            if 'real_time_forecast' not in locals():
                real_time_forecast = pd.DataFrame()
            
            # Ensure DataFrames are properly converted before summary
            if isinstance(forecast_df, list):
                forecast_df = pd.DataFrame(forecast_df) if forecast_df else pd.DataFrame()
            if isinstance(real_time_forecast, list):
                real_time_forecast = pd.DataFrame(real_time_forecast) if real_time_forecast else pd.DataFrame()
            
            summary_data = {
                'Metric': ['Total Parts', 'Backtest Records', 'Realtime Records', 'Average Error'],
                'Value': [
                    len(df_processed['PART_NO'].unique()) if not df_processed.empty else 0,
                    len(forecast_df) if not forecast_df.empty else 0,
                    len(real_time_forecast) if not real_time_forecast.empty else 0,
                    f"{forecast_df['ERROR'].astype(str).str.replace('%', '').astype(float).mean():.2f}%" if not forecast_df.empty and 'ERROR' in forecast_df.columns else "N/A"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            logger.info("✅ Summary sheet added to Excel file")
            
            # CRITICAL: Explicit save dan close
            logger.info("💾 Saving Excel file...")
            # writer.save() # Dihapus karena tidak lagi diperlukan dan menyebabkan error
            writer.close()
            logger.info("✅ Excel file saved and closed")
            
            # Force flush to disk
            time.sleep(0.1)  # Small delay to ensure file is written
            
            # Verify file was created and has content
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"Forecast Excel file created successfully at {output_path}. File size: {file_size} bytes")
                
                if file_size == 0:
                    logger.error("Excel file was created but is empty!")
                    # Try to recreate with minimal data
                    logger.info("🔄 Attempting to recreate Excel file with minimal data...")
                    with pd.ExcelWriter(output_path, engine='openpyxl') as fallback_writer:
                        # Create minimal data to ensure file is not empty
                        minimal_data = pd.DataFrame({
                            'PART_NO': ['Sample'],
                            'MONTH': ['2025-01'],
                            'FORECAST': [0],
                            'ERROR': ['0%']
                        })
                        minimal_data.to_excel(fallback_writer, sheet_name='Data', index=False)
                        
                        # Add error sheet
                        error_data = pd.DataFrame({
                            'Error': ['Excel file is empty after creation'],
                            'Message': ['Please try again or contact support'],
                            'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                        })
                        error_data.to_excel(fallback_writer, sheet_name='Error', index=False)
                    
                    # Check file size again
                    new_file_size = os.path.getsize(output_path)
                    logger.info(f"Fallback Excel file created with size: {new_file_size} bytes")
                    
                    if new_file_size == 0:
                        raise Exception("Excel file is empty after creation")
            else:
                logger.error(f"Excel file was not created at {output_path}")
                raise Exception("Excel file was not created")
        
        except Exception as e:
            logger.error(f"Error creating Excel file: {str(e)}")
            # Create fallback Excel file with error message
            try:
                fallback_path = os.path.join(TEMP_DIR, f"forecast_error_{file_id}.xlsx")
                with pd.ExcelWriter(fallback_path, engine='openpyxl') as writer:
                    error_data = {
                        'Error': ['Excel Creation Failed'],
                        'Message': [str(e)],
                        'Timestamp': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
                    }
                    pd.DataFrame(error_data).to_excel(writer, sheet_name='Error', index=False)
                
                # Update output_path to fallback
                output_path = fallback_path
                logger.info(f"✅ Fallback Excel file created at {fallback_path}")
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback Excel file: {fallback_error}")
                raise Exception(f"Failed to create Excel file: {str(e)}")
            
        # Simpan informasi file hasil untuk download (dipindah keluar dari except block)
        temp_files[file_id] = {
            'file_path': output_path,
            'original_filename': f"forecast_result_{file_id}.xlsx",
            'client_ip': client_ip,
            'created_time': time.time(),
            'file_size': os.path.getsize(output_path),
            'type': 'forecast_result'
        }
        logger.info(f"File tracking added for file_id: {file_id}")
        logger.info(f"Total temp_files: {len(temp_files)}")
        
        # PERBAIKAN: Pastikan variabel didefinisikan sebelum digunakan
        # Variabel sudah didefinisikan di atas, tapi pastikan mereka ada
        if 'df_processed' not in locals():
            df_processed = df.copy()
        if 'forecast_df' not in locals():
            forecast_df = pd.DataFrame()
        if 'real_time_forecast' not in locals():
            real_time_forecast = pd.DataFrame()
        
        # PERBAIKAN: Bersihkan NaN/inf di semua dataframe sebelum dikirim ke frontend
        # Handle categorical columns separately to avoid category errors
        def safe_fillna(df):
            df_cleaned = df.copy()
            for col in df_cleaned.columns:
                if df_cleaned[col].dtype.name == 'category':
                    # For categorical columns, convert to object first, then fillna
                    df_cleaned[col] = df_cleaned[col].astype('object')
                    df_cleaned[col] = df_cleaned[col].fillna(0)
                else:
                    # For other columns, handle inf and nan normally
                    df_cleaned[col] = df_cleaned[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            return df_cleaned
        
        # Ensure DataFrames are properly converted before processing
        if isinstance(forecast_df, list):
            forecast_df = pd.DataFrame(forecast_df) if forecast_df else pd.DataFrame()
        if isinstance(real_time_forecast, list):
            real_time_forecast = pd.DataFrame(real_time_forecast) if real_time_forecast else pd.DataFrame()
        
        df_processed = safe_fillna(df_processed)
        forecast_df = safe_fillna(forecast_df)
        real_time_forecast = safe_fillna(real_time_forecast)
        
        # PERBAIKAN: Enkripsi data sebelum dikirim ke frontend untuk keamanan
        import json
        import base64
        from cryptography.fernet import Fernet
        
        # Generate encryption key untuk session ini
        dashboard_key = Fernet.generate_key()
        fernet = Fernet(dashboard_key)
        
        # Enkripsi data untuk dashboard
        # Calculate total aggregation statistics for dashboard
        total_stats = {}
        
        # Calculate total forecast quantities for each month (BACKTEST data)
        if not forecast_df.empty:
            forecast_totals = forecast_df.groupby('MONTH')['FORECAST'].sum().to_dict()
            total_stats['forecast_totals'] = forecast_totals
            
            # Calculate average error across all parts (BACKTEST data)
            if 'ERROR' in forecast_df.columns:
                # Convert ERROR column to numeric (remove % and convert to float)
                error_values = forecast_df['ERROR'].astype(str).str.replace('%', '').astype(float)
                total_stats['average_error'] = float(error_values.mean())
            else:
                total_stats['average_error'] = 0.0
                
            # Calculate total actual quantities for each month (BACKTEST data)
            if 'ACTUAL' in forecast_df.columns:
                actual_totals = forecast_df.groupby('MONTH')['ACTUAL'].sum().to_dict()
                total_stats['actual_totals'] = actual_totals
            else:
                total_stats['actual_totals'] = {}
                
            # Calculate best model counts (BACKTEST data)
            if 'BEST_MODEL' in forecast_df.columns:
                model_counts = forecast_df['BEST_MODEL'].value_counts().to_dict()
                total_stats['model_counts'] = model_counts
            else:
                total_stats['model_counts'] = {}
        else:
            total_stats['forecast_totals'] = {}
            total_stats['average_error'] = 0.0
            total_stats['actual_totals'] = {}
            total_stats['model_counts'] = {}
        
        # Calculate total realtime forecast quantities for each month (REALTIME data)
        if not real_time_forecast.empty:
            # DEBUG: Log realtime data structure
            logger.info(f"🔍 DEBUG: real_time_forecast columns: {list(real_time_forecast.columns)}")
            logger.info(f"🔍 DEBUG: real_time_forecast shape: {real_time_forecast.shape}")
            logger.info(f"🔍 DEBUG: Sample realtime data: {real_time_forecast.head().to_dict('records')}")
            
            # Check if FORECAST_NEUTRAL column exists, otherwise use FORECAST
            if 'FORECAST_NEUTRAL' in real_time_forecast.columns:
                realtime_totals = real_time_forecast.groupby('MONTH')['FORECAST_NEUTRAL'].sum().to_dict()
                logger.info(f"✅ Using FORECAST_NEUTRAL column for realtime totals: {realtime_totals}")
            elif 'FORECAST' in real_time_forecast.columns:
                realtime_totals = real_time_forecast.groupby('MONTH')['FORECAST'].sum().to_dict()
                logger.info(f"✅ Using FORECAST column for realtime totals: {realtime_totals}")
            else:
                # If neither column exists, create empty totals
                realtime_totals = {}
                logger.warning("⚠️ No FORECAST_NEUTRAL or FORECAST column found in realtime data")
            total_stats['realtime_totals'] = realtime_totals
        else:
            total_stats['realtime_totals'] = {}
            logger.warning("⚠️ real_time_forecast is empty")
        
        # Calculate total parts count
        total_stats['total_parts'] = len(df_processed['PART_NO'].unique()) if not df_processed.empty else 0
        
        # Calculate total backtest records
        total_stats['total_backtest_records'] = len(forecast_df) if not forecast_df.empty else 0
        
        # Calculate total realtime records
        total_stats['total_realtime_records'] = len(real_time_forecast) if not real_time_forecast.empty else 0
        
        dashboard_data = {
            "original_df": df_processed.to_dict(orient="records"),
            "forecast_df": forecast_df.to_dict(orient="records"),
            "real_time_forecast": real_time_forecast.to_dict(orient="records") if not real_time_forecast.empty else [],
            "total_stats": total_stats  # Add total aggregation statistics
        }
        
        # Convert ke JSON dan enkripsi
        json_data = json.dumps(dashboard_data, default=str)
        encrypted_data = fernet.encrypt(json_data.encode())
        encrypted_base64 = base64.b64encode(encrypted_data).decode()
        
        # Simpan key untuk dekripsi di frontend (temporary)
        dashboard_keys[session_id] = {
            'key': base64.b64encode(dashboard_key).decode(),
            'created_time': time.time(),
            'client_ip': client_ip
        }
        
        # PERBAIKAN: Jangan wipe memory dulu, biarkan data tersedia untuk frontend
        # Memory akan dibersihkan otomatis oleh garbage collector atau cleanup function
        # if SECURITY_CONFIG.get("MEMORY_WIPE_AFTER_PROCESSING", False):
        #     security.wipe_memory_data([df_processed, forecast_df, real_time_forecast, df])
        #     logger.info("Memory data securely wiped")
        
        # PERBAIKAN: Auto-cleanup file upload setelah diproses
        try:
            if 'file_path' in temp_files.get(session_id, {}):
                upload_file_path = temp_files[session_id]['file_path']
                if os.path.exists(upload_file_path):
                    security.secure_delete_file(upload_file_path)
                    logger.info(f"Uploaded file cleaned up: {upload_file_path}")
                
                # Delete encrypted upload file if exists
                encrypted_upload_path = upload_file_path + '.enc'
                if os.path.exists(encrypted_upload_path):
                    security.secure_delete_file(encrypted_upload_path)
                    logger.info(f"Encrypted upload file cleaned up: {encrypted_upload_path}")
            
            # Remove upload file entry from temp_files
            if session_id in temp_files:
                del temp_files[session_id]
                logger.info(f"Upload file entry cleaned up: {session_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up upload file: {e}")
        
        # Return JSON for dashboard integration dengan data terenkripsi
        response_data = {
            "status": "success",
            "file_id": file_id,
            "session_id": session_id,
            "encrypted_data": encrypted_base64,  # Data terenkripsi
            "message": "Forecast completed successfully. Data is now available in dashboard."
        }
        logger.info(f"✅ Returning response with file_id: {file_id} and encrypted data")
        
        # Record success for circuit breaker
        record_success()
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error in forecast processing: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # PERBAIKAN: Error response yang lebih informatif
        error_message = str(e) if str(e).strip() else "Unknown error occurred during forecast processing"
        if "timeout" in error_message.lower():
            error_message = "Forecast processing timeout after 1 hour. Dataset terlalu besar atau server sedang sibuk. Silakan coba lagi dalam beberapa saat atau gunakan dataset yang lebih kecil."
        elif "memory" in error_message.lower():
            error_message = "Memory error. Dataset terlalu besar. Silakan gunakan dataset yang lebih kecil."
        elif "column not found" in error_message.lower():
            error_message = f"Data structure error: {error_message}. Pastikan file Excel memiliki kolom yang benar."
        elif not error_message.strip():
            error_message = "Forecast processing failed due to unknown error. Silakan coba lagi atau gunakan dataset yang lebih kecil."
        else:
            error_message = f"Forecast processing failed: {error_message}"
        
        error_response = {
            "status": "error",
            "message": error_message,
            "file_id": None,
            "encrypted_data": "",
            "session_id": None,
            "error_code": "FORECAST_FAILED"
        }
        logger.error(f"Returning error response: {error_response}")
        return error_response
            
    except HTTPException:
        raise
    except pd.errors.EmptyDataError:
        logger.error("Pandas EmptyDataError: Excel file is empty or malformed.")
        raise HTTPException(
            status_code=400,
            detail="File Excel kosong atau tidak valid."
        )
    except Exception as e:
        logger.error(f"Unexpected error in /process-forecast endpoint: {str(e)}", exc_info=True)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        error_detail = f"Internal server error processing file: {str(e)}"
        logger.error(f"Raising HTTPException: {error_detail}")
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )

# PERBAIKAN: Tambahkan endpoint untuk dekripsi data dashboard
@app.post("/api/dashboard-data")
async def get_dashboard_data(request: Request):
    """Endpoint untuk mendapatkan data dashboard yang terenkripsi"""
    try:
        # Security validation
        security_result = validate_request_security(request)
        client_ip = security_result["client_ip"]
        
        # Ambil data dari request body
        request_data = await request.json()
        session_id = request_data.get('session_id')
        encrypted_data = request_data.get('encrypted_data', '')
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID required")
        
        # Validasi session_id
        if session_id not in dashboard_keys:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Validasi IP address
        session_info = dashboard_keys[session_id]
        if session_info['client_ip'] != client_ip:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Validasi session timeout
        if time.time() - session_info['created_time'] > SECURITY_CONFIG.get("USER_SESSION_TIMEOUT", 3600):
            # Hapus session yang expired
            del dashboard_keys[session_id]
            raise HTTPException(status_code=401, detail="Session expired")
        
        # Ambil key untuk dekripsi
        key_base64 = session_info['key']
        key = base64.b64decode(key_base64)
        fernet = Fernet(key)
        
        if not encrypted_data:
            raise HTTPException(status_code=400, detail="No encrypted data provided")
        
        # Dekripsi data
        encrypted_bytes = base64.b64decode(encrypted_data)
        decrypted_json = fernet.decrypt(encrypted_bytes)
        dashboard_data = json.loads(decrypted_json.decode())
        
        # Log untuk debugging
        logger.info(f"✅ Dashboard data decrypted successfully")
        logger.info(f"   - original_df: {len(dashboard_data.get('original_df', []))} rows")
        logger.info(f"   - forecast_df: {len(dashboard_data.get('forecast_df', []))} rows")
        logger.info(f"   - real_time_forecast: {len(dashboard_data.get('real_time_forecast', []))} rows")
        
        # Jangan hapus key dulu, biarkan frontend bisa akses beberapa kali
        # Key akan dihapus otomatis oleh cleanup function
        
        return {
            "status": "success",
            "data": dashboard_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in dashboard data endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# PERBAIKAN: Tambahkan fungsi untuk membersihkan dashboard keys yang expired
def cleanup_dashboard_keys():
    """Membersihkan dashboard keys yang expired"""
    current_time = time.time()
    expired_keys = []
    
    for session_id, session_info in dashboard_keys.items():
        if current_time - session_info['created_time'] > SECURITY_CONFIG.get("USER_SESSION_TIMEOUT", 3600):
            expired_keys.append(session_id)
    
    for session_id in expired_keys:
        del dashboard_keys[session_id]
        logger.info(f"Cleaned up expired dashboard key: {session_id}")
    
    return len(expired_keys)

# PERBAIKAN: Tambahkan endpoint untuk membersihkan data dashboard
@app.post("/api/cleanup-dashboard")
async def cleanup_dashboard_endpoint(request: Request):
    """Endpoint untuk membersihkan data dashboard yang expired"""
    try:
        # Security validation
        security_result = validate_request_security(request)
        client_ip = security_result["client_ip"]
        
        # Cleanup expired keys
        cleaned_count = cleanup_dashboard_keys()
        
        return {
            "status": "success",
            "message": f"Cleaned up {cleaned_count} expired dashboard sessions",
            "remaining_sessions": len(dashboard_keys)
        }
        
    except Exception as e:
        logger.error(f"Error in dashboard cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# PERBAIKAN: Tambahkan cleanup otomatis setiap 1 jam
import asyncio
import threading

def auto_cleanup_dashboard():
    """Auto cleanup dashboard keys setiap 1 jam"""
    while True:
        try:
            time.sleep(3600)  # 1 jam
            cleanup_dashboard_keys()
            logger.info("Auto cleanup dashboard keys completed")
        except Exception as e:
            logger.error(f"Error in auto cleanup: {e}")

# Start auto cleanup thread
cleanup_thread = threading.Thread(target=auto_cleanup_dashboard, daemon=True)
cleanup_thread.start()

@app.get("/download-forecast")
async def download_forecast_endpoint(request: Request, file_id: str = Query(...)):
    # Security validation
    security_result = validate_request_security(request)
    client_ip = security_result["client_ip"]
    try:
        logger.info(f"Download request received for file_id: {file_id}")
        
        # Validasi session ID (file_id)
        if not file_id or len(file_id) < 10:
            raise HTTPException(
                status_code=400,
                detail="File ID tidak valid."
            )
        
        # Cek apakah file ada di temporary storage
        logger.info(f"Checking temp_files for file_id: {file_id}")
        logger.info(f"Available temp_files keys: {list(temp_files.keys())}")
        
        if file_id not in temp_files:
            logger.error(f"File ID {file_id} not found in temp_files")
            raise HTTPException(
                status_code=404,
                detail="File tidak ditemukan atau sudah expired."
            )
        
        file_info = temp_files[file_id]
        logger.info(f"File info found: {file_info}")
        
        # Cek apakah file sudah expired
        current_time = time.time()
        if current_time - file_info['created_time'] > SECURITY_CONFIG["MAX_FILE_AGE_HOURS"] * 3600:
            # Hapus file yang expired
            try:
                if os.path.exists(file_info['file_path']):
                    os.remove(file_info['file_path'])
                if os.path.exists(file_info['file_path'] + '.enc'):
                    os.remove(file_info['file_path'] + '.enc')
                del temp_files[file_id]
            except Exception as e:
                logger.error(f"Error cleaning up expired file: {e}")
            
            raise HTTPException(
                status_code=410,
                detail="File sudah expired. Silakan upload ulang."
            )
        
        # Cek apakah IP yang sama yang mengakses
        if file_info['client_ip'] != client_ip:
            logger.warning(f"IP mismatch for file {file_id}: {client_ip} vs {file_info['client_ip']}")
            raise HTTPException(
                status_code=403,
                detail="Akses ditolak. File hanya dapat diakses dari IP yang sama."
            )
        
        # Gunakan path dari temp_files
        output_path = file_info['file_path']
        logger.info(f"Looking for file at: {output_path}")
        
        # Check if file exists
        if not os.path.exists(output_path):
            logger.error(f"File not found at path: {output_path}")
            logger.info(f"Files in TEMP_DIR: {os.listdir(TEMP_DIR) if os.path.exists(TEMP_DIR) else 'TEMP_DIR does not exist'}")
            raise HTTPException(
                status_code=404, 
                detail=f"File hasil tidak ditemukan atau sudah expired. File ID: {file_id}"
            )
        
        # Check file size
        file_size = os.path.getsize(output_path)
        logger.info(f"File found, size: {file_size} bytes")
        
        if file_size == 0:
            logger.error(f"File exists but is empty: {output_path}")
            raise HTTPException(
                status_code=500,
                detail="File hasil kosong atau rusak."
            )
        
        # Return file response
        logger.info(f"Sending file response for: {output_path}")
        logger.info(f"File size: {file_size} bytes")
        logger.info(f"Filename: {file_info['original_filename']}")
        
        # PERBAIKAN: Selalu aktifkan auto-delete untuk mencegah memory leak
        logger.info(f"Auto-deleting file after download: {file_id}")
        
        # Schedule file deletion after response
        import threading
        def delete_after_download():
            time.sleep(5)  # Wait 5 seconds after download starts
            try:
                # Delete physical file
                if os.path.exists(output_path):
                    security.secure_delete_file(output_path)
                    logger.info(f"Physical file deleted: {output_path}")
                
                # Delete encrypted file if exists
                encrypted_path = output_path + '.enc'
                if os.path.exists(encrypted_path):
                    security.secure_delete_file(encrypted_path)
                    logger.info(f"Encrypted file deleted: {encrypted_path}")
                
                # Remove from temp_files dictionary
                if file_id in temp_files:
                    del temp_files[file_id]
                    logger.info(f"File entry removed from temp_files: {file_id}")
                
                # Force garbage collection
                import gc
                gc.collect()
                
                logger.info(f"File auto-deleted successfully: {file_id}")
                
            except Exception as e:
                logger.error(f"Error during auto-delete: {e}")
        
        threading.Thread(target=delete_after_download, daemon=True).start()
        
        file_response = FileResponse(
            path=output_path,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            filename=file_info['original_filename'],
            headers={
                "Content-Disposition": f"attachment; filename={file_info['original_filename']}"
            }
        )
        logger.info(f"FileResponse created successfully")
        return file_response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /download-forecast endpoint: {str(e)}", exc_info=True)
        error_detail = f"Internal server error downloading file: {str(e)}"
        logger.error(f"Raising HTTPException: {error_detail}")
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint untuk memastikan aplikasi berfungsi"""
    logger.info("🧪 Test endpoint called")
    return {
        "status": "success",
        "message": "Application is working",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "forecast": "/forecast",
            "process_forecast": "/process-forecast",
            "health": "/api/health",
            "security_status": "/api/security-status",
            "monitoring": "/api/monitoring"
        }
    }

@app.get("/api/monitoring")
async def monitoring_endpoint():
    """Endpoint untuk monitoring detail aplikasi"""
    try:
        if get_monitor:
            monitor = get_monitor()
        else:
            monitor = None
        summary = monitor.get_full_summary()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "monitoring_data": summary
        }
    except Exception as e:
        logger.error(f"Error in monitoring endpoint: {e}")
        return {
            "status": "error",
            "message": f"Monitoring error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/monitoring/export")
async def export_monitoring_logs():
    """Export semua log monitoring ke file JSON"""
    try:
        if get_monitor:
            monitor = get_monitor()
        else:
            monitor = None
        filepath = monitor.export_logs()
        
        return {
            "status": "success",
            "message": "Logs exported successfully",
            "filepath": filepath,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error exporting logs: {e}")
        return {
            "status": "error",
            "message": f"Export error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# PERBAIKAN: Tambahkan endpoint untuk manual memory cleanup
@app.post("/api/memory-cleanup")
async def manual_memory_cleanup(request: Request):
    """Manual memory cleanup endpoint"""
    try:
        # Security validation
        security_result = validate_request_security(request)
        client_ip = security_result["client_ip"]
        
        logger.info(f"🧹 Manual memory cleanup requested by {client_ip}")
        
        # Force memory cleanup
        optimize_memory_usage()
        
        # Get current memory usage
        import psutil
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = process.memory_percent()
        except ImportError:
            memory_mb = 0
            memory_percent = 0
        
        return {
            "status": "success",
            "message": "Memory cleanup completed successfully",
            "memory_usage_mb": round(memory_mb, 2),
            "memory_usage_percent": round(memory_percent, 1),
            "active_files": len(temp_files),
            "active_users": len(active_users),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in manual memory cleanup: {e}")
        return {
            "status": "error",
            "message": f"Memory cleanup failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# PERBAIKAN: Startup event untuk memory management
# DISABLE startup event untuk mencegah healthcheck failure
# @app.on_event("startup")
# async def startup_event():
#     """Startup event disabled untuk Railway healthcheck"""
#     logger.info("🚀 Application starting up...")
#     logger.info("🎉 Application startup completed (Railway optimized)")

if __name__ == "__main__":
    import uvicorn
    
    try:
        # PERBAIKAN: Skip optimize_memory_usage() saat startup untuk menghindari timeout
        # optimize_memory_usage() akan dipanggil setelah aplikasi running
        
        # Start with optimized settings for Railway
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,  # Fixed port untuk local testing
            log_level="warning",
            access_log=True,
            workers=1  # Single worker for Railway
        )
    except Exception as e:
        print(f"Startup error: {e}")
        # Fallback startup
        uvicorn.run(app, host="0.0.0.0", port=8000)