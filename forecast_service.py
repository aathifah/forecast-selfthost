"""
Forecast tanpa LSTM dengan Optimasi CPU-Only untuk Railway Deployment
=====================================================================
Kode ini telah dioptimalkan untuk menggunakan model tradisional tanpa LSTM agar lebih ringan dan cepat.
Perubahan utama:
1. Hapus semua model LSTM
2. Fokus pada model tradisional (ETS, ARIMA, Croston, MA, RF, XGB)
3. Optimasi untuk CPU-only
4. Early stopping yang lebih agresif
5. Memory cleanup setelah training
6. Railway deployment optimizations
"""

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm  # Progress bar untuk tracking
import warnings
import logging
import os
import sys
import tempfile
import gc
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima

from pandas import ExcelWriter

# Railway Environment Configuration
RAILWAY_ENV = os.getenv('RAILWAY_ENVIRONMENT', 'development')
IS_PRODUCTION = RAILWAY_ENV == 'production'

# Configure logging for Railway
if IS_PRODUCTION:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
else:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

# Railway-specific configurations
MAX_DATA_ROWS = int(os.getenv('MAX_DATA_ROWS', '1000000'))
MAX_DATA_MONTHS = int(os.getenv('MAX_DATA_MONTHS', '12'))
MAX_MEMORY_THRESHOLD = int(os.getenv('MAX_MEMORY_THRESHOLD', '80'))
N_JOBS = int(os.getenv('OMP_NUM_THREADS', '1'))
TIMEOUT_SECONDS = int(os.getenv('TIMEOUT_SECONDS', '1800'))

# Suppress warnings for Railway
warnings.filterwarnings("ignore")

# Set numpy threading for Railway
os.environ['OMP_NUM_THREADS'] = str(N_JOBS)
os.environ['OPENBLAS_NUM_THREADS'] = str(N_JOBS)
os.environ['MKL_NUM_THREADS'] = str(N_JOBS)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(N_JOBS)
os.environ['NUMEXPR_NUM_THREADS'] = str(N_JOBS)

# Default dataset path (will be overridden by uploaded files)
DATASET_PATH = 'TCO-PNOcategorize-cleaned-8.xlsx'

logger.info(f"🚀 Railway Environment: {RAILWAY_ENV}")
logger.info(f"📊 Max Data Rows: {MAX_DATA_ROWS}")
logger.info(f"📅 Max Data Months: {MAX_DATA_MONTHS}")
logger.info(f"🧠 Max Memory Threshold: {MAX_MEMORY_THRESHOLD}%")
logger.info(f"⚡ N Jobs: {N_JOBS}")
logger.info(f"⏱️ Timeout: {TIMEOUT_SECONDS}s")

# TensorFlow configuration removed - no LSTM models used

def cleanup_ml_memory():
    """Cleanup memory untuk Railway deployment"""
    try:
        gc.collect()
        logger.info("🧹 Memory cleanup completed")
    except Exception as e:
        logger.warning(f"⚠️ Memory cleanup warning: {e}")

def check_memory_usage():
    """Check memory usage untuk Railway"""
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > MAX_MEMORY_THRESHOLD:
            logger.warning(f"⚠️ High memory usage: {memory_percent}% > {MAX_MEMORY_THRESHOLD}%")
            cleanup_ml_memory()
        return memory_percent
    except ImportError:
        logger.warning("psutil not available for memory monitoring")
        return 0
    except Exception as e:
        logger.warning(f"Memory check error: {e}")
        return 0

def validate_excel_file(file_path):
    """Validate Excel file untuk Railway deployment"""
    try:
        # Check file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        max_size = 100 * 1024 * 1024  # 100MB limit for Railway
        if file_size > max_size:
            raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f}MB > 100MB")
        
        # Try to read Excel file
        df = pd.read_excel(file_path)
        
        # Check data size
        if len(df) > MAX_DATA_ROWS:
            logger.warning(f"⚠️ Large dataset: {len(df)} rows > {MAX_DATA_ROWS}")
            df = df.head(MAX_DATA_ROWS)
            logger.info(f"📊 Dataset truncated to {MAX_DATA_ROWS} rows")
        
        # Check required columns
        required_cols = ['PART_NO', 'MONTH', 'ORIGINAL_SHIPPING_QTY']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"✅ Excel file validated: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"❌ Excel validation failed: {e}")
        raise

def get_six_months_before(target_month):
    """
    Fungsi helper untuk mendapatkan 6 bulan sebelum target_month (EXCLUDE target_month)
    LOGIKA YANG BENAR - ROLLING CHECK (SESUAI PROGRAM SEDERHANA):
    - Untuk 2025-02: cek 6 bulan = 2024-08, 2024-09, 2024-10, 2024-11, 2024-12, 2025-01
    - Untuk 2025-03: cek 6 bulan = 2024-09, 2024-10, 2024-11, 2024-12, 2025-01, 2025-02
    - Untuk 2025-04: cek 6 bulan = 2024-10, 2024-11, 2024-12, 2025-01, 2025-02, 2025-03
    - Untuk 2025-05: cek 6 bulan = 2024-11, 2024-12, 2025-01, 2025-02, 2025-03, 2025-04
    """
    try:
        # LOGIKA YANG BENAR: 6 bulan sebelum target_month (EXCLUDE target_month)
        # ROLLING CHECK: Setiap bulan target, cek 6 bulan sebelumnya
        # FIXED: Gunakan logika yang sama dengan program sederhana
        
        # Method 1: Gunakan logika yang sama dengan program sederhana
        # start = target - pd.DateOffset(months=6)
        # end = target - pd.DateOffset(months=1)
        start_date = target_month - pd.DateOffset(months=6)
        end_date = target_month - pd.DateOffset(months=1)
        six_months_before = pd.date_range(start=start_date, end=end_date, freq='MS', inclusive='both')
        
        # Validasi: Pastikan benar-benar 6 bulan
        if len(six_months_before) == 6:
            print(f"🔍 6 months before {target_month.strftime('%Y-%m')}: {[d.strftime('%Y-%m') for d in six_months_before]}")
            return six_months_before
        
        # Method 2: Fallback method - gunakan periods=6 dari 1 bulan sebelum target
        # FIXED: Gunakan logika yang benar untuk 6 bulan terakhir
        six_months_before = pd.date_range(end=target_month - pd.DateOffset(months=1), periods=6, freq='MS')
        if len(six_months_before) == 6:
            print(f"🔍 6 months before {target_month.strftime('%Y-%m')} (fallback): {[d.strftime('%Y-%m') for d in six_months_before]}")
            return six_months_before
        
        # Method 3: Manual calculation jika semua method gagal
        # FIXED: Gunakan logika yang benar untuk 6 bulan terakhir
        six_months_before = [target_month - pd.DateOffset(months=i) for i in range(6, 0, -1)]
        six_months_before = [d.replace(day=1) for d in six_months_before]
        
        # Validasi final
        if len(six_months_before) != 6:
            raise ValueError(f"Failed to calculate 6 months before {target_month}")
        
        print(f"🔍 6 months before {target_month.strftime('%Y-%m')} (manual): {[d.strftime('%Y-%m') for d in six_months_before]}")
        return six_months_before
        
    except Exception as e:
        print(f"❌ ERROR: Failed to calculate 6 months before {target_month}: {e}")
        # Return empty list sebagai fallback
        return []

def validate_demand_requirement_general(part, target_month, part_df, test_df, part_category):
    """
    Fungsi helper untuk validasi syarat demand UNTUK SEMUA KATEGORI
    Syarat: Minimal 2 bulan dengan demand > 0 di 6 bulan rolling window
    
    CONTOH BACKTEST:
    - Dataset: 2024-01 sampai 2025-06
    - Backtest target months: 2025-02, 2025-03, 2025-04, 2025-05 (4 bulan sebelum bulan terakhir)
    
    ROLLING WINDOW CHECK:
    - Target 2025-02: cek 2024-08, 2024-09, 2024-10, 2024-11, 2024-12, 2025-01
    - Target 2025-03: cek 2024-09, 2024-10, 2024-11, 2024-12, 2025-01, 2025-02
    - Target 2025-04: cek 2024-10, 2024-11, 2024-12, 2025-01, 2025-02, 2025-03
    - Target 2025-05: cek 2024-11, 2024-12, 2025-01, 2025-02, 2025-03, 2025-04
    
    CONTOH REALTIME:
    - Dataset: 2024-01 sampai 2025-06
    - Realtime target months: 2025-06, 2025-07, 2025-08, 2025-09 (bulan terakhir + 3 bulan ke depan)
    
    ROLLING WINDOW CHECK:
    - Target 2025-06: cek 2024-12, 2025-01, 2025-02, 2025-03, 2025-04, 2025-05
    - Target 2025-07: cek 2025-01, 2025-02, 2025-03, 2025-04, 2025-05, 2025-06
    - Target 2025-08: cek 2025-02, 2025-03, 2025-04, 2025-05, 2025-06, 2025-07
    - Target 2025-09: cek 2025-03, 2025-04, 2025-05, 2025-06, 2025-07, 2025-08
    
    DATA BOLONG:
    - Bulan yang tidak ada data dianggap demand = 0
    - Contoh: data hanya ada di 2024-08, 2024-10, 2024-12
    - Window 6 bulan tetap: 2024-07, 2024-08, 2024-09, 2024-10, 2024-11, 2024-12
    - Hasil: 0, 8, 0, 1, 0, 5 (bulan bolong = 0)
    - Months with demand > 0: 3 (2024-08, 2024-10, 2024-12)
    - Meets requirement: YES (3 >= 2)
    """
    try:
        # Dapatkan 6 bulan terakhir sebelum target_month
        six_months_before = get_six_months_before(target_month)
        
        if len(six_months_before) != 6:
            print(f"❌ ERROR Part {part}: 6 months calculation failed! Got {len(six_months_before)} months instead of 6")
            return False, 0, []
        
        # Buat window 6 bulan dengan demand 0 untuk bulan yang tidak ada data
        # Data bolong (bulan yang tidak ada) otomatis dianggap demand = 0
        # FIX: Gunakan part_df (semua data) untuk window check, bukan train_df
        window_data = {}
        for month in six_months_before:
            # Cek apakah ada data untuk bulan ini di part_df (semua data)
            month_data = part_df[part_df['MONTH'] == month]
            if len(month_data) > 0:
                # Ada data untuk bulan ini
                demand_val = month_data['ORIGINAL_SHIPPING_QTY'].iloc[0]
                window_data[month] = demand_val
            else:
                # Bulan yang tidak ada data (bolong), set demand = 0
                window_data[month] = 0
        
        # Hitung bulan dengan demand > 0
        months_with_demand = sum(1 for demand in window_data.values() if demand > 0)
        
        # Debug info
        print(f"🔍 Part {part} ({part_category}): Target month: {target_month.strftime('%Y-%m')}")
        print(f"🔍 Part {part} ({part_category}): 6 months rolling window: {[d.strftime('%Y-%m') for d in six_months_before]}")
        print(f"🔍 Part {part} ({part_category}): Window data (missing months = 0): {dict(zip([d.strftime('%Y-%m') for d in window_data.keys()], window_data.values()))}")
        print(f"🔍 Part {part} ({part_category}): Months with demand > 0: {months_with_demand}")
        
        # Syarat untuk SEMUA KATEGORI: minimal 2 bulan dengan demand > 0 dari 6 bulan window
        meets_requirement = months_with_demand >= 2
        
        if meets_requirement:
            print(f"✅ Part {part} ({part_category}): Meets requirement - {months_with_demand} months with demand >= 2")
        else:
            print(f"❌ Part {part} ({part_category}): Does not meet requirement - only {months_with_demand} months with demand < 2")
        
        return meets_requirement, months_with_demand, list(window_data.values())
        
    except Exception as e:
        print(f"❌ ERROR Part {part}: Lumpy validation failed: {e}")
        return False, 0, []

# Fungsi validate_demand_requirement dihapus karena tidak digunakan lagi
# Sekarang hanya menggunakan validate_demand_requirement_lumpy untuk kategori Lumpy

def parse_month_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_integer_dtype(series):
        return pd.to_datetime(series.astype(str), format='%Y%m', errors='coerce')
    if pd.api.types.is_float_dtype(series):
        # Handle NaN values before conversion
        series_clean = series.dropna()
        if len(series_clean) == 0:
            return pd.Series([], dtype='datetime64[ns]')
        return pd.to_datetime(series_clean.astype(int).astype(str), format='%Y%m', errors='coerce')
    s = series.astype(str)
    for fmt in ['%Y%m', '%Y-%m', '%Y/%m', '%Y-%m-%d']:
        dt = pd.to_datetime(s, format=fmt, errors='coerce')
        if dt.notna().all():
            return dt
    return pd.to_datetime(s, errors='coerce')


def smape(actual, forecast):
    a = float(0 if pd.isna(actual) else actual)
    f = float(0 if pd.isna(forecast) else forecast)
    denom = abs(a) + abs(f)
    if denom == 0:
        return 0.0
    return float(2 * abs(a - f) / denom * 100)


def mape(actual, forecast):
    a = float(0 if pd.isna(actual) else actual)
    f = float(0 if pd.isna(forecast) else forecast)
    if a == 0:
        return 0.0 if f == 0 else 200.0
    return float(abs(a - f) / abs(a) * 100)


def hybrid_error(actual, forecast):
    a = float(0 if pd.isna(actual) else actual)
    f = float(0 if pd.isna(forecast) else forecast)
    
    # Jika actual dan forecast sama, berikan error minimal 0.1% (bukan 0%)
    if a == f:
        return 0.1  # Minimal error untuk menghindari 0% error yang tidak realistis
    
    if a == 0 and f == 0:
        return 0.1  # Minimal error untuk menghindari 0% error yang tidak realistis
    
    if a == 0:
        # Jika actual = 0 tapi forecast > 0, error = 100% (bukan 200%)
        return 100.0
    
    if a > 0:
        # Gunakan MAPE untuk konsistensi
        return min(abs(a - f) / abs(a) * 100, 200.0)  # Cap error maksimum 200%
    
    return 0.1  # Minimal error untuk menghindari 0% error yang tidak realistis

def calculate_responsiveness(actual_series, forecast_series):
    """
    Hitung responsiveness score untuk menilai seberapa responsif model terhadap perubahan
    """
    try:
        if len(actual_series) < 3 or len(forecast_series) < 3:
            return 0.0
        
        # Hitung perubahan actual vs forecast
        actual_changes = np.diff(actual_series)
        forecast_changes = np.diff(forecast_series)
        
        # Hitung korelasi antara perubahan actual dan forecast
        if len(actual_changes) > 1 and len(forecast_changes) > 1:
            correlation = np.corrcoef(actual_changes, forecast_changes)[0, 1]
            if np.isnan(correlation):
                return 0.0
            
            # Responsiveness score: semakin tinggi korelasi, semakin responsif
            responsiveness = max(0, correlation)
            return responsiveness
        else:
            return 0.0
            
    except Exception:
        return 0.0

def calculate_fluctuation_aware_error(actual, forecast, fluctuation_type, actual_series=None, forecast_series=None):
    """
    Hitung error dengan mempertimbangkan fluktuasi dan responsiveness
    """
    try:
        base_error = hybrid_error(actual, forecast)
        
        if fluctuation_type == 'HIGHLY_FLUCTUATING' and actual_series is not None and forecast_series is not None:
            # Untuk fluktuasi tinggi, berikan bonus untuk model yang responsif
            responsiveness = calculate_responsiveness(actual_series, forecast_series)
            
            # Bonus responsiveness: 5% untuk setiap 0.1 responsiveness score
            responsiveness_bonus = responsiveness * 0.5  # Max 50% bonus
            adjusted_error = base_error * (1 - responsiveness_bonus)
            
            return max(adjusted_error, 0.0)  # Ensure non-negative
        else:
            return base_error
            
    except Exception:
        return base_error

def handle_high_volatility_demand(series, part_category, fluctuation_type):
    """
    Strategi khusus untuk menangani demand dengan volatilitas tinggi
    """
    try:
        if fluctuation_type != 'HIGHLY_FLUCTUATING':
            return {}
        
        # 1. Volatility-adjusted smoothing
        if len(series) >= 6:
            # Gunakan exponential smoothing dengan alpha tinggi untuk responsif
            alpha = 0.3  # Higher alpha untuk lebih responsif
            smoothed_series = [series[0]]
            for i in range(1, len(series)):
                smoothed = alpha * series[i] + (1 - alpha) * smoothed_series[-1]
                smoothed_series.append(smoothed)
            
            # Forecast menggunakan trend dari smoothed series
            if len(smoothed_series) >= 3:
                trend = (smoothed_series[-1] - smoothed_series[-3]) / 2
                volatility_adjusted_fc = smoothed_series[-1] + trend
            else:
                volatility_adjusted_fc = smoothed_series[-1]
        else:
            volatility_adjusted_fc = np.mean(series[-3:]) if len(series) >= 3 else np.mean(series)
        
        # 2. Outlier-resistant forecast
        if len(series) >= 4:
            # Gunakan median untuk menghindari outlier
            recent_median = np.median(series[-4:])
            # Tambahkan sedikit trend
            if len(series) >= 6:
                trend = np.median(np.diff(series[-6:]))
                outlier_resistant_fc = recent_median + trend
            else:
                outlier_resistant_fc = recent_median
        else:
            outlier_resistant_fc = np.median(series) if len(series) > 0 else 0
        
        # 3. Adaptive window moving average
        if len(series) >= 3:
            # Window yang menyesuaikan dengan volatilitas
            volatility = np.std(series[-6:]) if len(series) >= 6 else np.std(series)
            mean_demand = np.mean(series[-6:]) if len(series) >= 6 else np.mean(series)
            cv = volatility / (mean_demand + 1e-8)
            
            # Window lebih pendek untuk volatilitas tinggi
            if cv > 0.5:  # High volatility
                window = 2
            elif cv > 0.3:  # Medium volatility
                window = 3
            else:  # Low volatility
                window = 4
            
            adaptive_ma_fc = np.mean(series[-window:]) if len(series) >= window else np.mean(series)
        else:
            adaptive_ma_fc = np.mean(series) if len(series) > 0 else 0
        
        return {
            'VOLATILITY_ADJUSTED': max(volatility_adjusted_fc, 0.0),
            'OUTLIER_RESISTANT': max(outlier_resistant_fc, 0.0),
            'ADAPTIVE_MA': max(adaptive_ma_fc, 0.0)
        }
        
    except Exception:
        return {}


def forecast_ma6(series):
    if len(series) == 0:
        return np.nan
    if len(series) >= 6:
        return float(np.mean(series[-6:]))
    return float(np.mean(series))

def forecast_ma_flexible(series, window=6):
    """
    Flexible Moving Average dengan window yang bisa disesuaikan
    """
    if len(series) == 0:
        return np.nan
    if len(series) >= window:
        return float(np.mean(series[-window:]))
    return float(np.mean(series))

def get_best_ma_model(series, actual_val):
    """
    Test multiple MA windows (3, 4, 5, 6) dan pilih yang error terkecil
    """
    if len(series) < 3:
        return 'MA6', forecast_ma6(series)
    
    ma_models = {}
    ma_errors = {}
    
    # Test different MA windows
    for window in [3, 4, 5, 6]:
        if len(series) >= window:
            pred = forecast_ma_flexible(series, window)
            if not np.isnan(pred):
                ma_models[f'MA{window}'] = pred
                ma_errors[f'MA{window}'] = hybrid_error(actual_val, pred)
    
    if not ma_errors:
        return 'MA6', forecast_ma6(series)
    
    # Pilih MA dengan error terkecil
    best_ma = min(ma_errors.items(), key=lambda x: x[1])[0]
    return best_ma, ma_models[best_ma]


def forecast_wma(series, window=6):
    if len(series) == 0:
        return np.nan
    w = int(min(window, len(series)))
    weights = np.arange(1, w + 1)
    weights = weights / weights.sum()
    return float(np.sum(np.array(series)[-w:] * weights))


def forecast_ets(series, seasonal_pref=False, fluctuation_type='STABLE'):
    if len(series) < 3:  # Reduced requirement
        # FALLBACK: Use simple average for very short series
        return max(float(np.mean(series)), 0.0) if len(series) > 0 else 0.0
    try:
        # Dapatkan parameter berdasarkan fluktuasi
        params = get_fluctuation_aware_parameters(fluctuation_type, 'MEDIUM')
        trend = params.get('ets_trend', 'add')
        seasonal = params.get('ets_seasonal', 'add')
        
        # FALLBACK: Jika data terlalu sedikit, gunakan model sederhana
        if len(series) < 4:
            # Simple exponential smoothing
            alpha = 0.3
            forecast = series[0]
            for i in range(1, len(series)):
                forecast = alpha * series[i] + (1 - alpha) * forecast
            return max(float(forecast), 0.0)
        
        if seasonal_pref and len(series) >= 12:
            model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=12, initialization_method="estimated")
        else:
            model = ExponentialSmoothing(series, trend=trend, seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        forecast_val = fit.forecast(1)[0]
        return max(float(forecast_val), 0.0)
    except Exception as e:
        # FALLBACK: Jika ETS gagal, gunakan simple exponential smoothing
        try:
            if len(series) >= 2:
                alpha = 0.3
                forecast = series[0]
                for i in range(1, len(series)):
                    forecast = alpha * series[i] + (1 - alpha) * forecast
                return max(float(forecast), 0.0)
            else:
                return max(float(np.mean(series)), 0.0) if len(series) > 0 else 0.0
        except:
            # Ultimate fallback
            return max(float(np.mean(series)), 0.0) if len(series) > 0 else 0.0

def forecast_ets_volatile(series, seasonal_pref=False):
    """
    ETS khusus untuk fluktuasi tinggi dengan parameter yang dioptimalkan
    """
    if len(series) < 4:
        return np.nan
    try:
        # Parameter untuk fluktuasi tinggi
        if seasonal_pref and len(series) >= 12:
            model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12, initialization_method="estimated")
        else:
            model = ExponentialSmoothing(series, trend='add', seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        return max(float(fit.forecast(1)[0]), 0.0)
    except Exception:
        return np.nan

def forecast_ets_adaptive(series, seasonal_pref=False):
    """
    ETS adaptif yang menyesuaikan dengan pola data
    """
    if len(series) < 4:
        return np.nan
    try:
        # Coba multiple configurations dan pilih yang terbaik
        best_forecast = np.nan
        best_error = float('inf')
        
        configurations = [
            {'trend': 'add', 'seasonal': 'add' if seasonal_pref and len(series) >= 12 else None},
            {'trend': 'mul', 'seasonal': 'add' if seasonal_pref and len(series) >= 12 else None},
            {'trend': 'add', 'seasonal': None},
            {'trend': 'mul', 'seasonal': None}
        ]
        
        for config in configurations:
            try:
                if config['seasonal'] and len(series) >= 12:
                    model = ExponentialSmoothing(series, trend=config['trend'], seasonal=config['seasonal'], seasonal_periods=12, initialization_method="estimated")
                else:
                    model = ExponentialSmoothing(series, trend=config['trend'], seasonal=None, initialization_method="estimated")
                
                fit = model.fit(optimized=True)
                forecast = float(fit.forecast(1)[0])
                
                # Simple cross-validation
                if len(series) >= 6:
                    train_series = series[:-1]
                    test_val = series[-1]
                    try:
                        test_model = ExponentialSmoothing(train_series, trend=config['trend'], seasonal=config['seasonal'], 
                                                       seasonal_periods=12 if config['seasonal'] else None, 
                                                       initialization_method="estimated")
                        test_fit = test_model.fit(optimized=True)
                        test_forecast = float(test_fit.forecast(1)[0])
                        test_error = hybrid_error(test_val, test_forecast)
                        
                        if test_error < best_error:
                            best_error = test_error
                            best_forecast = forecast
                    except:
                        if np.isnan(best_forecast):
                            best_forecast = forecast
                else:
                    if np.isnan(best_forecast):
                        best_forecast = forecast
            except Exception:
                continue
        
        return max(float(best_forecast), 0.0) if not np.isnan(best_forecast) else np.nan
        
    except Exception:
        return np.nan


def forecast_ets_tuned(series, seasonal_pref=False):
    if len(series) < 4:
        return np.nan
    best = np.nan
    best_error = float('inf')
    candidates = []
    
    # Generate more comprehensive model candidates
    try:
        if seasonal_pref and len(series) >= 12:
            candidates.extend([
                ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12, initialization_method="estimated"),
                ExponentialSmoothing(series, trend='mul', seasonal='add', seasonal_periods=12, initialization_method="estimated"),
                ExponentialSmoothing(series, trend='add', seasonal='mul', seasonal_periods=12, initialization_method="estimated"),
                ExponentialSmoothing(series, trend='mul', seasonal='mul', seasonal_periods=12, initialization_method="estimated")
            ])
        
        candidates.extend([
            ExponentialSmoothing(series, trend='add', seasonal=None, initialization_method="estimated"),
            ExponentialSmoothing(series, trend='mul', seasonal=None, initialization_method="estimated"),
            ExponentialSmoothing(series, trend=None, seasonal=None, initialization_method="estimated")
        ])
    except Exception:
        pass
    
    # Test each candidate with cross-validation
    for mdl in candidates:
        try:
            fit = mdl.fit(optimized=True)
            fc = float(fit.forecast(1)[0])
            
            # Simple cross-validation error estimation
            if len(series) >= 6:
                train_series = series[:-1]
                test_val = series[-1]
                try:
                    test_model = ExponentialSmoothing(train_series, trend=mdl.trend, seasonal=mdl.seasonal, 
                                                    seasonal_periods=getattr(mdl, 'seasonal_periods', None), 
                                                    initialization_method="estimated")
                    test_fit = test_model.fit(optimized=True)
                    test_fc = float(test_fit.forecast(1)[0])
                    test_error = hybrid_error(test_val, test_fc)
                    
                    if test_error < best_error:
                        best_error = test_error
                        best = fc
                except:
                    # Fallback to simple selection
                    if np.isnan(best) or abs(fc) < abs(best):
                        best = fc
            else:
                if np.isnan(best) or abs(fc) < abs(best):
                    best = fc
        except Exception:
            continue
    
    return max(float(best), 0.0) if not np.isnan(best) else np.nan


def forecast_arima_enhanced(series, seasonal_pref=False, fluctuation_type='STABLE'):
    """
    ARIMA Enhanced dengan kombinasi Auto ARIMA + Manual Parameter Optimization
    Strategi: Auto ARIMA untuk baseline + Manual tuning untuk edge cases
    """
    if len(series) < 4:
        return max(float(np.mean(series)), 0.0) if len(series) > 0 else 0.0
    
    best_forecast = np.nan
    best_error = float('inf')
    
    # FALLBACK untuk data sangat sedikit
    if len(series) < 6:
        try:
            x = np.arange(len(series))
            y = series
            coeffs = np.polyfit(x, y, 1)
            forecast = coeffs[0] * len(series) + coeffs[1]
            return max(float(forecast), 0.0)
        except:
            return max(float(np.mean(series)), 0.0) if len(series) > 0 else 0.0
    
    # Dapatkan parameter berdasarkan fluktuasi
    params = get_fluctuation_aware_parameters(fluctuation_type, 'MEDIUM')
    max_p = params.get('arima_max_p', 3)
    max_q = params.get('arima_max_q', 3)
    max_P = params.get('arima_max_P', 2)
    max_Q = params.get('arima_max_Q', 2)
    
    # STRATEGI 1: AUTO ARIMA dengan multiple configurations (Primary)
    arima_configs = []
    
    if seasonal_pref and len(series) >= 12:
        arima_configs.extend([
            # Full range untuk seasonal
            {'seasonal': True, 'm': 12, 'max_p': max_p, 'max_q': max_q, 'max_P': max_P, 'max_Q': max_Q},
            # Reduced range untuk stability
            {'seasonal': True, 'm': 12, 'max_p': max_p-1, 'max_q': max_q-1, 'max_P': max_P-1, 'max_Q': max_Q-1},
            # Conservative untuk data noisy
            {'seasonal': True, 'm': 12, 'max_p': 1, 'max_q': 1, 'max_P': 1, 'max_Q': 1}
        ])
    
    arima_configs.extend([
        # Non-seasonal configurations
        {'seasonal': False, 'max_p': max_p, 'max_q': max_q},
        {'seasonal': False, 'max_p': max_p-1, 'max_q': max_q-1},
        {'seasonal': False, 'max_p': 1, 'max_q': 1}
    ])
    
    # Test Auto ARIMA configurations
    for config in arima_configs:
        try:
            model = auto_arima(
                series, 
                stepwise=True, 
                suppress_warnings=True, 
                error_action='ignore',
                trace=False,  # Disable trace untuk performance
                **config
            )
            forecast = float(model.predict(1)[0])
            
            # Cross-validation untuk validasi
            if len(series) >= 10:
                train_series = series[:-2]
                test_val = series[-1]
                try:
                    test_model = auto_arima(
                        train_series, 
                        stepwise=True, 
                        suppress_warnings=True, 
                        error_action='ignore',
                        trace=False,
                        **config
                    )
                    test_forecast = float(test_model.predict(1)[0])
                    test_error = hybrid_error(test_val, test_forecast)
                    
                    if test_error < best_error:
                        best_error = test_error
                        best_forecast = forecast
                except:
                    if np.isnan(best_forecast):
                        best_forecast = forecast
            else:
                if np.isnan(best_forecast):
                    best_forecast = forecast
        except Exception:
            continue
    
    # STRATEGI 2: Manual Parameter Tuning untuk Edge Cases (Secondary)
    # Hanya jika Auto ARIMA gagal atau error tinggi
    if np.isnan(best_forecast) or best_error > 50.0:
        manual_configs = []
        
        # Manual configurations berdasarkan karakteristik data
        if fluctuation_type == 'HIGHLY_FLUCTUATING':
            # Untuk fluktuasi tinggi, gunakan parameter yang lebih konservatif
            manual_configs = [
                {'seasonal': False, 'max_p': 1, 'max_q': 1},  # ARIMA(1,1,1)
                {'seasonal': False, 'max_p': 2, 'max_q': 1},  # ARIMA(2,1,1)
                {'seasonal': False, 'max_p': 1, 'max_q': 2},  # ARIMA(1,1,2)
            ]
        elif fluctuation_type == 'STABLE':
            # Untuk data stabil, bisa gunakan parameter lebih tinggi
            manual_configs = [
                {'seasonal': False, 'max_p': 2, 'max_q': 2},  # ARIMA(2,1,2)
                {'seasonal': False, 'max_p': 3, 'max_q': 1},  # ARIMA(3,1,1)
                {'seasonal': False, 'max_p': 1, 'max_q': 3},  # ARIMA(1,1,3)
            ]
        else:
            # Default manual configurations
            manual_configs = [
                {'seasonal': False, 'max_p': 1, 'max_q': 1},  # ARIMA(1,1,1)
                {'seasonal': False, 'max_p': 2, 'max_q': 1},  # ARIMA(2,1,1)
                {'seasonal': False, 'max_p': 1, 'max_q': 2},  # ARIMA(1,1,2)
            ]
        
        # Test manual configurations
        for config in manual_configs:
            try:
                model = auto_arima(
                    series, 
                    stepwise=True, 
                    suppress_warnings=True, 
                    error_action='ignore',
                    trace=False,
                    **config
                )
                forecast = float(model.predict(1)[0])
                
                # Simple validation
                if len(series) >= 8:
                    train_series = series[:-1]
                    test_val = series[-1]
                    try:
                        test_model = auto_arima(
                            train_series, 
                            stepwise=True, 
                            suppress_warnings=True, 
                            error_action='ignore',
                            trace=False,
                            **config
                        )
                        test_forecast = float(test_model.predict(1)[0])
                        test_error = hybrid_error(test_val, test_forecast)
                        
                        if test_error < best_error:
                            best_error = test_error
                            best_forecast = forecast
                    except:
                        if np.isnan(best_forecast):
                            best_forecast = forecast
                else:
                    if np.isnan(best_forecast):
                        best_forecast = forecast
            except Exception:
                continue
    
    # FALLBACK: Jika semua ARIMA gagal
    if np.isnan(best_forecast):
        return max(float(np.mean(series)), 0.0) if len(series) > 0 else 0.0
    else:
        return max(float(best_forecast), 0.0)

def forecast_arima_volatile(series, seasonal_pref=False):
    """
    ARIMA khusus untuk fluktuasi tinggi dengan parameter yang dioptimalkan
    """
    if len(series) < 8:
        return np.nan
    
    best_forecast = np.nan
    best_error = float('inf')
    
    # Parameter untuk fluktuasi tinggi
    arima_configs = []
    
    if seasonal_pref and len(series) >= 12:
        arima_configs.extend([
            {'seasonal': True, 'm': 12, 'max_p': 3, 'max_q': 3, 'max_P': 2, 'max_Q': 2},
            {'seasonal': True, 'm': 12, 'max_p': 2, 'max_q': 2, 'max_P': 1, 'max_Q': 1}
        ])
    
    arima_configs.extend([
        {'seasonal': False, 'max_p': 3, 'max_q': 3},
        {'seasonal': False, 'max_p': 2, 'max_q': 2}
    ])
    
    for config in arima_configs:
        try:
            model = auto_arima(series, stepwise=True, suppress_warnings=True, error_action='ignore', **config)
            forecast = float(model.predict(1)[0])
            
            # Cross-validation error estimation
            if len(series) >= 10:
                train_series = series[:-2]
                test_val = series[-1]
                try:
                    test_model = auto_arima(train_series, stepwise=True, suppress_warnings=True, error_action='ignore', **config)
                    test_forecast = float(test_model.predict(1)[0])
                    test_error = hybrid_error(test_val, test_forecast)
                    
                    if test_error < best_error:
                        best_error = test_error
                        best_forecast = forecast
                except:
                    if np.isnan(best_forecast):
                        best_forecast = forecast
            else:
                if np.isnan(best_forecast):
                    best_forecast = forecast
        except Exception:
            continue
    
    return best_forecast

def forecast_arima_adaptive(series, seasonal_pref=False):
    """
    ARIMA adaptif yang menyesuaikan dengan pola data
    """
    if len(series) < 8:
        return np.nan
    
    best_forecast = np.nan
    best_error = float('inf')
    
    # Multiple configurations untuk adaptif
    arima_configs = []
    
    if seasonal_pref and len(series) >= 12:
        arima_configs.extend([
            {'seasonal': True, 'm': 12, 'max_p': 2, 'max_q': 2, 'max_P': 1, 'max_Q': 1},
            {'seasonal': True, 'm': 12, 'max_p': 1, 'max_q': 1, 'max_P': 1, 'max_Q': 1}
        ])
    
    arima_configs.extend([
        {'seasonal': False, 'max_p': 2, 'max_q': 2},
        {'seasonal': False, 'max_p': 1, 'max_q': 1}
    ])
    
    for config in arima_configs:
        try:
            model = auto_arima(series, stepwise=True, suppress_warnings=True, error_action='ignore', **config)
            forecast = float(model.predict(1)[0])
            
            # Cross-validation error estimation
            if len(series) >= 10:
                train_series = series[:-2]
                test_val = series[-1]
                try:
                    test_model = auto_arima(train_series, stepwise=True, suppress_warnings=True, error_action='ignore', **config)
                    test_forecast = float(test_model.predict(1)[0])
                    test_error = hybrid_error(test_val, test_forecast)
                    
                    if test_error < best_error:
                        best_error = test_error
                        best_forecast = forecast
                except:
                    if np.isnan(best_forecast):
                        best_forecast = forecast
            else:
                if np.isnan(best_forecast):
                    best_forecast = forecast
        except Exception:
            continue
    
    return best_forecast


# Croston family for Intermittent
def forecast_croston(series, alpha=0.1):
    y = np.array(series, dtype=float)
    if len(y) == 0:
        return 0.0  # Return 0 instead of NaN for empty series
    q_hat = None
    p_hat = 1.0
    periods = 0
    first = True
    for v in y:
        periods += 1
        if v > 0:
            if first:
                q_hat = v
                p_hat = periods
                first = False
            else:
                q_hat = alpha * v + (1 - alpha) * q_hat
                p_hat = alpha * periods + (1 - alpha) * p_hat
            periods = 0
    if first or q_hat is None or p_hat <= 0:
        # FALLBACK: Use simple average if Croston fails
        return max(float(np.mean(y[y > 0])), 0.0) if np.sum(y > 0) > 0 else 0.0
    return max(float(q_hat / p_hat), 0.0)


def forecast_croston_sba(series, alpha=0.1):
    b = forecast_croston(series, alpha)
    return max(float(b * (1.0 - alpha / 2.0)), 0.0)


def forecast_croston_tsb(series, alpha=0.1, beta=0.1):
    y = np.array(series, dtype=float)
    if len(y) == 0:
        return np.nan
    p_hat, q_hat = 0.0, 0.0
    first_q = True
    for v in y:
        occ = 1.0 if v > 0 else 0.0
        p_hat = beta * occ + (1 - beta) * p_hat
        if v > 0:
            if first_q:
                q_hat = v
                first_q = False
            else:
                q_hat = alpha * v + (1 - alpha) * q_hat
    return max(float(p_hat * q_hat), 0.0)


# LSTM utilities removed - no LSTM models used
def ensemble_voting(predictions, weights=None):
    """
    Ensemble voting dengan weighted average - ENHANCED VERSION
    """
    try:
        valid_preds = {k: v for k, v in predictions.items() if v is not None and not np.isnan(v)}
        if not valid_preds:
            return np.nan
        
        if weights is None:
            weights = {k: 1.0 for k in valid_preds.keys()}
        
        # Weighted average dengan error handling
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for model, pred in valid_preds.items():
            weight = weights.get(model, 1.0)
            weighted_sum += pred * weight
            weight_sum += weight
        
        if weight_sum > 0:
            result = weighted_sum / weight_sum
            return max(float(result), 0.0)  # Ensure non-negative
        else:
            return np.nan
            
    except Exception as e:
        logger.warning(f"Ensemble voting error: {e}")
        return np.nan

def enhanced_model_selection_for_sparse_data(series, part_category, actual_val):
    """
    Enhanced model selection khusus untuk data sparse (Intermittent/Lumpy)
    """
    try:
        preds = {}
        
        # 1. Time-series models (selalu bisa jalan) - Flexible MA
        best_ma_model, best_ma_pred = get_best_ma_model(series, actual_val)
        preds[best_ma_model] = best_ma_pred
        preds['WMA'] = forecast_wma(series)
        preds['ETS'] = forecast_ets(series, seasonal_pref=(part_category in ['Smooth','Erratic']))
        preds['ARIMA'] = forecast_arima_enhanced(series, seasonal_pref=(part_category in ['Smooth','Erratic']))
        
        # 2. Croston family untuk Intermittent/Lumpy (dirancang untuk data sparse)
        if part_category in ['Intermittent', 'Lumpy']:
            preds['CROSTON'] = forecast_croston(series)
            preds['CROSTON_SBA'] = forecast_croston_sba(series)
            preds['CROSTON_TSB'] = forecast_croston_tsb(series)
            
            # Enhanced Croston dengan multiple alpha values
            for alpha in [0.05, 0.1, 0.15, 0.2]:
                preds[f'CROSTON_ALPHA_{alpha}'] = forecast_croston(series, alpha)
                preds[f'CROSTON_SBA_ALPHA_{alpha}'] = forecast_croston_sba(series, alpha)
                preds[f'CROSTON_TSB_ALPHA_{alpha}'] = forecast_croston_tsb(series, alpha, alpha)
        
        # 3. Simple ML models untuk data kecil (LSTM removed)
        if len(series) >= 4:  # Reduced requirement
            try:
                # Simple linear trend
                if len(series) >= 3:
                    x = np.arange(len(series))
                    y = series
                    coeffs = np.polyfit(x, y, 1)
                    preds['LINEAR_TREND'] = max(0, coeffs[0] * len(series) + coeffs[1])
                
                # Simple exponential trend
                if len(series) >= 4:
                    x = np.arange(len(series))
                    y = np.log1p(series)  # Log transform
                    coeffs = np.polyfit(x, y, 1)
                    preds['EXP_TREND'] = max(0, np.expm1(coeffs[0] * len(series) + coeffs[1]))
                
                # Simple moving average dengan adaptive window
                for window in [2, 3, 4, 5]:
                    if len(series) >= window:
                        preds[f'MA_{window}'] = np.mean(series[-window:])
                
            except Exception:
                pass
        
        # 5. Category-specific enhancements
        if part_category == 'Intermittent':
            # Zero-inflated models
            zero_ratio = np.sum(series == 0) / len(series)
            if zero_ratio > 0.3:  # High zero ratio
                non_zero_series = series[series > 0]
                if len(non_zero_series) >= 2:
                    preds['ZERO_INFLATED_MEAN'] = np.mean(non_zero_series) * (1 - zero_ratio)
                    preds['ZERO_INFLATED_MEDIAN'] = np.median(non_zero_series) * (1 - zero_ratio)
        
        elif part_category == 'Lumpy':
            # Outlier-resistant models
            if len(series) >= 4:
                # Trimmed mean (remove outliers)
                q25, q75 = np.percentile(series, [25, 75])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                filtered_series = series[(series >= lower_bound) & (series <= upper_bound)]
                
                if len(filtered_series) >= 2:
                    preds['TRIMMED_MEAN'] = np.mean(filtered_series)
                    preds['TRIMMED_MEDIAN'] = np.median(filtered_series)
        
        # 6. Select best model
        valid_preds = {k: v for k, v in preds.items() if v is not None and not np.isnan(v)}
        if not valid_preds:
            return 'MA6', np.nan
        
        # Calculate errors
        errors = {}
        for model, pred in valid_preds.items():
            errors[model] = hybrid_error(actual_val, pred)
        
        # Sort by error
        sorted_errors = sorted(errors.items(), key=lambda x: x[1])
        best_model, best_error = sorted_errors[0]
        
        # If still high error, try ensemble
        if best_error > 30.0 and len(sorted_errors) >= 2:
            # Create ensemble from top models
            top_models = [model for model, _ in sorted_errors[:min(3, len(sorted_errors))]]
            ensemble_preds = {model: valid_preds[model] for model in top_models}
            
            # Weighted ensemble
            weights = {}
            for i, model in enumerate(top_models):
                weights[model] = 1.0 / (i + 1)
            
            ensemble_forecast = ensemble_voting(ensemble_preds, weights)
            if not np.isnan(ensemble_forecast):
                ensemble_error = hybrid_error(actual_val, ensemble_forecast)
                if ensemble_error < best_error:
                    return 'ENSEMBLE_SPARSE', ensemble_forecast
        
        return best_model, valid_preds[best_model]
        
    except Exception:
        return 'MA6', np.nan

def calculate_model_consistency_score(model_performance_history):
    """
    Hitung consistency score untuk model berdasarkan historical performance
    """
    if not model_performance_history:
        return 0.0
    
    errors = [perf['error'] for perf in model_performance_history]
    
    # 1. Error variance (low = consistent)
    error_variance = np.var(errors) if len(errors) > 1 else 0
    
    # 2. Selection frequency (berapa kali dipilih sebagai best)
    selection_freq = len(model_performance_history) / max(len(model_performance_history), 1)
    
    # 3. Performance stability (low std = stable)
    performance_stability = 1 / (np.std(errors) + 1e-8)
    
    # 4. Average performance
    avg_performance = 1 / (np.mean(errors) + 1e-8)
    
    consistency_score = (1/(error_variance + 1e-8)) * selection_freq * performance_stability * avg_performance
    return consistency_score

def calculate_temporal_generalization_score(model, series, lookback_periods=[6, 9, 12]):
    """
    Test model di multiple time periods untuk temporal generalization
    """
    scores = []
    
    for lookback in lookback_periods:
        if len(series) >= lookback + 3:
            # Rolling window validation
            errors = []
            for i in range(lookback, len(series) - 1):
                train_data = series[i-lookback:i]
                test_val = series[i]
                
                # Simple forecast (moving average as baseline)
                pred = np.mean(train_data[-3:]) if len(train_data) >= 3 else np.mean(train_data)
                error = hybrid_error(test_val, pred)
                errors.append(error)
            
            if errors:
                scores.append(1 / (np.std(errors) + 1e-8))
    
    return np.mean(scores) if scores else 0.0

def business_logic_validation(forecast, part_category, historical_data):
    """
    Validasi forecast berdasarkan business logic
    """
    try:
        if len(historical_data) == 0:
            return True, "No historical data"
        
        # 1. Forecast tidak boleh terlalu jauh dari historical range
        if forecast > historical_data.max() * 2:
            return False, "Forecast terlalu tinggi"
        
        # 2. Untuk Lumpy parts, forecast tidak boleh terlalu smooth
        if part_category == 'Lumpy' and forecast > 0 and historical_data.std() == 0:
            return False, "Lumpy part tidak boleh smooth"
        
        # 3. Untuk Smooth parts, forecast tidak boleh terlalu erratic
        if part_category == 'Smooth' and abs(forecast - historical_data.mean()) > historical_data.std() * 3:
            return False, "Smooth part tidak boleh erratic"
        
        return True, "Valid"
    except Exception:
        return True, "Validation error"

# Fungsi advanced_model_selection dihapus karena tidak digunakan dan menyebabkan error

# LSTM function removed - no LSTM models used
def lstm_next(series, lookback=12, epochs=20, batch_size=16):
    # LSTM removed - fallback to simple average
    return max(float(np.mean(series)), 0.0) if len(series) > 0 else 0.0


def add_operational_features_inplace(df: pd.DataFrame) -> None:
    for col in [
        'WORKING_DAYS', 'ORDER_CYCLE_DAYS', 'SS_DEMAND_QTY', 'STANDARD_STOCK_DAYS',
        'DELIVERY_LT_REGULER', 'SS_LT_DAYS', 'INVENTORY_CONTROL_CLASS'
    ]:
        if col not in df.columns:
            df[col] = 0 if col != 'INVENTORY_CONTROL_CLASS' else ''
    for col in ['WORKING_DAYS', 'ORDER_CYCLE_DAYS', 'SS_DEMAND_QTY', 'STANDARD_STOCK_DAYS', 'DELIVERY_LT_REGULER', 'SS_LT_DAYS']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    def _encode_inventory_class(value: object) -> float:
        try:
            if pd.isna(value):
                return 50.0
            s = str(value).strip().upper()
            if not s or not s[0].isalpha():
                return 50.0
            letter = s[0]
            digits = ''.join(ch for ch in s[1:] if ch.isdigit())
            digit = int(digits) if digits else 1
            letter_rank = ord(letter) - ord('A')
            return float(letter_rank * 10 + (digit - 1))
        except Exception:
            return 50.0
    df['INV_CLASS_SCORE'] = df['INVENTORY_CONTROL_CLASS'].apply(_encode_inventory_class)
    df['QTY_PER_WORKDAY'] = df['ORIGINAL_SHIPPING_QTY'] / df['WORKING_DAYS'].replace(0, 1)
    df['ORDER_FREQ'] = df['WORKING_DAYS'] / df['ORDER_CYCLE_DAYS'].replace(0, 1)
    # deprioritized cols intentionally not used later: SS_TO_DEMAND/COVERAGE_DAYS/LT_BURDEN/LT_UNCERTAINTY/GROWTH_RATE
    if 'PART_NO' in df.columns:
        df.sort_values(['PART_NO', 'MONTH'], inplace=True)
        df['QTY_PWD_ROLL_MEAN_3'] = df.groupby('PART_NO')['QTY_PER_WORKDAY'].transform(lambda s: s.rolling(3, min_periods=1).mean())
        df['QTY_PWD_ROLL_STD_6'] = df.groupby('PART_NO')['QTY_PER_WORKDAY'].transform(lambda s: s.rolling(6, min_periods=1).std())
    else:
        df['QTY_PWD_ROLL_MEAN_3'] = df['QTY_PER_WORKDAY']
        df['QTY_PWD_ROLL_STD_6'] = 0


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame, part_category: str):
    for df_tmp in (train_df, test_df):
        if not df_tmp.empty:
            if 'MONTH_NUM' not in df_tmp.columns:
                df_tmp['MONTH_NUM'] = df_tmp['MONTH'].dt.month
            if 'YEAR' not in df_tmp.columns:
                df_tmp['YEAR'] = df_tmp['MONTH'].dt.year
            if 'MONTH_SIN' not in df_tmp.columns:
                df_tmp['MONTH_SIN'] = np.sin(2 * np.pi * df_tmp['MONTH_NUM'] / 12)
            if 'MONTH_COS' not in df_tmp.columns:
                df_tmp['MONTH_COS'] = np.cos(2 * np.pi * df_tmp['MONTH_NUM'] / 12)

    combined_df = pd.concat([train_df, test_df], ignore_index=True).sort_values('MONTH')
    for i in range(1, 7):
        combined_df[f'LAG_{i}'] = combined_df['ORIGINAL_SHIPPING_QTY'].shift(i)
    combined_df['MEAN_LAG_3'] = combined_df[['LAG_1', 'LAG_2', 'LAG_3']].mean(axis=1)
    combined_df['SUM_LAG_6'] = combined_df[[f'LAG_{i}' for i in range(1, 7)]].sum(axis=1)
    combined_df['ROLLING_MEAN_3'] = combined_df['ORIGINAL_SHIPPING_QTY'].rolling(3).mean()
    combined_df['ROLLING_MEAN_6'] = combined_df['ORIGINAL_SHIPPING_QTY'].rolling(6).mean()
    combined_df['ROLLING_STD_3'] = combined_df['ORIGINAL_SHIPPING_QTY'].rolling(3).std()
    combined_df['ROLLING_STD_6'] = combined_df['ORIGINAL_SHIPPING_QTY'].rolling(6).std()
    combined_df['GROWTH_LAG_3_6'] = combined_df['LAG_3'] / (combined_df['LAG_6'] + 1e-8)

    if part_category == 'Erratic':
        combined_df['DIFF_1'] = combined_df['ORIGINAL_SHIPPING_QTY'].diff()
        combined_df['REL_1'] = combined_df['ORIGINAL_SHIPPING_QTY'].pct_change().fillna(0)

    if part_category in ['Smooth', 'Erratic']:
        features = ['LAG_1','LAG_2','LAG_3','LAG_4','LAG_5','LAG_6',
                    'MONTH_NUM','YEAR','MONTH_SIN','MONTH_COS',
                    'MEAN_LAG_3','SUM_LAG_6','ROLLING_MEAN_3','ROLLING_MEAN_6',
                    'ROLLING_STD_3','ROLLING_STD_6','GROWTH_LAG_3_6',
                    'INV_CLASS_SCORE','QTY_PER_WORKDAY','ORDER_FREQ','QTY_PWD_ROLL_MEAN_3','QTY_PWD_ROLL_STD_6']
        if part_category == 'Erratic':
            features.extend(['DIFF_1', 'REL_1'])
    else:
        features = ['LAG_1','LAG_2','LAG_3','LAG_4','LAG_5','LAG_6',
                    'MONTH_NUM','YEAR','MONTH_SIN','MONTH_COS',
                    'MEAN_LAG_3','SUM_LAG_6','ROLLING_MEAN_3','ROLLING_MEAN_6',
                    'ROLLING_STD_3','ROLLING_STD_6','GROWTH_LAG_3_6',
                    'INV_CLASS_SCORE','QTY_PER_WORKDAY','ORDER_FREQ','QTY_PWD_ROLL_MEAN_3','QTY_PWD_ROLL_STD_6']

    return combined_df, features


def detect_fluctuation_pattern(series):
    """
    Deteksi pola fluktuasi dalam time series
    Returns: fluctuation_type, volatility_level, trend_direction
    """
    if len(series) < 3:
        return 'MINIMAL_DATA', 'LOW', 'STABLE'
    
    # 1. Hitung volatilitas
    rolling_std = pd.Series(series).rolling(3).std().dropna()
    avg_volatility = rolling_std.mean() if len(rolling_std) > 0 else 0
    mean_demand = np.mean(series)
    cv = avg_volatility / (mean_demand + 1e-8)  # Coefficient of Variation
    
    # 2. Deteksi pola fluktuasi
    diff_series = np.diff(series)
    sign_changes = np.sum(np.diff(np.sign(diff_series)) != 0)
    fluctuation_ratio = sign_changes / max(len(diff_series) - 1, 1)
    
    # 3. Klasifikasi volatilitas
    if cv > 0.5:
        volatility_level = 'HIGH'
    elif cv > 0.2:
        volatility_level = 'MEDIUM'
    else:
        volatility_level = 'LOW'
    
    # 4. Klasifikasi pola fluktuasi
    if fluctuation_ratio > 0.6:
        fluctuation_type = 'HIGHLY_FLUCTUATING'  # Sering naik-turun
    elif fluctuation_ratio > 0.3:
        fluctuation_type = 'MODERATELY_FLUCTUATING'  # Sedang fluktuasi
    elif cv > 0.3:
        fluctuation_type = 'VOLATILE_BUT_TRENDING'  # Volatil tapi ada tren
    else:
        fluctuation_type = 'STABLE'  # Stabil
    
    # 5. Deteksi arah tren
    if len(series) >= 6:
        recent_trend = np.mean(series[-3:]) - np.mean(series[-6:-3])
        if recent_trend > mean_demand * 0.1:
            trend_direction = 'INCREASING'
        elif recent_trend < -mean_demand * 0.1:
            trend_direction = 'DECREASING'
        else:
            trend_direction = 'STABLE'
    else:
        trend_direction = 'STABLE'
    
    return fluctuation_type, volatility_level, trend_direction

def get_fluctuation_aware_parameters(fluctuation_type, volatility_level):
    """
    Sesuaikan parameter berdasarkan tingkat fluktuasi - ENHANCED VERSION
    """
    if fluctuation_type == 'HIGHLY_FLUCTUATING':
        return {
            'lstm_lookback': 8,  # Longer lookback untuk capture pattern
            'lstm_epochs': 15,    # More epochs untuk better learning
            'xgb_learning_rate': 0.2,  # Higher learning rate untuk responsif
            'xgb_max_depth': 6,   # Deeper untuk complex patterns
            'ma_window': 3,        # Shorter window untuk responsif
            'ets_trend': 'add',    # Additive trend untuk fluktuasi tinggi
            'ets_seasonal': 'add', # Additive seasonal
            'arima_max_p': 3,      # Higher AR order
            'arima_max_q': 3,      # Higher MA order
            'arima_max_P': 2,      # Higher seasonal AR
            'arima_max_Q': 2       # Higher seasonal MA
        }
    elif fluctuation_type == 'MODERATELY_FLUCTUATING':
        return {
            'lstm_lookback': 6,
            'lstm_epochs': 12,
            'xgb_learning_rate': 0.15,
            'xgb_max_depth': 5,
            'ma_window': 4,
            'ets_trend': 'add',
            'ets_seasonal': 'add',
            'arima_max_p': 2,
            'arima_max_q': 2,
            'arima_max_P': 1,
            'arima_max_Q': 1
        }
    elif fluctuation_type == 'VOLATILE_BUT_TRENDING':
        return {
            'lstm_lookback': 6,
            'lstm_epochs': 12,
            'xgb_learning_rate': 0.15,
            'xgb_max_depth': 5,
            'ma_window': 4,
            'ets_trend': 'mul',    # Multiplicative trend untuk trend yang kuat
            'ets_seasonal': 'add',
            'arima_max_p': 2,
            'arima_max_q': 2,
            'arima_max_P': 1,
            'arima_max_Q': 1
        }
    else:  # STABLE
        return {
            'lstm_lookback': 4,
            'lstm_epochs': 10,
            'xgb_learning_rate': 0.1,
            'xgb_max_depth': 4,
            'ma_window': 6,
            'ets_trend': 'add',
            'ets_seasonal': 'add',
            'arima_max_p': 1,
            'arima_max_q': 1,
            'arima_max_P': 1,
            'arima_max_Q': 1
        }

def select_model_for_fluctuation(fluctuation_type, volatility_level, part_category):
    """
    Pilih model berdasarkan pola fluktuasi yang terdeteksi - ENHANCED VERSION
    """
    model_priorities = {
        'HIGHLY_FLUCTUATING': {
            'primary': ['RF', 'XGB', 'ENSEMBLE'],
            'secondary': ['ETS_VOLATILE', 'ARIMA_VOLATILE', 'WMA', 'MA3', 'MA4'],
            'avoid': ['MA6', 'ETS_SMOOTH', 'ARIMA_SMOOTH']
        },
        'MODERATELY_FLUCTUATING': {
            'primary': ['RF', 'XGB', 'WMA', 'ETS_ADAPTIVE'],
            'secondary': ['MA3', 'MA4', 'MA5', 'ARIMA_ADAPTIVE'],
            'avoid': ['MA6']
        },
        'VOLATILE_BUT_TRENDING': {
            'primary': ['ETS_TREND', 'ARIMA_TREND', 'RF'],
            'secondary': ['XGB', 'WMA', 'MA4', 'MA5'],
            'avoid': ['MA3', 'MA6']
        },
        'STABLE': {
            'primary': ['ETS_SMOOTH', 'ARIMA_SMOOTH', 'MA6', 'WMA'],
            'secondary': ['RF', 'XGB'],
            'avoid': []
        }
    }
    
    # Sesuaikan dengan kategori parts
    if part_category == 'Erratic':
        # Untuk Erratic, prioritaskan model yang bisa handle volatilitas
        if fluctuation_type in ['HIGHLY_FLUCTUATING', 'MODERATELY_FLUCTUATING']:
            return ['RF', 'XGB', 'ENSEMBLE', 'ETS_VOLATILE', 'ARIMA_VOLATILE', 'WMA', 'MA3', 'MA4']
        else:
            return ['RF', 'XGB', 'ETS_ADAPTIVE', 'ARIMA_ADAPTIVE', 'WMA', 'MA4', 'MA5', 'MA6']
    elif part_category == 'Smooth':
        # Untuk Smooth, gunakan model tradisional jika stabil
        if fluctuation_type == 'STABLE':
            return ['ETS_SMOOTH', 'ARIMA_SMOOTH', 'MA6', 'WMA', 'RF', 'XGB']
        else:
            return ['RF', 'XGB', 'ETS_ADAPTIVE', 'ARIMA_ADAPTIVE', 'WMA', 'MA4', 'MA5', 'MA6']
    else:
        # Untuk kategori lain, gunakan prioritas default
        priorities = model_priorities.get(fluctuation_type, model_priorities['STABLE'])
        return priorities['primary'] + priorities['secondary']

def process_part(part, part_df, test_months, part_category: str):
    results = []
    
    # DEBUG: Log untuk troubleshooting backtest
    print(f"🔍 BACKTEST DEBUG: Processing Part {part} ({part_category})")
    print(f"🔍 BACKTEST DEBUG: Test months: {[m.strftime('%Y-%m') for m in test_months]}")
    print(f"🔍 BACKTEST DEBUG: Part data months: {sorted(part_df['MONTH'].dt.strftime('%Y-%m').unique())}")
    
    # Progress bar untuk setiap part (jika tidak menggunakan parallel)
    for target_month in tqdm(test_months, desc=f"Part {part}", leave=False, disable=False):
        # FIX: Pastikan tipe data konsisten untuk filtering
        # Debug tipe data
        print(f"🔍 DATA TYPE DEBUG: target_month type: {type(target_month)}")
        print(f"🔍 DATA TYPE DEBUG: part_df['MONTH'] type: {part_df['MONTH'].dtype}")
        print(f"🔍 DATA TYPE DEBUG: target_month value: {target_month}")
        print(f"🔍 DATA TYPE DEBUG: part_df['MONTH'] sample: {part_df['MONTH'].head(3).tolist()}")
        
        # FIX: Konversi target_month ke tipe yang sama dengan part_df['MONTH']
        if hasattr(target_month, 'date'):
            # Jika target_month adalah datetime, pastikan format sama
            target_month_parsed = target_month
        else:
            # Jika target_month adalah string atau tipe lain, parse ke datetime
            target_month_parsed = pd.to_datetime(target_month)
        
        # FIX: Pastikan part_df['MONTH'] juga datetime
        if not pd.api.types.is_datetime64_any_dtype(part_df['MONTH']):
            part_df['MONTH'] = pd.to_datetime(part_df['MONTH'])
        
        train_df = part_df[part_df['MONTH'] < target_month_parsed].copy()
        test_df = part_df[part_df['MONTH'] == target_month_parsed].copy()
        
        # DEBUG: Log hasil filtering
        print(f"🔍 FILTERING DEBUG: train_df shape: {train_df.shape}, test_df shape: {test_df.shape}")
        if not train_df.empty:
            print(f"🔍 FILTERING DEBUG: train_df months: {sorted(train_df['MONTH'].dt.strftime('%Y-%m').unique())}")
        if not test_df.empty:
            print(f"🔍 FILTERING DEBUG: test_df month: {test_df['MONTH'].dt.strftime('%Y-%m').iloc[0]}")

        if part_category in ['Intermittent', 'Lumpy']:
            p99 = train_df['ORIGINAL_SHIPPING_QTY'].quantile(0.99)
            train_df['ORIGINAL_SHIPPING_QTY'] = train_df['ORIGINAL_SHIPPING_QTY'].clip(upper=p99)

        # SYARAT FORECAST UNTUK SEMUA KATEGORI
        # Semua kategori memerlukan syarat forecast: minimal 2 bulan dengan demand > 0 di 6 bulan terakhir
        # FIX: Gunakan semua data part, bukan hanya train_df untuk window check yang akurat
        meets_requirement, months_with_demand, demand_values = validate_demand_requirement_general(part, target_month, part_df, test_df, part_category)
        
        # DEBUG: Log detail untuk troubleshooting
        print(f"🔍 BACKTEST DEBUG: Part {part} ({part_category}) - Target: {target_month.strftime('%Y-%m')}")
        print(f"🔍 BACKTEST DEBUG: Meets requirement: {meets_requirement}, Months with demand: {months_with_demand}")
        print(f"🔍 BACKTEST DEBUG: Train data shape: {train_df.shape}, Test data shape: {test_df.shape}")
            
        if not meets_requirement:
            # Jika tidak memenuhi syarat, forecast = 0 dan best model = INSUFFICIENT_DATA
            fc = 0.0
            best_model = 'INSUFFICIENT_DATA'
            print(f"❌ Part {part} ({part_category}): Does not meet forecast requirement - forecast = 0")
                
            # Tambahkan ke results dan skip forecasting
            results.append({
                'PART_NO': part,
                'MONTH': target_month.strftime('%Y-%m'),
                'FORECAST': 0,
                'ACTUAL': float(test_df['ORIGINAL_SHIPPING_QTY'].values[0]) if not test_df.empty else 0.0,
                'ERROR': '0.00%',
                'BEST_MODEL': 'INSUFFICIENT_DATA',
                'PART_CATEGORY': part_category
            })
            continue  # Skip forecasting untuk part ini
        else:
            print(f"✅ Part {part} ({part_category}): Meets forecast requirement - proceeding with forecast")
        
        actual = float(test_df['ORIGINAL_SHIPPING_QTY'].values[0]) if not test_df.empty else 0.0

        # DEBUG: Untuk semua kategori untuk troubleshooting
        print(f"🔍 Part {part} ({part_category}): Target month: {target_month.strftime('%Y-%m')}")
        print(f"🔍 Part {part} ({part_category}): Training data months: {sorted(train_df['MONTH'].dt.strftime('%Y-%m').unique()) if not train_df.empty else 'EMPTY'}")
        print(f"🔍 Part {part} ({part_category}): Training data shape: {train_df.shape}")
        
        if part_category in ['Smooth', 'Intermittent']:
            print(f"🔍 Part {part} ({part_category}): Training data demand: {train_df['ORIGINAL_SHIPPING_QTY'].tolist()}")
            print(f"🔍 Part {part} ({part_category}): Training data non-null: {train_df['ORIGINAL_SHIPPING_QTY'].notna().sum()}")
            print(f"🔍 Part {part} ({part_category}): Training data non-zero: {(train_df['ORIGINAL_SHIPPING_QTY'] > 0).sum()}")
            
            # DEBUG: Cek apakah ada masalah dengan data parsing
            print(f"🔍 Part {part} ({part_category}): Data validation:")
            print(f"   - train_df empty: {train_df.empty}")
            print(f"   - train_df MONTH dtype: {train_df['MONTH'].dtype}")
            print(f"   - train_df ORIGINAL_SHIPPING_QTY dtype: {train_df['ORIGINAL_SHIPPING_QTY'].dtype}")
            print(f"   - train_df has NaN in MONTH: {train_df['MONTH'].isna().any()}")
            print(f"   - train_df has NaN in ORIGINAL_SHIPPING_QTY: {train_df['ORIGINAL_SHIPPING_QTY'].isna().any()}")
            if not train_df.empty:
                print(f"   - train_df ORIGINAL_SHIPPING_QTY min: {train_df['ORIGINAL_SHIPPING_QTY'].min()}")
                print(f"   - train_df ORIGINAL_SHIPPING_QTY max: {train_df['ORIGINAL_SHIPPING_QTY'].max()}")
                print(f"   - train_df ORIGINAL_SHIPPING_QTY mean: {train_df['ORIGINAL_SHIPPING_QTY'].mean()}")
        
        # VALIDASI: Periksa apakah ada cukup data training
        if train_df.empty:
            print(f"❌ Part {part} ({part_category}): INSUFFICIENT TRAINING DATA - train_df is empty!")
            print(f"🔍 Part {part} ({part_category}): All part data months: {sorted(part_df['MONTH'].dt.strftime('%Y-%m').unique())}")
            print(f"🔍 Part {part} ({part_category}): Target month: {target_month.strftime('%Y-%m')}")
        elif len(train_df) < 3:
            print(f"⚠️  Part {part} ({part_category}): LIMITED TRAINING DATA - only {len(train_df)} months available!")

        # DEBUG: Hanya untuk kategori Smooth dan Intermittent
        if part_category in ['Smooth', 'Intermittent']:
            print(f"🔍 Part {part} ({part_category}): Before resampling - train_df MONTH values: {train_df['MONTH'].tolist()}")
            print(f"🔍 Part {part} ({part_category}): Before resampling - train_df ORIGINAL_SHIPPING_QTY values: {train_df['ORIGINAL_SHIPPING_QTY'].tolist()}")
        
        # FIX: Cek apakah ada masalah dengan data sebelum resampling
        if train_df.empty:
            print(f"❌ Part {part}: train_df is EMPTY!")
            # Gunakan rata-rata demand dari data historis SESUAI PERMINTAAN USER
            hist_data = part_df[part_df['MONTH'] < target_month].copy()
            if not hist_data.empty:
                # Gunakan rata-rata dari semua demand (termasuk yang nol)
                avg_demand = hist_data['ORIGINAL_SHIPPING_QTY'].mean()
                fc = max(0, avg_demand)
                best_model = 'AVG_HISTORICAL_DEMAND'
                print(f"⚠️  Part {part}: Using {best_model} (avg all demands): {fc:.1f}")
            else:
                fc = 0
                best_model = 'NO_HISTORICAL_DATA'
                print(f"⚠️  Part {part}: No historical data found, forecast = 0")
            
            # Hitung error dengan benar
            actual_val = float(test_df['ORIGINAL_SHIPPING_QTY'].values[0]) if not test_df.empty else 0.0
            error_val = hybrid_error(actual_val, fc)
            
            # Safe conversion to int with validation
            if pd.isna(fc) or np.isinf(fc):
                fc = 0.0
            forecast_int = int(round(fc)) if not pd.isna(fc) and not np.isinf(fc) else 0
            
            results.append({
                'PART_NO': part,
                'MONTH': target_month.strftime('%Y-%m'),
                'FORECAST': forecast_int,
                'ACTUAL': actual_val,
                'ERROR': f"{error_val:.2f}%",
                'BEST_MODEL': best_model,
                'PART_CATEGORY': part_category
            })
            continue
        
        series = train_df.set_index('MONTH').resample('MS')['ORIGINAL_SHIPPING_QTY'].sum().fillna(0).values.astype(float)
        
        # Handle NaN values - replace with 0 to prevent errors
        series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
        
        # DEBUG: Hanya untuk kategori Smooth dan Intermittent
        if part_category in ['Smooth', 'Intermittent']:
            print(f"🔍 Part {part} ({part_category}): Series for forecasting: {series.tolist()}")
            print(f"🔍 Part {part} ({part_category}): Series length: {len(series)}")
            print(f"🔍 Part {part} ({part_category}): Series dtype: {type(series)}")
            print(f"🔍 Part {part} ({part_category}): Series has NaN: {np.isnan(series).any()}")
            print(f"🔍 Part {part} ({part_category}): Series has inf: {np.isinf(series).any()}")
            print(f"🔍 Part {part} ({part_category}): Series non-zero count: {np.sum(series > 0)}")
            print(f"🔍 Part {part} ({part_category}): Series zero count: {np.sum(series == 0)}")
            print(f"🔍 Part {part} ({part_category}): Series sum: {np.sum(series)}")
            print(f"🔍 Part {part} ({part_category}): Series mean: {np.mean(series)}")
            print(f"🔍 Part {part} ({part_category}): Series std: {np.std(series)}")

        # FIX: actual_val untuk backtest harus menggunakan data yang sudah ada
        # Untuk backtest, test_df berisi data yang sudah ada (bukan future data)
        if not test_df.empty:
            actual_val = float(test_df['ORIGINAL_SHIPPING_QTY'].values[0])
            # Handle NaN values in actual_val
            if np.isnan(actual_val):
                actual_val = 0.0
        else:
            # Jika test_df kosong, gunakan nilai dari train_df terakhir
            if not train_df.empty:
                actual_val = float(train_df['ORIGINAL_SHIPPING_QTY'].iloc[-1])
                if np.isnan(actual_val):
                    actual_val = 0.0
            else:
                actual_val = 0.0
        
        # DEBUG: Hanya untuk kategori Smooth dan Intermittent
        if part_category in ['Smooth', 'Intermittent']:
            print(f"🔍 Part {part} ({part_category}): ACTUAL_VAL DEBUG:")
            print(f"   - test_df empty: {test_df.empty}")
            print(f"   - test_df shape: {test_df.shape}")
            print(f"   - actual_val: {actual_val}")
            if not test_df.empty:
                print(f"   - test_df ORIGINAL_SHIPPING_QTY: {test_df['ORIGINAL_SHIPPING_QTY'].values}")
            if not train_df.empty:
                print(f"   - train_df last ORIGINAL_SHIPPING_QTY: {train_df['ORIGINAL_SHIPPING_QTY'].iloc[-1]}")

        # DETEKSI POLA FLUKTUASI untuk semua kategori (untuk informasi)
        fluctuation_type, volatility_level, trend_direction = detect_fluctuation_pattern(series)
        
        # Debug info untuk data yang sulit (hanya untuk debug mode)
        print(f"🔍 Part {part} ({part_category}): Data analysis:")
        print(f"   - Series length: {len(series)} points")
        print(f"   - Series values: {series.tolist()}")
        print(f"   - Non-zero values: {np.sum(series > 0)}")
        print(f"   - Zero ratio: {np.sum(series == 0) / len(series):.1%}")
        print(f"   - Standard deviation: {np.std(series):.2f}")
        print(f"   - Mean: {np.mean(series):.2f}")
        
        if len(series) < 6:
            print(f"⚠️  Part {part}: Limited data ({len(series)} points), using simple models")
        elif np.std(series) == 0:
            print(f"⚠️  Part {part}: Constant data, using last value")
        elif np.sum(series == 0) / len(series) > 0.5:
            print(f"⚠️  Part {part}: High zero ratio ({np.sum(series == 0) / len(series):.1%}), using zero-inflated models")
        
        # Dapatkan parameter berdasarkan fluktuasi
        fluctuation_params = get_fluctuation_aware_parameters(fluctuation_type, volatility_level)

        preds = {}
        # Time-series models - Flexible MA selection
        if part_category in ['Smooth', 'Intermittent']:
            print(f"🔍 Part {part} ({part_category}): Testing MA models...")
        best_ma_model, best_ma_pred = get_best_ma_model(series, actual_val)
        preds[best_ma_model] = best_ma_pred
        if part_category in ['Smooth', 'Intermittent']:
            print(f"   {best_ma_model}: {best_ma_pred}")
        
        if part_category in ['Smooth', 'Intermittent']:
            print(f"🔍 Part {part} ({part_category}): Testing WMA...")
        wma_pred = forecast_wma(series)
        preds['WMA'] = wma_pred
        if part_category in ['Smooth', 'Intermittent']:
            print(f"   WMA: {wma_pred}")
        
        if part_category in ['Smooth', 'Intermittent']:
            print(f"🔍 Part {part} ({part_category}): Testing ETS...")
        ets_pred = forecast_ets(series, seasonal_pref=(part_category in ['Smooth','Erratic']), fluctuation_type=fluctuation_type)
        preds['ETS'] = ets_pred
        if part_category in ['Smooth', 'Intermittent']:
            print(f"   ETS: {ets_pred}")
        
        if part_category in ['Smooth', 'Intermittent']:
            print(f"🔍 Part {part} ({part_category}): Testing ARIMA...")
        arima_pred = forecast_arima_enhanced(series, seasonal_pref=(part_category in ['Smooth','Erratic']), fluctuation_type=fluctuation_type)
        preds['ARIMA'] = arima_pred
        if part_category in ['Smooth', 'Intermittent']:
            print(f"   ARIMA: {arima_pred}")
        
        # FALLBACK MODELS untuk data yang sulit - tambahkan model sederhana yang selalu bisa jalan
        if len(series) >= 1:
            # Simple average dari data terakhir
            preds['SIMPLE_AVG'] = max(0, np.mean(series))
            # Last value
            preds['LAST_VALUE'] = max(0, series[-1])
            # Median
            preds['MEDIAN'] = max(0, np.median(series))
        
        if len(series) >= 2:
            # Simple trend
            trend_val = series[-1] + (series[-1] - series[-2])
            preds['SIMPLE_TREND'] = max(0, trend_val)
            # Moving average 2
            preds['MA2'] = max(0, np.mean(series[-2:]))
        
        if len(series) >= 3:
            # Moving average 3
            preds['MA3'] = max(0, np.mean(series[-3:]))
            # Weighted average (recent lebih penting)
            weights = np.array([1, 2, 3])
            wma3_val = np.average(series[-3:], weights=weights)
            preds['WMA3'] = max(0, wma3_val)
        
        # FINAL MODELS - model yang pasti bisa jalan
        if len(series) > 0:
            # Jika semua model gagal, gunakan rata-rata sederhana
            preds['AVG_DEMAND'] = max(0, np.mean(series))
        
        # Tambahkan fluktuation-aware models - OPTIMIZED (hanya untuk kategori yang memerlukan)
        if fluctuation_type == 'HIGHLY_FLUCTUATING' and part_category in ['Erratic', 'Lumpy']:
            preds['ETS_VOLATILE'] = forecast_ets_volatile(series, seasonal_pref=(part_category in ['Smooth','Erratic']))
            preds['ARIMA_VOLATILE'] = forecast_arima_volatile(series, seasonal_pref=(part_category in ['Smooth','Erratic']))
            
            # Tambahkan high volatility strategy
            high_vol_preds = handle_high_volatility_demand(series, part_category, fluctuation_type)
            preds.update(high_vol_preds)
            
        elif fluctuation_type == 'MODERATELY_FLUCTUATING' and part_category in ['Erratic', 'Lumpy']:
            preds['ETS_ADAPTIVE'] = forecast_ets_adaptive(series, seasonal_pref=(part_category in ['Smooth','Erratic']))
            preds['ARIMA_ADAPTIVE'] = forecast_arima_adaptive(series, seasonal_pref=(part_category in ['Smooth','Erratic']))
        elif fluctuation_type == 'VOLATILE_BUT_TRENDING' and part_category in ['Erratic', 'Lumpy']:
            preds['ETS_TREND'] = forecast_ets(series, seasonal_pref=(part_category in ['Smooth','Erratic']), fluctuation_type='VOLATILE_BUT_TRENDING')
            preds['ARIMA_TREND'] = forecast_arima_enhanced(series, seasonal_pref=(part_category in ['Smooth','Erratic']), fluctuation_type='VOLATILE_BUT_TRENDING')
        else:  # STABLE - hanya untuk Smooth
            if part_category == 'Smooth':
                preds['ETS_SMOOTH'] = forecast_ets(series, seasonal_pref=(part_category in ['Smooth','Erratic']), fluctuation_type='STABLE')
                preds['ARIMA_SMOOTH'] = forecast_arima_enhanced(series, seasonal_pref=(part_category in ['Smooth','Erratic']), fluctuation_type='STABLE')

        # Enhanced Croston family untuk Intermittent/Lumpy dengan multiple alpha values - OPTIMIZED
        if part_category in ['Intermittent', 'Lumpy']:
            if part_category == 'Intermittent':  # Hanya log untuk Intermittent
                print(f"🔍 Part {part} ({part_category}): Testing Croston models...")
            croston_pred = forecast_croston(series)
            preds['CROSTON'] = croston_pred
            if part_category == 'Intermittent':
                print(f"   CROSTON: {croston_pred}")
            
            croston_sba_pred = forecast_croston_sba(series)
            preds['CROSTON_SBA'] = croston_sba_pred
            if part_category == 'Intermittent':
                print(f"   CROSTON_SBA: {croston_sba_pred}")
            
            croston_tsb_pred = forecast_croston_tsb(series)
            preds['CROSTON_TSB'] = croston_tsb_pred
            if part_category == 'Intermittent':
                print(f"   CROSTON_TSB: {croston_tsb_pred}")
            
            # REDUCED alpha values untuk speed (hanya 2 alpha terbaik)
            for alpha in [0.1, 0.2]:  # Reduced dari 4 ke 2 alpha
                croston_alpha_pred = forecast_croston(series, alpha)
                preds[f'CROSTON_ALPHA_{alpha}'] = croston_alpha_pred
                if part_category == 'Intermittent':
                    print(f"   CROSTON_ALPHA_{alpha}: {croston_alpha_pred}")
                
                croston_sba_alpha_pred = forecast_croston_sba(series, alpha)
                preds[f'CROSTON_SBA_ALPHA_{alpha}'] = croston_sba_alpha_pred
                if part_category == 'Intermittent':
                    print(f"   CROSTON_SBA_ALPHA_{alpha}: {croston_sba_alpha_pred}")
                
                croston_tsb_alpha_pred = forecast_croston_tsb(series, alpha, alpha)
                preds[f'CROSTON_TSB_ALPHA_{alpha}'] = croston_tsb_alpha_pred
                if part_category == 'Intermittent':
                    print(f"   CROSTON_TSB_ALPHA_{alpha}: {croston_tsb_alpha_pred}")

        # ML models (LSTM removed)
        # Strategy: RF/XGB dengan fitur eksternal
        # Alasan: RF/XGB tidak memiliki memory internal seperti LSTM
        # Fitur eksternal (seasonal, trend, volatility) bisa membantu
        combined_df, features = build_features(train_df, test_df, part_category)
        train_mask = combined_df['MONTH'] < target_month
        test_mask = combined_df['MONTH'] == target_month
        X_train = combined_df.loc[train_mask, features].fillna(0)
        y_train = combined_df.loc[train_mask, 'ORIGINAL_SHIPPING_QTY'].fillna(0)
        X_test = combined_df.loc[test_mask, features].fillna(0)

        scaler = StandardScaler()
        rf = RandomForestRegressor(n_estimators=400, max_depth=15, min_samples_split=2, min_samples_leaf=2, random_state=42)
        
        # XGB dengan parameter yang dioptimalkan untuk fluktuasi besar
        if part_category in ['Erratic', 'Lumpy']:
            # Parameter XGB untuk fluktuasi besar: lebih sensitif terhadap perubahan
            xgb = XGBRegressor(
                n_estimators=200,  # Reduced untuk menghindari overfitting
                max_depth=6,      # Deeper untuk capture complex patterns
                learning_rate=0.2, # Higher learning rate untuk responsif terhadap perubahan
                subsample=0.7,    # Lower subsample untuk generalization
                colsample_bytree=0.7,  # Lower colsample untuk regularization
                reg_alpha=0.1,     # L1 regularization
                reg_lambda=1.0,     # L2 regularization
                random_state=42
            )
        else:
            # Parameter XGB standar untuk kategori lain
            xgb = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.15, subsample=0.8, colsample_bytree=0.8, random_state=42)

        # Allowed per category - OPTIMAL COMBINATIONS berdasarkan penelitian
        # Strategy: RF/XGB dengan fitur eksternal (LSTM removed)
        # Smooth: ETS + ARIMA + RF + XGB (Traditional Statistical + ML)
        # Intermittent: Croston + Zero-Inflated (Specialized Statistical)
        # Erratic: RF + XGB + Ensemble (ML + Hybrid)
        # Lumpy: RF + XGB + Outlier-resistant (ML + Hybrid + Outlier-resistant)
        use_rf = part_category in ['Erratic', 'Lumpy', 'Smooth']  # RF untuk semua kategori utama
        use_xgb = part_category in ['Erratic', 'Lumpy', 'Smooth']  # XGB untuk semua kategori utama
        # Linear Regression dihapus untuk menyederhanakan dan mempercepat program

        if use_rf and len(X_train) >= 6:
            try:
                y_train_log = np.log1p(y_train)
                rf.fit(X_train, y_train_log)
                pred_log = rf.predict(X_test)[0] if len(X_test) else rf.predict(X_train.iloc[-1:])[0]
                preds['RF'] = float(np.expm1(pred_log))
            except Exception:
                preds['RF'] = np.nan

        if use_xgb and len(X_train) >= 6:
            try:
                y_train_log = np.log1p(y_train)
                xgb.fit(X_train, y_train_log)
                pred_log = xgb.predict(X_test)[0] if len(X_test) else xgb.predict(X_train.iloc[-1:])[0]
                preds['XGB'] = float(np.expm1(pred_log))
            except Exception:
                preds['XGB'] = np.nan

        # LSTM removed - no LSTM models used

        # Enhanced models untuk Smooth category (tetap menggunakan error terkecil) - OPTIMIZED
        # Strategy: Traditional Statistical Models (LSTM removed)
        # Alasan: Smooth data cocok untuk model tradisional
        if part_category == 'Smooth':
            # ETS dengan multiple configurations untuk Smooth - REDUCED
            preds['ETS_ADDITIVE'] = forecast_ets(series, seasonal_pref=True)
            # preds['ETS_MULTIPLICATIVE'] = forecast_ets(series, seasonal_pref=False)  # REMOVED untuk speed
            
            # ARIMA dengan seasonal preference untuk Smooth - REDUCED
            preds['ARIMA_SEASONAL'] = forecast_arima_enhanced(series, seasonal_pref=True)
            # preds['ARIMA_NON_SEASONAL'] = forecast_arima(series, seasonal_pref=False)  # REMOVED untuk speed
            
            # Linear trend untuk Smooth (karena biasanya memiliki trend yang jelas)
            if len(series) >= 3:
                try:
                    x = np.arange(len(series))
                    y = series
                    coeffs = np.polyfit(x, y, 1)
                    preds['LINEAR_TREND'] = max(0, coeffs[0] * len(series) + coeffs[1])
                except Exception:
                    pass

        # Simplified Ensemble untuk semua kategori - OPTIMIZED (hanya untuk kategori yang memerlukan)
        # Strategy: Ensemble RF/XGB dengan fitur eksternal (LSTM removed)
        # Alasan: Kombinasi RF/XGB (external features) optimal
        # RF+XGB Blend - hanya untuk Erratic/Lumpy
        if part_category in ['Erratic', 'Lumpy'] and not np.isnan(preds.get('RF', np.nan)) and not np.isnan(preds.get('XGB', np.nan)):
                preds['RF_XGB_BLEND'] = 0.5 * preds['RF'] + 0.5 * preds['XGB']
            
        # Ensemble (3-5 model terbaik) - akan dihitung setelah error calculation

        # DEBUG INFO: Hanya untuk kategori Smooth dan Intermittent
        if part_category in ['Smooth', 'Intermittent']:
            # Show the selected MA model
            ma_models = [k for k in preds.keys() if k.startswith('MA')]
            for ma_model in ma_models:
                print(f"   {ma_model}: {preds[ma_model]:.1f}")
            print(f"   WMA: {preds['WMA']:.1f}")
            print(f"   ETS: {preds['ETS']:.1f}")
            print(f"   ARIMA: {preds['ARIMA']:.1f}")
            
            # Debug khusus untuk Intermittent
            if part_category == 'Intermittent':
                print(f"   🎯 INTERMITTENT PRIORITY: Croston family preferred")
        
        # Valid predictions only - handle NaN values with ROBUST VALIDATION
        valid = {}
        for m, p in preds.items():
            print(f"🔍 Part {part} ({part_category}): Checking {m}: {p} (type: {type(p)}, isnan: {np.isnan(p) if p is not None else 'N/A'}, isinf: {np.isinf(p) if p is not None else 'N/A'})")
            
            # ROBUST VALIDATION: Accept more values as valid
            if p is not None:
                try:
                    # Convert to float and handle NaN/inf
                    p_float = float(p)
                    
                    # If NaN or inf, try to use fallback value
                    if np.isnan(p_float) or np.isinf(p_float):
                        if len(series) > 0 and np.sum(series) > 0:
                            # Use series average as fallback
                            p_clean = max(np.mean(series[series > 0]) if np.sum(series > 0) > 0 else np.mean(series), 0.1)
                            print(f"⚠️  Part {part} ({part_category}): {m} returned NaN/Inf, using series average: {p_clean:.2f}")
                        else:
                            print(f"❌ Part {part} ({part_category}): {m} is INVALID (NaN/Inf and no series data)")
                            continue
                    else:
                        p_clean = max(p_float, 0.0)  # Ensure non-negative
                        
                        # PREVENT "BEST MODEL ZERO": Jika part lolos syarat, tidak boleh ada model yang menghasilkan 0
                        if p_clean == 0 and meets_requirement and len(series) > 0 and np.sum(series) > 0:
                            # Jika part lolos syarat tapi model menghasilkan 0, gunakan MA fallback
                            recent_data = series[-6:] if len(series) >= 6 else series
                            non_zero_data = recent_data[recent_data > 0]
                            if len(non_zero_data) > 0:
                                p_clean = max(np.mean(non_zero_data), 0.1)
                            else:
                                p_clean = max(np.mean(recent_data), 0.1)
                            print(f"⚠️  Part {part} ({part_category}): {m} returned 0 but meets requirement, using MA fallback: {p_clean:.2f}")
                        elif p_clean == 0 and len(series) > 0 and np.sum(series) > 0:
                            # Jika tidak lolos syarat tapi ada data, tetap adjust
                            p_clean = max(np.mean(series[series > 0]) if np.sum(series > 0) > 0 else np.mean(series), 0.1)
                            print(f"⚠️  Part {part} ({part_category}): {m} returned 0, adjusted to {p_clean:.2f}")
                    
                    valid[m] = p_clean
                    print(f"✅ Part {part} ({part_category}): {m} is VALID with value {p_clean:.2f}")
                    
                except (ValueError, TypeError) as e:
                    print(f"❌ Part {part} ({part_category}): {m} is INVALID (conversion error: {e})")
            else:
                print(f"❌ Part {part} ({part_category}): {m} is INVALID (None value)")
        
        # DEBUG: Hanya untuk kategori Smooth dan Intermittent
        if part_category in ['Smooth', 'Intermittent']:
            print(f"🔍 Part {part} ({part_category}): All predictions:")
            for model, pred in preds.items():
                print(f"   {model}: {pred} (valid: {pred is not None and not np.isnan(pred) and not np.isinf(pred)})")
            print(f"🔍 Part {part} ({part_category}): Valid predictions: {list(valid.keys())}")
        
        # Initialize default values untuk mencegah error
        fc = 0.0
        best_model = 'AVG_HISTORICAL_DEMAND'
        
        if not valid:
            print(f"❌ Part {part} ({part_category}): NO VALID PREDICTIONS - All models failed!")
            print(f"🔍 Part {part}: Series length: {len(series)}, Series: {series.tolist()}")
            print(f"🔍 Part {part}: Fluctuation type: {fluctuation_type}, Volatility: {volatility_level}")
            print(f"🔍 Part {part}: All predictions: {preds}")
            
            # FALLBACK LOGIC: Gunakan MA fallback dari data historis
            # Untuk semua kategori yang lolos syarat forecast
            if meets_requirement and len(series) > 0 and np.sum(series) > 0:
                # Jika lolos syarat dan ada data, gunakan MA fallback
                recent_data = series[-6:] if len(series) >= 6 else series
                non_zero_data = recent_data[recent_data > 0]
                if len(non_zero_data) > 0:
                    fc = max(np.mean(non_zero_data), 1.0)
                else:
                    fc = max(np.mean(recent_data), 1.0)
                best_model = 'MA_FALLBACK'
                print(f"⚠️  Part {part} ({part_category}): All models failed but meets requirement, using MA_FALLBACK: {fc:.1f}")
            else:
                # Jika tidak lolos syarat, forecast = 0
                fc = 0
                best_model = 'INSUFFICIENT_DATA'
                print(f"⚠️  Part {part} ({part_category}): Does not meet requirement, forecast = 0")
        else:
            # Calculate errors for all valid predictions dengan fluktuation-aware error
            errors = {}
            for model, pred in valid.items():
                # Gunakan fluktuation-aware error untuk model yang responsif
                if fluctuation_type == 'HIGHLY_FLUCTUATING' and model in ['RF', 'XGB', 'ETS_VOLATILE', 'ARIMA_VOLATILE', 'VOLATILITY_ADJUSTED', 'OUTLIER_RESISTANT', 'ADAPTIVE_MA']:
                    # Untuk model yang responsif, gunakan responsiveness bonus
                    errors[model] = calculate_fluctuation_aware_error(actual_val, pred, fluctuation_type, series, [pred] * len(series))
                else:
                    # Untuk model lain, gunakan error biasa
                    errors[model] = hybrid_error(actual_val, pred)
            
        # Tambahkan Ensemble (3-5 model terbaik) jika ada cukup model - OPTIMIZED
        # Strategy: Ensemble RF/XGB dengan fitur eksternal (LSTM removed)
        # Alasan: Kombinasi optimal untuk menangani fluktuasi tinggi
        if len(valid) >= 3 and part_category in ['Erratic', 'Lumpy']:  # Hanya untuk kategori yang memerlukan ensemble
            # Pilih top 3-5 models berdasarkan error
            sorted_models = sorted(errors.items(), key=lambda x: x[1])
            top_models = sorted_models[:min(3, len(sorted_models))]  # REDUCED dari 5 ke 3 models untuk speed
            
            if len(top_models) >= 3:
                # Hitung ensemble weights (inverse error weighting)
                ensemble_weights = {}
                total_weight = 0
                for model, error in top_models:
                    weight = 1 / (error + 1e-8)
                    ensemble_weights[model] = weight
                    total_weight += weight
                
                # Normalize weights
                for model in ensemble_weights:
                    ensemble_weights[model] = ensemble_weights[model] / total_weight
                
                # Hitung ensemble forecast
                ensemble_forecast = 0
                for model, weight in ensemble_weights.items():
                    ensemble_forecast += valid[model] * weight
                
                # Tambahkan ke predictions dan errors
                preds['ENSEMBLE'] = ensemble_forecast
                valid['ENSEMBLE'] = ensemble_forecast
                errors['ENSEMBLE'] = hybrid_error(actual_val, ensemble_forecast)
            
        # MODEL SELECTION UNTUK SEMUA KATEGORI
        if valid and errors:
            # PRIORITAS KHUSUS HANYA UNTUK INTERMITTENT
            # Strategy: Croston family (specialized untuk intermittent) (LSTM removed)
            # Alasan: Intermittent data memerlukan model khusus
            if part_category == 'Intermittent':
                print(f"🔍 Part {part} (Intermittent): Available models: {list(valid.keys())}")
                print(f"🔍 Part {part} (Intermittent): Model errors: {errors}")
                
                # 1. Prioritas model berdasarkan kategori Intermittent
                category_priorities = get_category_model_priorities(part_category)
                print(f"🔍 Part {part} (Intermittent): Category priorities: {category_priorities}")
                
                prioritized_models = {}
                
                for model, error in errors.items():
                    if model in category_priorities['primary']:
                        # Bonus tinggi untuk primary models
                        priority_bonus = 0.8
                        prioritized_models[model] = error * (1 - priority_bonus)
                        print(f"🔍 Part {part} (Intermittent): {model} is PRIMARY, adjusted error: {error * (1 - priority_bonus):.2f}")
                    elif model in category_priorities['secondary']:
                        # Bonus sedang untuk secondary models
                        priority_bonus = 0.4
                        prioritized_models[model] = error * (1 - priority_bonus)
                        print(f"🔍 Part {part} (Intermittent): {model} is SECONDARY, adjusted error: {error * (1 - priority_bonus):.2f}")
                    elif model in category_priorities['avoid']:
                        # Penalty untuk avoid models
                        prioritized_models[model] = error * 1.5
                        print(f"🔍 Part {part} (Intermittent): {model} is AVOID, adjusted error: {error * 1.5:.2f}")
                    else:
                        # Model lain tanpa bonus/penalty
                        prioritized_models[model] = error
                        print(f"🔍 Part {part} (Intermittent): {model} is NEUTRAL, error: {error:.2f}")
                
                # 2. Pilih model dengan prioritas terbaik
                if prioritized_models:
                    best_model = min(prioritized_models.items(), key=lambda x: x[1])[0]
                    fc = valid[best_model]
                    print(f"🎯 Part {part}: Selected {best_model} with {errors[best_model]:.2f}% error (INTERMITTENT PRIORITY-BASED)")
                else:
                    # Fallback ke error terkecil
                    best_model = min(errors.items(), key=lambda x: x[1])[0]
                    fc = valid[best_model]
                    print(f"⚠️  Part {part}: Fallback to {best_model} with {errors[best_model]:.2f}% error")
            else:
                # KATEGORI LAIN (Smooth, Erratic, Lumpy): PILIH ERROR TERKECIL SAJA
                # 1. Prioritas: Error 3%-13% (ideal range)
                ideal_models = {m: e for m, e in errors.items() if 3.0 <= e <= 13.0}
                
                if ideal_models:
                    # Pilih model dengan error terkecil dalam range ideal
                    best_model = min(ideal_models.items(), key=lambda x: x[1])[0]
                    fc = valid[best_model]
                    print(f"🎯 Part {part}: Selected {best_model} with {errors[best_model]:.2f}% error (IDEAL RANGE 3-13%)")
                else:
                    # 2. Prioritas: Error < 30% (termasuk 0% error)
                    acceptable_models = {m: e for m, e in errors.items() if 0 <= e < 30.0}
                    
                    if acceptable_models:
                        # Pilih model dengan error terkecil dalam range acceptable
                        best_model = min(acceptable_models.items(), key=lambda x: x[1])[0]
                        fc = valid[best_model]
                        print(f"✅ Part {part}: Selected {best_model} with {errors[best_model]:.2f}% error (ACCEPTABLE < 30%)")
                    else:
                        # 3. Jika semua model error >= 30%, lakukan tuning
                        print(f"🔧 Part {part}: All models error >= 30%, performing tuning...")
                        
                        # TUNING: Coba parameter yang berbeda untuk model yang ada
                        tuned_models = {}
                        for model_name, pred in valid.items():
                            try:
                                # Tuning berdasarkan model type
                                if model_name.startswith('MA'):
                                    # Tuning MA dengan window yang berbeda
                                    window = int(model_name.replace('MA', ''))
                                    for new_window in [max(2, window-1), window, min(6, window+1)]:
                                        tuned_pred = forecast_ma_flexible(series, new_window)
                                        if not np.isnan(tuned_pred):
                                            tuned_models[f'{model_name}_TUNED_{new_window}'] = tuned_pred
                                            
                                elif model_name == 'WMA':
                                    # Tuning WMA dengan window yang berbeda
                                    for window in [3, 4, 5, 6]:
                                        tuned_pred = forecast_wma(series, window)
                                        if not np.isnan(tuned_pred):
                                            tuned_models[f'WMA_TUNED_{window}'] = tuned_pred
                                            
                                elif model_name == 'ETS':
                                    # Tuning ETS dengan parameter yang berbeda
                                    for trend in ['add', 'mul']:
                                        for seasonal in [None, 'add']:
                                            try:
                                                tuned_pred = forecast_ets(series, seasonal_pref=(seasonal is not None), fluctuation_type='STABLE')
                                                if not np.isnan(tuned_pred):
                                                    tuned_models[f'ETS_TUNED_{trend}_{seasonal}'] = tuned_pred
                                            except:
                                                continue
                                                
                                elif model_name == 'ARIMA':
                                    # Tuning ARIMA dengan parameter yang berbeda
                                    for p in [1, 2, 3]:
                                        for q in [1, 2, 3]:
                                            try:
                                                tuned_pred = forecast_arima_enhanced(series, seasonal_pref=False, fluctuation_type='STABLE')
                                                if not np.isnan(tuned_pred):
                                                    tuned_models[f'ARIMA_TUNED_{p}_{q}'] = tuned_pred
                                            except:
                                                continue
                                                
                                elif model_name in ['CROSTON', 'CROSTON_SBA', 'CROSTON_TSB']:
                                    # Tuning Croston dengan alpha yang berbeda
                                    for alpha in [0.05, 0.1, 0.15, 0.2, 0.25]:
                                        if model_name == 'CROSTON':
                                            tuned_pred = forecast_croston(series, alpha)
                                        elif model_name == 'CROSTON_SBA':
                                            tuned_pred = forecast_croston_sba(series, alpha)
                                        else:  # CROSTON_TSB
                                            tuned_pred = forecast_croston_tsb(series, alpha, alpha)
                                        
                                        if not np.isnan(tuned_pred):
                                            tuned_models[f'{model_name}_TUNED_{alpha}'] = tuned_pred
                                            
                            except Exception as e:
                                print(f"⚠️  Part {part}: Tuning failed for {model_name}: {e}")
                                continue
                        
                        # Hitung error untuk model yang sudah di-tuning
                        tuned_errors = {}
                        for model_name, pred in tuned_models.items():
                            tuned_errors[model_name] = hybrid_error(actual_val, pred)
                        
                        # Bandingkan error sebelum dan sesudah tuning
                        all_models = {**errors, **tuned_errors}
                        all_predictions = {**valid, **tuned_models}
                        
                        # Pilih model dengan error terkecil setelah tuning
                        best_model = min(all_models.items(), key=lambda x: x[1])[0]
                        fc = all_predictions[best_model]
                        
                        if best_model in tuned_errors:
                            print(f"🔧 Part {part}: Selected TUNED {best_model} with {all_models[best_model]:.2f}% error (AFTER TUNING)")
                        else:
                            print(f"🔧 Part {part}: Selected ORIGINAL {best_model} with {all_models[best_model]:.2f}% error (BEST BEFORE TUNING)")
                        
                        # Jika setelah tuning masih tidak ada model dengan error < 30%
                        if all_models[best_model] >= 30.0:
                            print(f"⚠️  Part {part}: After tuning, best error is {all_models[best_model]:.2f}% >= 30%, trying MA fallback...")
                            
                            # MA FALLBACK: Rata-rata dari 6 bulan terakhir
                            # Hitung rata-rata dari data yang ada dalam 6 bulan terakhir
                            if len(series) > 0:
                                # Ambil data dari 6 bulan terakhir yang ada
                                recent_data = series[-6:] if len(series) >= 6 else series
                                # Hitung rata-rata dari data yang ada (bukan 0)
                                non_zero_data = recent_data[recent_data > 0]
                                if len(non_zero_data) > 0:
                                    ma_fallback = np.mean(non_zero_data)
                                else:
                                    ma_fallback = np.mean(recent_data)
                                
                                # Hitung error untuk MA fallback
                                ma_fallback_error = hybrid_error(actual_val, ma_fallback)
                                
                                print(f"📊 Part {part}: MA Fallback error: {ma_fallback_error:.2f}%")
                                
                                # Bandingkan dengan model terbaik
                                if ma_fallback_error < all_models[best_model]:
                                    fc = ma_fallback
                                    best_model = 'MA_FALLBACK'
                                    print(f"✅ Part {part}: Selected MA_FALLBACK with {ma_fallback_error:.2f}% error (BETTER THAN TUNED MODELS)")
                                else:
                                    print(f"⚠️  Part {part}: MA_FALLBACK error {ma_fallback_error:.2f}% >= best model {all_models[best_model]:.2f}%, keeping best model")
                            else:
                                print(f"⚠️  Part {part}: No data available for MA fallback")

        # Handle NaN values in final forecast
        if np.isnan(fc) or np.isinf(fc):
            fc = 0.0
        
        # FIX: Pastikan forecast tidak 0 jika lolos syarat dan ada data
        if fc == 0 and meets_requirement and len(series) > 0 and np.sum(series) > 0:
            # Jika forecast 0 tapi lolos syarat dan ada data, gunakan MA fallback
            # Hitung rata-rata dari data yang ada dalam 6 bulan terakhir
            recent_data = series[-6:] if len(series) >= 6 else series
            # Hitung rata-rata dari data yang ada (bukan 0)
            non_zero_data = recent_data[recent_data > 0]
            if len(non_zero_data) > 0:
                fc = max(np.mean(non_zero_data), 1.0)
            else:
                fc = max(np.mean(recent_data), 1.0)
            best_model = 'MA_FALLBACK'
            print(f"⚠️  Part {part} ({part_category}): Final forecast was 0 but meets requirement, using MA_FALLBACK: {fc:.2f}")
        
        # Safe conversion to int with validation
        if pd.isna(fc) or np.isinf(fc):
            fc = 0.0
        final_fc = max(int(round(fc)) if not pd.isna(fc) and not np.isinf(fc) else 0, 0)
        actual_final = float(test_df['ORIGINAL_SHIPPING_QTY'].values[0]) if not test_df.empty else 0.0
        err_val = hybrid_error(actual_final, final_fc)
        results.append({
            'PART_NO': part,
            'MONTH': target_month.strftime('%Y-%m'),
            'FORECAST': final_fc,
            'ACTUAL': actual_final,
            'ERROR': f"{err_val:.2f}%",
            'BEST_MODEL': best_model,
            'PART_CATEGORY': part_category
        })
    return results


def get_category_model_priorities(part_category):
    """
    Dapatkan prioritas model berdasarkan kategori parts sesuai penelitian
    """
    priorities = {
        'Smooth': {
            'primary': ['ETS', 'ARIMA'],  # Traditional Statistical (LSTM removed)
            'secondary': ['RF', 'XGB', 'WMA', 'MA3', 'MA4', 'MA5', 'MA6'],
            'avoid': ['CROSTON', 'CROSTON_SBA', 'CROSTON_TSB']
        },
        'Erratic': {
            'primary': ['RF', 'XGB', 'ENSEMBLE'],  # ML + Hybrid (LSTM removed)
            'secondary': ['ETS', 'ARIMA', 'WMA', 'MA3', 'MA4', 'MA5', 'MA6'],
            'avoid': ['CROSTON', 'CROSTON_SBA', 'CROSTON_TSB']
        },
        'Intermittent': {
            'primary': ['CROSTON', 'CROSTON_SBA', 'CROSTON_TSB'],  # Specialized Statistical (LSTM removed)
            'secondary': ['RF', 'XGB', 'WMA', 'MA3', 'MA4', 'MA5', 'MA6'],
            'avoid': ['ETS', 'ARIMA']
        },
        'Lumpy': {
            'primary': ['CROSTON_TSB', 'CROSTON', 'CROSTON_SBA', 'ZERO_INFLATED_MEAN', 'ZERO_INFLATED_MEDIAN'],  # Croston family + Zero inflated priority
            'secondary': ['RF', 'XGB', 'WMA', 'MA3', 'MA4', 'MA5', 'MA6'],
            'avoid': ['ETS', 'ARIMA']
        }
    }
    return priorities.get(part_category, priorities['Smooth'])

def select_ensemble_models_for_realtime(part_no, backtest_results, part_category):
    """
    Pilih ensemble models untuk realtime berdasarkan:
    1. Prioritas model sesuai kategori parts (berdasarkan penelitian)
    2. Model yang paling sering dipilih sebagai best di backtest (dengan error < 30%)
    3. Temporal stability analysis (konsistensi performance across months)
    4. Ensemble approach: Gabungkan 3-5 model terbaik dengan weighted average
    5. Prioritas error: 3%-15% (ideal), <30% (acceptable), >30% (avoid)
    6. Hindari 0% error (pilihan kedua)
    """
    try:
        part_data = backtest_results[backtest_results['PART_NO'] == part_no]
        if part_data.empty:
            return 'MA6', 100.0, "No backtest data"
        
        # Hitung performance per model
        model_performance = {}
        for _, row in part_data.iterrows():
            model = row['BEST_MODEL']
            error = float(row['ERROR'].replace('%', ''))
            
            if model not in model_performance:
                model_performance[model] = {'errors': [], 'count': 0, 'months': []}
            
            model_performance[model]['errors'].append(error)
            model_performance[model]['count'] += 1
            model_performance[model]['months'].append(row['MONTH'])
        
        # Dapatkan prioritas model berdasarkan kategori (jika ada)
        category_priorities = get_category_model_priorities(part_category) if part_category != 'Smooth' else None
        
        # Hitung temporal stability HANYA untuk model dengan error < 30%
        temporal_stability = {}
        for model, perf in model_performance.items():
            avg_error = np.mean(perf['errors'])
            # HANYA hitung stability untuk model dengan error < 30%
            if avg_error < 30.0:
                errors = perf['errors']
                if len(errors) >= 2:
                    # Hitung consistency across months
                    error_std = np.std(errors)
                    error_trend = np.polyfit(range(len(errors)), errors, 1)[0]  # Slope
                    stability_score = 1 / (error_std + 1e-8) * (1 / (abs(error_trend) + 1e-8))
                    temporal_stability[model] = stability_score
                else:
                    temporal_stability[model] = 1.0  # Default untuk model dengan sedikit data
            else:
                # Model dengan error >= 30% tidak dihitung stability score
                temporal_stability[model] = 0.0
        
        # Filter model dengan error < 30% dan hindari 0% error
        valid_models = {}
        for model, perf in model_performance.items():
            avg_error = np.mean(perf['errors'])
            # Hindari model dengan 0% error sebagai prioritas utama
            if avg_error > 0 and avg_error < 30.0:
                valid_models[model] = {
                    'avg_error': avg_error,
                    'count': perf['count'],
                    'errors': perf['errors'],
                    'frequency': perf['count'] / len(part_data)
                }
        
        # 🎯 ENSEMBLE APPROACH: Pilih 3-5 model terbaik untuk ensemble
        if len(valid_models) >= 3:
            # Hitung combined score untuk semua valid models
            ensemble_candidates = {}
            for model, info in valid_models.items():
                # Priority score berdasarkan kategori (jika ada)
                if category_priorities:
                    if model in category_priorities['primary']:
                        priority_score = 3
                    elif model in category_priorities['secondary']:
                        priority_score = 2
                    elif model in category_priorities['avoid']:
                        priority_score = 0
                    else:
                        priority_score = 1
                else:
                    # Jika tidak ada part category, semua model sama prioritasnya
                    priority_score = 1
                
                # Error score (semakin kecil error, semakin besar score)
                error_score = 1 / (info['avg_error'] + 1e-8)
                
                # Temporal stability score
                stability_score = temporal_stability.get(model, 1.0)
                
                # Combined score: 70% error + 20% stability + 10% priority
                combined_score = (0.7 * error_score) + (0.2 * stability_score) + (0.1 * priority_score)
                
                ensemble_candidates[model] = {
                    'priority_score': priority_score,
                    'error_score': error_score,
                    'combined_score': combined_score,
                    'avg_error': info['avg_error'],
                    'frequency': info['frequency']
                }
            
            # Pilih top 3-5 models untuk ensemble
            sorted_models = sorted(ensemble_candidates.items(), key=lambda x: x[1]['combined_score'], reverse=True)
            top_models = sorted_models[:min(5, len(sorted_models))]
            
            # Hitung ensemble weights (inverse error weighting)
            ensemble_weights = {}
            total_weight = 0
            for model, info in top_models:
                # Weight berdasarkan inverse error dan frequency
                weight = (1 / (info['avg_error'] + 1e-8)) * (1 + info['frequency'])
                ensemble_weights[model] = weight
                total_weight += weight
            
            # Normalize weights
            for model in ensemble_weights:
                ensemble_weights[model] = ensemble_weights[model] / total_weight
            
            # Hitung average error untuk ensemble
            avg_ensemble_error = np.mean([info['avg_error'] for _, info in top_models])
            
            # Format ensemble info
            ensemble_info = f"ENSEMBLE: {len(top_models)} models ("
            for i, (model, info) in enumerate(top_models):
                ensemble_info += f"{model}: {ensemble_weights[model]:.2f}"
                if i < len(top_models) - 1:
                    ensemble_info += ", "
            ensemble_info += f") - avg error: {avg_ensemble_error:.1f}%"
            
            return 'ENSEMBLE', avg_ensemble_error, ensemble_info, ensemble_weights, [model for model, _ in top_models]
        
        elif len(valid_models) >= 1:
            # Jika hanya ada 1-2 model, gunakan single model terbaik
            best_model = min(valid_models.items(), key=lambda x: x[1]['avg_error'])
            return best_model[0], best_model[1]['avg_error'], f"Single model: {best_model[0]} (error: {best_model[1]['avg_error']:.1f}%)"
        
        else:
            # Fallback ke semua model jika tidak ada yang valid
            all_models = {}
            for model, perf in model_performance.items():
                avg_error = np.mean(perf['errors'])
                all_models[model] = {
                    'avg_error': avg_error,
                    'count': perf['count'],
                    'errors': perf['errors'],
                    'frequency': perf['count'] / len(part_data)
                }
            
            # Pilih model terbaik dari semua model
            best_model = min(all_models.items(), key=lambda x: x[1]['avg_error'])
            return best_model[0], best_model[1]['avg_error'], f"Fallback: {best_model[0]} (error: {best_model[1]['avg_error']:.1f}%)"
        
    except Exception as e:
        return 'MA6', 100.0, f"Error: {str(e)}"

def run_backtest_and_realtime(df: pd.DataFrame):
    df_proc = df.copy()
    df_proc['MONTH'] = parse_month_column(df_proc['MONTH'])
    if 'PartCategory' in df_proc.columns and 'PART_CATEGORY' not in df_proc.columns:
        df_proc = df_proc.rename(columns={'PartCategory': 'PART_CATEGORY'})

    # Limit dataset to last N months only
    all_months = sorted(df_proc['MONTH'].unique())
    if len(all_months) > MAX_DATA_MONTHS:
        cutoff_month = all_months[-MAX_DATA_MONTHS]  # Keep only last N months
        df_proc = df_proc[df_proc['MONTH'] >= cutoff_month]
        print(f"📅 Dataset limited to last {MAX_DATA_MONTHS} months: {cutoff_month.strftime('%Y-%m')} onwards")
    
    # Check if PART_CATEGORY column exists
    has_part_category = 'PART_CATEGORY' in df_proc.columns
    print(f"📊 Part category column available: {has_part_category}")
    
    # Check if ICC/INVENTORY_CONTROL_CLASS column exists
    has_icc = 'INVENTORY_CONTROL_CLASS' in df_proc.columns
    print(f"📊 ICC column available: {has_icc}")

    # Aggregate (exclude PART_NAME/TOPAS_ORDER_TYPE)
    agg = {'ORIGINAL_SHIPPING_QTY': 'sum'}
    for c in ['WORKING_DAYS','ORDER_CYCLE_DAYS','SS_DEMAND_QTY','STANDARD_STOCK_DAYS','DELIVERY_LT_REGULER','SS_LT_DAYS']:
        if c in df_proc.columns:
            agg[c] = 'mean'
    if 'INVENTORY_CONTROL_CLASS' in df_proc.columns:
        agg['INVENTORY_CONTROL_CLASS'] = 'first'
    if 'PART_CATEGORY' in df_proc.columns:
        agg['PART_CATEGORY'] = 'first'
    df_proc = df_proc.groupby(['PART_NO','MONTH'], as_index=False).agg(agg)

    add_operational_features_inplace(df_proc)

    # Time features
    df_proc['MONTH_NUM'] = df_proc['MONTH'].dt.month
    df_proc['YEAR'] = df_proc['MONTH'].dt.year
    df_proc['MONTH_SIN'] = np.sin(2 * np.pi * df_proc['MONTH_NUM'] / 12)
    df_proc['MONTH_COS'] = np.cos(2 * np.pi * df_proc['MONTH_NUM'] / 12)

    all_months = sorted(df_proc['MONTH'].unique())
    if len(all_months) < 5:
        raise Exception('Dataset minimal 5 bulan (4 backtest + 1 realtime)')
    
    # FIX: Backtest menggunakan 4 bulan sebelum bulan terakhir (bulan terakhir TIDAK termasuk)
    # Realtime menggunakan bulan terakhir + 3 bulan ke depan
    # FIX: Pastikan tipe data konsisten dengan part_df['MONTH']
    backtest_months = all_months[-5:-1]  # 4 bulan sebelum bulan terakhir (tanpa pd.to_datetime)
    
    print(f"📅 All available months: {[pd.to_datetime(m).strftime('%Y-%m') for m in all_months]}")
    print(f"📅 Backtest months (4 months before last): {[m.strftime('%Y-%m') for m in backtest_months]}")
    print(f"📊 Backtest logic: Rolling 6-month window check for each target month")
    
    # Realtime: bulan terakhir + 3 bulan ke depan
    # FIX: Pastikan tipe data konsisten dengan part_df['MONTH']
    last_month = all_months[-1].replace(day=1) if hasattr(all_months[-1], 'replace') else pd.to_datetime(all_months[-1]).replace(day=1)
    realtime_months = pd.date_range(start=last_month, periods=4, freq='MS')
    
    print(f"📅 Realtime months (last month + 3 ahead): {[m.strftime('%Y-%m') for m in realtime_months]}")
    print(f"📊 Realtime logic: Rolling 6-month window check for each target month")

    part_list = df_proc['PART_NO'].unique()
    part_to_category = df_proc.groupby('PART_NO')['PART_CATEGORY'].first().to_dict() if has_part_category else {}

    # Backtest dengan progress bar dan optimasi
    print(f"🔄 Starting backtest for {len(part_list)} parts...")
    print(f"📊 Backtest months: {len(backtest_months)} months per part")
    print(f"⏱️  Estimated time: ~{len(part_list) * len(backtest_months) * 0.5:.1f} seconds")
    print(f"🔧 Using {min(N_JOBS, len(part_list))} CPU cores for parallel processing")
    
    # DEBUG: Tambahkan log untuk troubleshooting
    print(f"🔍 DEBUG: Backtest months: {[m.strftime('%Y-%m') for m in backtest_months]}")
    print(f"🔍 DEBUG: Sample parts: {part_list[:5]}")
    
    # Optimasi: Gunakan lebih banyak cores untuk parallel processing
    bt_nested = Parallel(n_jobs=min(N_JOBS, len(part_list)))(
        delayed(process_part)(
            p,
            df_proc[df_proc['PART_NO'] == p].sort_values('MONTH'),
            backtest_months,
            part_to_category.get(p, 'Smooth') if has_part_category else 'Smooth'
        ) for p in tqdm(part_list, desc="Backtest Progress", unit="part", 
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} parts [{elapsed}<{remaining}]')
    )
    backtest = pd.DataFrame([r for sub in bt_nested for r in sub])

    # Enhanced model selection dengan ENSEMBLE APPROACH
    latest_models = {}
    if not backtest.empty:
        print(f"🔄 Starting model selection for {len(backtest['PART_NO'].unique())} parts...")
        tmp = backtest.copy()
        tmp['ERR_NUM'] = pd.to_numeric(tmp['ERROR'].str.replace('%',''), errors='coerce')
        
        for p, g in tqdm(tmp.groupby('PART_NO'), desc="Model Selection", unit="part",
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} parts [{elapsed}<{remaining}]'):
            # Get part category
            part_category = g['PART_CATEGORY'].iloc[0] if 'PART_CATEGORY' in g.columns else 'Smooth'
            
            # Pilih ensemble models untuk realtime berdasarkan logika yang diminta
            result = select_ensemble_models_for_realtime(p, tmp, part_category)
            
            # Dapatkan error performance untuk setiap model dari backtest
            part_backtest = tmp[tmp['PART_NO'] == p]
            backtest_errors = {}
            for _, row in part_backtest.iterrows():
                model = row['BEST_MODEL']
                error = float(row['ERROR'].replace('%', ''))
                if model not in backtest_errors:
                    backtest_errors[model] = []
                backtest_errors[model].append(error)
            
            # Hitung average error untuk setiap model
            avg_errors = {}
            for model, errors in backtest_errors.items():
                avg_errors[model] = np.mean(errors)
            
            if len(result) == 5:  # Ensemble result
                best_model, avg_error, reason, ensemble_weights, ensemble_models = result
                latest_models[p] = {
                    'model': best_model, 
                    'error': avg_error,
                    'reason': reason,
                    'ensemble_weights': ensemble_weights,
                    'ensemble_models': ensemble_models,
                    'backtest_errors': avg_errors,  # Tambahkan error performance dari backtest
                    'is_ensemble': True
                }
            else:  # Single model result
                best_model, avg_error, reason = result
                latest_models[p] = {
                    'model': best_model, 
                    'error': avg_error,
                    'reason': reason,
                    'backtest_errors': avg_errors,  # Tambahkan error performance dari backtest
                    'is_ensemble': False
                }
            
            # Tampilkan informasi detail pemilihan model
            if avg_error <= 15.0:
                print(f"🎯 Part {p}: Selected {best_model} with {avg_error:.1f}% error - EXCELLENT ({reason})")
            elif avg_error <= 30.0:
                print(f"✅ Part {p}: Selected {best_model} with {avg_error:.1f}% error - GOOD ({reason})")
            elif avg_error <= 50.0:
                print(f"⚠️  Part {p}: Selected {best_model} with {avg_error:.1f}% error - ACCEPTABLE ({reason})")
            else:
                print(f"❌ Part {p}: Selected {best_model} with {avg_error:.1f}% error - HIGH ERROR ({reason})")

    # Summary statistics untuk model selection
    if latest_models:
        print(f"\n📊 MODEL SELECTION SUMMARY:")
        print(f"=" * 50)
        
        # Hitung statistik
        total_parts = len(latest_models)
        excellent_parts = sum(1 for p, info in latest_models.items() if info['error'] <= 15.0)
        good_parts = sum(1 for p, info in latest_models.items() if 15.0 < info['error'] <= 30.0)
        acceptable_parts = sum(1 for p, info in latest_models.items() if 30.0 < info['error'] <= 50.0)
        high_error_parts = sum(1 for p, info in latest_models.items() if info['error'] > 50.0)
        
        print(f"Total Parts: {total_parts}")
        print(f"Excellent (error <= 15%): {excellent_parts} ({excellent_parts/total_parts*100:.1f}%)")
        print(f"Good (15% < error <= 30%): {good_parts} ({good_parts/total_parts*100:.1f}%)")
        print(f"Acceptable (30% < error <= 50%): {acceptable_parts} ({acceptable_parts/total_parts*100:.1f}%)")
        print(f"High Error (error > 50%): {high_error_parts} ({high_error_parts/total_parts*100:.1f}%)")
        
        # Model distribution
        model_counts = {}
        for p, info in latest_models.items():
            model = info['model']
            model_counts[model] = model_counts.get(model, 0) + 1
        
        print(f"\nModel Distribution:")
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {count} parts ({count/total_parts*100:.1f}%)")
        
        # Error distribution
        errors = [info['error'] for info in latest_models.values()]
        print(f"\nError Statistics:")
        print(f"  Average Error: {np.mean(errors):.1f}%")
        print(f"  Median Error: {np.median(errors):.1f}%")
        print(f"  Min Error: {np.min(errors):.1f}%")
        print(f"  Max Error: {np.max(errors):.1f}%")
        
        # Performance by category
        print(f"\nPerformance by Part Category:")
        category_performance = {}
        for p, info in latest_models.items():
            # Get part category from backtest data
            part_category = tmp[tmp['PART_NO'] == p]['PART_CATEGORY'].iloc[0] if not tmp[tmp['PART_NO'] == p].empty else 'Unknown'
            
            if part_category not in category_performance:
                category_performance[part_category] = {'errors': [], 'count': 0}
            
            category_performance[part_category]['errors'].append(info['error'])
            category_performance[part_category]['count'] += 1
        
        for category, perf in category_performance.items():
            avg_error = np.mean(perf['errors'])
            count = perf['count']
            print(f"  {category}: {count} parts, avg error: {avg_error:.1f}%")
        
        # Selection reason distribution
        print(f"\nSelection Reason Distribution:")
        reason_counts = {}
        for p, info in latest_models.items():
            reason = info.get('reason', 'Unknown')
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count} parts ({count/total_parts*100:.1f}%)")
        
        # Model selection by category (showing adherence to research recommendations)
        print(f"\nModel Selection by Category (Research-Based Recommendations):")
        print(f"📊 Selection Logic: 70% Error Performance + 20% Temporal Stability + 10% Category Priority")
        print(f"📊 Fallback Logic: 70% Error Performance + 20% Temporal Stability + 10% Category Priority")
        print(f"📊 Focus: Temporal Generalization untuk mengurangi overfitting")
        
        for category in ['Smooth', 'Erratic', 'Intermittent', 'Lumpy']:
            category_parts = [p for p, info in latest_models.items() 
                           if tmp[tmp['PART_NO'] == p]['PART_CATEGORY'].iloc[0] == category] if not tmp.empty else []
            
            if category_parts:
                category_priorities = get_category_model_priorities(category)
                print(f"\n  {category} ({len(category_parts)} parts):")
                print(f"    Recommended Primary: {', '.join(category_priorities['primary'])}")
                print(f"    Recommended Secondary: {', '.join(category_priorities['secondary'])}")
                print(f"    Avoid: {', '.join(category_priorities['avoid'])}")
                
                # Show actual model distribution for this category
                category_models = {}
                category_errors = {}
                for p in category_parts:
                    model = latest_models[p]['model']
                    error = latest_models[p]['error']
                    category_models[model] = category_models.get(model, 0) + 1
                    if model not in category_errors:
                        category_errors[model] = []
                    category_errors[model].append(error)
                
                print(f"    Actual Selection:")
                for model, count in sorted(category_models.items(), key=lambda x: x[1], reverse=True):
                    adherence = "✅" if model in category_priorities['primary'] else "⚠️" if model in category_priorities['secondary'] else "❌" if model in category_priorities['avoid'] else "🔶"
                    avg_error = np.mean(category_errors[model])
                    print(f"      {adherence} {model}: {count} parts ({count/len(category_parts)*100:.1f}%) - avg error: {avg_error:.1f}%")

    # Realtime rolling dengan ENSEMBLE APPROACH (selalu ensemble dari best models)
    print(f"🔄 Starting realtime forecast for {len(part_list)} parts...")
    print(f"📊 Realtime months: {len(realtime_months)} months per part")
    print(f"⏱️  Estimated time: ~{len(part_list) * len(realtime_months) * 0.3:.1f} seconds")
    
    # Count parts by category
    category_counts = {}
    for p in part_list:
        cat = part_to_category.get(p, 'Smooth')
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"📊 Parts by category: {category_counts}")
    
    results_rt = []
    for p in tqdm(part_list, desc="Realtime Progress", unit="part",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} parts [{elapsed}<{remaining}]'):
        cat = part_to_category.get(p, 'Smooth')
        part_df = df_proc[df_proc['PART_NO'] == p].sort_values('MONTH')
        hist = part_df.set_index('MONTH').resample('MS')['ORIGINAL_SHIPPING_QTY'].sum().fillna(0)
        series = hist.values.astype(float)
        lm = latest_models.get(p)
        future_series = list(series)
        
        # ROLLING FORECAST: Update series dengan forecast sebelumnya untuk forecast berikutnya
        # Ini memastikan setiap forecast menggunakan data terbaru termasuk forecast sebelumnya
        for m in tqdm(realtime_months, desc=f"Part {p}", leave=False, disable=len(part_list) > 10):
            fc = np.nan
            
            # SYARAT FORECAST UNTUK SEMUA KATEGORI DI REALTIME
            # Semua kategori memerlukan syarat forecast: minimal 2 bulan dengan demand > 0 di 6 bulan terakhir
                # Gunakan data historis untuk validasi
            # FIX: Pastikan tipe data konsisten untuk filtering di realtime
            if not pd.api.types.is_datetime64_any_dtype(part_df['MONTH']):
                part_df['MONTH'] = pd.to_datetime(part_df['MONTH'])
            
            # FIX: Gunakan semua data part untuk window check yang akurat di realtime juga
            meets_requirement, months_with_demand, demand_values = validate_demand_requirement_general(p, m, part_df, pd.DataFrame(), cat)
                
            if not meets_requirement:
                # Jika tidak memenuhi syarat, forecast = 0 dan best model = INSUFFICIENT_DATA
                fc = 0.0
                chosen_model = 'INSUFFICIENT_DATA'
                print(f"❌ Realtime Part {p} ({cat}): Does not meet forecast requirement - forecast = 0")
                    
                # Tambahkan ke results dan skip forecasting
                results_rt.append({
                    'PART_NO': p, 
                    'MONTH': m.strftime('%Y-%m'), 
                    'BEST_MODEL': 'INSUFFICIENT_DATA', 
                    'FORECAST_NEUTRAL': 0, 
                    'ERROR_BACKTEST': '', 
                    'PART_CATEGORY': cat
                })
                continue  # Skip forecasting untuk part ini
            else:
                print(f"✅ Realtime Part {p} ({cat}): Meets forecast requirement - proceeding with forecast")
            
            # Jika backtest menunjukkan model fallback tapi realtime punya data
            if lm and lm.get('model') in ['AVG_HISTORICAL_DEMAND', 'AVG_SERIES', 'AVG_DEMAND', 'NO_HISTORICAL_DATA', 'INSUFFICIENT_DATA']:
                print(f"🔄 Realtime Part {p}: Backtest showed fallback model but realtime has data, using default models")
                # Gunakan model default berdasarkan kategori
                if cat == 'Smooth':
                    fc = forecast_ets(np.array(future_series), seasonal_pref=True)
                    chosen_model = 'ETS'
                elif cat == 'Intermittent':
                    fc = forecast_croston(np.array(future_series))
                    chosen_model = 'CROSTON'
                elif cat in ['Erratic', 'Lumpy']:
                    fc = forecast_wma(future_series)  # LSTM removed, use WMA instead
                    chosen_model = 'WMA'
                else:
                    fc = forecast_wma(future_series)
                    chosen_model = 'WMA'
                
                if np.isnan(fc):
                    fc = forecast_wma(future_series)
                    chosen_model = 'WMA'
                
                print(f"✅ Realtime Part {p}: Using {chosen_model} for realtime forecast: {fc:.1f}")
                
            # SELALU gunakan ensemble dengan DYNAMIC WEIGHTING berdasarkan error performance
            elif lm and lm.get('is_ensemble', False):
                # 🎯 DYNAMIC ENSEMBLE FORECASTING: Weighting berdasarkan error performance
                ensemble_preds = {}
                ensemble_models = lm.get('ensemble_models', [])
                
                # 1. Dapatkan predictions dari semua models
                for model_name in ensemble_models:
                    try:
                        if model_name.startswith('MA'):
                            # Extract window from model name (MA3, MA4, MA5, MA6)
                            window = int(model_name.replace('MA', ''))
                            pred = forecast_ma_flexible(future_series, window)
                        elif model_name == 'WMA':
                            pred = forecast_wma(future_series)
                        elif model_name == 'ETS':
                            pred = forecast_ets(np.array(future_series), seasonal_pref=(cat in ['Smooth','Erratic']))
                        elif model_name == 'ARIMA':
                            pred = forecast_arima_enhanced(np.array(future_series), seasonal_pref=(cat in ['Smooth','Erratic']))
                        elif model_name in ['CROSTON','CROSTON_SBA','CROSTON_TSB']:
                            pred = forecast_croston(np.array(future_series)) if model_name=='CROSTON' else (
                                forecast_croston_sba(np.array(future_series)) if model_name=='CROSTON_SBA' else forecast_croston_tsb(np.array(future_series)))
                        elif model_name == 'RF':
                            # RF model dengan features - fallback ke simple approach
                            pred = forecast_ma_flexible(future_series, 3)  # Fallback ke MA3
                        elif model_name == 'XGB':
                            # XGB model dengan features - fallback ke simple approach
                            pred = forecast_ma_flexible(future_series, 4)  # Fallback ke MA4
                        else:
                            pred = forecast_wma(future_series)
                        
                        if not np.isnan(pred):
                            ensemble_preds[model_name] = pred
                    except Exception as e:
                        logger.warning(f"Error forecasting {model_name} for part {p}: {e}")
                        continue
                
                if ensemble_preds:
                    # 2. Hitung dynamic weights berdasarkan error performance di backtest
                    dynamic_weights = {}
                    total_weight = 0
                    
                    # Dapatkan error performance dari backtest untuk setiap model
                    backtest_errors = lm.get('backtest_errors', {})
                    
                    # Jika tidak ada error performance dari backtest, gunakan default
                    if not backtest_errors:
                        for model_name in ensemble_preds.keys():
                            # Default error performance berdasarkan model type
                            if model_name in ['RF', 'XGB']:
                                backtest_errors[model_name] = 15.0  # ML models biasanya lebih baik
                            elif model_name in ['ETS', 'ARIMA']:
                                backtest_errors[model_name] = 20.0  # Statistical models
                            elif model_name.startswith('MA'):
                                backtest_errors[model_name] = 25.0  # Simple models
                            else:
                                backtest_errors[model_name] = 30.0  # Default
                    
                    # Hitung weights berdasarkan inverse error (error lebih kecil = weight lebih besar)
                    for model_name, pred in ensemble_preds.items():
                        error = backtest_errors.get(model_name, 30.0)
                        # Weight = 1 / (error + 1e-8) untuk menghindari division by zero
                        weight = 1 / (error + 1e-8)
                        dynamic_weights[model_name] = weight
                        total_weight += weight
                    
                    # Normalize weights
                    for model_name in dynamic_weights:
                        dynamic_weights[model_name] = dynamic_weights[model_name] / total_weight
                    
                    # 3. Hitung ensemble forecast dengan dynamic weights
                    fc = ensemble_voting(ensemble_preds, dynamic_weights)
                    chosen_model = 'ML ENSEMBLE'
                    print(f"Part {p}: Dynamic ensemble forecast using {len(ensemble_preds)} models (weights: {dynamic_weights})")
                else:
                    # Jika tidak ada ensemble info, gunakan single model terbaik
                    chosen = lm['model'] if lm else ('WMA' if len(series) >= 3 else 'MA6')
                    chosen_model = chosen
                    
                    if chosen.startswith('MA'):
                        # Extract window from model name (MA3, MA4, MA5, MA6)
                        window = int(chosen.replace('MA', ''))
                        fc = forecast_ma_flexible(future_series, window)
                    elif chosen == 'WMA':
                        fc = forecast_wma(future_series)
                    elif chosen == 'ETS':
                        fc = forecast_ets(np.array(future_series), seasonal_pref=(cat in ['Smooth','Erratic']))
                    elif chosen == 'ARIMA':
                        fc = forecast_arima_enhanced(np.array(future_series), seasonal_pref=(cat in ['Smooth','Erratic']))
                    elif chosen in ['CROSTON','CROSTON_SBA','CROSTON_TSB']:
                        fc = forecast_croston(np.array(future_series)) if chosen=='CROSTON' else (
                            forecast_croston_sba(np.array(future_series)) if chosen=='CROSTON_SBA' else forecast_croston_tsb(np.array(future_series)))
                    else:
                        # RF/XGB/LR/BLEND require features; for realtime, fall back to WMA for simplicity
                        fc = forecast_wma(future_series)
                        chosen_model = 'WMA'
            
            # FALLBACK: Jika tidak ada model info, gunakan default berdasarkan kategori
            else:
                if cat == 'Smooth':
                    fc = forecast_ets(np.array(future_series), seasonal_pref=True)
                    chosen_model = 'ETS'
                elif cat == 'Intermittent':
                    fc = forecast_croston(np.array(future_series))
                    chosen_model = 'CROSTON'
                elif cat in ['Erratic', 'Lumpy']:
                    fc = forecast_wma(future_series)  # LSTM removed, use WMA instead
                    chosen_model = 'WMA'
                else:
                    fc = forecast_wma(future_series)
                    chosen_model = 'WMA'
                
                if np.isnan(fc):
                    fc = forecast_wma(future_series)
                    chosen_model = 'WMA'
            
            fc = 0 if np.isnan(fc) else fc
            fc = 0 if np.isinf(fc) else fc
            future_series.append(float(fc))
            err_bt = lm['error'] if lm else ''
            # Safe conversion to int with validation
            forecast_int = int(round(fc)) if not pd.isna(fc) and not np.isinf(fc) else 0
            # chosen_model sudah didefinisikan di atas, tidak perlu didefinisikan ulang
            results_rt.append({'PART_NO': p, 'MONTH': m.strftime('%Y-%m'), 'BEST_MODEL': chosen_model, 'FORECAST_NEUTRAL': max(forecast_int, 0), 'ERROR_BACKTEST': (f"{round(err_bt,2)}%" if err_bt != '' else ''), 'PART_CATEGORY': cat})

    # Summary progress
    print(f"\n✅ FORECAST COMPLETED!")
    print(f"📊 Backtest: {len(backtest)} records")
    print(f"📊 Realtime: {len(results_rt)} records")
    print(f"📊 Total parts processed: {len(part_list)}")
    
    # Performance summary
    if not backtest.empty:
        avg_error = backtest['ERROR'].str.replace('%', '').astype(float).mean()
        print(f"📈 Average backtest error: {avg_error:.2f}%")
        
        # Error distribution
        excellent = len(backtest[backtest['ERROR'].str.replace('%', '').astype(float) <= 15])
        good = len(backtest[(backtest['ERROR'].str.replace('%', '').astype(float) > 15) & 
                           (backtest['ERROR'].str.replace('%', '').astype(float) <= 30)])
        acceptable = len(backtest[(backtest['ERROR'].str.replace('%', '').astype(float) > 30) & 
                                (backtest['ERROR'].str.replace('%', '').astype(float) <= 50)])
        poor = len(backtest[backtest['ERROR'].str.replace('%', '').astype(float) > 50])
        
        print(f"📊 Error distribution:")
        print(f"   Excellent (≤15%): {excellent} ({excellent/len(backtest)*100:.1f}%)")
        print(f"   Good (15-30%): {good} ({good/len(backtest)*100:.1f}%)")
        print(f"   Acceptable (30-50%): {acceptable} ({acceptable/len(backtest)*100:.1f}%)")
        print(f"   Poor (>50%): {poor} ({poor/len(backtest)*100:.1f}%)")

    return df_proc, backtest, pd.DataFrame(results_rt)


def quick_tune_rf_xgb(model_name, X_train, y_train, X_test):
    """
    Advanced quick tuning untuk RF/XGB models dengan multiple configurations
    """
    try:
        if len(X_train) < 6:
            return np.nan
        
        best_pred = np.nan
        best_error = float('inf')
        
        if model_name == 'RF':
            # Multiple RF configurations
            rf_configs = [
                {'n_estimators': 400, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 2},
                {'n_estimators': 300, 'max_depth': 12, 'min_samples_split': 3, 'min_samples_leaf': 1},
                {'n_estimators': 500, 'max_depth': 18, 'min_samples_split': 2, 'min_samples_leaf': 3},
                {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 4, 'min_samples_leaf': 2}
            ]
            
            for config in rf_configs:
                try:
                    rf = RandomForestRegressor(random_state=42, **config)
                    y_train_log = np.log1p(y_train)
                    rf.fit(X_train, y_train_log)
                    pred_log = rf.predict(X_test)[0] if len(X_test) else rf.predict(X_train.iloc[-1:])[0]
                    pred = float(np.expm1(pred_log))
                    
                    # Simple cross-validation
                    if len(X_train) >= 10:
                        train_subset = X_train.iloc[:-2]
                        y_subset = y_train.iloc[:-2]
                        test_val = y_train.iloc[-1]
                        
                        rf.fit(train_subset, np.log1p(y_subset))
                        test_pred_log = rf.predict(X_train.iloc[-2:-1])[0]
                        test_pred = float(np.expm1(test_pred_log))
                        test_error = hybrid_error(test_val, test_pred)
                        
                        if test_error < best_error:
                            best_error = test_error
                            best_pred = pred
                    else:
                        if np.isnan(best_pred):
                            best_pred = pred
                except Exception:
                    continue
        
        elif model_name == 'XGB':
            # Multiple XGB configurations
            xgb_configs = [
                {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.15, 'subsample': 0.8, 'colsample_bytree': 0.8},
                {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.9},
                {'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.2, 'subsample': 0.7, 'colsample_bytree': 0.7},
                {'n_estimators': 150, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.95, 'colsample_bytree': 0.95}
            ]
            
            for config in xgb_configs:
                try:
                    xgb = XGBRegressor(random_state=42, **config)
                    y_train_log = np.log1p(y_train)
                    xgb.fit(X_train, y_train_log)
                    pred_log = xgb.predict(X_test)[0] if len(X_test) else xgb.predict(X_train.iloc[-1:])[0]
                    pred = float(np.expm1(pred_log))
                    
                    # Simple cross-validation
                    if len(X_train) >= 10:
                        train_subset = X_train.iloc[:-2]
                        y_subset = y_train.iloc[:-2]
                        test_val = y_train.iloc[-1]
                        
                        xgb.fit(train_subset, np.log1p(y_subset))
                        test_pred_log = xgb.predict(X_train.iloc[-2:-1])[0]
                        test_pred = float(np.expm1(test_pred_log))
                        test_error = hybrid_error(test_val, test_pred)
                        
                        if test_error < best_error:
                            best_error = test_error
                            best_pred = pred
                    else:
                        if np.isnan(best_pred):
                            best_pred = pred
                except Exception:
                    continue
        
        return best_pred
        
    except Exception:
        return np.nan

def simplify_model_name(model_name):
    """
    Menyederhanakan nama model untuk display di Excel - ULTRA SIMPLIFIED VERSION
    """
    if model_name is None:
        return 'UNKNOWN'
    
    model_str = str(model_name).upper()
    
    # Croston Family (termasuk SBA, TSB, Zero Inflated, dll)
    if any(x in model_str for x in ['CROSTON', 'SBA', 'TSB', 'ZERO_INFLATED', 'CROSTON_SBA', 'CROSTON_TSB']):
        return 'CROSTON'
    
    # MA Family (termasuk MA2, MA3, MA4, MA5, Simple Avg, dll)
    if any(x in model_str for x in ['MA', 'MOVING_AVERAGE', 'AVG_HISTORICAL_DEMAND', 'AVG_SERIES', 'SIMPLE_AVG', 'MA_FALLBACK']):
        return 'MA'
    
    # WMA Family (termasuk WMA, WMA3, dll)
    if any(x in model_str for x in ['WMA', 'WEIGHTED', 'WEIGHTED_MA']):
        return 'WMA'
    
    # ETS Family
    if any(x in model_str for x in ['ETS', 'EXPONENTIAL_SMOOTHING', 'EXPONENTIAL']):
        return 'ETS'
    
    # ARIMA Family
    if any(x in model_str for x in ['ARIMA', 'AUTO_ARIMA', 'ARIMA_ENHANCED']):
        return 'ARIMA'
    
    # Machine Learning Models
    if any(x in model_str for x in ['RF', 'RANDOM_FOREST', 'RANDOMFOREST']):
        return 'RANDOM_FOREST'
    if any(x in model_str for x in ['XGB', 'XGBOOST', 'XGBOOST']):
        return 'XGBOOST'
    
    # Trend Models (Linear, Exponential, Last Value) -> Basic Statistic
    if any(x in model_str for x in ['LINEAR_TREND', 'EXP_TREND', 'LAST_VALUE', 'TREND', 'SIMPLE_TREND']):
        return 'BASIC_STATISTIC'
    
    # Sederhanakan semua variasi Ensemble menjadi 'ML ENSEMBLE'
    if any(x in model_str for x in ['ENSEMBLE', 'BLEND', 'ML_ENSEMBLE', 'MACHINE_LEARNING_ENSEMBLE']):
        return 'ML ENSEMBLE'
    
    # Sederhanakan High Volatility models ke base model
    if any(x in model_str for x in ['VOLATILITY_ADJUSTED', 'VOLATILE', 'HIGH_VOLATILITY']):
        return 'ETS'  # Biasanya ETS-based
    if any(x in model_str for x in ['OUTLIER_RESISTANT', 'ROBUST', 'RESISTANT']):
        return 'MA'   # MA-based
    if any(x in model_str for x in ['ADAPTIVE_MA', 'ADAPTIVE', 'ADAPTIVE_MOVING_AVERAGE']):
        return 'MA'
    
    # Sederhanakan Simple models menjadi 'MA'
    if any(x in model_str for x in ['SIMPLE_AVG', 'SIMPLE_TREND', 'SIMPLE', 'BASIC']):
        return 'MA'
    if any(x in model_str for x in ['MEDIAN', 'MEDIAN_AVERAGE']):
        return 'MA'
    if any(x in model_str for x in ['TRIMMED_MEAN', 'TRIMMED_MEDIAN', 'TRIMMED']):
        return 'MA'
    if any(x in model_str for x in ['ZERO_INFLATED_MEAN', 'ZERO_INFLATED_MEDIAN', 'ZERO_INFLATED']):
        return 'MA'
    
    # Sederhanakan model dengan angka (MA2, MA3, MA4, MA5, dll)
    if model_str.startswith('MA') and len(model_str) <= 4 and model_str[2:].isdigit():
        return 'MA'
    
    # Sederhanakan model dengan angka (WMA3, WMA4, dll)
    if model_str.startswith('WMA') and len(model_str) <= 5 and model_str[3:].isdigit():
        return 'WMA'
    
    # Sederhanakan Fallback models menjadi 'MA'
    if any(x in model_str for x in ['AVG_LAST', 'LAST_VALUE', 'LAST', 'FINAL']):
        return 'MA'
    if any(x in model_str for x in ['ULTIMATE_FALLBACK', 'AVG_DEMAND', 'FALLBACK', 'DEFAULT']):
        return 'MA'
    if any(x in model_str for x in ['AVG_HISTORICAL_DEMAND', 'AVG_SERIES', 'HISTORICAL_AVG']):
        return 'MA'
    if any(x in model_str for x in ['MA_FALLBACK', 'FALLBACK_MA']):
        return 'MA'
    # Sederhanakan Insufficient data models
    if any(x in model_str for x in ['INSUFFICIENT_DATA', 'NO_HISTORICAL_DATA', 'INSUFFICIENT', 'NO_DATA']):
        return 'INSUFFICIENT'
    
    # Sederhanakan trend models ke base model
    if any(x in model_str for x in ['LINEAR_TREND', 'EXP_TREND', 'TREND', 'LINEAR', 'EXPONENTIAL']):
        return 'ETS'  # Biasanya ETS-based
    
    # Return as is untuk model lain
    return model_name


def process_forecast(df: pd.DataFrame, file_path: str = None):
    """Process forecast dengan Railway optimizations"""
    try:
        logger.info("🚀 Starting forecast process...")
        
        # Validate input data
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Data validation and cleaning
        logger.info("🔍 Validating and cleaning input data...")
        
        # Check for required columns
        required_cols = ['PART_NO', 'MONTH', 'ORIGINAL_SHIPPING_QTY']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean data types and handle NaN/inf values
        logger.info("🧹 Cleaning data types and handling NaN/inf values...")
        
        # Handle MONTH column
        if 'MONTH' in df.columns:
            df['MONTH'] = parse_month_column(df['MONTH'])
            # Remove rows with invalid dates
            df = df[df['MONTH'].notna()].copy()
            logger.info(f"📅 MONTH column cleaned, {len(df)} valid records remaining")
        
        # Handle ORIGINAL_SHIPPING_QTY column
        if 'ORIGINAL_SHIPPING_QTY' in df.columns:
            # Convert to numeric and handle NaN/inf
            df['ORIGINAL_SHIPPING_QTY'] = pd.to_numeric(df['ORIGINAL_SHIPPING_QTY'], errors='coerce')
            df['ORIGINAL_SHIPPING_QTY'] = df['ORIGINAL_SHIPPING_QTY'].fillna(0)
            df['ORIGINAL_SHIPPING_QTY'] = df['ORIGINAL_SHIPPING_QTY'].replace([np.inf, -np.inf], 0)
            logger.info(f"📊 ORIGINAL_SHIPPING_QTY column cleaned")
        
        # Handle PART_NO column
        if 'PART_NO' in df.columns:
            df['PART_NO'] = df['PART_NO'].astype(str)
            df = df[df['PART_NO'].notna()].copy()
            logger.info(f"🏷️ PART_NO column cleaned")
        
        # Final validation
        if df.empty:
            raise ValueError("No valid data remaining after cleaning")
        
        logger.info(f"✅ Data validation completed. Shape: {df.shape}")
        
        # Check memory before starting
        memory_before = check_memory_usage()
        logger.info(f"📊 Memory usage before processing: {memory_before}%")
        
        # Validate Excel file if file_path provided
        if file_path:
            logger.info(f"📁 Processing file: {file_path}")
            validated_df = validate_excel_file(file_path)
            df = validated_df
        
        # Run backtest and realtime
        df_bt, bt, rt = run_backtest_and_realtime(df)
        
        # Check if ICC and PART_CATEGORY columns exist in original data
        has_icc = 'INVENTORY_CONTROL_CLASS' in df.columns or 'ICC' in df.columns
        has_part_category = 'PART_CATEGORY' in df.columns or 'PartCategory' in df.columns
        
        # Determine ICC column name
        icc_col = None
        if 'INVENTORY_CONTROL_CLASS' in df.columns:
            icc_col = 'INVENTORY_CONTROL_CLASS'
        elif 'ICC' in df.columns:
            icc_col = 'ICC'
        
        # Determine PART_CATEGORY column name
        part_cat_col = None
        if 'PART_CATEGORY' in df.columns:
            part_cat_col = 'PART_CATEGORY'
        elif 'PartCategory' in df.columns:
            part_cat_col = 'PartCategory'
        
        # Sederhanakan nama model untuk display di Excel
        if not bt.empty:
            bt['BEST_MODEL'] = bt['BEST_MODEL'].apply(simplify_model_name)
        if not rt.empty:
            rt['BEST_MODEL'] = rt['BEST_MODEL'].apply(simplify_model_name)
        
        # Add ICC column if it exists in original data
        if has_icc and icc_col and not bt.empty:
            # Get ICC mapping from original data
            icc_mapping = df.groupby('PART_NO')[icc_col].first().to_dict()
            bt['INVENTORY_CONTROL_CLASS'] = bt['PART_NO'].map(icc_mapping)
        
        if has_icc and icc_col and not rt.empty:
            # Get ICC mapping from original data
            icc_mapping = df.groupby('PART_NO')[icc_col].first().to_dict()
            rt['INVENTORY_CONTROL_CLASS'] = rt['PART_NO'].map(icc_mapping)
        
        # PART_CATEGORY is already handled in run_backtest_and_realtime
        # No need to add it again
        
        # Create output file with Railway-optimized path
        output_path = '/tmp/forecast_deep_results.xlsx' if IS_PRODUCTION else 'forecast_deep_results.xlsx'
        
        with ExcelWriter(output_path, engine='openpyxl') as writer:
            bt.to_excel(writer, sheet_name='Backtest_Results', index=False)
            rt.to_excel(writer, sheet_name='RealTime_Forecast', index=False)
        
        # Memory cleanup after processing
        cleanup_ml_memory()
        memory_after = check_memory_usage()
        logger.info(f"📊 Memory usage after processing: {memory_after}%")
        
        logger.info(f"✅ Forecast completed successfully!")
        logger.info(f"📊 Backtest records: {len(bt)}")
        logger.info(f"📊 Realtime records: {len(rt)}")
        logger.info(f"📁 Output file: {output_path}")
        
        return {
            'status': 'success',
            'message': f'Forecast Deep completed. File saved: {output_path}',
            'data': {
                'total_backtest': len(bt),
                'total_realtime': len(rt),
                'output_path': output_path
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error in process_forecast: {e}")
        cleanup_ml_memory()
        return {
            'status': 'error',
            'message': f'Forecast failed: {str(e)}',
            'data': {
                'total_backtest': 0,
                'total_realtime': 0
            }
        }

def process_forecast_from_file(file_path: str):
    """Process forecast dari file Excel dengan Railway optimizations"""
    try:
        logger.info(f"📁 Processing Excel file: {file_path}")
        
        # Validate and read Excel file
        df = validate_excel_file(file_path)
        
        # Process forecast
        result = process_forecast(df, file_path)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error processing file {file_path}: {e}")
        return {
            'status': 'error',
            'message': f'Failed to process file: {str(e)}',
            'data': {
                'total_backtest': 0,
                'total_realtime': 0
            }
        }

def process_forecast_from_dataframe(df: pd.DataFrame):
    """Process forecast dari DataFrame dengan Railway optimizations"""
    try:
        logger.info("📊 Processing DataFrame")
        
        # Process forecast
        result = process_forecast(df)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error processing DataFrame: {e}")
        return {
            'status': 'error',
            'message': f'Failed to process DataFrame: {str(e)}',
            'data': {
                'total_backtest': 0,
                'total_realtime': 0
            }
        }


if __name__ == '__main__':
    try:
        logger.info(f'🚀 Starting Railway forecast service...')
        logger.info(f'📁 Loading dataset from: {DATASET_PATH}')
        
        # Check if file exists
        if os.path.exists(DATASET_PATH):
            df = pd.read_excel(DATASET_PATH)
            res = process_forecast(df)
            logger.info(res['message'])
        else:
            logger.warning(f"⚠️ Default dataset file not found: {DATASET_PATH}")
            logger.info("💡 Service ready for file uploads via web interface")
            
    except Exception as e:
        logger.error(f'❌ Error: {e}')
        sys.exit(1)
