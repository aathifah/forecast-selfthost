#!/usr/bin/env python3
"""
Activity Monitor - Monitoring Detail untuk Web Application
Memonitor semua aktivitas keamanan, proses forecast, dan detail setiap part number
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import pandas as pd
import numpy as np

# PERBAIKAN: Safe import untuk psutil
try:
    import psutil
except ImportError:
    psutil = None
    print("Warning: psutil not available, monitoring will be limited")

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Setup logging untuk monitoring
monitor_logger = logging.getLogger('activity_monitor')
monitor_logger.setLevel(logging.INFO)

# File handler untuk log monitoring
log_dir = os.path.join(current_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
monitor_handler = logging.FileHandler(os.path.join(log_dir, 'activity_monitor.log'))
monitor_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
monitor_handler.setFormatter(formatter)
monitor_logger.addHandler(monitor_handler)

@dataclass
class SecurityEvent:
    """Data class untuk event keamanan"""
    timestamp: str
    event_type: str
    client_ip: str
    user_agent: str
    endpoint: str
    file_size: Optional[int] = None
    file_extension: Optional[str] = None
    rate_limit_status: Optional[str] = None
    security_check_passed: bool = True
    details: Dict[str, Any] = None

@dataclass
class ForecastProcess:
    """Data class untuk proses forecast"""
    timestamp: str
    session_id: str
    client_ip: str
    total_parts: int
    current_part: int
    part_number: str
    dataset_size: int
    processing_stage: str
    model_selection_method: str
    error_metrics: Dict[str, float]
    best_model: str
    processing_time: float
    memory_usage: float
    details: Dict[str, Any] = None

@dataclass
class WebActivity:
    """Data class untuk aktivitas web"""
    timestamp: str
    client_ip: str
    endpoint: str
    method: str
    response_time: float
    status_code: int
    user_agent: str
    file_uploaded: bool = False
    file_size: Optional[int] = None
    session_id: Optional[str] = None

class ActivityMonitor:
    """Class untuk monitoring semua aktivitas aplikasi"""
    
    def __init__(self):
        self.security_events = deque(maxlen=1000)
        self.forecast_processes = deque(maxlen=1000)
        self.web_activities = deque(maxlen=1000)
        self.active_sessions = {}
        self.rate_limit_violations = defaultdict(int)
        self.memory_usage_history = deque(maxlen=100)
        self.cpu_usage_history = deque(maxlen=100)
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_system_resources, daemon=True)
        self.monitor_thread.start()
        
        monitor_logger.info("🚀 Activity Monitor initialized")
    
    def log_security_event(self, event_type: str, client_ip: str, user_agent: str, 
                          endpoint: str, **kwargs):
        """Log event keamanan"""
        event = SecurityEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            client_ip=client_ip,
            user_agent=user_agent,
            endpoint=endpoint,
            **kwargs
        )
        
        self.security_events.append(event)
        
        # Log detail ke file
        monitor_logger.info(f"🔒 SECURITY EVENT: {event_type} | IP: {client_ip} | Endpoint: {endpoint}")
        if kwargs:
            monitor_logger.info(f"   Details: {kwargs}")
    
    def log_forecast_process(self, session_id: str, client_ip: str, total_parts: int,
                           current_part: int, part_number: str, dataset_size: int,
                           processing_stage: str, model_selection_method: str,
                           error_metrics: Dict[str, float], best_model: str,
                           processing_time: float, memory_usage: float, **kwargs):
        """Log proses forecast detail"""
        process = ForecastProcess(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            client_ip=client_ip,
            total_parts=total_parts,
            current_part=current_part,
            part_number=part_number,
            dataset_size=dataset_size,
            processing_stage=processing_stage,
            model_selection_method=model_selection_method,
            error_metrics=error_metrics,
            best_model=best_model,
            processing_time=processing_time,
            memory_usage=memory_usage,
            details=kwargs
        )
        
        self.forecast_processes.append(process)
        
        # Log detail ke file
        monitor_logger.info(f"📊 FORECAST PROCESS: Part {current_part}/{total_parts} | Part: {part_number}")
        monitor_logger.info(f"   Stage: {processing_stage} | Model: {best_model} | Method: {model_selection_method}")
        monitor_logger.info(f"   Errors: {error_metrics} | Time: {processing_time:.2f}s | Memory: {memory_usage:.2f}MB")
        if kwargs:
            monitor_logger.info(f"   Details: {kwargs}")
    
    def log_web_activity(self, client_ip: str, endpoint: str, method: str,
                        response_time: float, status_code: int, user_agent: str,
                        file_uploaded: bool = False, file_size: Optional[int] = None,
                        session_id: Optional[str] = None):
        """Log aktivitas web"""
        # Kurangi kebisingan log: skip request statik/health yang sukses
        if endpoint.startswith('/static') or endpoint in ('/health', '/api/health'):
            if status_code < 400 and not file_uploaded:
                return
        activity = WebActivity(
            timestamp=datetime.now().isoformat(),
            client_ip=client_ip,
            endpoint=endpoint,
            method=method,
            response_time=response_time,
            status_code=status_code,
            user_agent=user_agent,
            file_uploaded=file_uploaded,
            file_size=file_size,
            session_id=session_id
        )
        
        self.web_activities.append(activity)
        
        # Log detail ke file
        monitor_logger.info(f"🌐 WEB ACTIVITY: {method} {endpoint} | IP: {client_ip} | Status: {status_code}")
        if response_time >= 1.0 or status_code >= 400 or file_uploaded:
            monitor_logger.info(f"   Response Time: {response_time:.3f}s | File Upload: {file_uploaded}")
        if file_size:
            monitor_logger.info(f"   File Size: {file_size} bytes")
    
    def log_model_selection_detail(self, part_number: str, models_tested: List[str],
                                 error_results: Dict[str, float], selected_model: str,
                                 selection_reason: str):
        """Log detail pemilihan model untuk setiap part"""
        monitor_logger.info(f"🎯 MODEL SELECTION for {part_number}:")
        monitor_logger.info(f"   Models tested: {models_tested}")
        monitor_logger.info(f"   Error results: {error_results}")
        monitor_logger.info(f"   Selected: {selected_model}")
        monitor_logger.info(f"   Reason: {selection_reason}")
    
    def log_error_detail(self, part_number: str, error_type: str, error_message: str,
                        error_traceback: str = None):
        """Log detail error untuk setiap part"""
        monitor_logger.error(f"❌ ERROR for {part_number}:")
        monitor_logger.error(f"   Type: {error_type}")
        monitor_logger.error(f"   Message: {error_message}")
        if error_traceback:
            monitor_logger.error(f"   Traceback: {error_traceback}")
    
    def log_memory_usage(self, memory_mb: float, memory_percent: float):
        """Log penggunaan memory"""
        self.memory_usage_history.append({
            'timestamp': datetime.now().isoformat(),
            'memory_mb': memory_mb,
            'memory_percent': memory_percent
        })
        
        monitor_logger.info(f"💾 MEMORY USAGE: {memory_mb:.2f}MB ({memory_percent:.1f}%)")
    
    def log_cpu_usage(self, cpu_percent: float):
        """Log penggunaan CPU"""
        self.cpu_usage_history.append({
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent
        })
        
        monitor_logger.info(f"🖥️ CPU USAGE: {cpu_percent:.1f}%")
    
    def _monitor_system_resources(self):
        """Monitor system resources secara berkala dengan ML memory tracking"""
        while self.monitoring_active:
            try:
                # PERBAIKAN: Check if psutil is available
                if psutil is None:
                    # Fallback monitoring without psutil
                    self.log_memory_usage(0.0, 0.0)
                    self.log_cpu_usage(0.0)
                    time.sleep(30)
                    continue
                
                # Process memory usage (RSS) instead of whole-machine usage
                process = psutil.Process(os.getpid())
                rss_bytes = process.memory_info().rss
                total_bytes = psutil.virtual_memory().total
                rss_mb = rss_bytes / 1024 / 1024
                mem_percent = (rss_bytes / total_bytes * 100.0) if total_bytes else 0.0
                self.log_memory_usage(rss_mb, mem_percent)
                
                # Process CPU usage (percentage of a single core baseline)
                # Note: psutil returns percentage over all cores; that's fine for trend monitoring
                cpu_percent = process.cpu_percent(interval=0.2)
                self.log_cpu_usage(cpu_percent)
                
                # ML Memory tracking
                try:
                    import tensorflow as tf
                    tf_memory = tf.config.experimental.get_memory_info('GPU:0') if tf.config.list_physical_devices('GPU') else None
                    if tf_memory:
                        monitor_logger.info(f"🧠 TensorFlow Memory: {tf_memory['current'] / 1024 / 1024:.2f}MB")
                except:
                    pass
                
                time.sleep(30)  # Update setiap 30 detik untuk monitoring yang lebih sering
                
            except Exception as e:
                monitor_logger.error(f"Error in system monitoring: {e}")
                time.sleep(60)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get summary keamanan"""
        if not self.security_events:
            return {"message": "No security events recorded"}
        
        recent_events = list(self.security_events)[-100:]  # 100 events terakhir
        
        event_counts = defaultdict(int)
        ip_counts = defaultdict(int)
        
        for event in recent_events:
            event_counts[event.event_type] += 1
            ip_counts[event.client_ip] += 1
        
        return {
            "total_events": len(recent_events),
            "event_types": dict(event_counts),
            "top_ips": dict(sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "recent_events": [asdict(event) for event in recent_events[-10:]]
        }
    
    def get_forecast_summary(self) -> Dict[str, Any]:
        """Get summary forecast"""
        if not self.forecast_processes:
            return {"message": "No forecast processes recorded"}
        
        recent_processes = list(self.forecast_processes)[-100:]
        
        total_parts_processed = len(recent_processes)
        avg_processing_time = np.mean([p.processing_time for p in recent_processes])
        avg_memory_usage = np.mean([p.memory_usage for p in recent_processes])
        
        model_counts = defaultdict(int)
        method_counts = defaultdict(int)
        
        for process in recent_processes:
            model_counts[process.best_model] += 1
            method_counts[process.model_selection_method] += 1
        
        return {
            "total_parts_processed": total_parts_processed,
            "avg_processing_time": avg_processing_time,
            "avg_memory_usage": avg_memory_usage,
            "model_distribution": dict(model_counts),
            "method_distribution": dict(method_counts),
            "recent_processes": [asdict(process) for process in recent_processes[-10:]]
        }
    
    def get_web_activity_summary(self) -> Dict[str, Any]:
        """Get summary aktivitas web"""
        if not self.web_activities:
            return {"message": "No web activities recorded"}
        
        recent_activities = list(self.web_activities)[-100:]
        
        endpoint_counts = defaultdict(int)
        status_counts = defaultdict(int)
        avg_response_time = np.mean([a.response_time for a in recent_activities])
        
        for activity in recent_activities:
            endpoint_counts[activity.endpoint] += 1
            status_counts[activity.status_code] += 1
        
        return {
            "total_requests": len(recent_activities),
            "avg_response_time": avg_response_time,
            "endpoint_distribution": dict(endpoint_counts),
            "status_distribution": dict(status_counts),
            "recent_activities": [asdict(activity) for activity in recent_activities[-10:]]
        }
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary system resources"""
        if not self.memory_usage_history:
            return {"message": "No system data available"}
        
        recent_memory = list(self.memory_usage_history)[-10:]
        recent_cpu = list(self.cpu_usage_history)[-10:]
        
        avg_memory = np.mean([m['memory_mb'] for m in recent_memory])
        avg_cpu = np.mean([c['cpu_percent'] for c in recent_cpu])
        
        return {
            "avg_memory_usage_mb": avg_memory,
            "avg_cpu_usage_percent": avg_cpu,
            "recent_memory": recent_memory,
            "recent_cpu": recent_cpu
        }
    
    def get_full_summary(self) -> Dict[str, Any]:
        """Get full summary semua monitoring"""
        return {
            "timestamp": datetime.now().isoformat(),
            "security": self.get_security_summary(),
            "forecast": self.get_forecast_summary(),
            "web_activity": self.get_web_activity_summary(),
            "system": self.get_system_summary()
        }
    
    def export_logs(self, filename: str = None):
        """Export semua logs ke file JSON"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"activity_logs_{timestamp}.json"
        
        log_data = {
            "export_timestamp": datetime.now().isoformat(),
            "security_events": [asdict(event) for event in self.security_events],
            "forecast_processes": [asdict(process) for process in self.forecast_processes],
            "web_activities": [asdict(activity) for activity in self.web_activities],
            "system_history": {
                "memory": list(self.memory_usage_history),
                "cpu": list(self.cpu_usage_history)
            }
        }
        
        filepath = os.path.join(log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        monitor_logger.info(f"📁 Logs exported to: {filepath}")
        return filepath

# Global instance
activity_monitor = ActivityMonitor()

def get_monitor():
    """Get global monitor instance"""
    return activity_monitor

if __name__ == "__main__":
    # Test monitoring
    print("🧪 Testing Activity Monitor...")
    
    # Test security event
    activity_monitor.log_security_event(
        event_type="FILE_UPLOAD",
        client_ip="192.168.1.100",
        user_agent="Mozilla/5.0",
        endpoint="/forecast",
        file_size=1024000,
        file_extension=".xlsx"
    )
    
    # Test forecast process
    activity_monitor.log_forecast_process(
        session_id="test_session_123",
        client_ip="192.168.1.100",
        total_parts=10,
        current_part=3,
        part_number="PART001",
        dataset_size=1000,
        processing_stage="MODEL_SELECTION",
        model_selection_method="ERROR_BASED",
        error_metrics={"MAPE": 0.15, "SMAPE": 0.12},
        best_model="XGBRegressor",
        processing_time=2.5,
        memory_usage=150.5
    )
    
    # Test web activity
    activity_monitor.log_web_activity(
        client_ip="192.168.1.100",
        endpoint="/forecast",
        method="POST",
        response_time=1.234,
        status_code=200,
        user_agent="Mozilla/5.0",
        file_uploaded=True,
        file_size=1024000
    )
    
    # Get summary
    summary = activity_monitor.get_full_summary()
    print("📊 Monitoring Summary:")
    print(json.dumps(summary, indent=2))
    
    print("✅ Activity Monitor test completed!")
