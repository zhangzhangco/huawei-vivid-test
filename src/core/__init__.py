"""
核心数学计算模块
包含Phoenix曲线计算、PQ转换、参数验证、质量指标计算等核心功能
"""

from .pq_converter import PQConverter
from .phoenix_calculator import PhoenixCurveCalculator
from .parameter_validator import ParameterValidator
from .safe_calculator import SafeCalculator
from .spline_calculator import SplineCurveCalculator, SplineCalculationError, SplineVisualizationHelper
from .quality_metrics import QualityMetricsCalculator, ImageQualityAnalyzer
from .temporal_smoothing import TemporalSmoothingProcessor, TemporalStats, TemporalState
from .image_processor import ImageProcessor, ImageStats, ImageProcessingError
from .auto_mode_estimator import AutoModeParameterEstimator, AutoModeInterface, AutoModeConfig, EstimationResult
from .state_manager import StateManager, SessionState, TemporalStateData, get_state_manager, reset_state_manager
from .export_manager import (
    ExportManager, LUTExporter, CSVExporter, DiagnosticPackageGenerator,
    CurveData, QualityMetrics, ExportMetadata, get_export_manager, reset_export_manager
)
from .ui_error_handler import UIErrorHandler, ErrorSeverity, ErrorMessage
from .error_recovery import ErrorRecoverySystem, RecoveryStrategy, SystemState, RecoveryAction
from .boundary_checker import BoundaryChecker, BoundaryViolation, BoundaryViolationType
from .performance_monitor import (
    PerformanceMonitor, AccelerationDetector, AutoDownsampler, SamplingDensityOptimizer,
    PerformanceMetrics, AccelerationStatus, get_performance_monitor, get_auto_downsampler, get_sampling_optimizer
)
from .progress_handler import (
    ProgressHandler, ProgressTracker, AsyncTaskManager, ProgressUpdate,
    get_progress_handler, create_gradio_progress_callback
)
from .metrics_extension import ExtendedMetrics
from .config_manager import ConfigManager
from .ui_integration import UIIntegration

__all__ = [
    'PQConverter',
    'PhoenixCurveCalculator', 
    'ParameterValidator',
    'SafeCalculator',
    'SplineCurveCalculator',
    'SplineCalculationError',
    'SplineVisualizationHelper',
    'QualityMetricsCalculator',
    'ImageQualityAnalyzer',
    'TemporalSmoothingProcessor',
    'TemporalStats',
    'TemporalState',
    'ImageProcessor',
    'ImageStats',
    'ImageProcessingError',
    'AutoModeParameterEstimator',
    'AutoModeInterface',
    'AutoModeConfig',
    'EstimationResult',
    'StateManager',
    'SessionState',
    'TemporalStateData',
    'get_state_manager',
    'reset_state_manager',
    'ExportManager',
    'LUTExporter',
    'CSVExporter',
    'DiagnosticPackageGenerator',
    'CurveData',
    'QualityMetrics',
    'ExportMetadata',
    'get_export_manager',
    'reset_export_manager',
    'UIErrorHandler',
    'ErrorSeverity',
    'ErrorMessage',
    'ErrorRecoverySystem',
    'RecoveryStrategy',
    'SystemState',
    'RecoveryAction',
    'BoundaryChecker',
    'BoundaryViolation',
    'BoundaryViolationType',
    'PerformanceMonitor',
    'AccelerationDetector',
    'AutoDownsampler',
    'SamplingDensityOptimizer',
    'PerformanceMetrics',
    'AccelerationStatus',
    'get_performance_monitor',
    'get_auto_downsampler',
    'get_sampling_optimizer',
    'ProgressHandler',
    'ProgressTracker',
    'AsyncTaskManager',
    'ProgressUpdate',
    'get_progress_handler',
    'create_gradio_progress_callback',
    'ExtendedMetrics',
    'ConfigManager',
    'UIIntegration'
]