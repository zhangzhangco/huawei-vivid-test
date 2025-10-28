"""
UI错误处理器
提供友好的错误提示、警告系统和可视化反馈
"""

import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import logging
from dataclasses import dataclass
import time


class ErrorSeverity(Enum):
    """错误严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ErrorMessage:
    """错误消息结构"""
    severity: ErrorSeverity
    title: str
    message: str
    suggestion: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class UIErrorHandler:
    """UI错误处理器"""
    
    def __init__(self):
        self.error_history: List[ErrorMessage] = []
        self.max_history = 50
        self.logger = logging.getLogger(__name__)
        
        # 错误消息模板
        self.error_templates = {
            # 参数错误
            'parameter_range': "参数 {param} = {value} 超出有效范围 {range}",
            'parameter_type': "参数 {param} 类型错误，期望 {expected}，实际 {actual}",
            'parameter_invalid': "参数 {param} 无效: {reason}",
            
            # 计算错误
            'calculation_failed': "计算失败: {operation}",
            'monotonicity_violation': "曲线非单调，已自动回退到安全配置",
            'numerical_instability': "数值不稳定，建议调整参数",
            
            # 图像处理错误
            'image_load_failed': "图像加载失败: {reason}",
            'image_format_unsupported': "不支持的图像格式: {format}",
            'image_too_large': "图像过大 ({size}MP)，建议小于 {limit}MP",
            
            # 系统错误
            'memory_insufficient': "内存不足，建议降低图像分辨率",
            'gpu_acceleration_failed': "GPU加速失败，已切换到CPU模式",
            'state_save_failed': "状态保存失败: {reason}",
            'state_load_failed': "状态加载失败: {reason}",
        }
        
        # 修正建议模板
        self.suggestion_templates = {
            'parameter_range': "请将参数调整到 {range} 范围内",
            'monotonicity_violation': "尝试减小参数p或增大参数a以保持单调性",
            'image_too_large': "使用图像编辑软件将图像缩放到 {limit}MP 以下",
            'numerical_instability': "建议使用默认参数或重置参数",
            'memory_insufficient': "关闭其他应用程序或降低图像分辨率",
        }
        
    def add_error(self, severity: ErrorSeverity, title: str, message: str, 
                  suggestion: Optional[str] = None) -> ErrorMessage:
        """添加错误消息"""
        error = ErrorMessage(severity, title, message, suggestion)
        self.error_history.append(error)
        
        # 限制历史记录长度
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
            
        # 记录到日志
        log_level = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        
        self.logger.log(log_level, f"{title}: {message}")
        
        return error
        
    def create_parameter_error(self, param_name: str, value: Any, 
                             valid_range: Tuple[float, float]) -> ErrorMessage:
        """创建参数错误"""
        message = self.error_templates['parameter_range'].format(
            param=param_name, value=value, range=valid_range
        )
        suggestion = self.suggestion_templates['parameter_range'].format(
            range=valid_range
        )
        return self.add_error(ErrorSeverity.ERROR, "参数错误", message, suggestion)
        
    def create_calculation_error(self, operation: str, details: str = "") -> ErrorMessage:
        """创建计算错误"""
        message = self.error_templates['calculation_failed'].format(operation=operation)
        if details:
            message += f" - {details}"
        return self.add_error(ErrorSeverity.ERROR, "计算错误", message)
        
    def create_monotonicity_warning(self) -> ErrorMessage:
        """创建单调性警告"""
        message = self.error_templates['monotonicity_violation']
        suggestion = self.suggestion_templates['monotonicity_violation']
        return self.add_error(ErrorSeverity.WARNING, "单调性警告", message, suggestion)
        
    def create_image_error(self, error_type: str, **kwargs) -> ErrorMessage:
        """创建图像处理错误"""
        if error_type == "too_large":
            message = self.error_templates['image_too_large'].format(**kwargs)
            suggestion = self.suggestion_templates['image_too_large'].format(**kwargs)
            severity = ErrorSeverity.WARNING
        elif error_type == "load_failed":
            message = self.error_templates['image_load_failed'].format(**kwargs)
            severity = ErrorSeverity.ERROR
            suggestion = "请检查文件格式和完整性"
        elif error_type == "format_unsupported":
            message = self.error_templates['image_format_unsupported'].format(**kwargs)
            severity = ErrorSeverity.ERROR
            suggestion = "支持的格式: JPG, PNG, EXR, HDR, TIFF"
        else:
            message = f"图像处理错误: {error_type}"
            severity = ErrorSeverity.ERROR
            suggestion = None
            
        return self.add_error(severity, "图像处理错误", message, suggestion)
        
    def create_system_error(self, error_type: str, **kwargs) -> ErrorMessage:
        """创建系统错误"""
        if error_type == "memory_insufficient":
            message = self.error_templates['memory_insufficient']
            suggestion = self.suggestion_templates['memory_insufficient']
            severity = ErrorSeverity.WARNING
        elif error_type == "gpu_failed":
            message = self.error_templates['gpu_acceleration_failed']
            severity = ErrorSeverity.INFO
            suggestion = "性能可能受到影响，但功能正常"
        else:
            message = f"系统错误: {error_type}"
            severity = ErrorSeverity.ERROR
            suggestion = None
            
        return self.add_error(severity, "系统错误", message, suggestion)
        
    def show_gradio_error(self, error: ErrorMessage) -> gr.Error:
        """显示Gradio错误"""
        full_message = error.message
        if error.suggestion:
            full_message += f"\n建议: {error.suggestion}"
            
        if error.severity == ErrorSeverity.ERROR or error.severity == ErrorSeverity.CRITICAL:
            return gr.Error(full_message)
        elif error.severity == ErrorSeverity.WARNING:
            return gr.Warning(full_message)
        else:
            return gr.Info(full_message)
            
    def show_parameter_error(self, param_name: str, value: Any, 
                           valid_range: Tuple[float, float]) -> gr.Error:
        """显示参数错误"""
        error = self.create_parameter_error(param_name, value, valid_range)
        return self.show_gradio_error(error)
        
    def show_calculation_warning(self, warning_msg: str) -> gr.Warning:
        """显示计算警告"""
        error = self.add_error(ErrorSeverity.WARNING, "计算警告", warning_msg)
        return gr.Warning(warning_msg)
        
    def show_image_error(self, error_msg: str) -> gr.Error:
        """显示图像处理错误"""
        error = self.add_error(ErrorSeverity.ERROR, "图像处理失败", error_msg)
        return gr.Error(error_msg)
        
    def validate_image_upload(self, image: np.ndarray, max_pixels: int = 10_000_000) -> Tuple[bool, str]:
        """验证上传图像"""
        if image is None:
            return False, "未检测到有效图像"
            
        if image.size > max_pixels:
            size_mp = image.size / 1_000_000
            limit_mp = max_pixels / 1_000_000
            self.create_image_error("too_large", size=size_mp, limit=limit_mp)
            return False, f"图像过大 ({size_mp:.1f}MP)，请上传小于 {limit_mp:.1f}MP 的图像"
            
        if len(image.shape) not in [2, 3]:
            self.create_image_error("format_unsupported", format="未知格式")
            return False, "不支持的图像格式"
            
        if image.shape[-1] not in [1, 3, 4] if len(image.shape) == 3 else True:
            return False, "不支持的通道数"
            
        return True, ""
        
    def create_error_plot(self, error_msg: str, plot_type: str = "curve") -> plt.Figure:
        """创建错误显示图表"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 设置背景色
        fig.patch.set_facecolor('#ffebee')  # 浅红色背景
        ax.set_facecolor('#ffffff')
        
        # 错误图标和文本
        ax.text(0.5, 0.6, '⚠️', fontsize=48, ha='center', va='center', 
                transform=ax.transAxes, color='#d32f2f')
        
        ax.text(0.5, 0.4, f'错误: {error_msg}', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='#d32f2f',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.text(0.5, 0.25, '请检查参数设置或联系技术支持', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=10, color='#666666')
        
        # 设置坐标轴
        if plot_type == "curve":
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('输入亮度 (PQ域)', fontsize=10)
            ax.set_ylabel('输出亮度 (PQ域)', fontsize=10)
            ax.set_title('曲线显示错误', fontsize=12, color='#d32f2f')
            
            # 添加网格
            ax.grid(True, alpha=0.3, color='#cccccc')
            
        elif plot_type == "image":
            ax.set_title('图像处理错误', fontsize=12, color='#d32f2f')
            ax.axis('off')
            
        # 移除坐标轴刻度
        ax.tick_params(colors='#666666', labelsize=8)
        
        plt.tight_layout()
        return fig
        
    def create_warning_plot(self, warning_msg: str, plot_type: str = "curve") -> plt.Figure:
        """创建警告显示图表"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 设置背景色
        fig.patch.set_facecolor('#fff3e0')  # 浅橙色背景
        ax.set_facecolor('#ffffff')
        
        # 警告图标和文本
        ax.text(0.5, 0.6, '⚠️', fontsize=48, ha='center', va='center', 
                transform=ax.transAxes, color='#f57c00')
        
        ax.text(0.5, 0.4, f'警告: {warning_msg}', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='#f57c00',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.text(0.5, 0.25, '系统已自动回退到安全配置', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=10, color='#666666')
        
        # 设置坐标轴
        if plot_type == "curve":
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('输入亮度 (PQ域)', fontsize=10)
            ax.set_ylabel('输出亮度 (PQ域)', fontsize=10)
            ax.set_title('曲线显示警告', fontsize=12, color='#f57c00')
            
            # 添加网格
            ax.grid(True, alpha=0.3, color='#cccccc')
            
        plt.tight_layout()
        return fig
        
    def create_status_indicator(self, status: str, is_error: bool = False) -> Dict[str, Any]:
        """创建状态指示器"""
        if is_error:
            return {
                "value": f"❌ {status}",
                "color": "#d32f2f",
                "background": "#ffebee"
            }
        else:
            return {
                "value": f"✅ {status}",
                "color": "#2e7d32",
                "background": "#e8f5e8"
            }
            
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        if not self.error_history:
            return {
                "total_errors": 0,
                "recent_errors": 0,
                "error_rate": 0.0,
                "status": "正常"
            }
            
        # 统计最近5分钟的错误
        current_time = time.time()
        recent_errors = [
            error for error in self.error_history 
            if current_time - error.timestamp < 300  # 5分钟
        ]
        
        error_count = len([e for e in recent_errors if e.severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]])
        warning_count = len([e for e in recent_errors if e.severity == ErrorSeverity.WARNING])
        
        # 确定系统状态
        if error_count > 5:
            status = "严重错误"
        elif error_count > 2:
            status = "错误较多"
        elif warning_count > 3:
            status = "警告较多"
        else:
            status = "正常"
            
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "error_count": error_count,
            "warning_count": warning_count,
            "error_rate": error_count / max(1, len(recent_errors)) * 100,
            "status": status
        }
        
    def clear_error_history(self):
        """清空错误历史"""
        self.error_history.clear()
        
    def get_recent_errors(self, count: int = 10) -> List[ErrorMessage]:
        """获取最近的错误"""
        return self.error_history[-count:] if self.error_history else []
        
    def format_error_for_display(self, error: ErrorMessage) -> str:
        """格式化错误用于显示"""
        timestamp_str = time.strftime("%H:%M:%S", time.localtime(error.timestamp))
        severity_icon = {
            ErrorSeverity.INFO: "ℹ️",
            ErrorSeverity.WARNING: "⚠️",
            ErrorSeverity.ERROR: "❌",
            ErrorSeverity.CRITICAL: "🚨"
        }[error.severity]
        
        formatted = f"{severity_icon} [{timestamp_str}] {error.title}: {error.message}"
        if error.suggestion:
            formatted += f"\n   💡 建议: {error.suggestion}"
            
        return formatted