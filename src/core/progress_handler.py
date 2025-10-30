"""
进度处理器模块
实现非阻塞UI和进度指示功能
"""

import time
import threading
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass
import gradio as gr
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import logging


@dataclass
class ProgressUpdate:
    """进度更新数据类"""
    progress: float  # 0.0 - 1.0
    description: str
    timestamp: float
    stage: str
    details: Optional[Dict[str, Any]] = None


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total_stages: int = 1):
        self.total_stages = total_stages
        self.current_stage = 0
        self.stage_progress = 0.0
        self.stage_descriptions = {}
        self.start_time = time.time()
        self.updates: List[ProgressUpdate] = []
        
    def set_stage_description(self, stage: int, description: str):
        """设置阶段描述"""
        self.stage_descriptions[stage] = description
        
    def update_stage_progress(self, progress: float, description: str = "", details: Optional[Dict] = None):
        """更新当前阶段进度"""
        self.stage_progress = max(0.0, min(1.0, progress))
        
        # 计算总体进度
        overall_progress = (self.current_stage + self.stage_progress) / self.total_stages
        
        # 获取阶段描述
        stage_desc = self.stage_descriptions.get(self.current_stage, f"阶段 {self.current_stage + 1}")
        full_description = f"{stage_desc}: {description}" if description else stage_desc
        
        update = ProgressUpdate(
            progress=overall_progress,
            description=full_description,
            timestamp=time.time(),
            stage=stage_desc,
            details=details
        )
        
        self.updates.append(update)
        return update
        
    def next_stage(self, description: str = ""):
        """进入下一阶段"""
        if self.current_stage < self.total_stages - 1:
            self.current_stage += 1
            self.stage_progress = 0.0
            
            if description:
                self.stage_descriptions[self.current_stage] = description
                
            return self.update_stage_progress(0.0, "开始")
        return None
        
    def complete(self, description: str = "完成"):
        """完成所有进度"""
        self.current_stage = self.total_stages - 1
        self.stage_progress = 1.0
        return self.update_stage_progress(1.0, description)
        
    def get_elapsed_time(self) -> float:
        """获取已用时间（秒）"""
        return time.time() - self.start_time
        
    def get_latest_update(self) -> Optional[ProgressUpdate]:
        """获取最新的进度更新"""
        return self.updates[-1] if self.updates else None


class AsyncTaskManager:
    """异步任务管理器"""
    
    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: Dict[str, Future] = {}
        self.task_results: Dict[str, Any] = {}
        self.task_errors: Dict[str, Exception] = {}
        
    def submit_task(self, task_id: str, func: Callable, *args, **kwargs) -> Future:
        """提交异步任务"""
        if task_id in self.active_tasks:
            # 取消现有任务
            self.active_tasks[task_id].cancel()
            
        future = self.executor.submit(func, *args, **kwargs)
        self.active_tasks[task_id] = future
        
        # 设置完成回调
        def task_completed(fut):
            try:
                result = fut.result()
                self.task_results[task_id] = result
            except Exception as e:
                self.task_errors[task_id] = e
                logging.error(f"任务 {task_id} 执行失败: {e}")
            finally:
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
                    
        future.add_done_callback(task_completed)
        return future
        
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """获取任务结果"""
        return self.task_results.get(task_id)
        
    def get_task_error(self, task_id: str) -> Optional[Exception]:
        """获取任务错误"""
        return self.task_errors.get(task_id)
        
    def is_task_running(self, task_id: str) -> bool:
        """检查任务是否正在运行"""
        return task_id in self.active_tasks and not self.active_tasks[task_id].done()
        
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].cancel()
        return False
        
    def cleanup_completed_tasks(self):
        """清理已完成的任务"""
        completed_tasks = [
            task_id for task_id, future in self.active_tasks.items()
            if future.done()
        ]
        
        for task_id in completed_tasks:
            del self.active_tasks[task_id]
            
    def shutdown(self):
        """关闭任务管理器"""
        self.executor.shutdown(wait=True)


def with_quality_assessment(func):
    """
    质量评估装饰器
    实现装饰器模式确保与Phoenix曲线处理的无缝集成
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)  # 原始处理
        if not args:
            return result
        
        self = args[0]
        
        # 如果处理成功且包含Lin/Lout数据，进行质量评估
        if (result.get('success', False) and 
            'lin_lout_data' in result and 
            result['lin_lout_data'] is not None):
            
            try:
                from .metrics_extension import ExtendedMetrics  # noqa: F401
                from .ui_integration import UIIntegration  # noqa: F401
                
                if not hasattr(self, 'extended_metrics') or self.extended_metrics is None:
                    from .metrics_extension import ExtendedMetrics as _Ext
                    self.extended_metrics = _Ext()
                if not hasattr(self, 'ui_integration') or self.ui_integration is None:
                    from .ui_integration import UIIntegration as _UI
                    self.ui_integration = _UI()
                
                lin_lout_data = result['lin_lout_data']
                
                # 计算质量指标（如果尚未计算）
                if 'quality_metrics' not in result or not result['quality_metrics']:
                    quality_metrics = self.extended_metrics.get_all_metrics(
                        lin_lout_data['lin'], 
                        lin_lout_data['lout']
                    )
                    result['quality_metrics'] = quality_metrics
                
                # 更新UI显示（如果需要）
                quality_metrics = result['quality_metrics']
                status = quality_metrics.get('Exposure_status', '未知')
                
                # 生成UI更新数据
                result['ui_updates'] = {
                    'quality_summary': self.ui_integration.update_quality_summary(quality_metrics, status),
                    'artist_tips': self.ui_integration.generate_artist_tips(quality_metrics, status),
                    'pq_histogram_data': {
                        'lin': lin_lout_data['lin'],
                        'lout': lin_lout_data['lout']
                    }
                }
                
            except Exception as e:
                logging.warning(f"质量评估装饰器执行失败: {e}")
                # 不影响主流程，只记录警告
        
        return result
    return wrapper


class ProgressHandler:
    """进度处理器主类"""
    
    def __init__(self, extended_metrics: Optional['ExtendedMetrics'] = None, ui_integration: Optional['UIIntegration'] = None):
        self.task_manager = AsyncTaskManager()
        self.progress_queue = queue.Queue()
        self._shutdown = False
        
        # 质量评估模块在实例级缓存，避免重复初始化
        self.extended_metrics = extended_metrics
        self.ui_integration = ui_integration
    
    def set_quality_modules(self, extended_metrics: Optional['ExtendedMetrics'], ui_integration: Optional['UIIntegration']) -> None:
        """注入质量评估相关模块，便于复用"""
        if extended_metrics is not None:
            self.extended_metrics = extended_metrics
        if ui_integration is not None:
            self.ui_integration = ui_integration
        
    @with_quality_assessment
    def process_image_with_progress(self, image: np.ndarray, 
                                  tone_curve_func: Callable,
                                  luminance_channel: str = "MaxRGB",
                                  progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """带进度指示的图像处理"""
        
        tracker = ProgressTracker(total_stages=5)
        tracker.set_stage_description(0, "图像预处理")
        tracker.set_stage_description(1, "降采样检查")
        tracker.set_stage_description(2, "色调映射")
        tracker.set_stage_description(3, "后处理")
        tracker.set_stage_description(4, "完成")
        
        try:
            # 阶段1: 图像预处理
            update = tracker.update_stage_progress(0.0, "验证图像格式")
            if progress_callback:
                progress_callback(update)
                
            if image is None or image.size == 0:
                raise ValueError("无效图像")
                
            original_shape = image.shape
            update = tracker.update_stage_progress(0.5, f"图像尺寸: {original_shape}")
            if progress_callback:
                progress_callback(update)
                
            # 导入必要模块
            from .performance_monitor import get_auto_downsampler
            from .image_processor import ImageProcessor
            
            processor = ImageProcessor()
            downsampler = get_auto_downsampler()
            
            update = tracker.update_stage_progress(1.0, "预处理完成")
            if progress_callback:
                progress_callback(update)
                
            # 阶段2: 降采样检查
            tracker.next_stage()
            update = tracker.update_stage_progress(0.0, "检查是否需要降采样")
            if progress_callback:
                progress_callback(update)
                
            should_downsample, scale, reason = downsampler.should_downsample(image.shape)
            
            if should_downsample:
                update = tracker.update_stage_progress(0.3, f"降采样: {reason}")
                if progress_callback:
                    progress_callback(update)
                    
                image = downsampler.downsample_image(image, scale)
                
                update = tracker.update_stage_progress(0.8, f"降采样完成，新尺寸: {image.shape}")
                if progress_callback:
                    progress_callback(update)
            else:
                update = tracker.update_stage_progress(0.8, "无需降采样")
                if progress_callback:
                    progress_callback(update)
                    
            update = tracker.update_stage_progress(1.0, "降采样检查完成")
            if progress_callback:
                progress_callback(update)
                
            # 阶段3: 色调映射
            tracker.next_stage()
            update = tracker.update_stage_progress(0.0, "开始色调映射")
            if progress_callback:
                progress_callback(update)
                
            # 转换到PQ域
            pq_image = processor.convert_to_pq_domain(image, "sRGB")
            update = tracker.update_stage_progress(0.3, "转换到PQ域")
            if progress_callback:
                progress_callback(update)
                
            # 应用色调映射并提取Lin/Lout数据
            mapped_image, lin_lout_data = processor.apply_tone_mapping_with_data_extraction(
                pq_image, tone_curve_func, luminance_channel)
            update = tracker.update_stage_progress(0.8, "应用色调映射")
            if progress_callback:
                progress_callback(update)
                
            update = tracker.update_stage_progress(1.0, "色调映射完成")
            if progress_callback:
                progress_callback(update)
                
            # 阶段4: 后处理
            tracker.next_stage()
            update = tracker.update_stage_progress(0.0, "开始后处理")
            if progress_callback:
                progress_callback(update)
                
            # 计算统计信息
            stats_before = processor.get_image_stats(pq_image, luminance_channel)
            stats_after = processor.get_image_stats(mapped_image, luminance_channel)
            
            update = tracker.update_stage_progress(0.3, "计算统计信息")
            if progress_callback:
                progress_callback(update)
            
            # 计算质量评估指标
            quality_metrics = {}
            try:
                from .metrics_extension import ExtendedMetrics  # noqa: F401
                from .ui_integration import UIIntegration  # noqa: F401
                
                if self.extended_metrics is None:
                    from .metrics_extension import ExtendedMetrics as _Ext
                    self.extended_metrics = _Ext()
                if self.ui_integration is None:
                    from .ui_integration import UIIntegration as _UI
                    self.ui_integration = _UI()
                
                # 计算质量指标
                quality_metrics = self.extended_metrics.get_all_metrics(
                    lin_lout_data['lin'], 
                    lin_lout_data['lout']
                )
                
                update = tracker.update_stage_progress(0.6, "计算质量指标")
                if progress_callback:
                    progress_callback(update)
                    
            except Exception as e:
                # 质量评估失败不应影响主流程
                quality_metrics = {'error': f'质量评估失败: {str(e)}'}
                
            # 转换为显示格式
            display_image = processor.convert_for_display(mapped_image)
            
            update = tracker.update_stage_progress(1.0, "后处理完成")
            if progress_callback:
                progress_callback(update)
                
            # 阶段5: 完成
            tracker.complete("处理完成")
            if progress_callback:
                progress_callback(tracker.get_latest_update())
                
            # 更新性能历史
            total_pixels = original_shape[0] * original_shape[1]
            elapsed_time = tracker.get_elapsed_time() * 1000  # 转换为毫秒
            downsampler.update_performance_history(elapsed_time, total_pixels)
            
            return {
                'success': True,
                'original_image': pq_image,
                'mapped_image': mapped_image,
                'display_image': display_image,
                'stats_before': stats_before,
                'stats_after': stats_after,
                'lin_lout_data': lin_lout_data,  # 新增：Lin/Lout数据
                'quality_metrics': quality_metrics,  # 新增：质量评估指标
                'processing_info': {
                    'original_shape': original_shape,
                    'final_shape': image.shape,
                    'downsampled': should_downsample,
                    'downsample_reason': reason if should_downsample else None,
                    'scale_factor': scale if should_downsample else 1.0,
                    'processing_time_ms': elapsed_time,
                    'luminance_channel': luminance_channel
                }
            }
            
        except Exception as e:
            error_update = ProgressUpdate(
                progress=0.0,
                description=f"处理失败: {str(e)}",
                timestamp=time.time(),
                stage="错误",
                details={'error': str(e)}
            )
            
            if progress_callback:
                progress_callback(error_update)
                
            return {
                'success': False,
                'error': str(e),
                'processing_info': {
                    'processing_time_ms': tracker.get_elapsed_time() * 1000
                }
            }
            
    def process_curve_with_progress(self, p: float, a: float, 
                                  enable_spline: bool = False,
                                  spline_params: Optional[Dict] = None,
                                  progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """带进度指示的曲线计算"""
        
        tracker = ProgressTracker(total_stages=4)
        tracker.set_stage_description(0, "参数验证")
        tracker.set_stage_description(1, "采样优化")
        tracker.set_stage_description(2, "曲线计算")
        tracker.set_stage_description(3, "验证完成")
        
        try:
            # 阶段1: 参数验证
            update = tracker.update_stage_progress(0.0, "验证参数")
            if progress_callback:
                progress_callback(update)
                
            from .phoenix_calculator import PhoenixCurveCalculator
            from .performance_monitor import get_sampling_optimizer
            from .parameter_validator import ParameterValidator
            
            validator = ParameterValidator()
            phoenix_calc = PhoenixCurveCalculator()
            sampling_optimizer = get_sampling_optimizer()
            
            # 验证参数
            is_valid, error_msg = validator.validate_phoenix_params(p, a)
            if not is_valid:
                raise ValueError(error_msg)
                
            update = tracker.update_stage_progress(1.0, "参数验证通过")
            if progress_callback:
                progress_callback(update)
                
            # 阶段2: 采样优化
            tracker.next_stage()
            update = tracker.update_stage_progress(0.0, "优化采样密度")
            if progress_callback:
                progress_callback(update)
                
            display_samples = sampling_optimizer.optimize_sampling_density("display")
            validation_samples = sampling_optimizer.optimize_sampling_density("validation")
            
            update = tracker.update_stage_progress(1.0, f"采样点数: 显示={display_samples}, 验证={validation_samples}")
            if progress_callback:
                progress_callback(update)
                
            # 阶段3: 曲线计算
            tracker.next_stage()
            update = tracker.update_stage_progress(0.0, "计算Phoenix曲线")
            if progress_callback:
                progress_callback(update)
                
            # 计算显示曲线
            L_display = np.linspace(0, 1, display_samples)
            L_out_display = phoenix_calc.compute_phoenix_curve(L_display, p, a)
            
            update = tracker.update_stage_progress(0.5, "计算验证曲线")
            if progress_callback:
                progress_callback(update)
                
            # 计算验证曲线
            L_validation = np.linspace(0, 1, validation_samples)
            L_out_validation = phoenix_calc.compute_phoenix_curve(L_validation, p, a)
            
            # 单调性验证
            is_monotonic = phoenix_calc.validate_monotonicity(L_out_validation)
            
            update = tracker.update_stage_progress(0.8, f"单调性检查: {'通过' if is_monotonic else '失败'}")
            if progress_callback:
                progress_callback(update)
                
            # 样条曲线处理（如果启用）
            final_curve = L_out_display.copy()
            spline_info = None
            
            if enable_spline and spline_params:
                try:
                    from .spline_calculator import SplineCurveCalculator
                    spline_calc = SplineCurveCalculator()
                    
                    nodes = spline_params.get('nodes', [0.2, 0.5, 0.8])
                    strength = spline_params.get('strength', 0.5)
                    
                    # 计算样条曲线
                    spline_curve = spline_calc.compute_pchip_spline(L_display, nodes)
                    final_curve = spline_calc.blend_with_phoenix(L_out_display, spline_curve, strength)
                    
                    # 验证样条曲线单调性
                    spline_monotonic = phoenix_calc.validate_monotonicity(final_curve)
                    
                    spline_info = {
                        'enabled': True,
                        'nodes': nodes,
                        'strength': strength,
                        'monotonic': spline_monotonic
                    }
                    
                    if not spline_monotonic:
                        final_curve = L_out_display.copy()  # 回退到Phoenix曲线
                        spline_info['fallback'] = True
                        
                except Exception as e:
                    spline_info = {
                        'enabled': True,
                        'error': str(e),
                        'fallback': True
                    }
                    
            update = tracker.update_stage_progress(1.0, "曲线计算完成")
            if progress_callback:
                progress_callback(update)
                
            # 阶段4: 完成
            tracker.complete("计算完成")
            if progress_callback:
                progress_callback(tracker.get_latest_update())
                
            return {
                'success': True,
                'input_luminance': L_display,
                'phoenix_curve': L_out_display,
                'final_curve': final_curve,
                'validation_curve': L_out_validation,
                'is_monotonic': is_monotonic,
                'spline_info': spline_info,
                'sampling_info': {
                    'display_samples': display_samples,
                    'validation_samples': validation_samples
                },
                'processing_info': {
                    'processing_time_ms': tracker.get_elapsed_time() * 1000,
                    'parameters': {'p': p, 'a': a}
                }
            }
            
        except Exception as e:
            error_update = ProgressUpdate(
                progress=0.0,
                description=f"计算失败: {str(e)}",
                timestamp=time.time(),
                stage="错误",
                details={'error': str(e)}
            )
            
            if progress_callback:
                progress_callback(error_update)
                
            return {
                'success': False,
                'error': str(e),
                'processing_info': {
                    'processing_time_ms': tracker.get_elapsed_time() * 1000
                }
            }
            
    def submit_async_task(self, task_id: str, task_func: Callable, *args, **kwargs) -> Future:
        """提交异步任务"""
        return self.task_manager.submit_task(task_id, task_func, *args, **kwargs)
        
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        is_running = self.task_manager.is_task_running(task_id)
        result = self.task_manager.get_task_result(task_id)
        error = self.task_manager.get_task_error(task_id)
        
        return {
            'running': is_running,
            'completed': result is not None,
            'error': error is not None,
            'result': result,
            'error_message': str(error) if error else None
        }
        
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        return self.task_manager.cancel_task(task_id)
        
    def cleanup(self):
        """清理资源"""
        self.task_manager.cleanup_completed_tasks()
        
    def shutdown(self):
        """关闭进度处理器"""
        self._shutdown = True
        self.task_manager.shutdown()


# 全局进度处理器实例
_global_progress_handler = None


def get_progress_handler() -> ProgressHandler:
    """获取全局进度处理器实例"""
    global _global_progress_handler
    if _global_progress_handler is None:
        _global_progress_handler = ProgressHandler()
    return _global_progress_handler


def create_gradio_progress_callback(progress_component: Optional[gr.Progress] = None):
    """创建Gradio进度回调函数"""
    def progress_callback(update: ProgressUpdate):
        if progress_component is not None:
            progress_component(update.progress, desc=update.description)
        else:
            # 如果没有进度组件，至少记录到日志
            logging.info(f"进度更新: {update.progress:.1%} - {update.description}")
            
    return progress_callback
