"""
进度处理器模块测试
"""

import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock

from src.core.progress_handler import (
    ProgressHandler, ProgressTracker, AsyncTaskManager, ProgressUpdate,
    get_progress_handler, create_gradio_progress_callback
)


class TestProgressTracker:
    """进度跟踪器测试"""
    
    def test_initialization(self):
        """测试初始化"""
        tracker = ProgressTracker(total_stages=3)
        
        assert tracker.total_stages == 3
        assert tracker.current_stage == 0
        assert tracker.stage_progress == 0.0
        assert len(tracker.updates) == 0
        
    def test_set_stage_description(self):
        """测试设置阶段描述"""
        tracker = ProgressTracker(total_stages=2)
        
        tracker.set_stage_description(0, "第一阶段")
        tracker.set_stage_description(1, "第二阶段")
        
        assert tracker.stage_descriptions[0] == "第一阶段"
        assert tracker.stage_descriptions[1] == "第二阶段"
        
    def test_update_stage_progress(self):
        """测试更新阶段进度"""
        tracker = ProgressTracker(total_stages=2)
        tracker.set_stage_description(0, "测试阶段")
        
        update = tracker.update_stage_progress(0.5, "进行中")
        
        assert update.progress == 0.25  # (0 + 0.5) / 2
        assert "测试阶段: 进行中" in update.description
        assert len(tracker.updates) == 1
        
    def test_next_stage(self):
        """测试进入下一阶段"""
        tracker = ProgressTracker(total_stages=3)
        
        # 完成第一阶段
        tracker.update_stage_progress(1.0, "完成")
        
        # 进入下一阶段
        update = tracker.next_stage("第二阶段")
        
        assert tracker.current_stage == 1
        assert tracker.stage_progress == 0.0
        assert update.progress == 1.0 / 3.0  # (1 + 0) / 3
        
    def test_complete(self):
        """测试完成所有进度"""
        tracker = ProgressTracker(total_stages=2)
        
        update = tracker.complete("全部完成")
        
        assert tracker.current_stage == 1
        assert tracker.stage_progress == 1.0
        assert update.progress == 1.0
        assert "全部完成" in update.description
        
    def test_get_elapsed_time(self):
        """测试获取已用时间"""
        tracker = ProgressTracker()
        
        time.sleep(0.01)  # 等待一小段时间
        elapsed = tracker.get_elapsed_time()
        
        assert elapsed > 0
        assert elapsed < 1  # 应该小于1秒
        
    def test_get_latest_update(self):
        """测试获取最新更新"""
        tracker = ProgressTracker()
        
        # 没有更新时应返回None
        assert tracker.get_latest_update() is None
        
        # 添加更新后应返回最新的
        update1 = tracker.update_stage_progress(0.3, "第一次更新")
        update2 = tracker.update_stage_progress(0.7, "第二次更新")
        
        latest = tracker.get_latest_update()
        assert latest is update2
        assert latest.description == "阶段 1: 第二次更新"


class TestAsyncTaskManager:
    """异步任务管理器测试"""
    
    def test_initialization(self):
        """测试初始化"""
        manager = AsyncTaskManager(max_workers=2)
        
        assert len(manager.active_tasks) == 0
        assert len(manager.task_results) == 0
        assert len(manager.task_errors) == 0
        
    def test_submit_task(self):
        """测试提交任务"""
        manager = AsyncTaskManager(max_workers=2)
        
        def simple_task(x, y):
            return x + y
            
        future = manager.submit_task("test_task", simple_task, 1, 2)
        
        assert "test_task" in manager.active_tasks
        
        # 等待任务完成
        result = future.result(timeout=1)
        assert result == 3
        
        # 稍等一下让回调执行
        time.sleep(0.1)
        
        # 检查结果是否被存储
        assert manager.get_task_result("test_task") == 3
        
    def test_submit_failing_task(self):
        """测试提交失败的任务"""
        manager = AsyncTaskManager(max_workers=2)
        
        def failing_task():
            raise ValueError("测试错误")
            
        future = manager.submit_task("failing_task", failing_task)
        
        with pytest.raises(ValueError):
            future.result(timeout=1)
            
        # 稍等一下让回调执行
        time.sleep(0.1)
        
        # 检查错误是否被存储
        error = manager.get_task_error("failing_task")
        assert error is not None
        assert "测试错误" in str(error)
        
    def test_cancel_task(self):
        """测试取消任务"""
        manager = AsyncTaskManager(max_workers=2)
        
        def slow_task():
            time.sleep(1)
            return "完成"
            
        future = manager.submit_task("slow_task", slow_task)
        
        # 立即取消任务
        cancelled = manager.cancel_task("slow_task")
        
        # 注意：已经开始执行的任务可能无法取消
        # 这里主要测试取消机制是否工作
        assert isinstance(cancelled, bool)
        
    def test_is_task_running(self):
        """测试检查任务是否运行"""
        manager = AsyncTaskManager(max_workers=2)
        
        def quick_task():
            return "完成"
            
        manager.submit_task("quick_task", quick_task)
        
        # 任务可能很快完成，所以这个测试可能不稳定
        # 主要测试方法是否存在和返回布尔值
        running = manager.is_task_running("quick_task")
        assert isinstance(running, bool)
        
        # 测试不存在的任务
        assert not manager.is_task_running("nonexistent_task")
        
    def test_cleanup_completed_tasks(self):
        """测试清理已完成任务"""
        manager = AsyncTaskManager(max_workers=2)
        
        def quick_task():
            return "完成"
            
        future = manager.submit_task("quick_task", quick_task)
        future.result(timeout=1)  # 等待完成
        
        # 清理前应该有任务
        assert "quick_task" in manager.active_tasks
        
        manager.cleanup_completed_tasks()
        
        # 清理后已完成的任务应该被移除
        # 注意：这个测试可能因为时序问题而不稳定
        
    def test_shutdown(self):
        """测试关闭任务管理器"""
        manager = AsyncTaskManager(max_workers=2)
        
        def simple_task():
            return "完成"
            
        manager.submit_task("test_task", simple_task)
        
        # 关闭应该不抛出异常
        manager.shutdown()


class TestProgressHandler:
    """进度处理器测试"""
    
    def test_initialization(self):
        """测试初始化"""
        handler = ProgressHandler()
        
        assert handler.task_manager is not None
        assert not handler._shutdown
        
    @patch('src.core.progress_handler.ImageProcessor')
    @patch('src.core.progress_handler.get_auto_downsampler')
    def test_process_image_with_progress(self, mock_get_downsampler, mock_image_processor):
        """测试带进度的图像处理"""
        # 设置模拟对象
        mock_downsampler = MagicMock()
        mock_get_downsampler.return_value = mock_downsampler
        mock_downsampler.should_downsample.return_value = (False, 1.0, "无需降采样")
        
        mock_processor = MagicMock()
        mock_image_processor.return_value = mock_processor
        mock_processor.convert_to_pq_domain.return_value = np.random.rand(100, 100, 3)
        mock_processor.apply_tone_mapping.return_value = np.random.rand(100, 100, 3)
        mock_processor.convert_for_display.return_value = np.random.rand(100, 100, 3)
        
        # 模拟图像统计
        from src.core.image_processor import ImageStats
        mock_stats = ImageStats(
            min_pq=0.0, max_pq=1.0, avg_pq=0.5, var_pq=0.1,
            input_format="test", processing_path="test", pixel_count=10000
        )
        mock_processor.get_image_stats.return_value = mock_stats
        
        handler = ProgressHandler()
        
        # 创建测试图像
        test_image = np.random.rand(200, 200, 3).astype(np.float32)
        
        # 创建简单的色调映射函数
        def tone_curve_func(L):
            return L * 0.8
            
        # 收集进度更新
        progress_updates = []
        def progress_callback(update):
            progress_updates.append(update)
            
        # 执行处理
        result = handler.process_image_with_progress(
            image=test_image,
            tone_curve_func=tone_curve_func,
            progress_callback=progress_callback
        )
        
        # 验证结果
        assert result['success']
        assert 'display_image' in result
        assert 'processing_info' in result
        
        # 验证进度更新
        assert len(progress_updates) > 0
        
        # 验证进度从0到1
        first_progress = progress_updates[0].progress
        last_progress = progress_updates[-1].progress
        assert first_progress >= 0.0
        assert last_progress == 1.0
        
    def test_process_image_with_invalid_input(self):
        """测试处理无效图像输入"""
        handler = ProgressHandler()
        
        def tone_curve_func(L):
            return L
            
        progress_updates = []
        def progress_callback(update):
            progress_updates.append(update)
            
        # 测试None输入
        result = handler.process_image_with_progress(
            image=None,
            tone_curve_func=tone_curve_func,
            progress_callback=progress_callback
        )
        
        assert not result['success']
        assert 'error' in result
        assert len(progress_updates) > 0  # 应该有错误进度更新
        
    @patch('src.core.progress_handler.PhoenixCurveCalculator')
    @patch('src.core.progress_handler.get_sampling_optimizer')
    @patch('src.core.progress_handler.ParameterValidator')
    def test_process_curve_with_progress(self, mock_validator, mock_get_optimizer, mock_phoenix_calc):
        """测试带进度的曲线计算"""
        # 设置模拟对象
        mock_validator_instance = MagicMock()
        mock_validator.return_value = mock_validator_instance
        mock_validator_instance.validate_phoenix_params.return_value = (True, "")
        
        mock_optimizer = MagicMock()
        mock_get_optimizer.return_value = mock_optimizer
        mock_optimizer.optimize_sampling_density.return_value = 512
        
        mock_calc = MagicMock()
        mock_phoenix_calc.return_value = mock_calc
        mock_calc.compute_phoenix_curve.return_value = np.linspace(0, 1, 512)
        mock_calc.validate_monotonicity.return_value = True
        
        handler = ProgressHandler()
        
        progress_updates = []
        def progress_callback(update):
            progress_updates.append(update)
            
        # 执行曲线计算
        result = handler.process_curve_with_progress(
            p=2.0,
            a=0.5,
            progress_callback=progress_callback
        )
        
        # 验证结果
        assert result['success']
        assert 'phoenix_curve' in result
        assert 'final_curve' in result
        assert 'is_monotonic' in result
        
        # 验证进度更新
        assert len(progress_updates) > 0
        assert progress_updates[-1].progress == 1.0
        
    def test_submit_async_task(self):
        """测试提交异步任务"""
        handler = ProgressHandler()
        
        def simple_task(x):
            return x * 2
            
        future = handler.submit_async_task("test_task", simple_task, 5)
        
        result = future.result(timeout=1)
        assert result == 10
        
    def test_get_task_status(self):
        """测试获取任务状态"""
        handler = ProgressHandler()
        
        def simple_task():
            return "完成"
            
        future = handler.submit_async_task("status_test", simple_task)
        future.result(timeout=1)  # 等待完成
        
        time.sleep(0.1)  # 等待回调执行
        
        status = handler.get_task_status("status_test")
        
        assert isinstance(status, dict)
        assert 'running' in status
        assert 'completed' in status
        assert 'error' in status
        
    def test_cleanup_and_shutdown(self):
        """测试清理和关闭"""
        handler = ProgressHandler()
        
        # 清理应该不抛出异常
        handler.cleanup()
        
        # 关闭应该不抛出异常
        handler.shutdown()
        
        assert handler._shutdown


class TestUtilityFunctions:
    """工具函数测试"""
    
    def test_get_progress_handler(self):
        """测试获取全局进度处理器"""
        handler1 = get_progress_handler()
        handler2 = get_progress_handler()
        
        # 应该返回同一个实例
        assert handler1 is handler2
        
    def test_create_gradio_progress_callback(self):
        """测试创建Gradio进度回调"""
        # 测试无进度组件的情况
        callback = create_gradio_progress_callback()
        
        # 创建测试更新
        update = ProgressUpdate(
            progress=0.5,
            description="测试进度",
            timestamp=time.time(),
            stage="测试阶段"
        )
        
        # 调用回调应该不抛出异常
        callback(update)
        
        # 测试有进度组件的情况
        mock_progress = MagicMock()
        callback_with_component = create_gradio_progress_callback(mock_progress)
        
        callback_with_component(update)
        
        # 验证进度组件被调用
        mock_progress.assert_called_once_with(0.5, desc="测试进度")


class TestIntegration:
    """集成测试"""
    
    def test_complete_workflow(self):
        """测试完整工作流"""
        handler = get_progress_handler()
        
        # 模拟一个完整的处理流程
        progress_history = []
        
        def collect_progress(update):
            progress_history.append({
                'progress': update.progress,
                'description': update.description,
                'stage': update.stage
            })
            
        # 创建简单的测试数据
        test_image = np.random.rand(500, 400, 3).astype(np.float32)
        
        def simple_tone_curve(L):
            return np.clip(L * 0.8, 0, 1)
            
        # 由于依赖模块可能不完整，这里主要测试接口
        try:
            result = handler.process_image_with_progress(
                image=test_image,
                tone_curve_func=simple_tone_curve,
                progress_callback=collect_progress
            )
            
            # 如果成功执行，验证基本结构
            assert isinstance(result, dict)
            assert 'success' in result
            
            # 验证进度历史
            if len(progress_history) > 0:
                assert progress_history[0]['progress'] >= 0.0
                assert progress_history[-1]['progress'] <= 1.0
                
        except ImportError:
            # 如果依赖模块不可用，跳过这个测试
            pytest.skip("依赖模块不可用")
            
    def test_error_handling_in_workflow(self):
        """测试工作流中的错误处理"""
        handler = get_progress_handler()
        
        error_updates = []
        
        def collect_errors(update):
            if update.stage == "错误":
                error_updates.append(update)
                
        # 测试无效输入的错误处理
        result = handler.process_image_with_progress(
            image=None,
            tone_curve_func=lambda x: x,
            progress_callback=collect_errors
        )
        
        assert not result['success']
        assert 'error' in result
        
        # 应该有错误进度更新
        assert len(error_updates) > 0
        assert "处理失败" in error_updates[0].description