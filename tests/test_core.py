"""
核心模块测试
测试Phoenix曲线计算、PQ转换、参数验证、质量指标计算等核心功能
"""

import pytest
import numpy as np
import os
import tempfile
import json
from src.core import (PQConverter, PhoenixCurveCalculator, ParameterValidator, 
                      SafeCalculator, QualityMetricsCalculator, ImageQualityAnalyzer,
                      TemporalSmoothingProcessor, TemporalStats, TemporalState)


class TestPQConverter:
    """PQ转换器测试"""
    
    def setup_method(self):
        self.converter = PQConverter()
        
    def test_linear_to_pq_basic(self):
        """测试基本线性到PQ转换"""
        # 测试边界值 (允许数值精度误差)
        assert abs(self.converter.linear_to_pq(0.0)) < 1e-6
        assert abs(self.converter.linear_to_pq(10000.0) - 1.0) < 1e-6
        
        # 测试中间值
        result = self.converter.linear_to_pq(1000.0)
        assert 0.0 < result < 1.0
        
    def test_pq_to_linear_basic(self):
        """测试基本PQ到线性转换"""
        # 测试边界值
        assert self.converter.pq_to_linear(0.0) == 0.0
        assert abs(self.converter.pq_to_linear(1.0) - 10000.0) < 1e-6
        
    def test_roundtrip_conversion(self):
        """测试往返转换精度"""
        test_values = [0.0, 100.0, 1000.0, 5000.0, 10000.0]
        for value in test_values:
            pq = self.converter.linear_to_pq(value)
            recovered = self.converter.pq_to_linear(pq)
            assert abs(recovered - value) < 1e-3
            
    def test_srgb_conversion(self):
        """测试sRGB转换"""
        # 测试边界值
        assert self.converter.srgb_to_linear(0.0) == 0.0
        assert self.converter.srgb_to_linear(1.0) == 1.0
        
        # 测试往返转换
        test_values = [0.0, 0.2, 0.5, 0.8, 1.0]
        for value in test_values:
            linear = self.converter.srgb_to_linear(value)
            recovered = self.converter.linear_to_srgb(linear)
            assert abs(recovered - value) < 1e-6


class TestPhoenixCurveCalculator:
    """Phoenix曲线计算器测试"""
    
    def setup_method(self):
        self.calculator = PhoenixCurveCalculator()
        
    def test_basic_calculation(self):
        """测试基本Phoenix曲线计算"""
        L = np.array([0.0, 0.5, 1.0])
        p, a = 2.0, 0.5
        
        result = self.calculator.compute_phoenix_curve(L, p, a)
        
        # 检查结果形状和范围
        assert result.shape == L.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        
    def test_monotonicity(self):
        """测试单调性"""
        L = np.linspace(0, 1, 100)
        p, a = 2.0, 0.5
        
        result = self.calculator.compute_phoenix_curve(L, p, a)
        
        # 检查单调性
        assert self.calculator.validate_monotonicity(result)
        
    def test_parameter_validation(self):
        """测试参数验证"""
        L = np.array([0.5])
        
        # 测试有效参数
        result = self.calculator.compute_phoenix_curve(L, 2.0, 0.5)
        assert len(result) == 1
        
        # 测试无效参数
        with pytest.raises(ValueError):
            self.calculator.compute_phoenix_curve(L, -1.0, 0.5)  # p < 0.1
        with pytest.raises(ValueError):
            self.calculator.compute_phoenix_curve(L, 2.0, -0.1)  # a < 0.0
            
    def test_endpoint_normalization(self):
        """测试端点归一化"""
        L = np.linspace(0, 1, 10)
        p, a = 2.0, 0.5
        
        L_out = self.calculator.compute_phoenix_curve(L, p, a)
        normalized = self.calculator.normalize_endpoints(L_out, 0.1, 0.9)
        
        # 检查端点
        assert abs(normalized[0] - 0.1) < 1e-6
        assert abs(normalized[-1] - 0.9) < 1e-6
        
    def test_display_curve(self):
        """测试显示曲线获取"""
        L, L_out = self.calculator.get_display_curve(2.0, 0.5)
        
        assert len(L) == self.calculator.display_samples
        assert len(L_out) == self.calculator.display_samples
        assert L[0] == 0.0
        assert L[-1] == 1.0


class TestParameterValidator:
    """参数验证器测试"""
    
    def test_phoenix_params_validation(self):
        """测试Phoenix参数验证"""
        # 有效参数
        valid, msg = ParameterValidator.validate_phoenix_params(2.0, 0.5)
        assert valid
        assert msg == ""
        
        # 无效参数
        valid, msg = ParameterValidator.validate_phoenix_params(0.05, 0.5)  # p太小
        assert not valid
        assert "p=" in msg
        
        valid, msg = ParameterValidator.validate_phoenix_params(2.0, 1.5)  # a太大
        assert not valid
        assert "a=" in msg
        
    def test_spline_nodes_validation(self):
        """测试样条节点验证"""
        # 有效节点
        nodes, warning = ParameterValidator.validate_spline_nodes([0.2, 0.5, 0.8])
        assert nodes == [0.2, 0.5, 0.8]
        assert warning == ""
        
        # 需要排序的节点
        nodes, warning = ParameterValidator.validate_spline_nodes([0.8, 0.2, 0.5])
        assert nodes == [0.2, 0.5, 0.8]
        
        # 间隔太小的节点
        nodes, warning = ParameterValidator.validate_spline_nodes([0.2, 0.201, 0.8])
        assert nodes[1] - nodes[0] >= 0.01
        assert "调整" in warning
        
    def test_parameter_sanitization(self):
        """测试参数清理"""
        # 测试缺失参数填充
        params = {'p': 3.0}
        sanitized = ParameterValidator.sanitize_parameters(params)
        
        assert 'a' in sanitized
        assert sanitized['a'] == ParameterValidator.DEFAULT_PARAMS['a']
        
        # 测试参数范围修正
        params = {'p': 10.0, 'a': -0.5}  # 超出范围
        sanitized = ParameterValidator.sanitize_parameters(params)
        
        assert ParameterValidator.PHOENIX_P_RANGE[0] <= sanitized['p'] <= ParameterValidator.PHOENIX_P_RANGE[1]
        assert ParameterValidator.PHOENIX_A_RANGE[0] <= sanitized['a'] <= ParameterValidator.PHOENIX_A_RANGE[1]


class TestSafeCalculator:
    """安全计算器测试"""
    
    def setup_method(self):
        self.calculator = SafeCalculator()
        
    def test_safe_phoenix_calculation(self):
        """测试安全Phoenix计算"""
        L = np.linspace(0, 1, 10)
        
        # 有效参数
        result, success, msg = self.calculator.safe_phoenix_calculation(L, 2.0, 0.5)
        assert success
        assert "成功" in msg
        assert len(result) == len(L)
        
        # 无效参数
        result, success, msg = self.calculator.safe_phoenix_calculation(L, -1.0, 0.5)
        assert not success
        assert np.array_equal(result, L)  # 应该返回恒等映射
        
    def test_safe_parameter_validation(self):
        """测试安全参数验证"""
        # 有效参数
        is_monotonic, msg = self.calculator.safe_phoenix_validation(2.0, 0.5)
        assert is_monotonic
        assert "通过" in msg
        
    def test_safe_display_curve(self):
        """测试安全显示曲线获取"""
        L, L_out, success, msg = self.calculator.get_safe_display_curve(2.0, 0.5)
        
        assert success
        assert len(L) == len(L_out)
        assert L[0] == 0.0
        assert L[-1] == 1.0
        
    def test_system_status(self):
        """测试系统状态"""
        status = self.calculator.get_system_status()
        
        assert 'error_count' in status
        assert 'system_stable' in status
        assert status['phoenix_calculator_ready']
        assert status['validator_ready']


class TestQualityMetricsCalculator:
    """质量指标计算器测试"""
    
    def setup_method(self):
        self.calculator = QualityMetricsCalculator()
        
    def test_luminance_extraction_maxrgb(self):
        """测试MaxRGB亮度通道提取"""
        # 测试RGB图像
        image = np.array([[[0.2, 0.5, 0.3], [0.8, 0.1, 0.9]]])  # (1, 2, 3)
        
        self.calculator.set_luminance_channel("MaxRGB")
        luminance = self.calculator.extract_luminance(image)
        
        expected = np.array([[0.5, 0.9]])  # max of each pixel
        np.testing.assert_array_almost_equal(luminance, expected)
        
    def test_luminance_extraction_y_channel(self):
        """测试Y通道亮度提取"""
        # 测试RGB图像
        image = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        
        self.calculator.set_luminance_channel("Y")
        luminance = self.calculator.extract_luminance(image)
        
        # BT.2100权重: [0.2627, 0.6780, 0.0593]
        expected = np.array([[0.2627, 0.6780, 0.0593]])
        np.testing.assert_array_almost_equal(luminance, expected, decimal=4)
        
    def test_luminance_extraction_grayscale(self):
        """测试灰度图像亮度提取"""
        image = np.array([[0.2, 0.5], [0.8, 0.1]])
        
        luminance = self.calculator.extract_luminance(image)
        np.testing.assert_array_equal(luminance, image)
        
    def test_perceptual_distortion_calculation(self):
        """测试感知失真计算"""
        L_in = np.array([0.2, 0.4, 0.6, 0.8])
        L_out = np.array([0.3, 0.5, 0.7, 0.9])
        
        distortion = self.calculator.compute_perceptual_distortion(L_in, L_out)
        
        # D' = |mean(L_out) - mean(L_in)| = |0.6 - 0.5| = 0.1
        expected = abs(np.mean(L_out) - np.mean(L_in))
        assert abs(distortion - expected) < 1e-10
        
    def test_local_contrast_1d(self):
        """测试1D局部对比度计算"""
        L = np.array([0.1, 0.3, 0.2, 0.8, 0.5])
        
        contrast = self.calculator.compute_local_contrast(L)
        
        # 相邻差分: [0.2, 0.1, 0.6, 0.3], 平均值 = 0.3
        expected = np.mean([0.2, 0.1, 0.6, 0.3])
        assert abs(contrast - expected) < 1e-10
        
    def test_local_contrast_2d(self):
        """测试2D局部对比度计算"""
        L = np.array([[0.1, 0.3], [0.2, 0.8]])
        
        contrast = self.calculator.compute_local_contrast(L)
        
        # 应该计算x和y方向的梯度平均值
        assert contrast > 0
        assert isinstance(contrast, float)
        
    def test_variance_distortion(self):
        """测试方差失真计算"""
        L_in = np.array([0.2, 0.4, 0.6, 0.8])
        L_out = np.array([0.1, 0.3, 0.5, 0.7])  # 方差更小
        
        var_distortion = self.calculator.compute_variance_distortion(L_in, L_out)
        
        var_in = np.var(L_in)
        var_out = np.var(L_out)
        expected = abs(var_out - var_in) / (var_in + self.calculator.eps)
        
        assert abs(var_distortion - expected) < 1e-10
        
    def test_mode_recommendation_hysteresis(self):
        """测试带滞回的模式推荐"""
        # 重置滞回状态
        self.calculator.reset_hysteresis()
        
        # 低失真 -> 自动模式
        mode = self.calculator.recommend_mode_with_hysteresis(0.03)
        assert mode == "自动模式"
        
        # 中等失真 -> 保持自动模式
        mode = self.calculator.recommend_mode_with_hysteresis(0.07)
        assert mode == "自动模式"
        
        # 高失真 -> 艺术模式
        mode = self.calculator.recommend_mode_with_hysteresis(0.12)
        assert mode == "艺术模式"
        
        # 中等失真 -> 保持艺术模式
        mode = self.calculator.recommend_mode_with_hysteresis(0.08)
        assert mode == "艺术模式"
        
        # 低失真 -> 切换到自动模式
        mode = self.calculator.recommend_mode_with_hysteresis(0.04)
        assert mode == "自动模式"
        
    def test_histogram_computation(self):
        """测试PQ域直方图计算"""
        L = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.25, 0.5])
        
        hist, bin_edges = self.calculator.compute_histogram(L, bins=4)
        
        assert len(hist) == 4
        assert len(bin_edges) == 5
        assert bin_edges[0] == 0.0
        assert bin_edges[-1] == 1.0
        assert np.sum(hist) == len(L)
        
    def test_histogram_stats(self):
        """测试直方图统计信息"""
        L = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        stats = self.calculator.compute_histogram_stats(L)
        
        assert 'min_pq' in stats
        assert 'max_pq' in stats
        assert 'avg_pq' in stats
        assert 'var_pq' in stats
        assert stats['min_pq'] == 0.1
        assert stats['max_pq'] == 0.5
        assert stats['avg_pq'] == 0.3
        
    def test_all_metrics_computation(self):
        """测试所有指标计算"""
        L_in = np.array([0.2, 0.4, 0.6, 0.8])
        L_out = np.array([0.3, 0.5, 0.7, 0.9])
        
        metrics = self.calculator.compute_all_metrics(L_in, L_out)
        
        # 检查必要的指标
        required_keys = [
            'perceptual_distortion', 'local_contrast_in', 'local_contrast_out',
            'variance_distortion', 'recommended_mode', 'luminance_channel'
        ]
        
        for key in required_keys:
            assert key in metrics
            
        assert metrics['luminance_channel'] == "MaxRGB"
        
    def test_hysteresis_threshold_setting(self):
        """测试滞回阈值设置"""
        # 有效阈值
        self.calculator.set_hysteresis_thresholds(0.03, 0.08)
        assert self.calculator.dt_low == 0.03
        assert self.calculator.dt_high == 0.08
        
        # 无效阈值
        with pytest.raises(ValueError):
            self.calculator.set_hysteresis_thresholds(0.08, 0.03)  # 下阈值 >= 上阈值
            
        with pytest.raises(ValueError):
            self.calculator.set_hysteresis_thresholds(-0.1, 0.5)  # 超出范围
            
    def test_input_validation(self):
        """测试输入验证"""
        L_in = np.array([0.2, 0.4, 0.6])
        L_out = np.array([0.3, 0.5, 0.7])
        
        # 有效输入
        valid, msg = self.calculator.validate_inputs(L_in, L_out)
        assert valid
        assert msg == ""
        
        # 形状不匹配
        L_out_wrong = np.array([0.3, 0.5])
        valid, msg = self.calculator.validate_inputs(L_in, L_out_wrong)
        assert not valid
        assert "形状不匹配" in msg
        
        # 包含NaN
        L_out_nan = np.array([0.3, np.nan, 0.7])
        valid, msg = self.calculator.validate_inputs(L_in, L_out_nan)
        assert not valid
        assert "非有限值" in msg
        
    def test_luminance_channel_switching(self):
        """测试亮度通道切换"""
        image = np.array([[[0.8, 0.2, 0.1]]])  # RGB
        
        # MaxRGB
        self.calculator.set_luminance_channel("MaxRGB")
        lum_maxrgb = self.calculator.extract_luminance(image)
        assert lum_maxrgb[0, 0] == 0.8
        
        # Y通道
        self.calculator.set_luminance_channel("Y")
        lum_y = self.calculator.extract_luminance(image)
        expected_y = 0.8 * 0.2627 + 0.2 * 0.6780 + 0.1 * 0.0593
        assert abs(lum_y[0, 0] - expected_y) < 1e-6
        
        # 无效通道
        with pytest.raises(ValueError):
            self.calculator.set_luminance_channel("Invalid")


class TestImageQualityAnalyzer:
    """图像质量分析器测试"""
    
    def setup_method(self):
        self.analyzer = ImageQualityAnalyzer()
        
    def test_image_quality_analysis(self):
        """测试图像质量分析"""
        # 创建测试图像
        image_in = np.random.rand(10, 10, 3) * 0.8  # 输入图像
        image_out = image_in * 1.2  # 稍微增亮的输出图像
        image_out = np.clip(image_out, 0, 1)
        
        result = self.analyzer.analyze_image_quality(image_in, image_out)
        
        # 检查结果包含必要的指标
        assert 'perceptual_distortion' in result
        assert 'recommended_mode' in result
        assert 'histogram_in' in result
        assert 'histogram_out' in result
        assert 'error' not in result  # 没有错误
        
    def test_compare_tone_mapping_results(self):
        """测试色调映射结果比较"""
        original = np.random.rand(5, 5, 3)
        
        # 创建两个不同的映射结果
        result1 = original * 0.8
        result2 = original * 1.2
        result2 = np.clip(result2, 0, 1)
        
        results = [
            ("方法1", result1),
            ("方法2", result2)
        ]
        
        comparison = self.analyzer.compare_tone_mapping_results(original, results)
        
        assert "方法1" in comparison
        assert "方法2" in comparison
        assert 'perceptual_distortion' in comparison["方法1"]
        assert 'perceptual_distortion' in comparison["方法2"]


class TestTemporalSmoothingProcessor:
    """时域平滑处理器测试"""
    
    def setup_method(self):
        # 使用临时文件避免测试间干扰
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
        self.processor = TemporalSmoothingProcessor(window_size=5, temporal_file=self.temp_file)
        
    def teardown_method(self):
        # 清理临时文件
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
            
    def test_initialization(self):
        """测试初始化"""
        # 测试默认参数
        processor = TemporalSmoothingProcessor()
        assert processor.window_size == 9
        assert processor.eps == 1e-8
        assert len(processor.parameter_history) == 0
        assert len(processor.distortion_history) == 0
        
        # 测试自定义参数
        assert self.processor.window_size == 5
        
        # 测试无效窗口大小
        with pytest.raises(ValueError):
            TemporalSmoothingProcessor(window_size=3)  # < 5
        with pytest.raises(ValueError):
            TemporalSmoothingProcessor(window_size=20)  # > 15
            
    def test_cold_start(self):
        """测试冷启动机制 (需求 11.3)"""
        # 添加一些数据
        self.processor.add_frame_parameters({'p': 2.0, 'a': 0.5}, 0.1)
        self.processor.add_frame_parameters({'p': 2.1, 'a': 0.6}, 0.2)
        
        assert len(self.processor.parameter_history) == 2
        assert self.processor.frame_count == 2
        
        # 执行冷启动
        self.processor.cold_start()
        
        assert len(self.processor.parameter_history) == 0
        assert len(self.processor.distortion_history) == 0
        assert self.processor.frame_count == 0
        
    def test_add_frame_parameters(self):
        """测试添加帧参数"""
        # 添加有效参数
        params = {'p': 2.0, 'a': 0.5}
        distortion = 0.1
        
        self.processor.add_frame_parameters(params, distortion)
        
        assert len(self.processor.parameter_history) == 1
        assert len(self.processor.distortion_history) == 1
        assert self.processor.parameter_history[0] == params
        assert self.processor.distortion_history[0] == distortion
        assert self.processor.frame_count == 1
        
        # 测试窗口大小限制 (需求 5.1)
        for i in range(10):  # 添加超过窗口大小的帧
            self.processor.add_frame_parameters({'p': 2.0 + i * 0.1, 'a': 0.5}, 0.1 + i * 0.01)
            
        assert len(self.processor.parameter_history) == self.processor.window_size
        assert len(self.processor.distortion_history) == self.processor.window_size
        assert self.processor.frame_count == 11  # 总帧数继续增加
        
        # 测试无效输入
        with pytest.raises(ValueError):
            self.processor.add_frame_parameters({}, 0.1)  # 空参数
        with pytest.raises(ValueError):
            self.processor.add_frame_parameters({'p': 2.0}, np.nan)  # 无效失真
        with pytest.raises(ValueError):
            self.processor.add_frame_parameters({'p': 2.0}, -0.1)  # 负失真
            
    def test_compute_weighted_average(self):
        """测试加权平均计算 (需求 11.1)"""
        # 空历史
        result = self.processor.compute_weighted_average()
        assert result == {}
        
        # 添加测试数据
        test_data = [
            ({'p': 2.0, 'a': 0.5}, 0.1),
            ({'p': 2.2, 'a': 0.6}, 0.05),  # 更低失真，权重更高
            ({'p': 1.8, 'a': 0.4}, 0.2)    # 更高失真，权重更低
        ]
        
        for params, distortion in test_data:
            self.processor.add_frame_parameters(params, distortion)
            
        result = self.processor.compute_weighted_average()
        
        # 检查结果
        assert 'p' in result
        assert 'a' in result
        
        # 权重计算: w = 1/(D + ε)
        weights = [1.0 / (d + self.processor.eps) for _, d in test_data]
        weight_sum = sum(weights)
        
        expected_p = sum(w * params['p'] for w, (params, _) in zip(weights, test_data)) / weight_sum
        expected_a = sum(w * params['a'] for w, (params, _) in zip(weights, test_data)) / weight_sum
        
        assert abs(result['p'] - expected_p) < 1e-10
        assert abs(result['a'] - expected_a) < 1e-10
        
    def test_apply_temporal_filter(self):
        """测试时域滤波 (需求 5.3)"""
        # 历史帧数不足时返回原参数 (需求 11.1)
        current_params = {'p': 2.0, 'a': 0.5}
        result = self.processor.apply_temporal_filter(current_params, 0.3)
        assert result == current_params
        
        # 添加历史数据
        self.processor.add_frame_parameters({'p': 1.8, 'a': 0.4}, 0.1)
        self.processor.add_frame_parameters({'p': 2.2, 'a': 0.6}, 0.05)
        
        # 应用滤波
        lambda_smooth = 0.3
        filtered = self.processor.apply_temporal_filter(current_params, lambda_smooth)
        
        assert 'p' in filtered
        assert 'a' in filtered
        
        # 验证滤波公式: filtered = current + λ * (smoothed - current)
        smoothed = self.processor.compute_weighted_average()
        expected_p = current_params['p'] + lambda_smooth * (smoothed['p'] - current_params['p'])
        expected_a = current_params['a'] + lambda_smooth * (smoothed['a'] - current_params['a'])
        
        assert abs(filtered['p'] - expected_p) < 1e-10
        assert abs(filtered['a'] - expected_a) < 1e-10
        
        # 测试无效平滑强度
        with pytest.raises(ValueError):
            self.processor.apply_temporal_filter(current_params, 0.1)  # < 0.2
        with pytest.raises(ValueError):
            self.processor.apply_temporal_filter(current_params, 0.6)  # > 0.5
            
    def test_get_smoothing_stats(self):
        """测试平滑统计信息 (需求 19.3)"""
        # 帧数不足时
        stats = self.processor.get_smoothing_stats()
        assert isinstance(stats, TemporalStats)
        assert stats.frame_count == 0
        assert stats.variance_reduction == 0.0
        
        # 添加足够的数据
        test_params = [
            {'p': 2.0, 'a': 0.5},
            {'p': 2.5, 'a': 0.6},
            {'p': 1.5, 'a': 0.4},
            {'p': 2.2, 'a': 0.55},
            {'p': 1.8, 'a': 0.45}
        ]
        test_distortions = [0.1, 0.05, 0.15, 0.08, 0.12]
        
        for params, distortion in zip(test_params, test_distortions):
            self.processor.add_frame_parameters(params, distortion)
            
        stats = self.processor.get_smoothing_stats()
        
        assert stats.frame_count == 5
        assert stats.p_var_raw > 0
        assert stats.p_var_filtered >= 0
        assert 0 <= stats.variance_reduction <= 1
        assert 0 <= stats.window_utilization <= 1
        
    def test_validate_smoothing_effectiveness(self):
        """测试平滑效果验证 (需求 11.3)"""
        # 帧数不足
        is_effective, msg = self.processor.validate_smoothing_effectiveness()
        assert not is_effective
        assert "帧数不足" in msg
        
        # 添加高方差数据
        high_variance_params = [
            {'p': 1.0, 'a': 0.2},
            {'p': 3.0, 'a': 0.8},
            {'p': 1.5, 'a': 0.3},
            {'p': 2.8, 'a': 0.7},
            {'p': 1.2, 'a': 0.25}
        ]
        distortions = [0.2, 0.1, 0.15, 0.05, 0.18]
        
        for params, distortion in zip(high_variance_params, distortions):
            self.processor.add_frame_parameters(params, distortion)
            
        is_effective, msg = self.processor.validate_smoothing_effectiveness(0.3)  # 30%阈值
        # 结果取决于具体的平滑算法效果
        assert isinstance(is_effective, bool)
        assert isinstance(msg, str)
        
    def test_window_size_adjustment(self):
        """测试窗口大小调整"""
        # 添加数据
        for i in range(8):
            self.processor.add_frame_parameters({'p': 2.0 + i * 0.1, 'a': 0.5}, 0.1)
            
        assert len(self.processor.parameter_history) == 5  # 原窗口大小
        
        # 扩大窗口
        self.processor.set_window_size(8)
        assert self.processor.window_size == 8
        assert len(self.processor.parameter_history) == 5  # 数据不变
        
        # 缩小窗口 (必须在5-15范围内)
        self.processor.set_window_size(6)
        assert self.processor.window_size == 6
        assert len(self.processor.parameter_history) == 5  # 数据不变，因为原来就小于6
        
        # 再次添加数据测试缩小效果
        for i in range(3):
            self.processor.add_frame_parameters({'p': 3.0 + i * 0.1, 'a': 0.7}, 0.1)
        assert len(self.processor.parameter_history) == 6  # 达到新窗口大小
        
        # 缩小到5
        self.processor.set_window_size(5)
        assert self.processor.window_size == 5
        assert len(self.processor.parameter_history) == 5  # 保留最新的5帧
        
        # 无效窗口大小
        with pytest.raises(ValueError):
            self.processor.set_window_size(2)
        with pytest.raises(ValueError):
            self.processor.set_window_size(20)
            
    def test_parameter_trends(self):
        """测试参数趋势获取"""
        # 空历史
        trends = self.processor.get_parameter_trends()
        assert trends == {}
        
        # 添加数据
        test_data = [
            {'p': 2.0, 'a': 0.5},
            {'p': 2.1, 'a': 0.6},
            {'p': 1.9, 'a': 0.4}
        ]
        
        for i, params in enumerate(test_data):
            self.processor.add_frame_parameters(params, 0.1 + i * 0.01)
            
        trends = self.processor.get_parameter_trends()
        
        assert 'p' in trends
        assert 'a' in trends
        assert trends['p'] == [2.0, 2.1, 1.9]
        assert trends['a'] == [0.5, 0.6, 0.4]
        
        distortion_trend = self.processor.get_distortion_trend()
        expected_distortions = [0.1, 0.11, 0.12]
        assert len(distortion_trend) == len(expected_distortions)
        for actual, expected in zip(distortion_trend, expected_distortions):
            assert abs(actual - expected) < 1e-10
        
    def test_state_persistence(self):
        """测试状态持久化 (需求 19.1)"""
        # 添加数据
        test_params = {'p': 2.0, 'a': 0.5}
        test_distortion = 0.1
        
        self.processor.add_frame_parameters(test_params, test_distortion)
        
        # 检查文件是否创建
        assert os.path.exists(self.temp_file)
        
        # 创建新处理器加载状态
        new_processor = TemporalSmoothingProcessor(window_size=5, temporal_file=self.temp_file)
        
        assert len(new_processor.parameter_history) == 1
        assert new_processor.parameter_history[0] == test_params
        assert new_processor.distortion_history[0] == test_distortion
        assert new_processor.frame_count == 1
        
    def test_state_info(self):
        """测试状态信息获取 (需求 19.3)"""
        info = self.processor.get_state_info()
        
        required_keys = [
            'window_size', 'current_frames', 'total_frames',
            'window_utilization', 'variance_reduction', 'is_effective',
            'last_update', 'state_file_exists'
        ]
        
        for key in required_keys:
            assert key in info
            
        assert info['window_size'] == 5
        assert info['current_frames'] == 0
        assert info['total_frames'] == 0
        assert info['window_utilization'] == 0.0
        
    def test_export_temporal_data(self):
        """测试时域数据导出"""
        # 添加测试数据
        for i in range(3):
            params = {'p': 2.0 + i * 0.1, 'a': 0.5 + i * 0.05}
            self.processor.add_frame_parameters(params, 0.1 + i * 0.02)
            
        export_data = self.processor.export_temporal_data()
        
        assert 'metadata' in export_data
        assert 'statistics' in export_data
        assert 'parameter_trends' in export_data
        assert 'distortion_trend' in export_data
        assert 'raw_data' in export_data
        
        # 检查元数据
        assert export_data['metadata']['window_size'] == 5
        assert export_data['metadata']['frame_count'] == 3
        
        # 检查原始数据
        assert len(export_data['raw_data']['parameter_history']) == 3
        assert len(export_data['raw_data']['distortion_history']) == 3
        
    def test_simulate_smoothing_effect(self):
        """测试平滑效果模拟"""
        # 准备测试数据
        test_params = [
            {'p': 2.0, 'a': 0.5},
            {'p': 2.5, 'a': 0.6},
            {'p': 1.5, 'a': 0.4},
            {'p': 2.2, 'a': 0.55}
        ]
        test_distortions = [0.1, 0.05, 0.15, 0.08]
        
        # 保存原始状态
        original_count = self.processor.frame_count
        
        # 执行模拟
        result = self.processor.simulate_smoothing_effect(test_params, test_distortions, 0.3)
        
        assert 'raw_sequence' in result
        assert 'filtered_sequence' in result
        assert 'statistics' in result
        assert 'effectiveness' in result
        
        assert len(result['raw_sequence']) == 4
        assert len(result['filtered_sequence']) == 4
        
        # 检查原始状态是否恢复
        assert self.processor.frame_count == original_count
        
        # 测试参数长度不匹配
        with pytest.raises(ValueError):
            self.processor.simulate_smoothing_effect(test_params, [0.1, 0.2])  # 长度不匹配
            
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试极小失真值
        self.processor.add_frame_parameters({'p': 2.0, 'a': 0.5}, 1e-10)
        self.processor.add_frame_parameters({'p': 2.1, 'a': 0.6}, 1e-9)
        
        # 应该能正常计算权重
        smoothed = self.processor.compute_weighted_average()
        assert 'p' in smoothed
        assert 'a' in smoothed
        
        # 测试相同失真值
        self.processor.cold_start()
        for i in range(3):
            self.processor.add_frame_parameters({'p': 2.0 + i * 0.1, 'a': 0.5}, 0.1)
            
        smoothed = self.processor.compute_weighted_average()
        # 相同权重应该得到均值
        expected_p = (2.0 + 2.1 + 2.2) / 3
        assert abs(smoothed['p'] - expected_p) < 1e-10
        
    def test_integration_with_quality_metrics(self):
        """测试与质量指标的集成"""
        # 模拟实际使用场景
        quality_calculator = QualityMetricsCalculator()
        
        # 模拟多帧处理
        for i in range(5):
            # 模拟参数变化
            p = 2.0 + np.sin(i * 0.5) * 0.3
            a = 0.5 + np.cos(i * 0.3) * 0.1
            params = {'p': p, 'a': a}
            
            # 模拟失真计算
            L_in = np.random.rand(100) * 0.8
            L_out = L_in ** p / (L_in ** p + a ** p)
            distortion = quality_calculator.compute_perceptual_distortion(L_in, L_out)
            
            # 添加到时域处理器
            self.processor.add_frame_parameters(params, distortion)
            
            # 应用平滑
            if i > 1:  # 有足够历史后开始平滑
                smoothed_params = self.processor.apply_temporal_filter(params, 0.3)
                
                # 验证平滑后的参数在合理范围内
                assert 0.1 <= smoothed_params['p'] <= 6.0
                assert 0.0 <= smoothed_params['a'] <= 1.0
                
        # 验证平滑效果
        stats = self.processor.get_smoothing_stats()
        assert stats.frame_count == 5
        assert stats.window_utilization == 1.0  # 窗口已满