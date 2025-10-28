#!/usr/bin/env python3
"""
Auto模式参数估算器测试
测试基于图像统计的Phoenix曲线参数自动估算功能
"""

import unittest
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core import (
    AutoModeParameterEstimator,
    AutoModeInterface,
    AutoModeConfig,
    EstimationResult,
    ImageStats
)


class TestAutoModeConfig(unittest.TestCase):
    """测试Auto模式配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = AutoModeConfig()
        
        self.assertEqual(config.p0, 1.0)
        self.assertEqual(config.a0, 0.3)
        self.assertEqual(config.alpha, 0.5)
        self.assertEqual(config.beta, 0.3)
        self.assertTrue(config.enable_adaptive_scaling)
        
    def test_config_serialization(self):
        """测试配置序列化"""
        config = AutoModeConfig(p0=1.5, a0=0.4, alpha=0.8, beta=0.5)
        
        # 转换为字典
        config_dict = config.to_dict()
        self.assertEqual(config_dict['p0'], 1.5)
        self.assertEqual(config_dict['a0'], 0.4)
        
        # 从字典创建
        config2 = AutoModeConfig.from_dict(config_dict)
        self.assertEqual(config2.p0, 1.5)
        self.assertEqual(config2.a0, 0.4)
        self.assertEqual(config2.alpha, 0.8)
        self.assertEqual(config2.beta, 0.5)


class TestAutoModeParameterEstimator(unittest.TestCase):
    """测试Auto模式参数估算器"""
    
    def setUp(self):
        """设置测试环境"""
        self.estimator = AutoModeParameterEstimator()
        
        # 创建测试用的图像统计数据
        self.normal_stats = ImageStats(
            min_pq=0.05,
            max_pq=0.85,
            avg_pq=0.45,
            var_pq=0.06,
            input_format="test",
            processing_path="test",
            pixel_count=1000000
        )
        
        self.low_light_stats = ImageStats(
            min_pq=0.01,
            max_pq=0.25,
            avg_pq=0.08,
            var_pq=0.003,
            input_format="test",
            processing_path="test",
            pixel_count=1000000
        )
        
    def test_basic_estimation(self):
        """测试基本参数估算"""
        result = self.estimator.estimate_parameters(self.normal_stats)
        
        # 检查结果类型
        self.assertIsInstance(result, EstimationResult)
        
        # 检查参数范围
        self.assertGreaterEqual(result.p_estimated, 0.1)
        self.assertLessEqual(result.p_estimated, 6.0)
        self.assertGreaterEqual(result.a_estimated, 0.0)
        self.assertLessEqual(result.a_estimated, 1.0)
        
        # 检查统计量
        self.assertEqual(result.min_pq, self.normal_stats.min_pq)
        self.assertEqual(result.max_pq, self.normal_stats.max_pq)
        self.assertEqual(result.avg_pq, self.normal_stats.avg_pq)
        self.assertEqual(result.var_pq, self.normal_stats.var_pq)
        
        # 检查置信度
        self.assertGreaterEqual(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)
        
    def test_linear_estimation_formula(self):
        """测试线性估参公式"""
        config = AutoModeConfig(p0=1.0, a0=0.3, alpha=0.5, beta=0.3)
        estimator = AutoModeParameterEstimator(config)
        
        result = estimator.estimate_parameters(self.normal_stats)
        
        # 验证线性公式
        expected_p_raw = config.p0 + config.alpha * (self.normal_stats.max_pq - self.normal_stats.avg_pq)
        expected_a_raw = config.a0 + config.beta * (self.normal_stats.avg_pq - self.normal_stats.min_pq)
        
        # 由于可能有自适应调整，检查原始值是否接近预期
        self.assertAlmostEqual(result.p_raw, expected_p_raw, delta=0.5)
        self.assertAlmostEqual(result.a_raw, expected_a_raw, delta=0.2)
        
    def test_parameter_clipping(self):
        """测试参数裁剪"""
        # 创建会导致参数超出范围的配置
        config = AutoModeConfig(p0=5.5, a0=0.9, alpha=2.0, beta=1.0)
        estimator = AutoModeParameterEstimator(config)
        
        # 使用极端统计值
        extreme_stats = ImageStats(
            min_pq=0.0,
            max_pq=1.0,
            avg_pq=0.1,
            var_pq=0.2,
            input_format="test",
            processing_path="test",
            pixel_count=1000000
        )
        
        result = estimator.estimate_parameters(extreme_stats)
        
        # 检查参数是否被正确裁剪
        self.assertGreaterEqual(result.p_estimated, 0.1)
        self.assertLessEqual(result.p_estimated, 6.0)
        self.assertGreaterEqual(result.a_estimated, 0.0)
        self.assertLessEqual(result.a_estimated, 1.0)
        
    def test_confidence_calculation(self):
        """测试置信度计算"""
        # 正常场景应该有较高置信度
        result_normal = self.estimator.estimate_parameters(self.normal_stats)
        
        # 极端场景应该有较低置信度
        extreme_stats = ImageStats(
            min_pq=0.001,
            max_pq=0.002,
            avg_pq=0.0015,
            var_pq=0.0,
            input_format="test",
            processing_path="test",
            pixel_count=1000000
        )
        result_extreme = self.estimator.estimate_parameters(extreme_stats)
        
        # 正常场景的置信度应该高于极端场景
        self.assertGreater(result_normal.confidence_score, result_extreme.confidence_score)
        
    def test_config_update(self):
        """测试配置更新"""
        original_p0 = self.estimator.config.p0
        
        # 更新配置
        self.estimator.update_config(p0=2.0, alpha=0.8)
        
        self.assertEqual(self.estimator.config.p0, 2.0)
        self.assertEqual(self.estimator.config.alpha, 0.8)
        self.assertNotEqual(self.estimator.config.p0, original_p0)
        
    def test_reset_to_defaults(self):
        """测试重置到默认配置"""
        # 修改配置
        self.estimator.update_config(p0=2.0, a0=0.8)
        
        # 重置
        self.estimator.reset_to_defaults()
        
        # 检查是否恢复默认值
        default_config = AutoModeConfig()
        self.assertEqual(self.estimator.config.p0, default_config.p0)
        self.assertEqual(self.estimator.config.a0, default_config.a0)
        
    def test_hyperparameter_validation(self):
        """测试超参数验证"""
        # 有效参数
        valid, msg = self.estimator.validate_hyperparameters(1.0, 0.3, 0.5, 0.3)
        self.assertTrue(valid)
        self.assertEqual(msg, "超参数验证通过")
        
        # 无效p0
        valid, msg = self.estimator.validate_hyperparameters(10.0, 0.3, 0.5, 0.3)
        self.assertFalse(valid)
        self.assertIn("p0", msg)
        
        # 无效a0
        valid, msg = self.estimator.validate_hyperparameters(1.0, 2.0, 0.5, 0.3)
        self.assertFalse(valid)
        self.assertIn("a0", msg)
        
    def test_estimation_summary(self):
        """测试估算摘要"""
        # 未进行估算时
        summary = self.estimator.get_estimation_summary()
        self.assertEqual(summary["status"], "no_estimation")
        
        # 进行估算后
        result = self.estimator.estimate_parameters(self.normal_stats)
        summary = self.estimator.get_estimation_summary()
        
        self.assertEqual(summary["status"], "success")
        self.assertIn("final_parameters", summary)
        self.assertIn("image_statistics", summary)
        self.assertIn("estimation_process", summary)
        self.assertIn("quality_assessment", summary)
        self.assertIn("hyperparameters", summary)
        
    def test_export_estimation_data(self):
        """测试导出估算数据"""
        # 未进行估算时
        export_data = self.estimator.export_estimation_data()
        self.assertIn("error", export_data)
        
        # 进行估算后
        result = self.estimator.estimate_parameters(self.normal_stats)
        export_data = self.estimator.export_estimation_data()
        
        self.assertIn("p_estimated", export_data)
        self.assertIn("a_estimated", export_data)
        self.assertIn("config", export_data)


class TestAutoModeInterface(unittest.TestCase):
    """测试Auto模式界面接口"""
    
    def setUp(self):
        """设置测试环境"""
        self.estimator = AutoModeParameterEstimator()
        self.interface = AutoModeInterface(self.estimator)
        
        self.test_stats = ImageStats(
            min_pq=0.05,
            max_pq=0.85,
            avg_pq=0.45,
            var_pq=0.06,
            input_format="test",
            processing_path="test",
            pixel_count=1000000
        )
        
    def test_observable_data_no_estimation(self):
        """测试未进行估算时的可观测数据"""
        data = self.interface.get_observable_data()
        
        self.assertFalse(data["has_estimation"])
        self.assertIn("hyperparameters", data)
        
    def test_observable_data_with_estimation(self):
        """测试进行估算后的可观测数据"""
        # 进行估算
        self.estimator.estimate_parameters(self.test_stats)
        
        data = self.interface.get_observable_data()
        
        self.assertTrue(data["has_estimation"])
        self.assertIn("hyperparameters", data)
        self.assertIn("image_statistics", data)
        self.assertIn("estimation_results", data)
        self.assertIn("quality_assessment", data)
        
    def test_apply_estimated_parameters(self):
        """测试一键应用估算参数"""
        # 未进行估算时
        success, params, msg = self.interface.apply_estimated_parameters()
        self.assertFalse(success)
        self.assertEqual(params, {})
        
        # 进行估算后
        self.estimator.estimate_parameters(self.test_stats)
        success, params, msg = self.interface.apply_estimated_parameters()
        
        self.assertTrue(success)
        self.assertIn("p", params)
        self.assertIn("a", params)
        self.assertIsInstance(params["p"], float)
        self.assertIsInstance(params["a"], float)
        
    def test_restore_default_hyperparameters(self):
        """测试恢复默认超参数"""
        # 修改超参数
        self.interface.update_hyperparameters(2.0, 0.8, 1.0, 0.6)
        
        # 恢复默认
        hyperparams, msg = self.interface.restore_default_hyperparameters()
        
        self.assertEqual(hyperparams["p0"], 1.0)
        self.assertEqual(hyperparams["a0"], 0.3)
        self.assertEqual(hyperparams["alpha"], 0.5)
        self.assertEqual(hyperparams["beta"], 0.3)
        self.assertIn("恢复", msg)
        
    def test_update_hyperparameters(self):
        """测试更新超参数"""
        # 有效更新
        success, msg = self.interface.update_hyperparameters(1.5, 0.4, 0.8, 0.5)
        self.assertTrue(success)
        self.assertIn("成功", msg)
        
        # 无效更新
        success, msg = self.interface.update_hyperparameters(10.0, 0.4, 0.8, 0.5)
        self.assertFalse(success)
        self.assertIn("有效范围", msg)
        
    def test_get_hyperparameter_ranges(self):
        """测试获取超参数范围"""
        ranges = self.interface.get_hyperparameter_ranges()
        
        self.assertIn("p0", ranges)
        self.assertIn("a0", ranges)
        self.assertIn("alpha", ranges)
        self.assertIn("beta", ranges)
        
        # 检查范围格式
        for param, (min_val, max_val) in ranges.items():
            self.assertIsInstance(min_val, (int, float))
            self.assertIsInstance(max_val, (int, float))
            self.assertLess(min_val, max_val)
            
    def test_format_estimation_report(self):
        """测试格式化估算报告"""
        # 未进行估算时
        report = self.interface.format_estimation_report()
        self.assertIn("尚未进行", report)
        
        # 进行估算后
        self.estimator.estimate_parameters(self.test_stats)
        report = self.interface.format_estimation_report()
        
        self.assertIn("估算报告", report)
        self.assertIn("图像统计", report)
        self.assertIn("超参数", report)
        self.assertIn("估算结果", report)
        self.assertIn("质量评估", report)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def setUp(self):
        """设置测试环境"""
        self.estimator = AutoModeParameterEstimator()
        
    def test_extreme_dark_image(self):
        """测试极暗图像"""
        dark_stats = ImageStats(
            min_pq=0.001,
            max_pq=0.01,
            avg_pq=0.005,
            var_pq=0.0001,
            input_format="test",
            processing_path="test",
            pixel_count=1000000
        )
        
        result = self.estimator.estimate_parameters(dark_stats)
        
        # 应该能正常估算，但置信度较低
        self.assertIsInstance(result, EstimationResult)
        self.assertLess(result.confidence_score, 0.8)
        
    def test_extreme_bright_image(self):
        """测试极亮图像"""
        bright_stats = ImageStats(
            min_pq=0.95,
            max_pq=0.999,
            avg_pq=0.98,
            var_pq=0.0002,
            input_format="test",
            processing_path="test",
            pixel_count=1000000
        )
        
        result = self.estimator.estimate_parameters(bright_stats)
        
        # 应该能正常估算，但置信度较低
        self.assertIsInstance(result, EstimationResult)
        self.assertLess(result.confidence_score, 0.8)
        
    def test_zero_variance_image(self):
        """测试零方差图像"""
        flat_stats = ImageStats(
            min_pq=0.5,
            max_pq=0.5,
            avg_pq=0.5,
            var_pq=0.0,
            input_format="test",
            processing_path="test",
            pixel_count=1000000
        )
        
        result = self.estimator.estimate_parameters(flat_stats)
        
        # 应该能处理，但置信度很低
        self.assertIsInstance(result, EstimationResult)
        self.assertLess(result.confidence_score, 0.7)
        
    def test_high_variance_image(self):
        """测试高方差图像"""
        high_var_stats = ImageStats(
            min_pq=0.0,
            max_pq=1.0,
            avg_pq=0.5,
            var_pq=0.25,
            input_format="test",
            processing_path="test",
            pixel_count=1000000
        )
        
        result = self.estimator.estimate_parameters(high_var_stats)
        
        # 应该能处理，置信度中等
        self.assertIsInstance(result, EstimationResult)
        self.assertGreater(result.confidence_score, 0.5)


class TestReproducibility(unittest.TestCase):
    """测试可复现性"""
    
    def test_deterministic_estimation(self):
        """测试估算的确定性"""
        stats = ImageStats(
            min_pq=0.05,
            max_pq=0.85,
            avg_pq=0.45,
            var_pq=0.06,
            input_format="test",
            processing_path="test",
            pixel_count=1000000
        )
        
        estimator1 = AutoModeParameterEstimator()
        estimator2 = AutoModeParameterEstimator()
        
        result1 = estimator1.estimate_parameters(stats)
        result2 = estimator2.estimate_parameters(stats)
        
        # 相同输入应该产生相同结果
        self.assertAlmostEqual(result1.p_estimated, result2.p_estimated, places=6)
        self.assertAlmostEqual(result1.a_estimated, result2.a_estimated, places=6)
        self.assertAlmostEqual(result1.confidence_score, result2.confidence_score, places=6)
        
    def test_config_consistency(self):
        """测试配置一致性"""
        config = AutoModeConfig(p0=1.5, a0=0.4, alpha=0.8, beta=0.5)
        
        estimator1 = AutoModeParameterEstimator(config)
        estimator2 = AutoModeParameterEstimator(config)
        
        stats = ImageStats(
            min_pq=0.1,
            max_pq=0.9,
            avg_pq=0.5,
            var_pq=0.08,
            input_format="test",
            processing_path="test",
            pixel_count=1000000
        )
        
        result1 = estimator1.estimate_parameters(stats)
        result2 = estimator2.estimate_parameters(stats)
        
        # 相同配置应该产生相同结果
        self.assertEqual(result1.p_estimated, result2.p_estimated)
        self.assertEqual(result1.a_estimated, result2.a_estimated)


if __name__ == '__main__':
    # 运行所有测试
    unittest.main(verbosity=2)