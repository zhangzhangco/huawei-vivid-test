"""
样条曲线计算器测试
测试PCHIP样条插值、C¹连续性验证和与Phoenix曲线的混合功能
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import (
    SplineCurveCalculator, 
    SplineCalculationError,
    SplineVisualizationHelper,
    PhoenixCurveCalculator,
    SafeCalculator
)


class TestSplineCurveCalculator:
    """样条曲线计算器测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.calculator = SplineCurveCalculator()
        self.phoenix_calc = PhoenixCurveCalculator()
        
        # 测试数据
        self.x = np.linspace(0, 1, 100)
        self.phoenix_curve = self.phoenix_calc.compute_phoenix_curve(self.x, 2.0, 0.5)
        self.test_nodes = [0.2, 0.5, 0.8]
        
    def test_initialization(self):
        """测试初始化"""
        assert self.calculator.default_nodes == [0.2, 0.5, 0.8]
        assert self.calculator.min_node_interval == 0.01
        assert self.calculator.eps == 1e-6
        assert self.calculator.continuity_tolerance == 1e-3
        
    def test_validate_and_correct_nodes_valid(self):
        """测试有效节点验证"""
        nodes = [0.2, 0.5, 0.8]
        corrected, warning = self.calculator.validate_and_correct_nodes(nodes)
        
        assert corrected == [0.2, 0.5, 0.8]
        assert warning == ""
        
    def test_validate_and_correct_nodes_sorting(self):
        """测试节点排序"""
        nodes = [0.8, 0.2, 0.5]
        corrected, warning = self.calculator.validate_and_correct_nodes(nodes)
        
        assert corrected == [0.2, 0.5, 0.8]
        # 简单排序不会产生警告，只有范围或间隔调整才会
        
    def test_validate_and_correct_nodes_interval(self):
        """测试最小间隔调整"""
        nodes = [0.2, 0.201, 0.8]  # 间隔太小
        corrected, warning = self.calculator.validate_and_correct_nodes(nodes)
        
        assert corrected[1] - corrected[0] >= 0.01
        assert "已自动调整" in warning
        
    def test_validate_and_correct_nodes_range(self):
        """测试范围约束"""
        nodes = [-0.1, 0.5, 1.1]  # 超出范围
        corrected, warning = self.calculator.validate_and_correct_nodes(nodes)
        
        assert all(0.0 <= node <= 1.0 for node in corrected)
        assert "已自动调整" in warning
        
    def test_compute_pchip_spline_basic(self):
        """测试基本PCHIP插值"""
        x_nodes = np.array([0.0, 0.3, 0.7, 1.0])
        y_nodes = np.array([0.0, 0.2, 0.8, 1.0])
        x_eval = np.linspace(0, 1, 50)
        
        result = self.calculator.compute_pchip_spline(x_nodes, y_nodes, x_eval)
        
        assert len(result) == len(x_eval)
        assert result[0] == pytest.approx(y_nodes[0], abs=1e-6)
        assert result[-1] == pytest.approx(y_nodes[-1], abs=1e-6)
        
    def test_compute_pchip_spline_monotonic(self):
        """测试PCHIP单调性保持"""
        x_nodes = np.array([0.0, 0.3, 0.7, 1.0])
        y_nodes = np.array([0.0, 0.3, 0.7, 1.0])  # 单调递增
        x_eval = np.linspace(0, 1, 100)
        
        result = self.calculator.compute_pchip_spline(x_nodes, y_nodes, x_eval)
        
        # 检查单调性
        assert np.all(np.diff(result) >= -1e-10)  # 允许数值误差
        
    def test_compute_pchip_spline_error(self):
        """测试PCHIP插值错误处理"""
        x_nodes = np.array([0.0, 0.5])  # 节点太少
        y_nodes = np.array([0.0])       # 长度不匹配
        x_eval = np.linspace(0, 1, 10)
        
        with pytest.raises(SplineCalculationError):
            self.calculator.compute_pchip_spline(x_nodes, y_nodes, x_eval)
            
    def test_verify_c1_continuity_continuous(self):
        """测试C¹连续性验证 - 连续情况"""
        x_nodes = np.array([0.0, 0.5, 1.0])
        y_nodes = np.array([0.0, 0.25, 1.0])  # 简单单调函数
        
        is_continuous, error = self.calculator.verify_c1_continuity(x_nodes, y_nodes)
        
        assert is_continuous
        assert error <= self.calculator.continuity_tolerance
        
    def test_verify_c1_continuity_error_handling(self):
        """测试C¹连续性验证错误处理"""
        x_nodes = np.array([0.0, 0.5])  # 节点太少
        y_nodes = np.array([0.0])       # 长度不匹配
        
        is_continuous, error = self.calculator.verify_c1_continuity(x_nodes, y_nodes)
        
        assert not is_continuous
        assert error == float('inf')
        
    def test_create_spline_from_phoenix_success(self):
        """测试从Phoenix曲线创建样条 - 成功情况"""
        spline_curve, success, message = self.calculator.create_spline_from_phoenix(
            self.phoenix_curve, self.x, self.test_nodes
        )
        
        assert success
        assert len(spline_curve) == len(self.phoenix_curve)
        assert spline_curve[0] == pytest.approx(self.phoenix_curve[0], abs=1e-6)
        assert spline_curve[-1] == pytest.approx(self.phoenix_curve[-1], abs=1e-6)
        
    def test_create_spline_from_phoenix_bad_nodes(self):
        """测试从Phoenix曲线创建样条 - 坏节点"""
        bad_nodes = [0.5, 0.5, 0.5]  # 重复节点
        
        spline_curve, success, message = self.calculator.create_spline_from_phoenix(
            self.phoenix_curve, self.x, bad_nodes
        )
        
        assert success  # 应该自动修正
        assert "已自动调整" in message
        
    def test_blend_with_phoenix_zero_strength(self):
        """测试与Phoenix曲线混合 - 零强度"""
        spline_curve = self.phoenix_curve * 1.1  # 稍微不同的曲线
        
        result = self.calculator.blend_with_phoenix(
            self.phoenix_curve, spline_curve, 0.0
        )
        
        np.testing.assert_array_almost_equal(result, self.phoenix_curve)
        
    def test_blend_with_phoenix_full_strength(self):
        """测试与Phoenix曲线混合 - 满强度"""
        spline_curve = self.phoenix_curve * 1.1
        
        result = self.calculator.blend_with_phoenix(
            self.phoenix_curve, spline_curve, 1.0
        )
        
        np.testing.assert_array_almost_equal(result, spline_curve)
        
    def test_blend_with_phoenix_half_strength(self):
        """测试与Phoenix曲线混合 - 半强度"""
        spline_curve = self.phoenix_curve * 1.2
        
        result = self.calculator.blend_with_phoenix(
            self.phoenix_curve, spline_curve, 0.5
        )
        
        expected = 0.5 * self.phoenix_curve + 0.5 * spline_curve
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_check_monotonicity_monotonic(self):
        """测试单调性检查 - 单调情况"""
        monotonic_curve = np.linspace(0, 1, 100)
        
        assert self.calculator.check_monotonicity(monotonic_curve)
        
    def test_check_monotonicity_non_monotonic(self):
        """测试单调性检查 - 非单调情况"""
        non_monotonic = np.array([0, 0.5, 0.3, 1.0])  # 有下降
        
        assert not self.calculator.check_monotonicity(non_monotonic)
        
    def test_check_monotonicity_with_tolerance(self):
        """测试单调性检查 - 数值误差容忍"""
        # 有微小数值误差但基本单调的曲线
        nearly_monotonic = np.linspace(0, 1, 100)
        nearly_monotonic[50] -= 1e-8  # 微小下降
        
        assert self.calculator.check_monotonicity(nearly_monotonic)
        
    def test_compute_spline_with_fallback_zero_strength(self):
        """测试带回退的样条计算 - 零强度"""
        final_curve, used_spline, status = self.calculator.compute_spline_with_fallback(
            self.phoenix_curve, self.x, self.test_nodes, 0.0
        )
        
        assert not used_spline
        assert "强度为0" in status
        np.testing.assert_array_almost_equal(final_curve, self.phoenix_curve)
        
    def test_compute_spline_with_fallback_success(self):
        """测试带回退的样条计算 - 成功情况"""
        final_curve, used_spline, status = self.calculator.compute_spline_with_fallback(
            self.phoenix_curve, self.x, self.test_nodes, 0.5
        )
        
        assert used_spline
        assert "成功" in status
        assert len(final_curve) == len(self.phoenix_curve)
        
    def test_compute_spline_with_fallback_bad_nodes(self):
        """测试带回退的样条计算 - 坏节点自动修正"""
        bad_nodes = [0.1, 0.101, 0.9]  # 间隔太小
        
        final_curve, used_spline, status = self.calculator.compute_spline_with_fallback(
            self.phoenix_curve, self.x, bad_nodes, 0.5
        )
        
        assert used_spline  # 应该自动修正并成功
        assert "已自动调整" in status
        
    @patch('core.spline_calculator.PchipInterpolator')
    def test_compute_spline_with_fallback_spline_error(self, mock_pchip):
        """测试带回退的样条计算 - 样条计算错误"""
        mock_pchip.side_effect = Exception("插值失败")
        
        final_curve, used_spline, status = self.calculator.compute_spline_with_fallback(
            self.phoenix_curve, self.x, self.test_nodes, 0.5
        )
        
        assert not used_spline
        assert "样条创建失败" in status
        np.testing.assert_array_almost_equal(final_curve, self.phoenix_curve)
        
    def test_get_spline_segments_info(self):
        """测试样条段信息获取"""
        info = self.calculator.get_spline_segments_info(self.test_nodes)
        
        assert info['node_count'] == 5  # 包含端点
        assert info['segment_count'] == 4
        assert len(info['segments']) == 4
        assert info['corrected_nodes'] == self.test_nodes
        
        # 检查段信息
        for segment in info['segments']:
            assert 'start' in segment
            assert 'end' in segment
            assert 'length' in segment
            assert segment['end'] > segment['start']
            
    def test_edge_cases(self):
        """测试边界情况"""
        # 空节点列表
        empty_nodes = []
        corrected, warning = self.calculator.validate_and_correct_nodes(empty_nodes)
        assert len(corrected) > 0  # 应该使用默认值
        
        # 单个节点
        single_node = [0.5]
        corrected, warning = self.calculator.validate_and_correct_nodes(single_node)
        assert len(corrected) >= 1
        
        # 极端强度值
        extreme_curve, used_spline, status = self.calculator.compute_spline_with_fallback(
            self.phoenix_curve, self.x, self.test_nodes, 2.0  # 超出范围
        )
        assert used_spline  # 应该被夹取到1.0
        
    def test_numerical_stability(self):
        """测试数值稳定性"""
        # 使用极端参数的Phoenix曲线
        extreme_phoenix = self.phoenix_calc.compute_phoenix_curve(self.x, 0.1, 0.001)
        
        final_curve, used_spline, status = self.calculator.compute_spline_with_fallback(
            extreme_phoenix, self.x, self.test_nodes, 0.5
        )
        
        # 检查结果的有效性
        assert np.all(np.isfinite(final_curve))
        assert np.all(final_curve >= 0)
        assert np.all(final_curve <= 1)


class TestSplineVisualizationHelper:
    """样条曲线可视化辅助类测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.x = np.linspace(0, 1, 100)
        self.phoenix_curve = np.power(self.x, 2)  # 简单的平方曲线
        self.spline_curve = np.power(self.x, 1.5)  # 稍微不同的曲线
        self.nodes = [0.2, 0.5, 0.8]
        
    def test_generate_comparison_data(self):
        """测试对比数据生成"""
        viz_data = SplineVisualizationHelper.generate_comparison_data(
            self.phoenix_curve, self.spline_curve, self.x, self.nodes
        )
        
        assert 'x' in viz_data
        assert 'phoenix' in viz_data
        assert 'spline' in viz_data
        assert 'difference' in viz_data
        assert 'nodes' in viz_data
        
        np.testing.assert_array_equal(viz_data['x'], self.x)
        np.testing.assert_array_equal(viz_data['phoenix'], self.phoenix_curve)
        np.testing.assert_array_equal(viz_data['spline'], self.spline_curve)
        
        expected_diff = self.spline_curve - self.phoenix_curve
        np.testing.assert_array_almost_equal(viz_data['difference'], expected_diff)
        
        # 检查节点数据
        assert len(viz_data['nodes']['x']) == len(self.nodes)
        assert len(viz_data['nodes']['y_phoenix']) == len(self.nodes)
        assert len(viz_data['nodes']['y_spline']) == len(self.nodes)
        
    def test_compute_spline_statistics(self):
        """测试样条统计计算"""
        stats = SplineVisualizationHelper.compute_spline_statistics(
            self.phoenix_curve, self.spline_curve
        )
        
        required_keys = [
            'max_deviation', 'mean_deviation', 'rms_deviation',
            'positive_deviation_ratio', 'spline_range', 'phoenix_range'
        ]
        
        for key in required_keys:
            assert key in stats
            
        # 检查统计值的合理性
        assert stats['max_deviation'] >= 0
        assert stats['mean_deviation'] >= 0
        assert stats['rms_deviation'] >= 0
        assert 0 <= stats['positive_deviation_ratio'] <= 1
        
        # 检查范围
        assert len(stats['spline_range']) == 2
        assert len(stats['phoenix_range']) == 2
        assert stats['spline_range'][0] <= stats['spline_range'][1]
        assert stats['phoenix_range'][0] <= stats['phoenix_range'][1]
        
    def test_statistics_identical_curves(self):
        """测试相同曲线的统计"""
        stats = SplineVisualizationHelper.compute_spline_statistics(
            self.phoenix_curve, self.phoenix_curve
        )
        
        assert stats['max_deviation'] == pytest.approx(0, abs=1e-10)
        assert stats['mean_deviation'] == pytest.approx(0, abs=1e-10)
        assert stats['rms_deviation'] == pytest.approx(0, abs=1e-10)


class TestSplineIntegration:
    """样条曲线集成测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.safe_calc = SafeCalculator()
        
    def test_safe_calculator_spline_integration(self):
        """测试安全计算器的样条集成"""
        p, a = 2.0, 0.5
        th_nodes = [0.2, 0.5, 0.8]
        th_strength = 0.6
        
        x, final_curve, phoenix_ok, spline_ok, status = self.safe_calc.safe_combined_curve_calculation(
            p, a, th_nodes, th_strength
        )
        
        assert phoenix_ok
        assert spline_ok
        assert len(final_curve) == len(x)
        assert np.all(np.isfinite(final_curve))
        assert "Phoenix: 计算成功" in status
        assert "样条: 样条曲线应用成功" in status
        
    def test_safe_calculator_spline_fallback(self):
        """测试安全计算器的样条回退"""
        p, a = 2.0, 0.5
        bad_nodes = []  # 空节点列表
        th_strength = 0.5
        
        x, final_curve, phoenix_ok, spline_ok, status = self.safe_calc.safe_combined_curve_calculation(
            p, a, bad_nodes, th_strength
        )
        
        assert phoenix_ok
        # spline_ok可能为True（自动修正）或False（回退）
        assert len(final_curve) == len(x)
        assert np.all(np.isfinite(final_curve))
        
    def test_safe_calculator_zero_strength(self):
        """测试安全计算器零强度情况"""
        p, a = 2.0, 0.5
        th_nodes = [0.2, 0.5, 0.8]
        th_strength = 0.0
        
        x, final_curve, phoenix_ok, spline_ok, status = self.safe_calc.safe_combined_curve_calculation(
            p, a, th_nodes, th_strength
        )
        
        assert phoenix_ok
        assert not spline_ok  # 强度为0，不使用样条
        assert "样条强度为0" in status
        
    def test_system_status_with_spline(self):
        """测试包含样条的系统状态"""
        status = self.safe_calc.get_system_status()
        
        assert 'spline_calculator_ready' in status
        assert status['spline_calculator_ready'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])