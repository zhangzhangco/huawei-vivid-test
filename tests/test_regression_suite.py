#!/usr/bin/env python3
"""
自动化回归测试套件
集成所有验证测试，提供完整的回归测试功能
"""

import pytest
import numpy as np
import os
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any
import sys

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from test_validation_framework import (
    AutomatedRegressionTestSuite, ValidationResult,
    GoldenStandardTests, MonotonicityStressTests, 
    HysteresisStabilityTests, ExportImportConsistencyTests
)
from golden_test_data import GoldenTestDataGenerator


class TestRegressionSuite:
    """回归测试套件测试类"""
    
    @classmethod
    def setup_class(cls):
        """类级别设置"""
        cls.test_suite = AutomatedRegressionTestSuite()
        cls.temp_dir = tempfile.mkdtemp()
        
        # 生成测试数据
        cls.data_generator = GoldenTestDataGenerator()
        cls.golden_images = cls.data_generator.generate_golden_images()
        cls.golden_curves = cls.data_generator.generate_golden_curves()
        
    @classmethod
    def teardown_class(cls):
        """类级别清理"""
        import shutil
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)
            
    def test_golden_standard_phoenix_curves(self):
        """测试Phoenix曲线金标测试"""
        golden_tests = GoldenStandardTests()
        results = golden_tests.run_golden_tests()
        
        # 筛选Phoenix曲线测试
        phoenix_results = [r for r in results if r.test_name.startswith("phoenix_")]
        
        assert len(phoenix_results) > 0, "应该有Phoenix曲线测试"
        
        # 检查基础测试通过
        basic_tests = [r for r in phoenix_results if "basic" in r.test_name or "gamma22" in r.test_name]
        for test in basic_tests:
            assert test.passed, f"基础Phoenix测试失败: {test.test_name} - {test.error_message}"
            
        # 统计通过率
        passed_count = sum(1 for r in phoenix_results if r.passed)
        total_count = len(phoenix_results)
        pass_rate = passed_count / total_count
        
        print(f"Phoenix曲线测试通过率: {passed_count}/{total_count} ({pass_rate*100:.1f}%)")
        
        # 至少80%的测试应该通过
        assert pass_rate >= 0.8, f"Phoenix曲线测试通过率过低: {pass_rate*100:.1f}%"
        
    def test_golden_standard_quality_metrics(self):
        """测试质量指标金标测试"""
        golden_tests = GoldenStandardTests()
        results = golden_tests.run_golden_tests()
        
        # 筛选质量指标测试
        quality_results = [r for r in results if r.test_name.startswith("quality_")]
        
        assert len(quality_results) > 0, "应该有质量指标测试"
        
        # 均匀图像测试应该通过
        uniform_test = next((r for r in quality_results if "uniform" in r.test_name), None)
        if uniform_test:
            assert uniform_test.passed, f"均匀图像质量指标测试失败: {uniform_test.error_message}"
            
            # 检查具体指标值
            if uniform_test.actual_values:
                assert uniform_test.actual_values.get("perceptual_distortion", 1.0) < 1e-6
                assert uniform_test.actual_values.get("local_contrast", 1.0) < 1e-6
                assert uniform_test.actual_values.get("variance_distortion", 1.0) < 1e-6
                
    def test_golden_standard_temporal_smoothing(self):
        """测试时域平滑金标测试"""
        golden_tests = GoldenStandardTests()
        results = golden_tests.run_golden_tests()
        
        # 筛选时域平滑测试
        temporal_results = [r for r in results if r.test_name.startswith("temporal_")]
        
        assert len(temporal_results) > 0, "应该有时域平滑测试"
        
        # 恒定参数测试应该通过
        constant_test = next((r for r in temporal_results if "constant" in r.test_name), None)
        if constant_test:
            assert constant_test.passed, f"恒定参数时域平滑测试失败: {constant_test.error_message}"
            
    def test_monotonicity_stress_comprehensive(self):
        """测试全面的单调性压力测试"""
        mono_tests = MonotonicityStressTests()
        results = mono_tests.run_monotonicity_stress_tests()
        
        assert len(results) > 0, "应该有单调性压力测试结果"
        
        # 检查Phoenix单调性压力测试
        phoenix_stress = next((r for r in results if "phoenix_monotonicity_stress" in r.test_name), None)
        assert phoenix_stress is not None, "应该有Phoenix单调性压力测试"
        
        if not phoenix_stress.passed:
            print(f"Phoenix单调性压力测试失败: {phoenix_stress.error_message}")
            if phoenix_stress.actual_values and "failed_combinations" in phoenix_stress.actual_values:
                failed = phoenix_stress.actual_values["failed_combinations"]
                print(f"失败的参数组合: {failed[:5]}...")  # 只显示前5个
                
        # 检查样条单调性压力测试
        spline_stress = next((r for r in results if "spline_monotonicity_stress" in r.test_name), None)
        if spline_stress:
            if not spline_stress.passed:
                print(f"样条单调性压力测试失败: {spline_stress.error_message}")
                
        # 检查极端参数测试
        extreme_results = [r for r in results if "extreme_monotonicity" in r.test_name]
        extreme_passed = sum(1 for r in extreme_results if r.passed)
        extreme_total = len(extreme_results)
        
        if extreme_total > 0:
            extreme_pass_rate = extreme_passed / extreme_total
            print(f"极端参数单调性测试通过率: {extreme_passed}/{extreme_total} ({extreme_pass_rate*100:.1f}%)")
            
            # 至少70%的极端参数测试应该通过
            assert extreme_pass_rate >= 0.7, f"极端参数单调性测试通过率过低: {extreme_pass_rate*100:.1f}%"
            
    def test_hysteresis_stability_comprehensive(self):
        """测试全面的滞回稳定性测试"""
        hysteresis_tests = HysteresisStabilityTests()
        results = hysteresis_tests.run_hysteresis_stability_tests()
        
        assert len(results) > 0, "应该有滞回稳定性测试结果"
        
        # 基本滞回测试必须通过
        basic_test = next((r for r in results if "basic_hysteresis" in r.test_name), None)
        assert basic_test is not None, "应该有基本滞回测试"
        assert basic_test.passed, f"基本滞回测试失败: {basic_test.error_message}"
        
        # 边界振荡测试
        oscillation_test = next((r for r in results if "boundary_oscillation" in r.test_name), None)
        if oscillation_test:
            if not oscillation_test.passed:
                print(f"边界振荡测试失败: {oscillation_test.error_message}")
            else:
                print("边界振荡测试通过")
                
        # 长期稳定性测试
        stability_test = next((r for r in results if "long_term_stability" in r.test_name), None)
        if stability_test:
            if not stability_test.passed:
                print(f"长期稳定性测试失败: {stability_test.error_message}")
            else:
                print("长期稳定性测试通过")
                
        # 至少基本测试必须通过
        critical_tests = [basic_test]
        critical_passed = sum(1 for t in critical_tests if t and t.passed)
        assert critical_passed == len(critical_tests), "关键滞回测试必须全部通过"
        
    def test_export_import_consistency_comprehensive(self):
        """测试全面的导出/导入一致性测试"""
        consistency_tests = ExportImportConsistencyTests()
        results = consistency_tests.run_consistency_tests()
        
        assert len(results) > 0, "应该有一致性测试结果"
        
        # LUT一致性测试
        lut_test = next((r for r in results if "lut_consistency" in r.test_name), None)
        assert lut_test is not None, "应该有LUT一致性测试"
        
        if not lut_test.passed:
            print(f"LUT一致性测试失败: {lut_test.error_message}")
            if lut_test.actual_values:
                max_error = lut_test.actual_values.get("max_error", "未知")
                print(f"LUT最大误差: {max_error}")
        else:
            print("LUT一致性测试通过")
            
        # CSV一致性测试
        csv_test = next((r for r in results if "csv_consistency" in r.test_name), None)
        if csv_test:
            if not csv_test.passed:
                print(f"CSV一致性测试失败: {csv_test.error_message}")
            else:
                print("CSV一致性测试通过")
                
        # 诊断包完整性测试
        diagnostic_test = next((r for r in results if "diagnostic_package" in r.test_name), None)
        if diagnostic_test:
            if not diagnostic_test.passed:
                print(f"诊断包完整性测试失败: {diagnostic_test.error_message}")
            else:
                print("诊断包完整性测试通过")
                
        # 会话状态一致性测试
        session_test = next((r for r in results if "session_state" in r.test_name), None)
        if session_test:
            if not session_test.passed:
                print(f"会话状态一致性测试失败: {session_test.error_message}")
            else:
                print("会话状态一致性测试通过")
                
        # 统计通过率
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        pass_rate = passed_count / total_count
        
        print(f"导出/导入一致性测试通过率: {passed_count}/{total_count} ({pass_rate*100:.1f}%)")
        
        # 至少90%的一致性测试应该通过
        assert pass_rate >= 0.9, f"导出/导入一致性测试通过率过低: {pass_rate*100:.1f}%"
        
    def test_full_regression_suite_execution(self):
        """测试完整回归测试套件执行"""
        results = self.test_suite.run_full_regression_suite()
        
        # 检查结果结构完整性
        required_keys = [
            "test_summary",
            "golden_standard_tests", 
            "monotonicity_stress_tests",
            "hysteresis_stability_tests",
            "export_import_consistency_tests",
            "detailed_failures"
        ]
        
        for key in required_keys:
            assert key in results, f"结果中缺少必需的键: {key}"
            
        # 检查测试摘要
        summary = results["test_summary"]
        assert "total_tests" in summary
        assert "passed_tests" in summary
        assert "failed_tests" in summary
        assert "start_time" in summary
        assert "end_time" in summary
        
        # 验证测试数量一致性
        total_from_categories = (
            len(results["golden_standard_tests"]) +
            len(results["monotonicity_stress_tests"]) +
            len(results["hysteresis_stability_tests"]) +
            len(results["export_import_consistency_tests"])
        )
        
        assert summary["total_tests"] == total_from_categories, "测试总数不一致"
        assert summary["total_tests"] == summary["passed_tests"] + summary["failed_tests"], "通过和失败测试数量不匹配"
        
        # 检查是否有测试执行
        assert summary["total_tests"] > 0, "应该有测试执行"
        
        # 打印测试摘要
        print(f"\n=== 回归测试套件执行摘要 ===")
        print(f"总测试数: {summary['total_tests']}")
        print(f"通过: {summary['passed_tests']}")
        print(f"失败: {summary['failed_tests']}")
        print(f"通过率: {summary['passed_tests']/summary['total_tests']*100:.1f}%")
        
        # 如果有失败，显示详细信息
        if summary["failed_tests"] > 0:
            print(f"\n=== 失败测试详情 ===")
            for failure in results["detailed_failures"][:5]:  # 只显示前5个
                print(f"- {failure['test_name']}: {failure['error_message']}")
                
        # 各类测试通过率
        categories = [
            ("golden_standard_tests", "金标测试"),
            ("monotonicity_stress_tests", "单调性压力测试"),
            ("hysteresis_stability_tests", "滞回稳定性测试"),
            ("export_import_consistency_tests", "导出/导入一致性测试")
        ]
        
        print(f"\n=== 各类测试通过率 ===")
        for category_key, category_name in categories:
            category_results = results[category_key]
            if category_results:
                passed = sum(1 for r in category_results if r.passed)
                total = len(category_results)
                rate = passed / total * 100
                print(f"{category_name}: {passed}/{total} ({rate:.1f}%)")
                
    def test_report_generation_and_content(self):
        """测试报告生成和内容"""
        results = self.test_suite.run_full_regression_suite()
        
        # 生成报告
        report_file = os.path.join(self.temp_dir, "test_report.md")
        report_content = self.test_suite.generate_test_report(results, report_file)
        
        # 检查报告文件生成
        assert os.path.exists(report_file), "报告文件应该被创建"
        
        # 检查报告内容
        assert "HDR色调映射专利可视化工具验证报告" in report_content
        assert "测试摘要" in report_content
        assert "通过率" in report_content
        
        # 检查各类测试在报告中
        test_categories = ["金标测试用例", "单调性压力测试", "滞回稳定性测试", "导出/导入一致性测试"]
        for category in test_categories:
            assert category in report_content, f"报告中应该包含{category}"
            
        # 检查报告文件内容与返回内容一致
        with open(report_file, 'r', encoding='utf-8') as f:
            file_content = f.read()
            
        assert file_content == report_content, "文件内容应该与返回内容一致"
        
        print(f"测试报告已生成: {report_file}")
        print(f"报告长度: {len(report_content)} 字符")
        
    def test_performance_benchmarks(self):
        """测试性能基准"""
        import time
        
        # 测试单个Phoenix曲线计算性能
        from core import PhoenixCurveCalculator
        
        calc = PhoenixCurveCalculator()
        L = np.linspace(0, 1, 1000)
        
        start_time = time.time()
        for _ in range(100):  # 100次计算
            calc.compute_phoenix_curve(L, 2.0, 0.5)
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) / 100 * 1000
        print(f"Phoenix曲线计算平均时间: {avg_time_ms:.2f} ms")
        
        # 性能要求: 单次计算应该在10ms以内
        assert avg_time_ms < 10.0, f"Phoenix曲线计算性能不达标: {avg_time_ms:.2f} ms > 10 ms"
        
        # 测试质量指标计算性能
        from core import QualityMetricsCalculator
        
        quality_calc = QualityMetricsCalculator()
        test_image = np.random.rand(256, 256).astype(np.float32)
        L_in = quality_calc.extract_luminance(test_image)
        L_out = L_in ** 2.0
        
        start_time = time.time()
        for _ in range(50):  # 50次计算
            quality_calc.compute_all_metrics(L_in, L_out)
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) / 50 * 1000
        print(f"质量指标计算平均时间: {avg_time_ms:.2f} ms")
        
        # 性能要求: 质量指标计算应该在50ms以内
        assert avg_time_ms < 50.0, f"质量指标计算性能不达标: {avg_time_ms:.2f} ms > 50 ms"
        
    def test_memory_usage_validation(self):
        """测试内存使用验证"""
        import psutil
        import gc
        
        # 获取初始内存使用
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行大量计算
        from core import PhoenixCurveCalculator, QualityMetricsCalculator
        
        calc = PhoenixCurveCalculator()
        quality_calc = QualityMetricsCalculator()
        
        # 创建大量数据
        large_arrays = []
        for i in range(10):
            L = np.linspace(0, 1, 10000)
            L_out = calc.compute_phoenix_curve(L, 2.0 + i * 0.1, 0.5)
            large_arrays.append(L_out)
            
            # 创建大图像
            large_image = np.random.rand(512, 512).astype(np.float32)
            L_in = quality_calc.extract_luminance(large_image)
            quality_calc.compute_all_metrics(L_in, L_out[:512*512])
            
        # 检查内存使用
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"初始内存: {initial_memory:.1f} MB")
        print(f"峰值内存: {peak_memory:.1f} MB")
        print(f"内存增长: {memory_increase:.1f} MB")
        
        # 清理内存
        del large_arrays
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_recovered = peak_memory - final_memory
        
        print(f"清理后内存: {final_memory:.1f} MB")
        print(f"回收内存: {memory_recovered:.1f} MB")
        
        # 内存使用应该合理 (增长不超过500MB)
        assert memory_increase < 500, f"内存使用过多: {memory_increase:.1f} MB"
        
        # 内存应该能够回收 (至少回收50%)
        recovery_rate = memory_recovered / memory_increase if memory_increase > 0 else 1.0
        assert recovery_rate > 0.5, f"内存回收不足: {recovery_rate*100:.1f}%"
        
    def test_numerical_stability_validation(self):
        """测试数值稳定性验证"""
        from core import PhoenixCurveCalculator, SafeCalculator
        
        calc = PhoenixCurveCalculator()
        safe_calc = SafeCalculator()
        
        # 测试极端输入值
        extreme_inputs = [
            np.array([0.0, 1e-10, 1e-8, 1e-6, 0.5, 1.0-1e-6, 1.0-1e-8, 1.0-1e-10, 1.0]),
            np.array([1e-15, 1e-12, 1e-9, 0.1, 0.9, 1.0-1e-9, 1.0-1e-12, 1.0-1e-15]),
        ]
        
        extreme_params = [
            (0.1, 0.0),    # 最小p, 最小a
            (6.0, 1.0),    # 最大p, 最大a
            (0.1, 1.0),    # 最小p, 最大a
            (6.0, 0.0),    # 最大p, 最小a
        ]
        
        for L_input in extreme_inputs:
            for p, a in extreme_params:
                try:
                    # 直接计算
                    L_out = calc.compute_phoenix_curve(L_input, p, a)
                    
                    # 检查结果有效性
                    assert np.all(np.isfinite(L_out)), f"结果包含非有限值: p={p}, a={a}"
                    assert np.all(L_out >= 0), f"结果包含负值: p={p}, a={a}"
                    assert np.all(L_out <= 1), f"结果超出范围: p={p}, a={a}"
                    
                    # 检查单调性
                    is_monotonic = calc.validate_monotonicity(L_out)
                    if not is_monotonic:
                        print(f"警告: 极端参数({p}, {a})产生非单调曲线")
                        
                    # 安全计算验证
                    L_safe, success, msg = safe_calc.safe_phoenix_calculation(L_input, p, a)
                    assert np.all(np.isfinite(L_safe)), f"安全计算结果包含非有限值: p={p}, a={a}"
                    
                except Exception as e:
                    pytest.fail(f"数值稳定性测试失败: p={p}, a={a}, 错误: {e}")
                    
        print("数值稳定性验证通过")
        
    def test_edge_cases_comprehensive(self):
        """测试全面的边界情况"""
        from core import (PhoenixCurveCalculator, QualityMetricsCalculator, 
                         TemporalSmoothingProcessor, SplineCurveCalculator)
        
        # Phoenix曲线边界情况
        calc = PhoenixCurveCalculator()
        
        # 空数组
        empty_array = np.array([])
        try:
            result = calc.compute_phoenix_curve(empty_array, 2.0, 0.5)
            assert len(result) == 0, "空数组应该返回空结果"
        except Exception:
            pass  # 允许抛出异常
            
        # 单点数组
        single_point = np.array([0.5])
        result = calc.compute_phoenix_curve(single_point, 2.0, 0.5)
        assert len(result) == 1, "单点数组应该返回单点结果"
        assert np.isfinite(result[0]), "单点结果应该是有限值"
        
        # 质量指标边界情况
        quality_calc = QualityMetricsCalculator()
        
        # 单像素图像
        single_pixel = np.array([[0.5]])
        L_in = quality_calc.extract_luminance(single_pixel)
        L_out = L_in.copy()
        
        distortion = quality_calc.compute_perceptual_distortion(L_in, L_out)
        assert np.isfinite(distortion), "单像素失真应该是有限值"
        
        contrast = quality_calc.compute_local_contrast(L_out)
        assert np.isfinite(contrast), "单像素对比度应该是有限值"
        
        # 时域平滑边界情况
        temporal_proc = TemporalSmoothingProcessor(window_size=5)
        
        # 空历史
        smoothed = temporal_proc.compute_weighted_average()
        assert smoothed == {}, "空历史应该返回空字典"
        
        # 单帧历史
        temporal_proc.add_frame_parameters({"p": 2.0, "a": 0.5}, 0.1)
        smoothed = temporal_proc.compute_weighted_average()
        assert "p" in smoothed, "单帧历史应该返回参数"
        assert abs(smoothed["p"] - 2.0) < 1e-10, "单帧平滑应该等于原值"
        
        print("边界情况测试通过")


def run_comprehensive_regression_tests():
    """运行全面的回归测试"""
    print("开始运行全面回归测试...")
    
    # 创建测试实例
    test_instance = TestRegressionSuite()
    test_instance.setup_class()
    
    try:
        # 运行各项测试
        test_methods = [
            "test_golden_standard_phoenix_curves",
            "test_golden_standard_quality_metrics", 
            "test_golden_standard_temporal_smoothing",
            "test_monotonicity_stress_comprehensive",
            "test_hysteresis_stability_comprehensive",
            "test_export_import_consistency_comprehensive",
            "test_full_regression_suite_execution",
            "test_report_generation_and_content",
            "test_performance_benchmarks",
            "test_memory_usage_validation",
            "test_numerical_stability_validation",
            "test_edge_cases_comprehensive"
        ]
        
        passed_tests = 0
        failed_tests = 0
        
        for method_name in test_methods:
            try:
                print(f"\n运行测试: {method_name}")
                method = getattr(test_instance, method_name)
                method()
                print(f"✅ {method_name} 通过")
                passed_tests += 1
            except Exception as e:
                print(f"❌ {method_name} 失败: {e}")
                failed_tests += 1
                
        # 输出总结
        total_tests = passed_tests + failed_tests
        pass_rate = passed_tests / total_tests * 100 if total_tests > 0 else 0
        
        print(f"\n=== 全面回归测试总结 ===")
        print(f"总测试方法: {total_tests}")
        print(f"通过: {passed_tests}")
        print(f"失败: {failed_tests}")
        print(f"通过率: {pass_rate:.1f}%")
        
        return pass_rate >= 80  # 80%通过率为合格
        
    finally:
        test_instance.teardown_class()


if __name__ == "__main__":
    # 运行全面回归测试
    success = run_comprehensive_regression_tests()
    
    if success:
        print("\n🎉 全面回归测试通过!")
        exit(0)
    else:
        print("\n❌ 全面回归测试失败!")
        exit(1)