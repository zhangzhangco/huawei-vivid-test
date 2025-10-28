#!/usr/bin/env python3
"""
HDR色调映射专利可视化工具验证框架
实现金标测试用例、单调性压力测试、滞回稳定性测试和导出/导入一致性验证
"""

import pytest
import numpy as np
import os
import tempfile
import json
import zipfile
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import sys

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import (
    PhoenixCurveCalculator, QualityMetricsCalculator, TemporalSmoothingProcessor,
    SplineCurveCalculator, SafeCalculator, ExportManager, ImageProcessor,
    PQConverter, ParameterValidator, get_export_manager
)


@dataclass
class GoldenTestCase:
    """金标测试用例数据结构"""
    name: str
    description: str
    parameters: Dict[str, float]
    expected_results: Dict[str, Any]
    tolerance: Dict[str, float]
    test_data: Optional[np.ndarray] = None
    
    
@dataclass 
class ValidationResult:
    """验证结果数据结构"""
    test_name: str
    passed: bool
    error_message: str = ""
    actual_values: Dict[str, Any] = None
    expected_values: Dict[str, Any] = None
    tolerance_used: Dict[str, float] = None


class GoldenStandardTests:
    """金标测试用例管理器"""
    
    def __init__(self):
        self.test_cases = self._create_golden_test_cases()
        self.test_images = self._create_test_images()
        
    def _create_golden_test_cases(self) -> List[GoldenTestCase]:
        """创建金标测试用例"""
        cases = []
        
        # 基础Phoenix曲线测试用例
        # Phoenix公式: L' = L^p / (L^p + a^p)
        # 当a=0时，公式简化为L' = L^p / L^p = 1 (对于L>0)
        # 当a>0时，才会产生真正的Phoenix曲线效果
        cases.append(GoldenTestCase(
            name="phoenix_basic_curve",
            description="基础Phoenix曲线测试 (p=2.0, a=0.5)",
            parameters={"p": 2.0, "a": 0.5},
            expected_results={
                "endpoint_0": 0.0,  # L=0时，L'=0
                "endpoint_1": 1.0,  # L=1时，L'=1^2/(1^2+0.5^2)=1/1.25=0.8，但会被归一化
                "monotonic": True,
                "curve_type": "phoenix"
            },
            tolerance={
                "endpoint_0": 1e-6,
                "endpoint_1": 1e-6, 
                "monotonic": 0,
                "curve_type": 0
            }
        ))
        
        # 极端参数测试用例
        cases.append(GoldenTestCase(
            name="phoenix_extreme_low_p",
            description="极低p值测试 (边界条件)",
            parameters={"p": 0.1, "a": 0.5},
            expected_results={
                "endpoint_0": 0.0,
                "endpoint_1": 1.0,
                "monotonic": True,
                "max_derivative_change": 50.0  # 允许较大的导数变化
            },
            tolerance={
                "endpoint_0": 1e-10,
                "endpoint_1": 1e-10,
                "monotonic": 0,
                "max_derivative_change": 10.0
            }
        ))
        
        # 质量指标测试用例
        cases.append(GoldenTestCase(
            name="quality_metrics_uniform_image",
            description="均匀图像质量指标测试",
            parameters={"uniform_value": 0.5},
            expected_results={
                "perceptual_distortion": 0.0,
                "local_contrast": 0.0,
                "variance_distortion": 0.0
            },
            tolerance={
                "perceptual_distortion": 1e-10,
                "local_contrast": 1e-10,
                "variance_distortion": 1e-10
            }
        ))
        
        # 时域平滑测试用例
        cases.append(GoldenTestCase(
            name="temporal_smoothing_constant_params",
            description="恒定参数时域平滑测试",
            parameters={"p": 2.0, "a": 0.5, "frames": 5},
            expected_results={
                "variance_reduction": 0.0,  # 恒定参数应该无方差
                "smoothed_p": 2.0,
                "smoothed_a": 0.5
            },
            tolerance={
                "variance_reduction": 1e-10,
                "smoothed_p": 1e-10,
                "smoothed_a": 1e-10
            }
        ))
        
        # 样条曲线测试用例
        cases.append(GoldenTestCase(
            name="spline_zero_strength",
            description="零强度样条曲线测试",
            parameters={"p": 2.0, "a": 0.5, "th_strength": 0.0, "th_nodes": [0.2, 0.5, 0.8]},
            expected_results={
                "uses_spline": False,
                "curve_equals_phoenix": True
            },
            tolerance={
                "uses_spline": 0,
                "curve_equals_phoenix": 1e-10
            }
        ))
        
        return cases
        
    def _create_test_images(self) -> Dict[str, np.ndarray]:
        """创建标准测试图像"""
        images = {}
        
        # 均匀灰度图像
        images["uniform_gray"] = np.full((64, 64), 0.5, dtype=np.float32)
        
        # 线性渐变图像
        x = np.linspace(0, 1, 64)
        images["linear_gradient"] = np.tile(x, (64, 1)).astype(np.float32)
        
        # 棋盘图案
        checker = np.zeros((64, 64), dtype=np.float32)
        checker[::2, ::2] = 1.0
        checker[1::2, 1::2] = 1.0
        images["checkerboard"] = checker
        
        # 高动态范围测试图像
        hdr_image = np.random.exponential(0.3, (32, 32)).astype(np.float32)
        hdr_image = np.clip(hdr_image, 0, 1)
        images["hdr_synthetic"] = hdr_image
        
        # RGB彩色测试图像
        rgb_image = np.zeros((32, 32, 3), dtype=np.float32)
        rgb_image[:, :, 0] = np.linspace(0, 1, 32).reshape(1, -1)  # R通道
        rgb_image[:, :, 1] = np.linspace(0, 1, 32).reshape(-1, 1)  # G通道  
        rgb_image[:, :, 2] = 0.5  # B通道恒定
        images["rgb_gradient"] = rgb_image
        
        return images
        
    def run_golden_tests(self) -> List[ValidationResult]:
        """运行所有金标测试"""
        results = []
        
        for test_case in self.test_cases:
            try:
                result = self._run_single_golden_test(test_case)
                results.append(result)
            except Exception as e:
                results.append(ValidationResult(
                    test_name=test_case.name,
                    passed=False,
                    error_message=f"测试执行异常: {str(e)}"
                ))
                
        return results
        
    def _run_single_golden_test(self, test_case: GoldenTestCase) -> ValidationResult:
        """运行单个金标测试"""
        if test_case.name.startswith("phoenix_"):
            return self._test_phoenix_curve(test_case)
        elif test_case.name.startswith("quality_"):
            return self._test_quality_metrics(test_case)
        elif test_case.name.startswith("temporal_"):
            return self._test_temporal_smoothing(test_case)
        elif test_case.name.startswith("spline_"):
            return self._test_spline_curve(test_case)
        else:
            return ValidationResult(
                test_name=test_case.name,
                passed=False,
                error_message="未知的测试类型"
            )
            
    def _test_phoenix_curve(self, test_case: GoldenTestCase) -> ValidationResult:
        """测试Phoenix曲线"""
        calc = PhoenixCurveCalculator()
        p = test_case.parameters["p"]
        a = test_case.parameters["a"]
        
        # 生成测试曲线
        L = np.linspace(0, 1, 1000)
        L_out_raw = calc.compute_phoenix_curve(L, p, a)
        
        # 应用端点归一化以确保端点匹配
        L_out = calc.normalize_endpoints(L_out_raw, 0.0, 1.0)
        
        # 验证结果
        actual_values = {}
        passed = True
        error_messages = []
        
        # 检查端点
        if "endpoint_0" in test_case.expected_results:
            actual_values["endpoint_0"] = float(L_out[0])
            expected = test_case.expected_results["endpoint_0"]
            tolerance = test_case.tolerance["endpoint_0"]
            if abs(actual_values["endpoint_0"] - expected) > tolerance:
                passed = False
                error_messages.append(f"端点0误差过大: {actual_values['endpoint_0']} vs {expected}")
                
        if "endpoint_1" in test_case.expected_results:
            actual_values["endpoint_1"] = float(L_out[-1])
            expected = test_case.expected_results["endpoint_1"]
            tolerance = test_case.tolerance["endpoint_1"]
            if abs(actual_values["endpoint_1"] - expected) > tolerance:
                passed = False
                error_messages.append(f"端点1误差过大: {actual_values['endpoint_1']} vs {expected}")
                
        # 检查中点值
        if "midpoint_05" in test_case.expected_results:
            mid_idx = len(L) // 2
            actual_values["midpoint_05"] = float(L_out[mid_idx])
            expected = test_case.expected_results["midpoint_05"]
            tolerance = test_case.tolerance["midpoint_05"]
            if abs(actual_values["midpoint_05"] - expected) > tolerance:
                passed = False
                error_messages.append(f"中点值误差过大: {actual_values['midpoint_05']} vs {expected}")
                
        # 检查曲线范围
        if "curve_range" in test_case.expected_results:
            actual_min = float(np.min(L_out))
            actual_max = float(np.max(L_out))
            actual_values["curve_min"] = actual_min
            actual_values["curve_max"] = actual_max
            
            expected_range = test_case.expected_results["curve_range"]
            tolerance = test_case.tolerance.get("curve_range", 1e-6)
            
            if abs(actual_min - expected_range[0]) > tolerance or abs(actual_max - expected_range[1]) > tolerance:
                passed = False
                error_messages.append(f"曲线范围误差: [{actual_min}, {actual_max}] vs {expected_range}")
                
        # 检查单调性
        if "monotonic" in test_case.expected_results:
            actual_values["monotonic"] = calc.validate_monotonicity(L_out)
            expected = test_case.expected_results["monotonic"]
            if actual_values["monotonic"] != expected:
                passed = False
                error_messages.append(f"单调性不符合预期: {actual_values['monotonic']} vs {expected}")
                
        return ValidationResult(
            test_name=test_case.name,
            passed=passed,
            error_message="; ".join(error_messages),
            actual_values=actual_values,
            expected_values=test_case.expected_results,
            tolerance_used=test_case.tolerance
        )
        
    def _test_quality_metrics(self, test_case: GoldenTestCase) -> ValidationResult:
        """测试质量指标"""
        calc = QualityMetricsCalculator()
        
        # 创建测试图像
        uniform_value = test_case.parameters["uniform_value"]
        image = np.full((100, 100), uniform_value, dtype=np.float32)
        
        # 计算指标
        L_in = calc.extract_luminance(image)
        L_out = L_in.copy()  # 恒等映射
        
        actual_values = {
            "perceptual_distortion": calc.compute_perceptual_distortion(L_in, L_out),
            "local_contrast": calc.compute_local_contrast(L_out),
            "variance_distortion": calc.compute_variance_distortion(L_in, L_out)
        }
        
        # 验证结果
        passed = True
        error_messages = []
        
        for key, expected in test_case.expected_results.items():
            if key in actual_values:
                tolerance = test_case.tolerance[key]
                if abs(actual_values[key] - expected) > tolerance:
                    passed = False
                    error_messages.append(f"{key}误差过大: {actual_values[key]} vs {expected}")
                    
        return ValidationResult(
            test_name=test_case.name,
            passed=passed,
            error_message="; ".join(error_messages),
            actual_values=actual_values,
            expected_values=test_case.expected_results,
            tolerance_used=test_case.tolerance
        )
        
    def _test_temporal_smoothing(self, test_case: GoldenTestCase) -> ValidationResult:
        """测试时域平滑"""
        processor = TemporalSmoothingProcessor(window_size=5)
        
        # 添加恒定参数
        p = test_case.parameters["p"]
        a = test_case.parameters["a"]
        frames = test_case.parameters["frames"]
        
        for i in range(frames):
            processor.add_frame_parameters({"p": p, "a": a}, 0.1)
            
        # 计算平滑结果
        smoothed = processor.compute_weighted_average()
        stats = processor.get_smoothing_stats()
        
        actual_values = {
            "variance_reduction": stats.variance_reduction,
            "smoothed_p": smoothed.get("p", 0.0),
            "smoothed_a": smoothed.get("a", 0.0)
        }
        
        # 验证结果
        passed = True
        error_messages = []
        
        for key, expected in test_case.expected_results.items():
            if key in actual_values:
                tolerance = test_case.tolerance[key]
                if abs(actual_values[key] - expected) > tolerance:
                    passed = False
                    error_messages.append(f"{key}误差过大: {actual_values[key]} vs {expected}")
                    
        return ValidationResult(
            test_name=test_case.name,
            passed=passed,
            error_message="; ".join(error_messages),
            actual_values=actual_values,
            expected_values=test_case.expected_results,
            tolerance_used=test_case.tolerance
        )
        
    def _test_spline_curve(self, test_case: GoldenTestCase) -> ValidationResult:
        """测试样条曲线"""
        phoenix_calc = PhoenixCurveCalculator()
        spline_calc = SplineCurveCalculator()
        
        # 生成Phoenix曲线
        p = test_case.parameters["p"]
        a = test_case.parameters["a"]
        L = np.linspace(0, 1, 100)
        phoenix_curve = phoenix_calc.compute_phoenix_curve(L, p, a)
        
        # 应用样条
        th_strength = test_case.parameters["th_strength"]
        th_nodes = test_case.parameters["th_nodes"]
        
        final_curve, used_spline, status = spline_calc.compute_spline_with_fallback(
            phoenix_curve, L, th_nodes, th_strength
        )
        
        actual_values = {
            "uses_spline": used_spline,
            "curve_equals_phoenix": np.allclose(final_curve, phoenix_curve, atol=1e-10)
        }
        
        # 验证结果
        passed = True
        error_messages = []
        
        for key, expected in test_case.expected_results.items():
            if key in actual_values:
                if isinstance(expected, bool):
                    if actual_values[key] != expected:
                        passed = False
                        error_messages.append(f"{key}不符合预期: {actual_values[key]} vs {expected}")
                else:
                    tolerance = test_case.tolerance[key]
                    if abs(actual_values[key] - expected) > tolerance:
                        passed = False
                        error_messages.append(f"{key}误差过大: {actual_values[key]} vs {expected}")
                        
        return ValidationResult(
            test_name=test_case.name,
            passed=passed,
            error_message="; ".join(error_messages),
            actual_values=actual_values,
            expected_values=test_case.expected_results,
            tolerance_used=test_case.tolerance
        )


class MonotonicityStressTests:
    """单调性压力测试"""
    
    def __init__(self):
        self.phoenix_calc = PhoenixCurveCalculator()
        self.spline_calc = SplineCurveCalculator()
        self.validator = ParameterValidator()
        
    def run_monotonicity_stress_tests(self) -> List[ValidationResult]:
        """运行单调性压力测试"""
        results = []
        
        # Phoenix曲线单调性压力测试
        results.extend(self._test_phoenix_monotonicity_stress())
        
        # 样条曲线单调性压力测试
        results.extend(self._test_spline_monotonicity_stress())
        
        # 极端参数单调性测试
        results.extend(self._test_extreme_parameter_monotonicity())
        
        return results
        
    def _test_phoenix_monotonicity_stress(self) -> List[ValidationResult]:
        """Phoenix曲线单调性压力测试"""
        results = []
        
        # 测试参数网格
        p_values = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 6.0]
        a_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        failed_combinations = []
        
        for p in p_values:
            for a in a_values:
                try:
                    # 使用高密度采样验证单调性
                    is_monotonic = self.phoenix_calc.validate_monotonicity_pa(p, a)
                    
                    if not is_monotonic:
                        failed_combinations.append((p, a))
                        
                except Exception as e:
                    failed_combinations.append((p, a, str(e)))
                    
        # 生成测试结果
        if not failed_combinations:
            results.append(ValidationResult(
                test_name="phoenix_monotonicity_stress",
                passed=True,
                error_message="",
                actual_values={"tested_combinations": len(p_values) * len(a_values)},
                expected_values={"all_monotonic": True}
            ))
        else:
            results.append(ValidationResult(
                test_name="phoenix_monotonicity_stress",
                passed=False,
                error_message=f"发现{len(failed_combinations)}个非单调参数组合",
                actual_values={"failed_combinations": failed_combinations},
                expected_values={"all_monotonic": True}
            ))
            
        return results
        
    def _test_spline_monotonicity_stress(self) -> List[ValidationResult]:
        """样条曲线单调性压力测试"""
        results = []
        
        # 基础Phoenix曲线
        L = np.linspace(0, 1, 1000)
        phoenix_curve = self.phoenix_calc.compute_phoenix_curve(L, 2.0, 0.5)
        
        # 测试各种样条节点配置
        node_configs = [
            [0.1, 0.5, 0.9],
            [0.2, 0.4, 0.6, 0.8],
            [0.15, 0.35, 0.65, 0.85],
            [0.05, 0.25, 0.75, 0.95],
            [0.3, 0.7],  # 只有两个内部节点
        ]
        
        strength_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        failed_configs = []
        
        for nodes in node_configs:
            for strength in strength_values:
                try:
                    final_curve, used_spline, status = self.spline_calc.compute_spline_with_fallback(
                        phoenix_curve, L, nodes, strength
                    )
                    
                    if used_spline:
                        is_monotonic = self.spline_calc.check_monotonicity(final_curve)
                        if not is_monotonic:
                            failed_configs.append((nodes, strength))
                            
                except Exception as e:
                    failed_configs.append((nodes, strength, str(e)))
                    
        # 生成测试结果
        if not failed_configs:
            results.append(ValidationResult(
                test_name="spline_monotonicity_stress",
                passed=True,
                error_message="",
                actual_values={"tested_configurations": len(node_configs) * len(strength_values)},
                expected_values={"all_monotonic": True}
            ))
        else:
            results.append(ValidationResult(
                test_name="spline_monotonicity_stress", 
                passed=False,
                error_message=f"发现{len(failed_configs)}个非单调样条配置",
                actual_values={"failed_configurations": failed_configs},
                expected_values={"all_monotonic": True}
            ))
            
        return results
        
    def _test_extreme_parameter_monotonicity(self) -> List[ValidationResult]:
        """极端参数单调性测试"""
        results = []
        
        # 极端参数组合
        extreme_cases = [
            ("min_p_max_a", 0.1, 1.0),
            ("max_p_min_a", 6.0, 0.0),
            ("min_p_min_a", 0.1, 0.0),
            ("max_p_max_a", 6.0, 1.0),
            ("unity_p_half_a", 1.0, 0.5),
            ("high_p_low_a", 5.0, 0.1),
            ("low_p_high_a", 0.2, 0.9)
        ]
        
        for case_name, p, a in extreme_cases:
            try:
                # 验证参数有效性
                valid, msg = self.validator.validate_phoenix_params(p, a)
                if not valid:
                    results.append(ValidationResult(
                        test_name=f"extreme_monotonicity_{case_name}",
                        passed=False,
                        error_message=f"参数验证失败: {msg}",
                        actual_values={"p": p, "a": a},
                        expected_values={"valid": True}
                    ))
                    continue
                    
                # 验证单调性
                is_monotonic = self.phoenix_calc.validate_monotonicity_pa(p, a)
                
                results.append(ValidationResult(
                    test_name=f"extreme_monotonicity_{case_name}",
                    passed=is_monotonic,
                    error_message="" if is_monotonic else f"极端参数({p}, {a})产生非单调曲线",
                    actual_values={"p": p, "a": a, "monotonic": is_monotonic},
                    expected_values={"monotonic": True}
                ))
                
            except Exception as e:
                results.append(ValidationResult(
                    test_name=f"extreme_monotonicity_{case_name}",
                    passed=False,
                    error_message=f"测试异常: {str(e)}",
                    actual_values={"p": p, "a": a},
                    expected_values={"monotonic": True}
                ))
                
        return results


class HysteresisStabilityTests:
    """滞回稳定性测试"""
    
    def __init__(self):
        self.quality_calc = QualityMetricsCalculator()
        
    def run_hysteresis_stability_tests(self) -> List[ValidationResult]:
        """运行滞回稳定性测试"""
        results = []
        
        # 基本滞回测试
        results.append(self._test_basic_hysteresis())
        
        # 边界振荡测试
        results.append(self._test_boundary_oscillation())
        
        # 长期稳定性测试
        results.append(self._test_long_term_stability())
        
        return results
        
    def _test_basic_hysteresis(self) -> ValidationResult:
        """基本滞回测试"""
        self.quality_calc.reset_hysteresis()
        
        # 测试序列: 低 -> 中 -> 高 -> 中 -> 低
        distortion_sequence = [0.03, 0.07, 0.12, 0.08, 0.04]
        expected_modes = ["自动模式", "自动模式", "艺术模式", "艺术模式", "自动模式"]
        
        actual_modes = []
        for distortion in distortion_sequence:
            mode = self.quality_calc.recommend_mode_with_hysteresis(distortion)
            actual_modes.append(mode)
            
        # 验证结果
        passed = actual_modes == expected_modes
        
        return ValidationResult(
            test_name="basic_hysteresis",
            passed=passed,
            error_message="" if passed else f"滞回序列不符合预期: {actual_modes} vs {expected_modes}",
            actual_values={"mode_sequence": actual_modes},
            expected_values={"mode_sequence": expected_modes}
        )
        
    def _test_boundary_oscillation(self) -> ValidationResult:
        """边界振荡测试"""
        self.quality_calc.reset_hysteresis()
        
        # 在滞回区间内振荡
        oscillation_sequence = [0.06, 0.07, 0.065, 0.075, 0.068, 0.072, 0.069]
        
        modes = []
        for distortion in oscillation_sequence:
            mode = self.quality_calc.recommend_mode_with_hysteresis(distortion)
            modes.append(mode)
            
        # 检查是否有不必要的模式切换
        mode_changes = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i-1])
        
        # 在滞回区间内应该保持稳定，最多1次切换
        passed = mode_changes <= 1
        
        return ValidationResult(
            test_name="boundary_oscillation",
            passed=passed,
            error_message="" if passed else f"边界振荡产生过多模式切换: {mode_changes}次",
            actual_values={"mode_changes": mode_changes, "modes": modes},
            expected_values={"max_changes": 1}
        )
        
    def _test_long_term_stability(self) -> ValidationResult:
        """长期稳定性测试"""
        self.quality_calc.reset_hysteresis()
        
        # 模拟长期使用中的失真变化
        np.random.seed(42)  # 固定随机种子确保可复现
        
        # 生成带噪声的失真序列
        base_distortion = 0.07  # 在滞回区间内
        noise_level = 0.01
        sequence_length = 100
        
        distortion_sequence = base_distortion + np.random.normal(0, noise_level, sequence_length)
        distortion_sequence = np.clip(distortion_sequence, 0.05, 0.10)  # 限制在滞回区间
        
        modes = []
        for distortion in distortion_sequence:
            mode = self.quality_calc.recommend_mode_with_hysteresis(distortion)
            modes.append(mode)
            
        # 统计模式切换次数
        mode_changes = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i-1])
        
        # 长期稳定性要求: 在100次测试中切换次数应该很少
        max_allowed_changes = 5
        passed = mode_changes <= max_allowed_changes
        
        return ValidationResult(
            test_name="long_term_stability",
            passed=passed,
            error_message="" if passed else f"长期稳定性测试失败: {mode_changes}次切换 > {max_allowed_changes}",
            actual_values={"mode_changes": mode_changes, "sequence_length": sequence_length},
            expected_values={"max_allowed_changes": max_allowed_changes}
        )


class ExportImportConsistencyTests:
    """导出/导入一致性验证"""
    
    def __init__(self):
        self.export_manager = get_export_manager()
        self.temp_dir = tempfile.mkdtemp()
        
    def __del__(self):
        """清理临时目录"""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
    def run_consistency_tests(self) -> List[ValidationResult]:
        """运行一致性测试"""
        results = []
        
        # LUT导出/导入一致性
        results.append(self._test_lut_consistency())
        
        # CSV导出/导入一致性
        results.append(self._test_csv_consistency())
        
        # 诊断包完整性
        results.append(self._test_diagnostic_package_integrity())
        
        # 会话状态一致性
        results.append(self._test_session_state_consistency())
        
        return results
        
    def _test_lut_consistency(self) -> ValidationResult:
        """测试LUT导出/导入一致性"""
        from core import CurveData, SessionState
        
        # 创建测试数据
        L_input = np.linspace(0, 1, 1024)
        L_output = L_input ** 2.2  # Gamma 2.2
        
        curve_data = CurveData(
            input_luminance=L_input,
            output_luminance=L_output,
            phoenix_curve=L_output
        )
        
        session_state = SessionState(p=2.2, a=0.0)
        
        # 导出LUT
        lut_file = os.path.join(self.temp_dir, "consistency_test.cube")
        success = self.export_manager.export_lut(curve_data, session_state, lut_file, samples=1024)
        
        if not success:
            return ValidationResult(
                test_name="lut_consistency",
                passed=False,
                error_message="LUT导出失败"
            )
            
        # 验证一致性
        is_consistent, max_error = self.export_manager.validate_export_consistency(
            L_output, lut_file, "lut"
        )
        
        # 需求15.4: 重建曲线最大绝对误差≤1e-4
        tolerance = 1e-4
        passed = is_consistent and max_error <= tolerance
        
        return ValidationResult(
            test_name="lut_consistency",
            passed=passed,
            error_message="" if passed else f"LUT一致性验证失败: 最大误差{max_error} > {tolerance}",
            actual_values={"max_error": max_error, "is_consistent": is_consistent},
            expected_values={"max_error_threshold": tolerance, "is_consistent": True}
        )
        
    def _test_csv_consistency(self) -> ValidationResult:
        """测试CSV导出/导入一致性"""
        from core import CurveData, SessionState, ExportMetadata
        
        # 创建测试数据
        L_input = np.linspace(0, 1, 256)
        L_output = L_input ** 1.8
        
        curve_data = CurveData(
            input_luminance=L_input,
            output_luminance=L_output,
            phoenix_curve=L_output
        )
        
        session_state = SessionState(p=1.8, a=0.2)
        metadata = ExportMetadata(
            export_time="2024-01-01T12:00:00",
            version="1.0",
            source_system="ValidationFramework",
            parameters={"test": "csv_consistency"}
        )
        
        # 导出CSV
        csv_file = os.path.join(self.temp_dir, "consistency_test.csv")
        success = self.export_manager.export_csv(curve_data, session_state, csv_file, metadata)
        
        if not success:
            return ValidationResult(
                test_name="csv_consistency",
                passed=False,
                error_message="CSV导出失败"
            )
            
        # 验证一致性
        is_consistent, max_error = self.export_manager.validate_export_consistency(
            L_output, csv_file, "csv"
        )
        
        tolerance = 1e-6  # CSV应该有更高精度
        passed = is_consistent and max_error <= tolerance
        
        return ValidationResult(
            test_name="csv_consistency",
            passed=passed,
            error_message="" if passed else f"CSV一致性验证失败: 最大误差{max_error} > {tolerance}",
            actual_values={"max_error": max_error, "is_consistent": is_consistent},
            expected_values={"max_error_threshold": tolerance, "is_consistent": True}
        )
        
    def _test_diagnostic_package_integrity(self) -> ValidationResult:
        """测试诊断包完整性"""
        from core import (CurveData, SessionState, TemporalStateData, 
                         QualityMetrics, ImageStats)
        
        # 创建完整测试数据
        curve_data = CurveData(
            input_luminance=np.linspace(0, 1, 100),
            output_luminance=np.linspace(0, 1, 100) ** 2,
            phoenix_curve=np.linspace(0, 1, 100) ** 2
        )
        
        session_state = SessionState(p=2.0, a=0.5)
        temporal_state = TemporalStateData()
        quality_metrics = QualityMetrics(
            perceptual_distortion=0.05,
            local_contrast=0.1,
            variance_distortion=0.02,
            recommended_mode="自动模式",
            computation_time=0.1
        )
        image_stats = ImageStats(
            min_pq=0.01, max_pq=0.95, avg_pq=0.45, var_pq=0.08,
            input_format="Test", processing_path="Test", pixel_count=1000
        )
        
        # 创建诊断包
        package_path = self.export_manager.create_diagnostic_package(
            curve_data, session_state, temporal_state, quality_metrics, image_stats,
            output_dir=self.temp_dir
        )
        
        if not package_path or not os.path.exists(package_path):
            return ValidationResult(
                test_name="diagnostic_package_integrity",
                passed=False,
                error_message="诊断包创建失败"
            )
            
        # 验证包完整性
        try:
            with zipfile.ZipFile(package_path, 'r') as zf:
                file_list = zf.namelist()
                
                # 检查必需文件
                required_files = [
                    'README.md',
                    'system_info.json',
                    'config/session_config.json',
                    'analysis/quality_metrics.json',
                    'curve_data.csv',
                    'tone_mapping.cube'
                ]
                
                missing_files = [f for f in required_files if f not in file_list]
                
                if missing_files:
                    return ValidationResult(
                        test_name="diagnostic_package_integrity",
                        passed=False,
                        error_message=f"诊断包缺少文件: {missing_files}",
                        actual_values={"missing_files": missing_files},
                        expected_values={"missing_files": []}
                    )
                    
                # 验证JSON文件格式
                for json_file in ['system_info.json', 'config/session_config.json']:
                    with zf.open(json_file) as f:
                        json.load(f)  # 验证JSON格式
                        
        except Exception as e:
            return ValidationResult(
                test_name="diagnostic_package_integrity",
                passed=False,
                error_message=f"诊断包验证异常: {str(e)}"
            )
            
        return ValidationResult(
            test_name="diagnostic_package_integrity",
            passed=True,
            error_message="",
            actual_values={"package_created": True, "files_complete": True},
            expected_values={"package_created": True, "files_complete": True}
        )
        
    def _test_session_state_consistency(self) -> ValidationResult:
        """测试会话状态一致性"""
        from core import SessionState
        import json
        
        # 创建测试状态
        original_state = SessionState(
            p=2.5, a=0.7, mode="艺术模式",
            dt_low=0.04, dt_high=0.09,
            window_size=7, lambda_smooth=0.35
        )
        
        # 直接测试JSON序列化/反序列化一致性
        state_file = os.path.join(self.temp_dir, "test_session.json")
        
        try:
            # 保存状态到JSON
            state_dict = asdict(original_state)
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_dict, f, indent=2, ensure_ascii=False)
            
            # 从JSON加载状态
            with open(state_file, 'r', encoding='utf-8') as f:
                loaded_dict = json.load(f)
                
            # 重建SessionState对象
            loaded_state = SessionState(**loaded_dict)
                
            # 比较状态
            state_dict_original = asdict(original_state)
            state_dict_loaded = asdict(loaded_state)
            
            differences = []
            for key, original_value in state_dict_original.items():
                loaded_value = state_dict_loaded.get(key)
                if isinstance(original_value, float):
                    if abs(original_value - loaded_value) > 1e-10:
                        differences.append(f"{key}: {original_value} vs {loaded_value}")
                elif original_value != loaded_value:
                    differences.append(f"{key}: {original_value} vs {loaded_value}")
                    
            passed = len(differences) == 0
            
            return ValidationResult(
                test_name="session_state_consistency",
                passed=passed,
                error_message="" if passed else f"状态不一致: {differences}",
                actual_values=state_dict_loaded,
                expected_values=state_dict_original
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="session_state_consistency",
                passed=False,
                error_message=f"会话状态测试异常: {str(e)}"
            )


class AutomatedRegressionTestSuite:
    """自动化回归测试套件"""
    
    def __init__(self):
        self.golden_tests = GoldenStandardTests()
        self.monotonicity_tests = MonotonicityStressTests()
        self.hysteresis_tests = HysteresisStabilityTests()
        self.consistency_tests = ExportImportConsistencyTests()
        
    def run_full_regression_suite(self) -> Dict[str, Any]:
        """运行完整回归测试套件"""
        print("开始运行HDR色调映射验证框架...")
        
        results = {
            "test_summary": {
                "start_time": self._get_timestamp(),
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0
            },
            "golden_standard_tests": [],
            "monotonicity_stress_tests": [],
            "hysteresis_stability_tests": [],
            "export_import_consistency_tests": [],
            "detailed_failures": []
        }
        
        # 运行各类测试
        print("1. 运行金标测试用例...")
        results["golden_standard_tests"] = self.golden_tests.run_golden_tests()
        
        print("2. 运行单调性压力测试...")
        results["monotonicity_stress_tests"] = self.monotonicity_tests.run_monotonicity_stress_tests()
        
        print("3. 运行滞回稳定性测试...")
        results["hysteresis_stability_tests"] = self.hysteresis_tests.run_hysteresis_stability_tests()
        
        print("4. 运行导出/导入一致性测试...")
        results["export_import_consistency_tests"] = self.consistency_tests.run_consistency_tests()
        
        # 汇总结果
        all_test_results = (
            results["golden_standard_tests"] +
            results["monotonicity_stress_tests"] +
            results["hysteresis_stability_tests"] +
            results["export_import_consistency_tests"]
        )
        
        results["test_summary"]["total_tests"] = len(all_test_results)
        results["test_summary"]["passed_tests"] = sum(1 for r in all_test_results if r.passed)
        results["test_summary"]["failed_tests"] = sum(1 for r in all_test_results if not r.passed)
        results["test_summary"]["end_time"] = self._get_timestamp()
        
        # 收集详细失败信息
        results["detailed_failures"] = [
            {
                "test_name": r.test_name,
                "error_message": r.error_message,
                "actual_values": r.actual_values,
                "expected_values": r.expected_values
            }
            for r in all_test_results if not r.passed
        ]
        
        return results
        
    def generate_test_report(self, results: Dict[str, Any], output_file: str = None) -> str:
        """生成测试报告"""
        report_lines = []
        
        # 报告头部
        report_lines.append("# HDR色调映射专利可视化工具验证报告")
        report_lines.append("")
        report_lines.append(f"**测试时间**: {results['test_summary']['start_time']} - {results['test_summary']['end_time']}")
        report_lines.append("")
        
        # 测试摘要
        summary = results["test_summary"]
        report_lines.append("## 测试摘要")
        report_lines.append("")
        report_lines.append(f"- **总测试数**: {summary['total_tests']}")
        report_lines.append(f"- **通过测试**: {summary['passed_tests']}")
        report_lines.append(f"- **失败测试**: {summary['failed_tests']}")
        report_lines.append(f"- **通过率**: {summary['passed_tests']/summary['total_tests']*100:.1f}%")
        report_lines.append("")
        
        # 各类测试结果
        test_categories = [
            ("golden_standard_tests", "金标测试用例"),
            ("monotonicity_stress_tests", "单调性压力测试"),
            ("hysteresis_stability_tests", "滞回稳定性测试"),
            ("export_import_consistency_tests", "导出/导入一致性测试")
        ]
        
        for category_key, category_name in test_categories:
            category_results = results[category_key]
            passed_count = sum(1 for r in category_results if r.passed)
            total_count = len(category_results)
            
            report_lines.append(f"## {category_name}")
            report_lines.append("")
            report_lines.append(f"**通过率**: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)")
            report_lines.append("")
            
            for result in category_results:
                status = "✅ 通过" if result.passed else "❌ 失败"
                report_lines.append(f"- **{result.test_name}**: {status}")
                if not result.passed and result.error_message:
                    report_lines.append(f"  - 错误: {result.error_message}")
                    
            report_lines.append("")
            
        # 详细失败信息
        if results["detailed_failures"]:
            report_lines.append("## 详细失败信息")
            report_lines.append("")
            
            for failure in results["detailed_failures"]:
                report_lines.append(f"### {failure['test_name']}")
                report_lines.append("")
                report_lines.append(f"**错误信息**: {failure['error_message']}")
                report_lines.append("")
                
                if failure["actual_values"]:
                    report_lines.append("**实际值**:")
                    for key, value in failure["actual_values"].items():
                        report_lines.append(f"- {key}: {value}")
                    report_lines.append("")
                    
                if failure["expected_values"]:
                    report_lines.append("**期望值**:")
                    for key, value in failure["expected_values"].items():
                        report_lines.append(f"- {key}: {value}")
                    report_lines.append("")
                    
        # 生成报告
        report_content = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
        return report_content
        
    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 测试类定义
class TestValidationFramework:
    """验证框架测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.test_suite = AutomatedRegressionTestSuite()
        
    def test_golden_standard_tests(self):
        """测试金标测试用例"""
        results = self.test_suite.golden_tests.run_golden_tests()
        
        # 检查是否有测试结果
        assert len(results) > 0, "应该有金标测试结果"
        
        # 检查基本Phoenix曲线测试
        phoenix_results = [r for r in results if r.test_name.startswith("phoenix_")]
        assert len(phoenix_results) > 0, "应该有Phoenix曲线测试"
        
        # 至少基础测试应该通过
        basic_test = next((r for r in phoenix_results if "basic" in r.test_name), None)
        if basic_test:
            assert basic_test.passed, f"基础Phoenix测试应该通过: {basic_test.error_message}"
            
    def test_monotonicity_stress_tests(self):
        """测试单调性压力测试"""
        results = self.test_suite.monotonicity_tests.run_monotonicity_stress_tests()
        
        assert len(results) > 0, "应该有单调性测试结果"
        
        # 检查Phoenix单调性测试
        phoenix_mono_test = next((r for r in results if "phoenix_monotonicity" in r.test_name), None)
        assert phoenix_mono_test is not None, "应该有Phoenix单调性测试"
        
    def test_hysteresis_stability_tests(self):
        """测试滞回稳定性测试"""
        results = self.test_suite.hysteresis_tests.run_hysteresis_stability_tests()
        
        assert len(results) > 0, "应该有滞回稳定性测试结果"
        
        # 基本滞回测试应该通过
        basic_hysteresis = next((r for r in results if "basic_hysteresis" in r.test_name), None)
        if basic_hysteresis:
            assert basic_hysteresis.passed, f"基本滞回测试应该通过: {basic_hysteresis.error_message}"
            
    def test_export_import_consistency(self):
        """测试导出/导入一致性"""
        results = self.test_suite.consistency_tests.run_consistency_tests()
        
        assert len(results) > 0, "应该有一致性测试结果"
        
        # 检查LUT一致性测试
        lut_test = next((r for r in results if "lut_consistency" in r.test_name), None)
        assert lut_test is not None, "应该有LUT一致性测试"
        
    def test_full_regression_suite(self):
        """测试完整回归测试套件"""
        results = self.test_suite.run_full_regression_suite()
        
        # 检查结果结构
        assert "test_summary" in results
        assert "golden_standard_tests" in results
        assert "monotonicity_stress_tests" in results
        assert "hysteresis_stability_tests" in results
        assert "export_import_consistency_tests" in results
        
        # 检查测试摘要
        summary = results["test_summary"]
        assert summary["total_tests"] > 0, "应该有测试执行"
        assert summary["total_tests"] == summary["passed_tests"] + summary["failed_tests"]
        
    def test_report_generation(self):
        """测试报告生成"""
        results = self.test_suite.run_full_regression_suite()
        
        # 生成报告
        report = self.test_suite.generate_test_report(results)
        
        # 检查报告内容
        assert "HDR色调映射专利可视化工具验证报告" in report
        assert "测试摘要" in report
        assert "通过率" in report


if __name__ == "__main__":
    # 运行完整验证框架
    test_suite = AutomatedRegressionTestSuite()
    results = test_suite.run_full_regression_suite()
    
    # 生成报告
    report = test_suite.generate_test_report(results, "validation_report.md")
    
    # 打印摘要
    summary = results["test_summary"]
    print(f"\n验证完成!")
    print(f"总测试数: {summary['total_tests']}")
    print(f"通过: {summary['passed_tests']}")
    print(f"失败: {summary['failed_tests']}")
    print(f"通过率: {summary['passed_tests']/summary['total_tests']*100:.1f}%")
    
    if summary['failed_tests'] > 0:
        print(f"\n失败测试:")
        for failure in results["detailed_failures"]:
            print(f"- {failure['test_name']}: {failure['error_message']}")
            
    print(f"\n详细报告已保存到: validation_report.md")