"""
边界条件检查器
提供全面的参数边界条件检查和约束处理
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from enum import Enum

from .parameter_validator import ParameterValidator


class BoundaryViolationType(Enum):
    """边界违反类型"""
    RANGE_VIOLATION = "range_violation"
    CONSTRAINT_VIOLATION = "constraint_violation"
    DEPENDENCY_VIOLATION = "dependency_violation"
    NUMERICAL_LIMIT = "numerical_limit"
    LOGICAL_INCONSISTENCY = "logical_inconsistency"


@dataclass
class BoundaryViolation:
    """边界违反记录"""
    parameter: str
    violation_type: BoundaryViolationType
    current_value: Any
    expected_range: Optional[Tuple[float, float]] = None
    constraint_description: str = ""
    severity: str = "error"  # "error", "warning", "info"
    auto_correctable: bool = True
    suggested_value: Optional[Any] = None


class BoundaryChecker:
    """边界条件检查器"""
    
    def __init__(self):
        self.validator = ParameterValidator()
        self.logger = logging.getLogger(__name__)
        
        # 扩展的参数约束定义
        self.parameter_constraints = {
            # Phoenix曲线参数
            'p': {
                'range': (0.1, 6.0),
                'type': (int, float),
                'description': '亮度控制因子',
                'critical_values': [0.0, float('inf')],
                'recommended_range': (0.5, 4.0),
                'stability_range': (0.2, 5.0)
            },
            'a': {
                'range': (0.0, 1.0),
                'type': (int, float),
                'description': '缩放因子',
                'critical_values': [float('inf')],
                'recommended_range': (0.1, 0.8),
                'stability_range': (0.0, 1.0)
            },
            
            # 显示设备参数
            'min_display_pq': {
                'range': (0.0, 1.0),
                'type': (int, float),
                'description': '显示设备最小PQ值',
                'critical_values': [float('inf'), float('-inf')],
                'recommended_range': (0.0, 0.1)
            },
            'max_display_pq': {
                'range': (0.0, 1.0),
                'type': (int, float),
                'description': '显示设备最大PQ值',
                'critical_values': [float('inf'), float('-inf')],
                'recommended_range': (0.9, 1.0)
            },
            
            # 质量指标参数
            'dt_low': {
                'range': (0.01, 0.20),
                'type': (int, float),
                'description': '失真下阈值',
                'recommended_range': (0.03, 0.08)
            },
            'dt_high': {
                'range': (0.01, 0.20),
                'type': (int, float),
                'description': '失真上阈值',
                'recommended_range': (0.08, 0.15)
            },
            
            # 时域平滑参数
            'window_size': {
                'range': (3, 20),
                'type': int,
                'description': '时域窗口大小',
                'recommended_range': (5, 15)
            },
            'lambda_smooth': {
                'range': (0.1, 0.8),
                'type': (int, float),
                'description': '平滑强度',
                'recommended_range': (0.2, 0.5)
            },
            
            # 样条曲线参数
            'th_strength': {
                'range': (0.0, 1.0),
                'type': (int, float),
                'description': '样条强度',
                'recommended_range': (0.0, 0.7)
            },
            'th1': {
                'range': (0.0, 1.0),
                'type': (int, float),
                'description': '样条节点1',
                'recommended_range': (0.1, 0.4)
            },
            'th2': {
                'range': (0.0, 1.0),
                'type': (int, float),
                'description': '样条节点2',
                'recommended_range': (0.4, 0.6)
            },
            'th3': {
                'range': (0.0, 1.0),
                'type': (int, float),
                'description': '样条节点3',
                'recommended_range': (0.6, 0.9)
            },
            
            # 系统参数
            'display_samples': {
                'range': (64, 4096),
                'type': int,
                'description': '显示采样点数',
                'recommended_range': (256, 1024)
            },
            'validation_samples': {
                'range': (256, 8192),
                'type': int,
                'description': '验证采样点数',
                'recommended_range': (512, 2048)
            }
        }
        
        # 参数依赖关系
        self.parameter_dependencies = {
            ('min_display_pq', 'max_display_pq'): self._check_display_range_dependency,
            ('dt_low', 'dt_high'): self._check_threshold_dependency,
            ('th1', 'th2', 'th3'): self._check_spline_nodes_dependency,
            ('p', 'a'): self._check_phoenix_stability_dependency,
            ('display_samples', 'validation_samples'): self._check_sampling_dependency
        }
        
        # 数值稳定性检查
        self.numerical_limits = {
            'epsilon': 1e-8,
            'max_finite': 1e10,
            'min_positive': 1e-10
        }
        
    def check_all_boundaries(self, parameters: Dict[str, Any]) -> Tuple[bool, List[BoundaryViolation]]:
        """检查所有边界条件"""
        violations = []
        
        # 1. 基本范围检查
        range_violations = self._check_parameter_ranges(parameters)
        violations.extend(range_violations)
        
        # 2. 类型检查
        type_violations = self._check_parameter_types(parameters)
        violations.extend(type_violations)
        
        # 3. 数值稳定性检查
        numerical_violations = self._check_numerical_stability(parameters)
        violations.extend(numerical_violations)
        
        # 4. 依赖关系检查
        dependency_violations = self._check_parameter_dependencies(parameters)
        violations.extend(dependency_violations)
        
        # 5. 逻辑一致性检查
        logic_violations = self._check_logical_consistency(parameters)
        violations.extend(logic_violations)
        
        # 6. 特殊值检查
        special_violations = self._check_special_values(parameters)
        violations.extend(special_violations)
        
        is_valid = len([v for v in violations if v.severity == "error"]) == 0
        
        return is_valid, violations
        
    def _check_parameter_ranges(self, parameters: Dict[str, Any]) -> List[BoundaryViolation]:
        """检查参数范围"""
        violations = []
        
        for param_name, value in parameters.items():
            if param_name not in self.parameter_constraints:
                continue
                
            constraint = self.parameter_constraints[param_name]
            param_range = constraint['range']
            
            if not isinstance(value, (int, float)):
                continue
                
            if not (param_range[0] <= value <= param_range[1]):
                # 确定建议值
                suggested_value = np.clip(value, param_range[0], param_range[1])
                
                # 超出基本范围是错误，超出推荐范围是警告
                severity = "error"
                
                violations.append(BoundaryViolation(
                    parameter=param_name,
                    violation_type=BoundaryViolationType.RANGE_VIOLATION,
                    current_value=value,
                    expected_range=param_range,
                    constraint_description=f"{constraint['description']}必须在范围{param_range}内",
                    severity=severity,
                    auto_correctable=True,
                    suggested_value=suggested_value
                ))
                
        return violations
        
    def _check_parameter_types(self, parameters: Dict[str, Any]) -> List[BoundaryViolation]:
        """检查参数类型"""
        violations = []
        
        for param_name, value in parameters.items():
            if param_name not in self.parameter_constraints:
                continue
                
            constraint = self.parameter_constraints[param_name]
            expected_types = constraint['type']
            
            if not isinstance(value, expected_types):
                violations.append(BoundaryViolation(
                    parameter=param_name,
                    violation_type=BoundaryViolationType.CONSTRAINT_VIOLATION,
                    current_value=value,
                    constraint_description=f"{constraint['description']}必须为{expected_types}类型",
                    severity="error",
                    auto_correctable=True,
                    suggested_value=float(value) if expected_types == (int, float) else int(value)
                ))
                
        return violations
        
    def _check_numerical_stability(self, parameters: Dict[str, Any]) -> List[BoundaryViolation]:
        """检查数值稳定性"""
        violations = []
        
        for param_name, value in parameters.items():
            if not isinstance(value, (int, float)):
                continue
                
            # 检查NaN和无穷大
            if np.isnan(value) or np.isinf(value):
                violations.append(BoundaryViolation(
                    parameter=param_name,
                    violation_type=BoundaryViolationType.NUMERICAL_LIMIT,
                    current_value=value,
                    constraint_description="参数值不能为NaN或无穷大",
                    severity="error",
                    auto_correctable=True,
                    suggested_value=self.validator.DEFAULT_PARAMS.get(param_name, 1.0)
                ))
                
            # 检查极端值
            elif abs(value) > self.numerical_limits['max_finite']:
                violations.append(BoundaryViolation(
                    parameter=param_name,
                    violation_type=BoundaryViolationType.NUMERICAL_LIMIT,
                    current_value=value,
                    constraint_description=f"参数值过大，可能导致数值不稳定",
                    severity="warning",
                    auto_correctable=True,
                    suggested_value=np.sign(value) * self.numerical_limits['max_finite'] / 10
                ))
                
            # 检查接近零的正值参数
            elif param_name in ['a', 'dt_low', 'dt_high', 'lambda_smooth'] and 0 < value < self.numerical_limits['min_positive']:
                violations.append(BoundaryViolation(
                    parameter=param_name,
                    violation_type=BoundaryViolationType.NUMERICAL_LIMIT,
                    current_value=value,
                    constraint_description="参数值过小，可能导致数值不稳定",
                    severity="warning",
                    auto_correctable=True,
                    suggested_value=self.numerical_limits['min_positive']
                ))
                
        return violations
        
    def _check_parameter_dependencies(self, parameters: Dict[str, Any]) -> List[BoundaryViolation]:
        """检查参数依赖关系"""
        violations = []
        
        for param_group, check_func in self.parameter_dependencies.items():
            # 检查所有相关参数是否存在
            if all(param in parameters for param in param_group):
                param_values = [parameters[param] for param in param_group]
                dependency_violations = check_func(param_group, param_values)
                violations.extend(dependency_violations)
                
        return violations
        
    def _check_display_range_dependency(self, param_names: Tuple[str, ...], 
                                      values: List[Any]) -> List[BoundaryViolation]:
        """检查显示范围依赖关系"""
        violations = []
        min_pq, max_pq = values
        
        if not isinstance(min_pq, (int, float)) or not isinstance(max_pq, (int, float)):
            return violations
            
        if max_pq <= min_pq:
            violations.append(BoundaryViolation(
                parameter="max_display_pq",
                violation_type=BoundaryViolationType.DEPENDENCY_VIOLATION,
                current_value=max_pq,
                constraint_description=f"最大PQ值({max_pq})必须大于最小PQ值({min_pq})",
                severity="error",
                auto_correctable=True,
                suggested_value=min_pq + 0.1
            ))
            
        # 检查范围是否过小
        if max_pq - min_pq < 0.01:
            violations.append(BoundaryViolation(
                parameter="display_range",
                violation_type=BoundaryViolationType.DEPENDENCY_VIOLATION,
                current_value=(min_pq, max_pq),
                constraint_description="显示范围过小，可能导致映射效果不明显",
                severity="warning",
                auto_correctable=True,
                suggested_value=(min_pq, min_pq + 0.1)
            ))
            
        return violations
        
    def _check_threshold_dependency(self, param_names: Tuple[str, ...], 
                                  values: List[Any]) -> List[BoundaryViolation]:
        """检查阈值依赖关系"""
        violations = []
        dt_low, dt_high = values
        
        if not isinstance(dt_low, (int, float)) or not isinstance(dt_high, (int, float)):
            return violations
            
        if dt_high <= dt_low:
            violations.append(BoundaryViolation(
                parameter="dt_high",
                violation_type=BoundaryViolationType.DEPENDENCY_VIOLATION,
                current_value=dt_high,
                constraint_description=f"上阈值({dt_high})必须大于下阈值({dt_low})",
                severity="error",
                auto_correctable=True,
                suggested_value=dt_low + 0.02
            ))
            
        # 检查阈值间隔是否合理
        if dt_high - dt_low < 0.01:
            violations.append(BoundaryViolation(
                parameter="threshold_interval",
                violation_type=BoundaryViolationType.DEPENDENCY_VIOLATION,
                current_value=(dt_low, dt_high),
                constraint_description="阈值间隔过小，滞回效果可能不明显",
                severity="warning",
                auto_correctable=True
            ))
            
        return violations
        
    def _check_spline_nodes_dependency(self, param_names: Tuple[str, ...], 
                                     values: List[Any]) -> List[BoundaryViolation]:
        """检查样条节点依赖关系"""
        violations = []
        th1, th2, th3 = values
        
        if not all(isinstance(v, (int, float)) for v in values):
            return violations
            
        nodes = [th1, th2, th3]
        sorted_nodes = sorted(nodes)
        
        # 检查节点顺序
        if nodes != sorted_nodes:
            violations.append(BoundaryViolation(
                parameter="spline_nodes",
                violation_type=BoundaryViolationType.DEPENDENCY_VIOLATION,
                current_value=nodes,
                constraint_description="样条节点必须按升序排列",
                severity="warning",
                auto_correctable=True,
                suggested_value=sorted_nodes
            ))
            
        # 检查节点间隔
        min_interval = 0.01
        for i in range(1, len(sorted_nodes)):
            if sorted_nodes[i] - sorted_nodes[i-1] < min_interval:
                violations.append(BoundaryViolation(
                    parameter=f"th{i+1}",
                    violation_type=BoundaryViolationType.DEPENDENCY_VIOLATION,
                    current_value=sorted_nodes[i],
                    constraint_description=f"节点间隔过小，最小间隔应为{min_interval}",
                    severity="warning",
                    auto_correctable=True,
                    suggested_value=sorted_nodes[i-1] + min_interval
                ))
                
        return violations
        
    def _check_phoenix_stability_dependency(self, param_names: Tuple[str, ...], 
                                          values: List[Any]) -> List[BoundaryViolation]:
        """检查Phoenix参数稳定性依赖关系"""
        violations = []
        p, a = values
        
        if not isinstance(p, (int, float)) or not isinstance(a, (int, float)):
            return violations
            
        # 检查极端参数组合
        if p > 4.0 and a < 0.1:
            violations.append(BoundaryViolation(
                parameter="phoenix_combination",
                violation_type=BoundaryViolationType.DEPENDENCY_VIOLATION,
                current_value=(p, a),
                constraint_description="高p值配合低a值可能导致数值不稳定",
                severity="warning",
                auto_correctable=True,
                suggested_value=(min(p, 3.0), max(a, 0.2))
            ))
            
        # 检查可能导致非单调的组合
        if p > 5.0 or (p > 3.0 and a > 0.8):
            violations.append(BoundaryViolation(
                parameter="phoenix_monotonicity",
                violation_type=BoundaryViolationType.DEPENDENCY_VIOLATION,
                current_value=(p, a),
                constraint_description="参数组合可能导致曲线非单调",
                severity="warning",
                auto_correctable=True
            ))
            
        return violations
        
    def _check_sampling_dependency(self, param_names: Tuple[str, ...], 
                                 values: List[Any]) -> List[BoundaryViolation]:
        """检查采样参数依赖关系"""
        violations = []
        display_samples, validation_samples = values
        
        if not isinstance(display_samples, int) or not isinstance(validation_samples, int):
            return violations
            
        # 验证采样数应该大于等于显示采样数
        if validation_samples < display_samples:
            violations.append(BoundaryViolation(
                parameter="validation_samples",
                violation_type=BoundaryViolationType.DEPENDENCY_VIOLATION,
                current_value=validation_samples,
                constraint_description=f"验证采样数({validation_samples})应大于等于显示采样数({display_samples})",
                severity="warning",
                auto_correctable=True,
                suggested_value=max(display_samples, 1024)
            ))
            
        return violations
        
    def _check_logical_consistency(self, parameters: Dict[str, Any]) -> List[BoundaryViolation]:
        """检查逻辑一致性"""
        violations = []
        
        # 检查样条强度与节点的一致性
        th_strength = parameters.get('th_strength', 0.0)
        if th_strength > 0:
            # 如果启用样条，检查节点是否合理
            nodes = [parameters.get('th1'), parameters.get('th2'), parameters.get('th3')]
            if any(node is None for node in nodes):
                violations.append(BoundaryViolation(
                    parameter="spline_configuration",
                    violation_type=BoundaryViolationType.LOGICAL_INCONSISTENCY,
                    current_value=th_strength,
                    constraint_description="启用样条曲线时必须设置所有节点",
                    severity="error",
                    auto_correctable=True,
                    suggested_value=0.0
                ))
                
        # 检查模式与参数的一致性
        mode = parameters.get('mode', '艺术模式')
        if mode == '自动模式':
            # 自动模式下某些参数应该被忽略
            manual_params = ['p', 'a']
            for param in manual_params:
                if param in parameters:
                    violations.append(BoundaryViolation(
                        parameter=param,
                        violation_type=BoundaryViolationType.LOGICAL_INCONSISTENCY,
                        current_value=parameters[param],
                        constraint_description=f"自动模式下参数{param}将被自动计算覆盖",
                        severity="info",
                        auto_correctable=False
                    ))
                    
        return violations
        
    def _check_special_values(self, parameters: Dict[str, Any]) -> List[BoundaryViolation]:
        """检查特殊值"""
        violations = []
        
        for param_name, value in parameters.items():
            if param_name not in self.parameter_constraints:
                continue
                
            constraint = self.parameter_constraints[param_name]
            critical_values = constraint.get('critical_values', [])
            
            if not isinstance(value, (int, float)):
                continue
                
            # 检查临界值
            for critical_value in critical_values:
                if np.isclose(value, critical_value, rtol=1e-10):
                    violations.append(BoundaryViolation(
                        parameter=param_name,
                        violation_type=BoundaryViolationType.NUMERICAL_LIMIT,
                        current_value=value,
                        constraint_description=f"参数值接近临界值{critical_value}，可能导致计算问题",
                        severity="warning",
                        auto_correctable=True,
                        suggested_value=self.validator.DEFAULT_PARAMS.get(param_name, 1.0)
                    ))
                    
        return violations
        
    def auto_correct_violations(self, parameters: Dict[str, Any], 
                              violations: List[BoundaryViolation]) -> Dict[str, Any]:
        """自动修正违反条件"""
        corrected_params = parameters.copy()
        
        for violation in violations:
            if not violation.auto_correctable or violation.severity == "info":
                continue
                
            if violation.suggested_value is not None:
                if violation.parameter in corrected_params:
                    corrected_params[violation.parameter] = violation.suggested_value
                    self.logger.info(f"自动修正参数 {violation.parameter}: {violation.current_value} -> {violation.suggested_value}")
                    
        return corrected_params
        
    def get_violation_summary(self, violations: List[BoundaryViolation]) -> Dict[str, Any]:
        """获取违反条件摘要"""
        summary = {
            'total_violations': len(violations),
            'error_count': len([v for v in violations if v.severity == "error"]),
            'warning_count': len([v for v in violations if v.severity == "warning"]),
            'info_count': len([v for v in violations if v.severity == "info"]),
            'auto_correctable_count': len([v for v in violations if v.auto_correctable]),
            'violation_types': {}
        }
        
        # 按类型统计
        for violation in violations:
            vtype = violation.violation_type.value
            if vtype not in summary['violation_types']:
                summary['violation_types'][vtype] = 0
            summary['violation_types'][vtype] += 1
            
        return summary
        
    def create_violation_report(self, violations: List[BoundaryViolation]) -> str:
        """创建违反条件报告"""
        if not violations:
            return "✅ 所有边界条件检查通过"
            
        report = "# 边界条件检查报告\n\n"
        
        # 按严重程度分组
        errors = [v for v in violations if v.severity == "error"]
        warnings = [v for v in violations if v.severity == "warning"]
        infos = [v for v in violations if v.severity == "info"]
        
        if errors:
            report += "## ❌ 错误 (必须修正)\n"
            for violation in errors:
                report += f"- **{violation.parameter}**: {violation.constraint_description}\n"
                report += f"  当前值: {violation.current_value}\n"
                if violation.suggested_value is not None:
                    report += f"  建议值: {violation.suggested_value}\n"
                report += "\n"
                
        if warnings:
            report += "## ⚠️ 警告 (建议修正)\n"
            for violation in warnings:
                report += f"- **{violation.parameter}**: {violation.constraint_description}\n"
                report += f"  当前值: {violation.current_value}\n"
                if violation.suggested_value is not None:
                    report += f"  建议值: {violation.suggested_value}\n"
                report += "\n"
                
        if infos:
            report += "## ℹ️ 信息\n"
            for violation in infos:
                report += f"- **{violation.parameter}**: {violation.constraint_description}\n"
                report += f"  当前值: {violation.current_value}\n"
                report += "\n"
                
        return report