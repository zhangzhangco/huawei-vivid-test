"""
安全计算器
提供带错误处理和自动回退的安全计算功能
"""

import numpy as np
import logging
from typing import Tuple, Union, Dict, Any, List, Optional

from .phoenix_calculator import PhoenixCurveCalculator
from .parameter_validator import ParameterValidator
from .spline_calculator import SplineCurveCalculator, SplineCalculationError
from .ui_error_handler import UIErrorHandler, ErrorSeverity
from .error_recovery import ErrorRecoverySystem
from .boundary_checker import BoundaryChecker


class CalculationError(Exception):
    """计算错误基类"""
    pass


class ParameterRangeError(CalculationError):
    """参数范围错误"""
    pass


class MonotonicityError(CalculationError):
    """单调性错误"""
    pass


class SafeCalculator:
    """安全计算器"""
    
    def __init__(self):
        self.phoenix_calculator = PhoenixCurveCalculator()
        self.spline_calculator = SplineCurveCalculator()
        self.validator = ParameterValidator()
        self.ui_error_handler = UIErrorHandler()
        self.error_recovery = ErrorRecoverySystem()
        self.boundary_checker = BoundaryChecker()
        
        # 错误计数器
        self.error_count = 0
        self.max_errors = 10
        
        # 启用自动恢复
        self.auto_recovery_enabled = True
        
    def safe_phoenix_calculation(self, L: Union[np.ndarray, float], p: float, a: float) -> Tuple[Union[np.ndarray, float], bool, str]:
        """
        安全的Phoenix曲线计算
        
        Args:
            L: 输入亮度
            p: 亮度控制因子
            a: 缩放因子
            
        Returns:
            (计算结果, 是否成功, 状态信息)
        """
        try:
            # 参数验证
            valid, msg = self.validator.validate_phoenix_params(p, a)
            if not valid:
                logging.warning(f"Phoenix参数验证失败: {msg}")
                return L, False, msg  # 返回恒等映射
                
            # 执行计算
            L_out = self.phoenix_calculator.compute_phoenix_curve(L, p, a)
            
            # 单调性检查 (仅对数组进行检查)
            if isinstance(L_out, np.ndarray) and len(L_out) > 1:
                if not self.phoenix_calculator.validate_monotonicity(L_out):
                    logging.warning("Phoenix曲线非单调，已回退到恒等映射")
                    return L, False, "曲线非单调，已回退到恒等映射"
                    
            return L_out, True, "计算成功"
            
        except Exception as e:
            self.error_count += 1
            error_msg = f"Phoenix曲线计算失败: {str(e)}"
            logging.error(error_msg)
            
            if self.error_count > self.max_errors:
                logging.critical("错误次数过多，系统可能不稳定")
                
            return L, False, error_msg
            
    def safe_phoenix_validation(self, p: float, a: float) -> Tuple[bool, str]:
        """
        安全的Phoenix参数单调性验证
        
        Args:
            p: 亮度控制因子
            a: 缩放因子
            
        Returns:
            (是否单调, 状态信息)
        """
        try:
            # 参数验证
            valid, msg = self.validator.validate_phoenix_params(p, a)
            if not valid:
                return False, f"参数验证失败: {msg}"
                
            # 单调性验证
            is_monotonic = self.phoenix_calculator.validate_monotonicity_pa(p, a)
            
            if is_monotonic:
                return True, "参数组合单调性验证通过"
            else:
                return False, "参数组合导致非单调曲线"
                
        except Exception as e:
            error_msg = f"单调性验证失败: {str(e)}"
            logging.error(error_msg)
            return False, error_msg
            
    def safe_endpoint_normalization(self, L_out: np.ndarray, L_min: float = 0.0, 
                                   L_max: float = 1.0) -> Tuple[np.ndarray, bool, str]:
        """
        安全的端点归一化
        
        Args:
            L_out: Phoenix曲线输出
            L_min: 显示设备最小PQ值
            L_max: 显示设备最大PQ值
            
        Returns:
            (归一化结果, 是否成功, 状态信息)
        """
        try:
            # 显示范围验证
            valid, msg = self.validator.validate_display_range(L_min, L_max)
            if not valid:
                return L_out, False, f"显示范围验证失败: {msg}"
                
            # 输入验证
            if not isinstance(L_out, np.ndarray):
                return L_out, False, "输入必须为numpy数组"
                
            if len(L_out) < 2:
                return L_out, True, "数组长度不足，跳过归一化"
                
            # 执行归一化
            normalized = self.phoenix_calculator.normalize_endpoints(L_out, L_min, L_max)
            
            # 检查端点精度
            endpoint_error = self.phoenix_calculator.check_endpoint_accuracy(normalized, L_min, L_max)
            
            if endpoint_error > 1e-3:
                logging.warning(f"端点匹配精度较低: {endpoint_error}")
                return normalized, True, f"归一化完成，端点误差: {endpoint_error:.6f}"
            else:
                return normalized, True, "端点归一化成功"
                
        except Exception as e:
            error_msg = f"端点归一化失败: {str(e)}"
            logging.error(error_msg)
            return L_out, False, error_msg
            
    def safe_parameter_sanitization(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        安全的参数清理和修正
        
        Args:
            params: 输入参数字典
            
        Returns:
            (修正后的参数, 警告信息列表)
        """
        try:
            warnings = []
            
            # 验证所有参数
            valid, errors = self.validator.validate_all_parameters(params)
            if not valid:
                warnings.extend(errors)
                
            # 清理参数
            sanitized_params = self.validator.sanitize_parameters(params)
            
            # 再次验证清理后的参数
            valid_after, errors_after = self.validator.validate_all_parameters(sanitized_params)
            if not valid_after:
                warnings.append("参数清理后仍存在问题")
                warnings.extend(errors_after)
                
            return sanitized_params, warnings
            
        except Exception as e:
            error_msg = f"参数清理失败: {str(e)}"
            logging.error(error_msg)
            # 返回默认参数
            return self.validator.DEFAULT_PARAMS.copy(), [error_msg]
            
    def get_safe_display_curve(self, p: float, a: float) -> Tuple[np.ndarray, np.ndarray, bool, str]:
        """
        获取安全的显示曲线
        
        Args:
            p: 亮度控制因子
            a: 缩放因子
            
        Returns:
            (输入数组, 输出数组, 是否成功, 状态信息)
        """
        try:
            # 获取输入数组
            L = np.linspace(0, 1, self.phoenix_calculator.display_samples)
            
            # 安全计算
            L_out, success, msg = self.safe_phoenix_calculation(L, p, a)
            
            return L, L_out, success, msg
            
        except Exception as e:
            error_msg = f"获取显示曲线失败: {str(e)}"
            logging.error(error_msg)
            
            # 返回恒等线
            L = np.linspace(0, 1, self.phoenix_calculator.display_samples)
            return L, L.copy(), False, error_msg
            
    def reset_error_count(self):
        """重置错误计数器"""
        self.error_count = 0
        
    def safe_spline_calculation(self, phoenix_curve: np.ndarray, x_input: np.ndarray,
                               th_nodes: List[float], th_strength: float) -> Tuple[np.ndarray, bool, str]:
        """
        安全的样条曲线计算
        
        Args:
            phoenix_curve: Phoenix曲线
            x_input: 输入x坐标
            th_nodes: 样条节点
            th_strength: 样条强度
            
        Returns:
            (最终曲线, 是否使用样条, 状态信息)
        """
        try:
            # 参数验证
            valid, msg = self.validator.validate_spline_strength(th_strength)
            if not valid:
                return phoenix_curve.copy(), False, f"样条强度验证失败: {msg}"
                
            # 节点验证
            corrected_nodes, warning = self.validator.validate_spline_nodes(th_nodes)
            
            # 执行样条计算
            final_curve, used_spline, status = self.spline_calculator.compute_spline_with_fallback(
                phoenix_curve, x_input, corrected_nodes, th_strength
            )
            
            # 组合状态信息
            full_status = status
            if warning:
                full_status = f"{warning}; {status}"
                
            return final_curve, used_spline, full_status
            
        except SplineCalculationError as e:
            error_msg = f"样条计算错误: {str(e)}"
            logging.error(error_msg)
            return phoenix_curve.copy(), False, error_msg
            
        except Exception as e:
            self.error_count += 1
            error_msg = f"样条计算失败: {str(e)}"
            logging.error(error_msg)
            return phoenix_curve.copy(), False, error_msg
            
    def safe_combined_curve_calculation(self, p: float, a: float, th_nodes: List[float], 
                                      th_strength: float) -> Tuple[np.ndarray, np.ndarray, bool, bool, str]:
        """
        安全的组合曲线计算 (Phoenix + 样条)
        
        Args:
            p: Phoenix参数p
            a: Phoenix参数a
            th_nodes: 样条节点
            th_strength: 样条强度
            
        Returns:
            (输入数组, 输出数组, Phoenix成功, 样条成功, 状态信息)
        """
        try:
            # 获取输入数组
            L = np.linspace(0, 1, self.phoenix_calculator.display_samples)
            
            # 计算Phoenix曲线
            phoenix_curve, phoenix_success, phoenix_msg = self.safe_phoenix_calculation(L, p, a)
            
            if not phoenix_success:
                return L, L.copy(), False, False, f"Phoenix计算失败: {phoenix_msg}"
                
            # 如果样条强度为0，直接返回Phoenix曲线
            if th_strength <= 1e-6:
                return L, phoenix_curve, True, False, "样条强度为0，使用Phoenix曲线"
                
            # 计算样条曲线
            final_curve, spline_success, spline_msg = self.safe_spline_calculation(
                phoenix_curve, L, th_nodes, th_strength
            )
            
            # 组合状态信息
            status = f"Phoenix: {phoenix_msg}; 样条: {spline_msg}"
            
            return L, final_curve, phoenix_success, spline_success, status
            
        except Exception as e:
            error_msg = f"组合曲线计算失败: {str(e)}"
            logging.error(error_msg)
            L = np.linspace(0, 1, self.phoenix_calculator.display_samples)
            return L, L.copy(), False, False, error_msg

    def comprehensive_parameter_validation(self, parameters: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        全面的参数验证
        
        Args:
            parameters: 输入参数字典
            
        Returns:
            (是否有效, 修正后的参数, 错误信息列表)
        """
        try:
            # 1. 边界条件检查
            is_valid, violations = self.boundary_checker.check_all_boundaries(parameters)
            
            error_messages = []
            
            # 2. 处理违反条件
            if violations:
                # 创建错误消息
                for violation in violations:
                    if violation.severity == "error":
                        error = self.ui_error_handler.create_parameter_error(
                            violation.parameter, 
                            violation.current_value,
                            violation.expected_range or (0, 1)
                        )
                        error_messages.append(error.message)
                    elif violation.severity == "warning":
                        warning = self.ui_error_handler.add_error(
                            ErrorSeverity.WARNING,
                            "参数警告",
                            violation.constraint_description,
                            violation.suggested_value
                        )
                        error_messages.append(f"警告: {warning.message}")
                        
                # 自动修正可修正的违反条件
                corrected_params = self.boundary_checker.auto_correct_violations(parameters, violations)
                
                # 如果启用自动恢复且存在错误
                if not is_valid and self.auto_recovery_enabled:
                    success, recovery_msg, recovered_params = self.error_recovery.auto_recover(
                        "parameter_validation", parameters, str(violations)
                    )
                    
                    if success:
                        error_messages.append(f"自动恢复: {recovery_msg}")
                        return True, recovered_params, error_messages
                    else:
                        corrected_params = recovered_params
                        
                return len([v for v in violations if v.severity == "error"]) == 0, corrected_params, error_messages
            
            return True, parameters, []
            
        except Exception as e:
            error_msg = f"参数验证失败: {str(e)}"
            self.ui_error_handler.create_system_error("validation_failed", reason=error_msg)
            return False, self.validator.DEFAULT_PARAMS.copy(), [error_msg]
            
    def safe_phoenix_calculation_enhanced(self, L: Union[np.ndarray, float], p: float, a: float) -> Tuple[Union[np.ndarray, float], bool, str, Dict[str, Any]]:
        """
        增强的安全Phoenix曲线计算
        
        Args:
            L: 输入亮度
            p: 亮度控制因子
            a: 缩放因子
            
        Returns:
            (计算结果, 是否成功, 状态信息, 详细状态)
        """
        detailed_status = {
            'parameter_validation': False,
            'computation_success': False,
            'monotonicity_check': False,
            'numerical_stability': False,
            'recovery_applied': False
        }
        
        try:
            # 1. 参数验证
            params = {'p': p, 'a': a}
            is_valid, corrected_params, validation_errors = self.comprehensive_parameter_validation(params)
            detailed_status['parameter_validation'] = is_valid
            
            if not is_valid and not self.auto_recovery_enabled:
                return L, False, f"参数验证失败: {'; '.join(validation_errors)}", detailed_status
                
            # 使用修正后的参数
            p_corrected = corrected_params.get('p', p)
            a_corrected = corrected_params.get('a', a)
            
            # 2. 执行计算
            try:
                L_out = self.phoenix_calculator.compute_phoenix_curve(L, p_corrected, a_corrected)
                detailed_status['computation_success'] = True
            except Exception as calc_error:
                if self.auto_recovery_enabled:
                    success, recovery_msg, recovered_params = self.error_recovery.auto_recover(
                        "computation_failure", params, str(calc_error)
                    )
                    if success:
                        p_corrected = recovered_params.get('p', 2.0)
                        a_corrected = recovered_params.get('a', 0.5)
                        L_out = self.phoenix_calculator.compute_phoenix_curve(L, p_corrected, a_corrected)
                        detailed_status['recovery_applied'] = True
                        detailed_status['computation_success'] = True
                    else:
                        return L, False, f"计算失败且恢复失败: {str(calc_error)}", detailed_status
                else:
                    return L, False, f"计算失败: {str(calc_error)}", detailed_status
                    
            # 3. 数值稳定性检查
            if isinstance(L_out, np.ndarray):
                if np.any(np.isnan(L_out)) or np.any(np.isinf(L_out)):
                    detailed_status['numerical_stability'] = False
                    if self.auto_recovery_enabled:
                        success, recovery_msg, recovered_params = self.error_recovery.auto_recover(
                            "numerical_instability", params, "NaN或Inf值"
                        )
                        if success:
                            p_safe = recovered_params.get('p', 2.0)
                            a_safe = recovered_params.get('a', 0.5)
                            L_out = self.phoenix_calculator.compute_phoenix_curve(L, p_safe, a_safe)
                            detailed_status['recovery_applied'] = True
                            detailed_status['numerical_stability'] = True
                        else:
                            return L, False, "数值不稳定且恢复失败", detailed_status
                    else:
                        return L, False, "计算结果包含NaN或Inf值", detailed_status
                else:
                    detailed_status['numerical_stability'] = True
                    
            # 4. 单调性检查 (仅对数组进行检查)
            if isinstance(L_out, np.ndarray) and len(L_out) > 1:
                is_monotonic = self.phoenix_calculator.validate_monotonicity(L_out)
                detailed_status['monotonicity_check'] = is_monotonic
                
                if not is_monotonic:
                    self.ui_error_handler.create_monotonicity_warning()
                    
                    if self.auto_recovery_enabled:
                        success, recovery_msg, recovered_params = self.error_recovery.auto_recover(
                            "monotonicity_violation", params, "曲线非单调"
                        )
                        if success:
                            p_mono = recovered_params.get('p', 2.0)
                            a_mono = recovered_params.get('a', 0.5)
                            L_out = self.phoenix_calculator.compute_phoenix_curve(L, p_mono, a_mono)
                            detailed_status['recovery_applied'] = True
                            detailed_status['monotonicity_check'] = self.phoenix_calculator.validate_monotonicity(L_out)
                        else:
                            return L, False, "曲线非单调且恢复失败", detailed_status
                    else:
                        return L, False, "曲线非单调，已回退到恒等映射", detailed_status
            else:
                detailed_status['monotonicity_check'] = True
                
            # 5. 保存成功状态
            if all([detailed_status['parameter_validation'], detailed_status['computation_success'], 
                   detailed_status['monotonicity_check'], detailed_status['numerical_stability']]):
                self.error_recovery.save_state(corrected_params, is_valid=True)
                
            status_msg = "计算成功"
            if detailed_status['recovery_applied']:
                status_msg += " (已应用自动恢复)"
            if validation_errors:
                status_msg += f" (参数已修正: {len(validation_errors)}个问题)"
                
            return L_out, True, status_msg, detailed_status
            
        except Exception as e:
            self.error_count += 1
            error_msg = f"Phoenix曲线计算失败: {str(e)}"
            self.ui_error_handler.create_calculation_error("Phoenix曲线计算", str(e))
            
            if self.error_count > self.max_errors:
                self.ui_error_handler.create_system_error("too_many_errors")
                
            return L, False, error_msg, detailed_status
            
    def safe_image_validation(self, image: np.ndarray, max_pixels: int = 10_000_000) -> Tuple[bool, str, Optional[np.ndarray]]:
        """
        安全的图像验证
        
        Args:
            image: 输入图像
            max_pixels: 最大像素数限制
            
        Returns:
            (是否有效, 状态信息, 处理后的图像)
        """
        try:
            # 使用UI错误处理器的验证方法
            is_valid, error_msg = self.ui_error_handler.validate_image_upload(image, max_pixels)
            
            if not is_valid:
                return False, error_msg, None
                
            # 额外的安全检查
            if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
                processed_image = image.astype(np.float32)
                warning_msg = f"图像数据类型已转换为float32 (原类型: {image.dtype})"
                self.ui_error_handler.add_error(ErrorSeverity.INFO, "数据类型转换", warning_msg)
                return True, warning_msg, processed_image
                
            # 检查值范围
            if image.dtype in [np.uint8, np.uint16]:
                # 整数类型，正常范围
                return True, "图像验证通过", image
            else:
                # 浮点类型，检查范围
                if np.any(image < 0) or np.any(image > 1):
                    if np.max(image) > 1:
                        # 可能是HDR图像，不需要归一化
                        return True, "检测到HDR图像", image
                    else:
                        # 包含负值，需要处理
                        processed_image = np.clip(image, 0, 1)
                        warning_msg = "图像包含负值，已裁剪到[0,1]范围"
                        self.ui_error_handler.add_error(ErrorSeverity.WARNING, "图像值范围", warning_msg)
                        return True, warning_msg, processed_image
                        
            return True, "图像验证通过", image
            
        except Exception as e:
            error_msg = f"图像验证失败: {str(e)}"
            self.ui_error_handler.create_image_error("validation_failed", reason=str(e))
            return False, error_msg, None
            
    def get_comprehensive_system_status(self) -> Dict[str, Any]:
        """
        获取全面的系统状态
        
        Returns:
            系统状态字典
        """
        base_status = {
            'error_count': self.error_count,
            'max_errors': self.max_errors,
            'system_stable': self.error_count < self.max_errors,
            'phoenix_calculator_ready': self.phoenix_calculator is not None,
            'spline_calculator_ready': self.spline_calculator is not None,
            'validator_ready': self.validator is not None,
            'display_samples': self.phoenix_calculator.display_samples,
            'validation_samples': self.phoenix_calculator.validation_samples,
            'auto_recovery_enabled': self.auto_recovery_enabled
        }
        
        # 添加错误处理器状态
        error_summary = self.ui_error_handler.get_error_summary()
        base_status.update({
            'ui_error_handler': error_summary,
            'recent_errors': len(self.ui_error_handler.get_recent_errors(10))
        })
        
        # 添加恢复系统状态
        recovery_status = self.error_recovery.get_recovery_status()
        base_status.update({
            'error_recovery': recovery_status
        })
        
        return base_status
        
    def create_system_diagnostic_report(self) -> str:
        """创建系统诊断报告"""
        status = self.get_comprehensive_system_status()
        
        report = f"""
# 系统诊断报告

## 基本状态
- 系统稳定性: {'稳定' if status['system_stable'] else '不稳定'}
- 错误计数: {status['error_count']}/{status['max_errors']}
- 自动恢复: {'启用' if status['auto_recovery_enabled'] else '禁用'}

## 组件状态
- Phoenix计算器: {'就绪' if status['phoenix_calculator_ready'] else '未就绪'}
- 样条计算器: {'就绪' if status['spline_calculator_ready'] else '未就绪'}
- 参数验证器: {'就绪' if status['validator_ready'] else '未就绪'}

## 错误处理状态
- 总错误数: {status['ui_error_handler']['total_errors']}
- 最近错误数: {status['ui_error_handler']['recent_errors']}
- 错误率: {status['ui_error_handler']['error_rate']:.1f}%
- 系统状态: {status['ui_error_handler']['status']}

## 恢复系统状态
- 恢复尝试: {status['error_recovery']['recovery_attempts']}/{status['error_recovery']['max_recovery_attempts']}
- 状态历史: {status['error_recovery']['state_history_length']}个
- 最后有效状态: {'可用' if status['error_recovery']['last_valid_state_available'] else '不可用'}

## 采样配置
- 显示采样点数: {status['display_samples']}
- 验证采样点数: {status['validation_samples']}
"""
        
        return report
        
    def enable_auto_recovery(self, enabled: bool = True):
        """启用/禁用自动恢复"""
        self.auto_recovery_enabled = enabled
        if enabled:
            self.ui_error_handler.add_error(ErrorSeverity.INFO, "自动恢复", "自动恢复已启用")
        else:
            self.ui_error_handler.add_error(ErrorSeverity.INFO, "自动恢复", "自动恢复已禁用")
            
    def reset_error_handling_system(self):
        """重置错误处理系统"""
        self.error_count = 0
        self.ui_error_handler.clear_error_history()
        self.error_recovery.reset_recovery_system()
        self.ui_error_handler.add_error(ErrorSeverity.INFO, "系统重置", "错误处理系统已重置")

    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态 (保持向后兼容)
        
        Returns:
            系统状态字典
        """
        return self.get_comprehensive_system_status()