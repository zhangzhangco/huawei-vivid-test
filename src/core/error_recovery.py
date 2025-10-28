"""
错误恢复系统
提供自动回退机制和系统状态恢复功能
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import copy

from .parameter_validator import ParameterValidator
from .phoenix_calculator import PhoenixCurveCalculator
from .ui_error_handler import UIErrorHandler, ErrorSeverity


class RecoveryStrategy(Enum):
    """恢复策略"""
    FALLBACK_TO_DEFAULT = "fallback_to_default"
    FALLBACK_TO_LAST_VALID = "fallback_to_last_valid"
    PARAMETER_CORRECTION = "parameter_correction"
    IDENTITY_MAPPING = "identity_mapping"
    SAFE_APPROXIMATION = "safe_approximation"


@dataclass
class SystemState:
    """系统状态快照"""
    parameters: Dict[str, Any]
    timestamp: float
    is_valid: bool
    validation_errors: List[str] = field(default_factory=list)
    computation_success: bool = True
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class RecoveryAction:
    """恢复动作"""
    strategy: RecoveryStrategy
    description: str
    parameters: Dict[str, Any]
    success_probability: float
    side_effects: List[str] = field(default_factory=list)


class ErrorRecoverySystem:
    """错误恢复系统"""
    
    def __init__(self):
        self.validator = ParameterValidator()
        self.phoenix_calc = PhoenixCurveCalculator()
        self.ui_error_handler = UIErrorHandler()
        self.logger = logging.getLogger(__name__)
        
        # 状态历史
        self.state_history: List[SystemState] = []
        self.max_history = 20
        
        # 默认安全参数
        self.safe_parameters = {
            'p': 2.0,
            'a': 0.5,
            'dt_low': 0.05,
            'dt_high': 0.10,
            'min_display_pq': 0.0,
            'max_display_pq': 1.0,
            'window_size': 9,
            'lambda_smooth': 0.3,
            'th_nodes': [0.2, 0.5, 0.8],
            'th_strength': 0.0,
            'luminance_channel': 'MaxRGB'
        }
        
        # 恢复策略优先级
        self.recovery_priorities = [
            RecoveryStrategy.PARAMETER_CORRECTION,
            RecoveryStrategy.FALLBACK_TO_LAST_VALID,
            RecoveryStrategy.SAFE_APPROXIMATION,
            RecoveryStrategy.FALLBACK_TO_DEFAULT,
            RecoveryStrategy.IDENTITY_MAPPING
        ]
        
        # 错误计数器
        self.error_counts = {}
        self.recovery_attempts = 0
        self.max_recovery_attempts = 5
        
    def save_state(self, parameters: Dict[str, Any], is_valid: bool = True, 
                   validation_errors: List[str] = None) -> SystemState:
        """保存系统状态"""
        state = SystemState(
            parameters=copy.deepcopy(parameters),
            timestamp=time.time(),
            is_valid=is_valid,
            validation_errors=validation_errors or []
        )
        
        self.state_history.append(state)
        
        # 限制历史长度
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
            
        return state
        
    def get_last_valid_state(self) -> Optional[SystemState]:
        """获取最后一个有效状态"""
        for state in reversed(self.state_history):
            if state.is_valid and state.computation_success:
                return state
        return None
        
    def analyze_error(self, error_type: str, parameters: Dict[str, Any], 
                     error_details: str = "") -> List[RecoveryAction]:
        """分析错误并生成恢复策略"""
        recovery_actions = []
        
        # 增加错误计数
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        if error_type == "parameter_validation":
            recovery_actions.extend(self._analyze_parameter_errors(parameters, error_details))
        elif error_type == "monotonicity_violation":
            recovery_actions.extend(self._analyze_monotonicity_errors(parameters))
        elif error_type == "numerical_instability":
            recovery_actions.extend(self._analyze_numerical_errors(parameters))
        elif error_type == "computation_failure":
            recovery_actions.extend(self._analyze_computation_errors(parameters, error_details))
        elif error_type == "image_processing":
            recovery_actions.extend(self._analyze_image_errors(parameters, error_details))
        else:
            # 通用错误处理
            recovery_actions.extend(self._get_generic_recovery_actions(parameters))
            
        # 按成功概率排序
        recovery_actions.sort(key=lambda x: x.success_probability, reverse=True)
        
        return recovery_actions
        
    def _analyze_parameter_errors(self, parameters: Dict[str, Any], 
                                error_details: str) -> List[RecoveryAction]:
        """分析参数错误"""
        actions = []
        
        # 参数修正策略
        corrected_params = self.validator.sanitize_parameters(parameters)
        if corrected_params != parameters:
            actions.append(RecoveryAction(
                strategy=RecoveryStrategy.PARAMETER_CORRECTION,
                description="自动修正参数到有效范围",
                parameters=corrected_params,
                success_probability=0.9,
                side_effects=["参数值可能发生变化"]
            ))
            
        # 回退到最后有效状态
        last_valid = self.get_last_valid_state()
        if last_valid:
            actions.append(RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK_TO_LAST_VALID,
                description="回退到最后一个有效状态",
                parameters=last_valid.parameters,
                success_probability=0.8,
                side_effects=["丢失当前参数修改"]
            ))
            
        # 回退到默认参数
        actions.append(RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK_TO_DEFAULT,
            description="重置为默认安全参数",
            parameters=self.safe_parameters.copy(),
            success_probability=0.95,
            side_effects=["丢失所有自定义参数"]
        ))
        
        return actions
        
    def _analyze_monotonicity_errors(self, parameters: Dict[str, Any]) -> List[RecoveryAction]:
        """分析单调性错误"""
        actions = []
        
        # 尝试调整参数保持单调性
        adjusted_params = parameters.copy()
        
        # 策略1: 减小p值
        if adjusted_params.get('p', 2.0) > 1.5:
            adjusted_params['p'] = max(1.5, adjusted_params['p'] * 0.8)
            actions.append(RecoveryAction(
                strategy=RecoveryStrategy.PARAMETER_CORRECTION,
                description=f"减小p值到 {adjusted_params['p']:.2f} 以保持单调性",
                parameters=adjusted_params.copy(),
                success_probability=0.7,
                side_effects=["对比度可能降低"]
            ))
            
        # 策略2: 增大a值
        adjusted_params = parameters.copy()
        if adjusted_params.get('a', 0.5) < 0.8:
            adjusted_params['a'] = min(0.8, adjusted_params['a'] + 0.2)
            actions.append(RecoveryAction(
                strategy=RecoveryStrategy.PARAMETER_CORRECTION,
                description=f"增大a值到 {adjusted_params['a']:.2f} 以保持单调性",
                parameters=adjusted_params.copy(),
                success_probability=0.6,
                side_effects=["亮度映射范围可能改变"]
            ))
            
        # 策略3: 禁用样条曲线
        if parameters.get('th_strength', 0) > 0:
            adjusted_params = parameters.copy()
            adjusted_params['th_strength'] = 0.0
            actions.append(RecoveryAction(
                strategy=RecoveryStrategy.PARAMETER_CORRECTION,
                description="禁用样条曲线以保持单调性",
                parameters=adjusted_params,
                success_probability=0.8,
                side_effects=["失去样条曲线的局部优化效果"]
            ))
            
        return actions
        
    def _analyze_numerical_errors(self, parameters: Dict[str, Any]) -> List[RecoveryAction]:
        """分析数值稳定性错误"""
        actions = []
        
        # 检查极端参数值
        adjusted_params = parameters.copy()
        
        # 避免极端的p值
        if adjusted_params.get('p', 2.0) > 5.0:
            adjusted_params['p'] = 4.0
        elif adjusted_params.get('p', 2.0) < 0.2:
            adjusted_params['p'] = 0.5
            
        # 避免极端的a值
        if adjusted_params.get('a', 0.5) < 0.01:
            adjusted_params['a'] = 0.1
            
        if adjusted_params != parameters:
            actions.append(RecoveryAction(
                strategy=RecoveryStrategy.PARAMETER_CORRECTION,
                description="调整极端参数值以提高数值稳定性",
                parameters=adjusted_params,
                success_probability=0.8,
                side_effects=["参数值被限制在稳定范围内"]
            ))
            
        # 使用安全近似
        safe_params = self._get_safe_approximation(parameters)
        actions.append(RecoveryAction(
            strategy=RecoveryStrategy.SAFE_APPROXIMATION,
            description="使用数值稳定的参数近似",
            parameters=safe_params,
            success_probability=0.9,
            side_effects=["可能与原始参数有轻微差异"]
        ))
        
        return actions
        
    def _analyze_computation_errors(self, parameters: Dict[str, Any], 
                                  error_details: str) -> List[RecoveryAction]:
        """分析计算错误"""
        actions = []
        
        # 检查是否是内存问题
        if "memory" in error_details.lower() or "out of memory" in error_details.lower():
            # 减少采样点数
            reduced_params = parameters.copy()
            reduced_params['display_samples'] = min(256, parameters.get('display_samples', 512))
            actions.append(RecoveryAction(
                strategy=RecoveryStrategy.PARAMETER_CORRECTION,
                description="减少采样点数以降低内存使用",
                parameters=reduced_params,
                success_probability=0.7,
                side_effects=["曲线精度可能降低"]
            ))
            
        # 使用恒等映射作为最后手段
        identity_params = parameters.copy()
        identity_params.update({'p': 1.0, 'a': 0.0, 'th_strength': 0.0})
        actions.append(RecoveryAction(
            strategy=RecoveryStrategy.IDENTITY_MAPPING,
            description="使用恒等映射确保系统稳定",
            parameters=identity_params,
            success_probability=1.0,
            side_effects=["无色调映射效果"]
        ))
        
        return actions
        
    def _analyze_image_errors(self, parameters: Dict[str, Any], 
                            error_details: str) -> List[RecoveryAction]:
        """分析图像处理错误"""
        actions = []
        
        # 图像过大
        if "too large" in error_details.lower():
            # 启用自动降采样
            adjusted_params = parameters.copy()
            adjusted_params['auto_downsample'] = True
            adjusted_params['max_image_pixels'] = 2_000_000  # 2MP
            actions.append(RecoveryAction(
                strategy=RecoveryStrategy.PARAMETER_CORRECTION,
                description="启用自动降采样处理大图像",
                parameters=adjusted_params,
                success_probability=0.8,
                side_effects=["图像分辨率可能降低"]
            ))
            
        # 格式不支持
        elif "format" in error_details.lower():
            actions.append(RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK_TO_DEFAULT,
                description="跳过图像处理，使用合成数据",
                parameters=parameters,
                success_probability=0.9,
                side_effects=["无法显示真实图像效果"]
            ))
            
        return actions
        
    def _get_generic_recovery_actions(self, parameters: Dict[str, Any]) -> List[RecoveryAction]:
        """获取通用恢复策略"""
        actions = []
        
        # 回退到最后有效状态
        last_valid = self.get_last_valid_state()
        if last_valid:
            actions.append(RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK_TO_LAST_VALID,
                description="回退到最后一个有效状态",
                parameters=last_valid.parameters,
                success_probability=0.7,
                side_effects=["丢失当前修改"]
            ))
            
        # 回退到默认参数
        actions.append(RecoveryAction(
            strategy=RecoveryStrategy.FALLBACK_TO_DEFAULT,
            description="重置为默认安全参数",
            parameters=self.safe_parameters.copy(),
            success_probability=0.9,
            side_effects=["丢失所有自定义设置"]
        ))
        
        return actions
        
    def _get_safe_approximation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """获取安全的参数近似"""
        safe_params = parameters.copy()
        
        # 将参数调整到安全范围
        safe_params['p'] = np.clip(safe_params.get('p', 2.0), 0.5, 4.0)
        safe_params['a'] = np.clip(safe_params.get('a', 0.5), 0.1, 0.8)
        safe_params['th_strength'] = min(safe_params.get('th_strength', 0.0), 0.5)
        
        return safe_params
        
    def execute_recovery(self, recovery_action: RecoveryAction) -> Tuple[bool, str, Dict[str, Any]]:
        """执行恢复动作"""
        try:
            self.recovery_attempts += 1
            
            if self.recovery_attempts > self.max_recovery_attempts:
                return False, "超过最大恢复尝试次数", self.safe_parameters.copy()
                
            # 验证恢复参数
            valid, errors = self.validator.validate_all_parameters(recovery_action.parameters)
            
            if not valid:
                return False, f"恢复参数验证失败: {'; '.join(errors)}", recovery_action.parameters
                
            # 测试计算是否成功
            test_success = self._test_computation(recovery_action.parameters)
            
            if test_success:
                # 保存成功的状态
                self.save_state(recovery_action.parameters, is_valid=True)
                self.recovery_attempts = 0  # 重置计数器
                
                # 记录成功的恢复
                self.logger.info(f"恢复成功: {recovery_action.description}")
                
                return True, f"恢复成功: {recovery_action.description}", recovery_action.parameters
            else:
                return False, f"恢复测试失败: {recovery_action.description}", recovery_action.parameters
                
        except Exception as e:
            error_msg = f"恢复执行失败: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg, recovery_action.parameters
            
    def _test_computation(self, parameters: Dict[str, Any]) -> bool:
        """测试计算是否成功"""
        try:
            # 测试Phoenix曲线计算
            p = parameters.get('p', 2.0)
            a = parameters.get('a', 0.5)
            
            L = np.linspace(0, 1, 64)  # 使用较少的点进行测试
            L_out = self.phoenix_calc.compute_phoenix_curve(L, p, a)
            
            # 检查单调性
            if not self.phoenix_calc.validate_monotonicity(L_out):
                return False
                
            # 检查数值稳定性
            if np.any(np.isnan(L_out)) or np.any(np.isinf(L_out)):
                return False
                
            return True
            
        except Exception:
            return False
            
    def auto_recover(self, error_type: str, parameters: Dict[str, Any], 
                    error_details: str = "") -> Tuple[bool, str, Dict[str, Any]]:
        """自动恢复"""
        # 分析错误并生成恢复策略
        recovery_actions = self.analyze_error(error_type, parameters, error_details)
        
        if not recovery_actions:
            return False, "无可用的恢复策略", parameters
            
        # 尝试执行恢复策略
        for action in recovery_actions:
            success, message, recovered_params = self.execute_recovery(action)
            
            if success:
                # 创建成功消息
                self.ui_error_handler.add_error(
                    ErrorSeverity.INFO,
                    "自动恢复成功",
                    message,
                    "系统已恢复正常运行"
                )
                return True, message, recovered_params
            else:
                # 记录失败的尝试
                self.logger.warning(f"恢复策略失败: {action.description} - {message}")
                
        # 所有策略都失败，使用最后的安全策略
        return False, "所有恢复策略都失败，请手动重置", self.safe_parameters.copy()
        
    def get_recovery_status(self) -> Dict[str, Any]:
        """获取恢复系统状态"""
        return {
            'recovery_attempts': self.recovery_attempts,
            'max_recovery_attempts': self.max_recovery_attempts,
            'error_counts': self.error_counts.copy(),
            'state_history_length': len(self.state_history),
            'last_valid_state_available': self.get_last_valid_state() is not None,
            'system_stable': self.recovery_attempts < self.max_recovery_attempts
        }
        
    def reset_recovery_system(self):
        """重置恢复系统"""
        self.recovery_attempts = 0
        self.error_counts.clear()
        self.state_history.clear()
        self.ui_error_handler.clear_error_history()
        self.logger.info("恢复系统已重置")
        
    def create_recovery_report(self) -> str:
        """创建恢复报告"""
        status = self.get_recovery_status()
        error_summary = self.ui_error_handler.get_error_summary()
        
        report = f"""
# 系统恢复报告

## 恢复状态
- 恢复尝试次数: {status['recovery_attempts']}/{status['max_recovery_attempts']}
- 系统稳定性: {'稳定' if status['system_stable'] else '不稳定'}
- 状态历史长度: {status['state_history_length']}
- 最后有效状态: {'可用' if status['last_valid_state_available'] else '不可用'}

## 错误统计
- 总错误数: {error_summary['total_errors']}
- 最近错误数: {error_summary['recent_errors']}
- 错误率: {error_summary['error_rate']:.1f}%
- 系统状态: {error_summary['status']}

## 错误类型分布
"""
        
        for error_type, count in status['error_counts'].items():
            report += f"- {error_type}: {count}次\n"
            
        return report