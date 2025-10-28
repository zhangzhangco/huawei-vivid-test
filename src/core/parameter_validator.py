"""
参数验证器
提供Phoenix曲线参数、样条节点、显示范围等的验证和修正功能
"""

from typing import List, Tuple, Dict, Any
import numpy as np


class ParameterValidator:
    """参数验证器"""
    
    # 参数范围定义
    PHOENIX_P_RANGE = (0.1, 6.0)
    PHOENIX_A_RANGE = (0.0, 1.0)
    DISPLAY_PQ_RANGE = (0.0, 1.0)
    DISTORTION_THRESHOLD_RANGE = (0.01, 0.20)
    TEMPORAL_WINDOW_RANGE = (3, 20)
    TEMPORAL_LAMBDA_RANGE = (0.1, 0.8)
    SPLINE_STRENGTH_RANGE = (0.0, 1.0)
    
    # 默认值
    DEFAULT_PARAMS = {
        'p': 2.0,
        'a': 0.5,
        'dt_low': 0.05,
        'dt_high': 0.10,
        'min_display_pq': 0.0,
        'max_display_pq': 1.0,
        'window_size': 9,
        'lambda_smooth': 0.3,
        'th_strength': 0.0,
        'th_nodes': [0.2, 0.5, 0.8]
    }
    
    @staticmethod
    def validate_phoenix_params(p: float, a: float) -> Tuple[bool, str]:
        """
        验证Phoenix参数
        
        Args:
            p: 亮度控制因子
            a: 缩放因子
            
        Returns:
            (是否有效, 错误信息)
        """
        if not isinstance(p, (int, float)):
            return False, f"参数p必须为数值类型，当前类型: {type(p)}"
        if not isinstance(a, (int, float)):
            return False, f"参数a必须为数值类型，当前类型: {type(a)}"
            
        if not (ParameterValidator.PHOENIX_P_RANGE[0] <= p <= ParameterValidator.PHOENIX_P_RANGE[1]):
            return False, f"参数p={p}超出范围{ParameterValidator.PHOENIX_P_RANGE}"
        if not (ParameterValidator.PHOENIX_A_RANGE[0] <= a <= ParameterValidator.PHOENIX_A_RANGE[1]):
            return False, f"参数a={a}超出范围{ParameterValidator.PHOENIX_A_RANGE}"
            
        return True, ""
        
    @staticmethod
    def validate_spline_nodes(nodes: List[float], min_interval: float = 0.01) -> Tuple[List[float], str]:
        """
        验证并修正样条节点
        
        Args:
            nodes: 样条节点列表
            min_interval: 最小间隔
            
        Returns:
            (修正后的节点列表, 警告信息)
        """
        if not isinstance(nodes, (list, tuple, np.ndarray)):
            return [0.2, 0.5, 0.8], "节点必须为列表类型，已使用默认值"
            
        if len(nodes) == 0:
            return [0.2, 0.5, 0.8], "节点列表为空，已使用默认值"
            
        # 转换为浮点数并排序
        try:
            float_nodes = [float(node) for node in nodes]
        except (ValueError, TypeError):
            return [0.2, 0.5, 0.8], "节点包含非数值元素，已使用默认值"
            
        sorted_nodes = sorted(float_nodes)
        original_nodes = sorted_nodes.copy()
        
        # 确保在[0,1]范围内
        sorted_nodes = [max(0.0, min(1.0, node)) for node in sorted_nodes]
        
        # 确保最小间隔
        for i in range(1, len(sorted_nodes)):
            if sorted_nodes[i] - sorted_nodes[i-1] < min_interval:
                sorted_nodes[i] = sorted_nodes[i-1] + min_interval
                
        # 再次确保在[0,1]范围内
        sorted_nodes = [max(0.0, min(1.0, node)) for node in sorted_nodes]
        
        warning = ""
        if sorted_nodes != original_nodes:
            warning = "样条节点已自动调整以满足最小间隔和范围约束"
            
        return sorted_nodes, warning
        
    @staticmethod
    def validate_display_range(min_pq: float, max_pq: float) -> Tuple[bool, str]:
        """
        验证显示范围
        
        Args:
            min_pq: 最小PQ值
            max_pq: 最大PQ值
            
        Returns:
            (是否有效, 错误信息)
        """
        if not isinstance(min_pq, (int, float)) or not isinstance(max_pq, (int, float)):
            return False, "显示范围参数必须为数值类型"
            
        if max_pq <= min_pq:
            return False, f"MaxDisplay_PQ({max_pq})必须大于MinDisplay_PQ({min_pq})"
            
        pq_range = ParameterValidator.DISPLAY_PQ_RANGE
        if not (pq_range[0] <= min_pq <= pq_range[1]) or not (pq_range[0] <= max_pq <= pq_range[1]):
            return False, f"显示范围必须在PQ域{pq_range}内"
            
        return True, ""
        
    @staticmethod
    def validate_distortion_thresholds(dt_low: float, dt_high: float) -> Tuple[bool, str]:
        """
        验证失真阈值
        
        Args:
            dt_low: 下阈值
            dt_high: 上阈值
            
        Returns:
            (是否有效, 错误信息)
        """
        if not isinstance(dt_low, (int, float)) or not isinstance(dt_high, (int, float)):
            return False, "失真阈值必须为数值类型"
            
        if dt_high <= dt_low:
            return False, f"上阈值({dt_high})必须大于下阈值({dt_low})"
            
        threshold_range = ParameterValidator.DISTORTION_THRESHOLD_RANGE
        if not (threshold_range[0] <= dt_low <= threshold_range[1]):
            return False, f"下阈值{dt_low}超出范围{threshold_range}"
        if not (threshold_range[0] <= dt_high <= threshold_range[1]):
            return False, f"上阈值{dt_high}超出范围{threshold_range}"
            
        return True, ""
        
    @staticmethod
    def validate_temporal_params(window_size: int, lambda_smooth: float) -> Tuple[bool, str]:
        """
        验证时域平滑参数
        
        Args:
            window_size: 窗口大小
            lambda_smooth: 平滑强度
            
        Returns:
            (是否有效, 错误信息)
        """
        if not isinstance(window_size, int):
            return False, f"窗口大小必须为整数，当前类型: {type(window_size)}"
        if not isinstance(lambda_smooth, (int, float)):
            return False, f"平滑强度必须为数值类型，当前类型: {type(lambda_smooth)}"
            
        window_range = ParameterValidator.TEMPORAL_WINDOW_RANGE
        if not (window_range[0] <= window_size <= window_range[1]):
            return False, f"窗口大小{window_size}超出范围{window_range}"
            
        lambda_range = ParameterValidator.TEMPORAL_LAMBDA_RANGE
        if not (lambda_range[0] <= lambda_smooth <= lambda_range[1]):
            return False, f"平滑强度{lambda_smooth}超出范围{lambda_range}"
            
        return True, ""
        
    @staticmethod
    def validate_spline_strength(strength: float) -> Tuple[bool, str]:
        """
        验证样条强度
        
        Args:
            strength: 样条强度
            
        Returns:
            (是否有效, 错误信息)
        """
        if not isinstance(strength, (int, float)):
            return False, f"样条强度必须为数值类型，当前类型: {type(strength)}"
            
        strength_range = ParameterValidator.SPLINE_STRENGTH_RANGE
        if not (strength_range[0] <= strength <= strength_range[1]):
            return False, f"样条强度{strength}超出范围{strength_range}"
            
        return True, ""
        
    @staticmethod
    def validate_all_parameters(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证所有参数
        
        Args:
            params: 参数字典
            
        Returns:
            (是否全部有效, 错误信息列表)
        """
        errors = []
        
        # Phoenix参数
        if 'p' in params and 'a' in params:
            valid, msg = ParameterValidator.validate_phoenix_params(params['p'], params['a'])
            if not valid:
                errors.append(msg)
                
        # 显示范围
        if 'min_display_pq' in params and 'max_display_pq' in params:
            valid, msg = ParameterValidator.validate_display_range(
                params['min_display_pq'], params['max_display_pq'])
            if not valid:
                errors.append(msg)
                
        # 失真阈值
        if 'dt_low' in params and 'dt_high' in params:
            valid, msg = ParameterValidator.validate_distortion_thresholds(
                params['dt_low'], params['dt_high'])
            if not valid:
                errors.append(msg)
                
        # 时域参数
        if 'window_size' in params and 'lambda_smooth' in params:
            valid, msg = ParameterValidator.validate_temporal_params(
                params['window_size'], params['lambda_smooth'])
            if not valid:
                errors.append(msg)
                
        # 样条强度
        if 'th_strength' in params:
            valid, msg = ParameterValidator.validate_spline_strength(params['th_strength'])
            if not valid:
                errors.append(msg)
                
        # 样条节点
        if 'th_nodes' in params:
            _, warning = ParameterValidator.validate_spline_nodes(params['th_nodes'])
            if warning:
                errors.append(f"样条节点警告: {warning}")
                
        return len(errors) == 0, errors
        
    @staticmethod
    def sanitize_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        清理和修正参数
        
        Args:
            params: 输入参数字典
            
        Returns:
            修正后的参数字典
        """
        sanitized = params.copy()
        defaults = ParameterValidator.DEFAULT_PARAMS
        
        # 使用默认值填充缺失参数
        for key, default_value in defaults.items():
            if key not in sanitized:
                sanitized[key] = default_value
                
        # 修正Phoenix参数
        if 'p' in sanitized:
            p_range = ParameterValidator.PHOENIX_P_RANGE
            sanitized['p'] = max(p_range[0], min(p_range[1], float(sanitized['p'])))
        if 'a' in sanitized:
            a_range = ParameterValidator.PHOENIX_A_RANGE
            sanitized['a'] = max(a_range[0], min(a_range[1], float(sanitized['a'])))
            
        # 修正样条节点
        if 'th_nodes' in sanitized:
            sanitized['th_nodes'], _ = ParameterValidator.validate_spline_nodes(sanitized['th_nodes'])
            
        return sanitized