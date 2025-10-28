"""
Auto模式参数估算器
实现基于图像统计的Phoenix曲线参数自动估算功能
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
import logging

from .image_processor import ImageStats


@dataclass
class AutoModeConfig:
    """Auto模式配置参数"""
    # 基础参数
    p0: float = 1.0              # p基础值
    a0: float = 0.3              # a基础值
    alpha: float = 0.5           # p调节系数
    beta: float = 0.3            # a调节系数
    
    # 参数范围约束
    p_min: float = 0.1
    p_max: float = 6.0
    a_min: float = 0.0
    a_max: float = 1.0
    
    # 估算策略参数
    enable_adaptive_scaling: bool = True    # 启用自适应缩放
    contrast_weight: float = 0.2           # 对比度权重
    brightness_weight: float = 0.8         # 亮度权重
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AutoModeConfig':
        """从字典创建"""
        return cls(**data)


@dataclass
class EstimationResult:
    """估算结果"""
    # 最终参数
    p_estimated: float
    a_estimated: float
    
    # 中间统计量
    min_pq: float
    max_pq: float
    avg_pq: float
    var_pq: float
    
    # 估算过程
    p_raw: float                # 原始估算值
    a_raw: float                # 原始估算值
    p_clipped: bool             # p是否被裁剪
    a_clipped: bool             # a是否被裁剪
    
    # 配置参数
    config: AutoModeConfig
    
    # 质量评估
    confidence_score: float     # 估算置信度 (0-1)
    estimation_quality: str     # 估算质量描述
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['config'] = self.config.to_dict()
        return result


class AutoModeParameterEstimator:
    """Auto模式参数估算器
    
    基于图像统计信息自动估算Phoenix曲线的最优参数p和a
    实现线性估参公式和可配置的超参数系统
    """
    
    def __init__(self, config: Optional[AutoModeConfig] = None):
        """初始化估算器
        
        Args:
            config: Auto模式配置，如果为None则使用默认配置
        """
        self.config = config or AutoModeConfig()
        self.last_estimation = None
        
    def estimate_parameters(self, image_stats: ImageStats) -> EstimationResult:
        """基于图像统计估算参数
        
        Args:
            image_stats: 图像统计信息
            
        Returns:
            估算结果
        """
        # 提取统计量
        min_pq = image_stats.min_pq
        max_pq = image_stats.max_pq
        avg_pq = image_stats.avg_pq
        var_pq = image_stats.var_pq
        
        # 线性估参公式
        # p = p0 + α*(max_pq - avg_pq)
        p_raw = self.config.p0 + self.config.alpha * (max_pq - avg_pq)
        
        # a = a0 + β*(avg_pq - min_pq)
        a_raw = self.config.a0 + self.config.beta * (avg_pq - min_pq)
        
        # 自适应调整（如果启用）
        if self.config.enable_adaptive_scaling:
            p_raw, a_raw = self._apply_adaptive_scaling(
                p_raw, a_raw, min_pq, max_pq, avg_pq, var_pq
            )
        
        # 参数裁剪到有效范围
        p_clipped = not (self.config.p_min <= p_raw <= self.config.p_max)
        a_clipped = not (self.config.a_min <= a_raw <= self.config.a_max)
        
        p_estimated = np.clip(p_raw, self.config.p_min, self.config.p_max)
        a_estimated = np.clip(a_raw, self.config.a_min, self.config.a_max)
        
        # 计算置信度和质量评估
        confidence_score = self._calculate_confidence(
            min_pq, max_pq, avg_pq, var_pq, p_clipped, a_clipped
        )
        
        estimation_quality = self._assess_estimation_quality(
            confidence_score, p_clipped, a_clipped
        )
        
        # 创建结果
        result = EstimationResult(
            p_estimated=float(p_estimated),
            a_estimated=float(a_estimated),
            min_pq=min_pq,
            max_pq=max_pq,
            avg_pq=avg_pq,
            var_pq=var_pq,
            p_raw=float(p_raw),
            a_raw=float(a_raw),
            p_clipped=p_clipped,
            a_clipped=a_clipped,
            config=self.config,
            confidence_score=confidence_score,
            estimation_quality=estimation_quality
        )
        
        self.last_estimation = result
        return result
        
    def _apply_adaptive_scaling(self, p_raw: float, a_raw: float,
                              min_pq: float, max_pq: float, 
                              avg_pq: float, var_pq: float) -> Tuple[float, float]:
        """应用自适应缩放调整
        
        Args:
            p_raw, a_raw: 原始估算值
            min_pq, max_pq, avg_pq, var_pq: 图像统计量
            
        Returns:
            调整后的参数值
        """
        # 计算动态范围和对比度指标
        dynamic_range = max_pq - min_pq
        contrast_ratio = var_pq / (avg_pq + 1e-8)
        
        # 基于动态范围调整p
        if dynamic_range > 0.8:  # 高动态范围
            p_adjustment = 0.2 * self.config.brightness_weight
        elif dynamic_range < 0.3:  # 低动态范围
            p_adjustment = -0.1 * self.config.brightness_weight
        else:
            p_adjustment = 0.0
            
        # 基于对比度调整a
        if contrast_ratio > 0.1:  # 高对比度
            a_adjustment = 0.1 * self.config.contrast_weight
        elif contrast_ratio < 0.02:  # 低对比度
            a_adjustment = -0.05 * self.config.contrast_weight
        else:
            a_adjustment = 0.0
            
        return p_raw + p_adjustment, a_raw + a_adjustment
        
    def _calculate_confidence(self, min_pq: float, max_pq: float, 
                            avg_pq: float, var_pq: float,
                            p_clipped: bool, a_clipped: bool) -> float:
        """计算估算置信度
        
        Args:
            min_pq, max_pq, avg_pq, var_pq: 图像统计量
            p_clipped, a_clipped: 参数是否被裁剪
            
        Returns:
            置信度分数 (0-1)
        """
        confidence = 1.0
        
        # 动态范围因子
        dynamic_range = max_pq - min_pq
        if dynamic_range < 0.1:  # 动态范围过小
            confidence *= 0.7
        elif dynamic_range > 0.9:  # 动态范围过大
            confidence *= 0.8
            
        # 亮度分布因子
        if avg_pq < 0.1 or avg_pq > 0.9:  # 过暗或过亮
            confidence *= 0.8
            
        # 方差因子
        if var_pq < 0.01:  # 方差过小，图像过于平坦
            confidence *= 0.6
        elif var_pq > 0.2:  # 方差过大，图像过于复杂
            confidence *= 0.9
            
        # 参数裁剪惩罚
        if p_clipped:
            confidence *= 0.7
        if a_clipped:
            confidence *= 0.8
            
        return max(0.0, min(1.0, confidence))
        
    def _assess_estimation_quality(self, confidence_score: float,
                                 p_clipped: bool, a_clipped: bool) -> str:
        """评估估算质量
        
        Args:
            confidence_score: 置信度分数
            p_clipped, a_clipped: 参数是否被裁剪
            
        Returns:
            质量描述字符串
        """
        if confidence_score >= 0.9:
            quality = "优秀"
        elif confidence_score >= 0.7:
            quality = "良好"
        elif confidence_score >= 0.5:
            quality = "一般"
        else:
            quality = "较差"
            
        # 添加警告信息
        warnings = []
        if p_clipped:
            warnings.append("p参数被裁剪")
        if a_clipped:
            warnings.append("a参数被裁剪")
            
        if warnings:
            quality += f" ({', '.join(warnings)})"
            
        return quality
        
    def update_config(self, **kwargs) -> None:
        """更新配置参数
        
        Args:
            **kwargs: 要更新的配置参数
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logging.warning(f"未知的配置参数: {key}")
                
    def reset_to_defaults(self) -> None:
        """重置到默认配置"""
        self.config = AutoModeConfig()
        
    def get_estimation_summary(self) -> Dict[str, Any]:
        """获取估算摘要信息
        
        Returns:
            包含估算过程关键信息的字典
        """
        if self.last_estimation is None:
            return {"status": "no_estimation", "message": "尚未进行参数估算"}
            
        result = self.last_estimation
        
        return {
            "status": "success",
            "final_parameters": {
                "p": result.p_estimated,
                "a": result.a_estimated
            },
            "image_statistics": {
                "min_pq": result.min_pq,
                "max_pq": result.max_pq,
                "avg_pq": result.avg_pq,
                "var_pq": result.var_pq
            },
            "estimation_process": {
                "p_raw": result.p_raw,
                "a_raw": result.a_raw,
                "p_clipped": result.p_clipped,
                "a_clipped": result.a_clipped
            },
            "quality_assessment": {
                "confidence_score": result.confidence_score,
                "estimation_quality": result.estimation_quality
            },
            "hyperparameters": {
                "p0": result.config.p0,
                "a0": result.config.a0,
                "alpha": result.config.alpha,
                "beta": result.config.beta
            }
        }
        
    def export_estimation_data(self) -> Dict[str, Any]:
        """导出完整的估算数据
        
        Returns:
            完整的估算数据字典
        """
        if self.last_estimation is None:
            return {"error": "no_estimation_available"}
            
        return self.last_estimation.to_dict()
        
    def validate_hyperparameters(self, p0: float, a0: float, 
                                alpha: float, beta: float) -> Tuple[bool, str]:
        """验证超参数的有效性
        
        Args:
            p0, a0: 基础参数
            alpha, beta: 调节系数
            
        Returns:
            (is_valid, message): 验证结果和消息
        """
        # 检查基础参数范围
        if not (0.1 <= p0 <= 6.0):
            return False, f"p0={p0}超出有效范围[0.1, 6.0]"
        if not (0.0 <= a0 <= 1.0):
            return False, f"a0={a0}超出有效范围[0.0, 1.0]"
            
        # 检查调节系数范围（建议范围）
        if not (-2.0 <= alpha <= 2.0):
            return False, f"alpha={alpha}超出建议范围[-2.0, 2.0]"
        if not (-1.0 <= beta <= 1.0):
            return False, f"beta={beta}超出建议范围[-1.0, 1.0]"
            
        return True, "超参数验证通过"


class AutoModeInterface:
    """Auto模式用户界面接口
    
    提供估参过程的可观测界面和一键操作功能
    """
    
    def __init__(self, estimator: AutoModeParameterEstimator):
        """初始化界面接口
        
        Args:
            estimator: 参数估算器实例
        """
        self.estimator = estimator
        self._default_config = AutoModeConfig()
        
    def get_observable_data(self) -> Dict[str, Any]:
        """获取可观测的估参数据
        
        Returns:
            用于界面显示的数据字典
        """
        summary = self.estimator.get_estimation_summary()
        
        if summary["status"] != "success":
            return {
                "status": summary["status"],
                "message": summary.get("message", ""),
                "hyperparameters": self.estimator.config.to_dict(),
                "has_estimation": False
            }
            
        return {
            "status": "success",
            "has_estimation": True,
            
            # 超参数显示
            "hyperparameters": {
                "p0": summary["hyperparameters"]["p0"],
                "a0": summary["hyperparameters"]["a0"],
                "alpha": summary["hyperparameters"]["alpha"],
                "beta": summary["hyperparameters"]["beta"]
            },
            
            # 图像统计显示
            "image_statistics": {
                "min_pq": f"{summary['image_statistics']['min_pq']:.4f}",
                "max_pq": f"{summary['image_statistics']['max_pq']:.4f}",
                "avg_pq": f"{summary['image_statistics']['avg_pq']:.4f}",
                "var_pq": f"{summary['image_statistics']['var_pq']:.6f}"
            },
            
            # 估算结果显示
            "estimation_results": {
                "p_estimated": f"{summary['final_parameters']['p']:.3f}",
                "a_estimated": f"{summary['final_parameters']['a']:.3f}",
                "p_raw": f"{summary['estimation_process']['p_raw']:.3f}",
                "a_raw": f"{summary['estimation_process']['a_raw']:.3f}",
                "p_clipped": summary['estimation_process']['p_clipped'],
                "a_clipped": summary['estimation_process']['a_clipped']
            },
            
            # 质量评估显示
            "quality_assessment": {
                "confidence_score": f"{summary['quality_assessment']['confidence_score']:.2f}",
                "estimation_quality": summary['quality_assessment']['estimation_quality']
            }
        }
        
    def apply_estimated_parameters(self) -> Tuple[bool, Dict[str, float], str]:
        """一键应用估算参数到滑块
        
        Returns:
            (success, parameters, message): 应用结果
        """
        if self.estimator.last_estimation is None:
            return False, {}, "尚未进行参数估算，请先上传图像"
            
        result = self.estimator.last_estimation
        parameters = {
            "p": result.p_estimated,
            "a": result.a_estimated
        }
        
        message = f"已应用估算参数: p={result.p_estimated:.3f}, a={result.a_estimated:.3f}"
        if result.p_clipped or result.a_clipped:
            message += " (部分参数已裁剪到有效范围)"
            
        return True, parameters, message
        
    def restore_default_hyperparameters(self) -> Tuple[Dict[str, float], str]:
        """恢复默认超参数
        
        Returns:
            (hyperparameters, message): 默认超参数和消息
        """
        self.estimator.reset_to_defaults()
        
        hyperparams = {
            "p0": self._default_config.p0,
            "a0": self._default_config.a0,
            "alpha": self._default_config.alpha,
            "beta": self._default_config.beta
        }
        
        message = "已恢复默认超参数设置"
        return hyperparams, message
        
    def update_hyperparameters(self, p0: float, a0: float, 
                             alpha: float, beta: float) -> Tuple[bool, str]:
        """更新超参数
        
        Args:
            p0, a0: 基础参数
            alpha, beta: 调节系数
            
        Returns:
            (success, message): 更新结果
        """
        # 验证参数
        valid, message = self.estimator.validate_hyperparameters(p0, a0, alpha, beta)
        if not valid:
            return False, message
            
        # 更新配置
        self.estimator.update_config(p0=p0, a0=a0, alpha=alpha, beta=beta)
        
        return True, "超参数更新成功"
        
    def get_hyperparameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """获取超参数的建议范围
        
        Returns:
            超参数范围字典
        """
        return {
            "p0": (0.1, 6.0),
            "a0": (0.0, 1.0),
            "alpha": (-2.0, 2.0),
            "beta": (-1.0, 1.0)
        }
        
    def format_estimation_report(self) -> str:
        """格式化估算报告
        
        Returns:
            格式化的估算报告字符串
        """
        data = self.get_observable_data()
        
        if not data["has_estimation"]:
            return "尚未进行参数估算"
            
        report = []
        report.append("=== Auto模式参数估算报告 ===")
        report.append("")
        
        # 图像统计
        stats = data["image_statistics"]
        report.append("图像统计信息:")
        report.append(f"  最小PQ值: {stats['min_pq']}")
        report.append(f"  最大PQ值: {stats['max_pq']}")
        report.append(f"  平均PQ值: {stats['avg_pq']}")
        report.append(f"  方差: {stats['var_pq']}")
        report.append("")
        
        # 超参数
        hyper = data["hyperparameters"]
        report.append("当前超参数:")
        report.append(f"  p0 = {hyper['p0']}")
        report.append(f"  a0 = {hyper['a0']}")
        report.append(f"  α = {hyper['alpha']}")
        report.append(f"  β = {hyper['beta']}")
        report.append("")
        
        # 估算结果
        results = data["estimation_results"]
        report.append("估算结果:")
        report.append(f"  原始估算: p = {results['p_raw']}, a = {results['a_raw']}")
        report.append(f"  最终参数: p = {results['p_estimated']}, a = {results['a_estimated']}")
        
        if results['p_clipped'] or results['a_clipped']:
            clipped = []
            if results['p_clipped']:
                clipped.append("p")
            if results['a_clipped']:
                clipped.append("a")
            report.append(f"  注意: {', '.join(clipped)} 参数已被裁剪到有效范围")
        report.append("")
        
        # 质量评估
        quality = data["quality_assessment"]
        report.append("质量评估:")
        report.append(f"  置信度: {quality['confidence_score']}")
        report.append(f"  估算质量: {quality['estimation_quality']}")
        
        return "\n".join(report)