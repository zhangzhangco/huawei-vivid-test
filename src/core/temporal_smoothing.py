"""
时域平滑处理器
实现加权平均时域平滑算法、冷启动机制和状态管理
支持时域缓冲的分离存储和平滑效果统计
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from dataclasses import dataclass, asdict


@dataclass
class TemporalStats:
    """时域平滑统计信息"""
    frame_count: int
    p_var_raw: float
    p_var_filtered: float
    variance_reduction: float
    delta_p_raw: float
    delta_p_filtered: float
    window_utilization: float  # 窗口利用率 (实际帧数/窗口大小)
    
    
@dataclass
class TemporalState:
    """时域状态数据结构"""
    parameter_history: List[Dict[str, float]]
    distortion_history: List[float]
    window_size: int
    last_update: str
    frame_count: int = 0
    

class TemporalSmoothingProcessor:
    """
    时域平滑处理器
    
    实现需求:
    - 5.1: 时域窗口长度M控制 (5-15帧)
    - 5.3: 平滑强度λ控制 (0.2-0.5)
    - 11.1: 权重w_k = 1/(D[N-k]+ε)加权平均计算
    - 11.3: 冷启动机制和时域缓冲清空
    - 19.1: 时域缓冲分离存储
    - 19.3: 时域状态可视化指示
    """
    
    def __init__(self, window_size: int = 9, temporal_file: str = "temporal_state.json"):
        """
        初始化时域平滑处理器
        
        Args:
            window_size: 时域窗口大小 (5-15帧)
            temporal_file: 时域状态文件路径
        """
        # 参数验证
        if not (5 <= window_size <= 15):
            raise ValueError(f"窗口大小{window_size}超出范围[5, 15]")
            
        self.window_size = window_size
        self.temporal_file = temporal_file
        self.eps = 1e-8
        
        # 时域缓冲
        self.parameter_history: List[Dict[str, float]] = []
        self.distortion_history: List[float] = []
        self.frame_count = 0
        
        # 默认参数范围
        self.lambda_range = (0.2, 0.5)
        self.default_lambda = 0.3
        
        # 加载已有状态
        self._load_temporal_state()
        
    def cold_start(self):
        """
        冷启动：清空时域缓冲 (需求 11.3)
        在切换模式、载入新图像或调整通道时调用
        """
        logging.info("执行时域平滑冷启动")
        self.parameter_history.clear()
        self.distortion_history.clear()
        self.frame_count = 0
        
        # 清空持久化状态
        self._clear_temporal_state_file()
        
    def add_frame_parameters(self, params: Dict[str, float], distortion: float):
        """
        添加新帧参数到时域缓冲
        
        Args:
            params: 参数字典 (如 {'p': 2.0, 'a': 0.5})
            distortion: 感知失真值
        """
        # 参数验证
        if not isinstance(params, dict) or not params:
            raise ValueError("参数必须是非空字典")
        if not np.isfinite(distortion) or distortion < 0:
            raise ValueError(f"失真值{distortion}无效")
            
        # 添加到历史记录
        self.parameter_history.append(params.copy())
        self.distortion_history.append(float(distortion))
        self.frame_count += 1
        
        # 保持窗口大小 (需求 5.1)
        if len(self.parameter_history) > self.window_size:
            self.parameter_history.pop(0)
            self.distortion_history.pop(0)
            
        # 自动保存状态
        self._save_temporal_state()
        
        logging.debug(f"添加帧参数: {params}, 失真: {distortion:.6f}, "
                     f"缓冲大小: {len(self.parameter_history)}/{self.window_size}")
        
    def compute_weighted_average(self) -> Dict[str, float]:
        """
        计算加权平均参数 (需求 11.1)
        权重公式: w_k = 1/(D[N-k]+ε)
        
        Returns:
            加权平均后的参数字典
        """
        if not self.parameter_history:
            return {}
            
        # 计算权重 w_k = 1/(D[N-k]+ε)
        weights = [1.0 / (d + self.eps) for d in self.distortion_history]
        weight_sum = sum(weights)
        
        if weight_sum <= self.eps:
            # 所有权重都很小，使用均匀权重
            weights = [1.0] * len(weights)
            weight_sum = len(weights)
            
        # 加权平均计算
        smoothed_params = {}
        for key in self.parameter_history[0].keys():
            weighted_sum = sum(w * params[key] for w, params in zip(weights, self.parameter_history))
            smoothed_params[key] = weighted_sum / weight_sum
            
        return smoothed_params
        
    def apply_temporal_filter(self, current_params: Dict[str, float], 
                            lambda_smooth: float = None) -> Dict[str, float]:
        """
        应用时域滤波 (需求 5.3)
        
        Args:
            current_params: 当前帧参数
            lambda_smooth: 平滑强度 (0.2-0.5)
            
        Returns:
            滤波后的参数
        """
        if lambda_smooth is None:
            lambda_smooth = self.default_lambda
            
        # 参数验证
        if not (self.lambda_range[0] <= lambda_smooth <= self.lambda_range[1]):
            raise ValueError(f"平滑强度{lambda_smooth}超出范围{self.lambda_range}")
            
        # 历史帧数不足时直接返回当前参数 (需求 11.1)
        if len(self.parameter_history) < 2:
            return current_params.copy()
            
        # 计算加权平均
        smoothed = self.compute_weighted_average()
        filtered_params = {}
        
        # 应用时域滤波
        for key, current_val in current_params.items():
            if key in smoothed:
                # 滤波公式: filtered = current + λ * (smoothed - current)
                delta_filtered = lambda_smooth * (smoothed[key] - current_val)
                filtered_params[key] = current_val + delta_filtered
            else:
                filtered_params[key] = current_val
                
        return filtered_params
        
    def get_smoothing_stats(self) -> TemporalStats:
        """
        获取平滑统计信息 (需求 19.3)
        
        Returns:
            时域平滑统计数据
        """
        if len(self.parameter_history) < 3:
            return TemporalStats(
                frame_count=self.frame_count,
                p_var_raw=0.0,
                p_var_filtered=0.0,
                variance_reduction=0.0,
                delta_p_raw=0.0,
                delta_p_filtered=0.0,
                window_utilization=len(self.parameter_history) / self.window_size
            )
            
        # 提取p参数序列
        p_raw = np.array([params.get('p', 0.0) for params in self.parameter_history], dtype=np.float64)
        
        # 计算原始方差
        var_raw = float(np.var(p_raw))
        
        # 构造滤波序列用于方差计算
        # 使用指数平滑重建滤波序列
        p_filtered = np.zeros_like(p_raw)
        p_filtered[0] = p_raw[0]
        
        for i in range(1, len(p_raw)):
            # 模拟滤波过程
            weights = 1.0 / (np.array(self.distortion_history[:i+1], dtype=np.float64) + self.eps)
            weights /= weights.sum()
            smoothed_val = np.sum(weights * p_raw[:i+1])
            
            # 应用默认平滑强度
            delta = self.default_lambda * (smoothed_val - p_raw[i])
            p_filtered[i] = p_raw[i] + delta
            
        var_filtered = float(np.var(p_filtered))
        
        # 计算方差减少率
        reduction = (var_raw - var_filtered) / (var_raw + self.eps) if var_raw > self.eps else 0.0
        
        # 计算最近的delta值
        delta_p_raw = float(p_raw[-1] - p_raw[-2]) if len(p_raw) >= 2 else 0.0
        delta_p_filtered = float(p_filtered[-1] - p_filtered[-2]) if len(p_filtered) >= 2 else 0.0
        
        return TemporalStats(
            frame_count=self.frame_count,
            p_var_raw=var_raw,
            p_var_filtered=var_filtered,
            variance_reduction=max(0.0, reduction),
            delta_p_raw=delta_p_raw,
            delta_p_filtered=delta_p_filtered,
            window_utilization=len(self.parameter_history) / self.window_size
        )
        
    def get_parameter_trends(self) -> Dict[str, List[float]]:
        """
        获取参数变化趋势
        
        Returns:
            参数趋势字典
        """
        if not self.parameter_history:
            return {}
            
        trends = {}
        for key in self.parameter_history[0].keys():
            trends[key] = [params[key] for params in self.parameter_history]
            
        return trends
        
    def get_distortion_trend(self) -> List[float]:
        """
        获取失真变化趋势
        
        Returns:
            失真值序列
        """
        return self.distortion_history.copy()
        
    def validate_smoothing_effectiveness(self, threshold: float = 0.5) -> Tuple[bool, str]:
        """
        验证平滑效果 (需求 11.3: 方差减少50%以上)
        
        Args:
            threshold: 方差减少阈值
            
        Returns:
            (是否有效, 说明信息)
        """
        stats = self.get_smoothing_stats()
        
        if stats.frame_count < 3:
            return False, f"帧数不足({stats.frame_count}<3)，无法验证平滑效果"
            
        if stats.variance_reduction >= threshold:
            return True, f"平滑有效：方差减少{stats.variance_reduction:.1%} >= {threshold:.1%}"
        else:
            return False, f"平滑效果不足：方差减少{stats.variance_reduction:.1%} < {threshold:.1%}"
            
    def set_window_size(self, new_size: int):
        """
        设置窗口大小
        
        Args:
            new_size: 新的窗口大小 (5-15)
        """
        if not (5 <= new_size <= 15):
            raise ValueError(f"窗口大小{new_size}超出范围[5, 15]")
            
        old_size = self.window_size
        self.window_size = new_size
        
        # 调整现有缓冲
        if new_size < len(self.parameter_history):
            # 缩小窗口：保留最新的帧
            excess = len(self.parameter_history) - new_size
            self.parameter_history = self.parameter_history[excess:]
            self.distortion_history = self.distortion_history[excess:]
            
        logging.info(f"窗口大小从{old_size}调整为{new_size}")
        self._save_temporal_state()
        
    def get_state_info(self) -> Dict[str, Any]:
        """
        获取状态信息 (需求 19.3)
        
        Returns:
            状态信息字典
        """
        stats = self.get_smoothing_stats()
        
        return {
            'window_size': self.window_size,
            'current_frames': len(self.parameter_history),
            'total_frames': self.frame_count,
            'window_utilization': stats.window_utilization,
            'variance_reduction': stats.variance_reduction,
            'is_effective': stats.variance_reduction >= 0.5,
            'last_update': datetime.now().isoformat(),
            'state_file_exists': os.path.exists(self.temporal_file)
        }
        
    def _save_temporal_state(self):
        """保存时域状态到文件 (需求 19.1)"""
        try:
            temporal_data = {
                'parameter_history': self.parameter_history,
                'distortion_history': self.distortion_history,
                'window_size': self.window_size,
                'frame_count': self.frame_count,
                'last_update': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(self.temporal_file, 'w', encoding='utf-8') as f:
                json.dump(temporal_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logging.warning(f"保存时域状态失败: {e}")
            
    def _load_temporal_state(self):
        """加载时域状态"""
        try:
            if os.path.exists(self.temporal_file):
                with open(self.temporal_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 验证数据完整性
                if all(key in data for key in ['parameter_history', 'distortion_history', 'window_size']):
                    self.parameter_history = data['parameter_history']
                    self.distortion_history = data['distortion_history']
                    self.frame_count = data.get('frame_count', len(self.parameter_history))
                    
                    # 如果窗口大小改变，调整缓冲
                    if data['window_size'] != self.window_size:
                        logging.info(f"窗口大小从{data['window_size']}调整为{self.window_size}")
                        if self.window_size < len(self.parameter_history):
                            excess = len(self.parameter_history) - self.window_size
                            self.parameter_history = self.parameter_history[excess:]
                            self.distortion_history = self.distortion_history[excess:]
                            
                    logging.info(f"加载时域状态: {len(self.parameter_history)}帧")
                    
        except Exception as e:
            logging.warning(f"加载时域状态失败: {e}")
            self.cold_start()
            
    def _clear_temporal_state_file(self):
        """清空时域状态文件"""
        try:
            if os.path.exists(self.temporal_file):
                os.remove(self.temporal_file)
                logging.debug("时域状态文件已清空")
        except Exception as e:
            logging.warning(f"清空时域状态文件失败: {e}")
            
    def export_temporal_data(self) -> Dict[str, Any]:
        """
        导出时域数据用于分析
        
        Returns:
            完整的时域数据
        """
        stats = self.get_smoothing_stats()
        trends = self.get_parameter_trends()
        
        return {
            'metadata': {
                'window_size': self.window_size,
                'frame_count': self.frame_count,
                'export_time': datetime.now().isoformat()
            },
            'statistics': asdict(stats),
            'parameter_trends': trends,
            'distortion_trend': self.distortion_history.copy(),
            'raw_data': {
                'parameter_history': self.parameter_history.copy(),
                'distortion_history': self.distortion_history.copy()
            }
        }
        
    def simulate_smoothing_effect(self, test_params: List[Dict[str, float]], 
                                test_distortions: List[float],
                                lambda_smooth: float = None) -> Dict[str, Any]:
        """
        模拟平滑效果 (用于测试和演示)
        
        Args:
            test_params: 测试参数序列
            test_distortions: 测试失真序列
            lambda_smooth: 平滑强度
            
        Returns:
            模拟结果
        """
        if lambda_smooth is None:
            lambda_smooth = self.default_lambda
            
        if len(test_params) != len(test_distortions):
            raise ValueError("参数序列和失真序列长度不匹配")
            
        # 保存当前状态
        original_history = self.parameter_history.copy()
        original_distortions = self.distortion_history.copy()
        original_count = self.frame_count
        
        # 清空缓冲进行模拟
        self.cold_start()
        
        raw_sequence = []
        filtered_sequence = []
        
        try:
            for params, distortion in zip(test_params, test_distortions):
                # 添加参数
                self.add_frame_parameters(params, distortion)
                
                # 记录原始值
                raw_sequence.append(params.copy())
                
                # 计算滤波值
                filtered = self.apply_temporal_filter(params, lambda_smooth)
                filtered_sequence.append(filtered)
                
            # 计算统计
            stats = self.get_smoothing_stats()
            
            return {
                'raw_sequence': raw_sequence,
                'filtered_sequence': filtered_sequence,
                'statistics': asdict(stats),
                'effectiveness': stats.variance_reduction >= 0.5
            }
            
        finally:
            # 恢复原始状态
            self.parameter_history = original_history
            self.distortion_history = original_distortions
            self.frame_count = original_count