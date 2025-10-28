"""
状态管理器 - 实现会话状态和时域状态的分离存储
支持JSON序列化、自动保存/加载和状态一致性验证
"""

import json
import os
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np
from datetime import datetime

from .temporal_smoothing import TemporalStats, TemporalState


@dataclass
class SessionState:
    """会话状态 - UI参数和用户配置"""
    
    # Phoenix曲线参数
    p: float = 2.0
    a: float = 0.5
    
    # 工作模式
    mode: str = "艺术模式"  # "自动模式" 或 "艺术模式"
    
    # 质量指标参数
    dt_low: float = 0.05
    dt_high: float = 0.10
    luminance_channel: str = "MaxRGB"  # "MaxRGB" 或 "Y"
    
    # 样条曲线参数
    enable_spline: bool = False
    th1: float = 0.2
    th2: float = 0.5
    th3: float = 0.8
    th_strength: float = 0.0
    
    # 时域平滑参数
    window_size: int = 9
    lambda_smooth: float = 0.3
    
    # Auto模式参数
    auto_alpha: float = 1.0
    auto_beta: float = 0.5
    auto_p0: float = 2.0
    auto_a0: float = 0.5
    
    # UI设置
    auto_save_enabled: bool = True
    performance_mode: bool = False
    max_image_size: int = 1048576  # 1MP
    
    # 元数据
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        """从字典创建"""
        # 过滤掉不存在的字段
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class TemporalStateData:
    """时域状态数据 - 时域平滑缓冲和统计"""
    
    # 时域缓冲
    parameter_history: List[Tuple[float, float]] = field(default_factory=list)  # (p, a) 历史
    distortion_history: List[float] = field(default_factory=list)  # D' 历史
    timestamp_history: List[float] = field(default_factory=list)  # 时间戳历史
    
    # 当前状态
    current_frame: int = 0
    total_frames: int = 0
    
    # 统计信息
    variance_reduction: float = 0.0
    smoothing_active: bool = False
    last_mode: str = ""
    last_channel: str = ""
    last_image_hash: str = ""
    
    # 元数据
    created_time: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # JSON序列化会将元组转换为列表，这里保持一致
        return data
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemporalStateData':
        """从字典创建"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        # 将parameter_history中的列表转换回元组
        if 'parameter_history' in filtered_data:
            filtered_data['parameter_history'] = [
                tuple(item) if isinstance(item, list) and len(item) == 2 else item
                for item in filtered_data['parameter_history']
            ]
            
        return cls(**filtered_data)
        
    def should_reset(self, mode: str, channel: str, image_hash: str) -> bool:
        """判断是否需要重置时域状态"""
        return (self.last_mode != mode or 
                self.last_channel != channel or 
                self.last_image_hash != image_hash)
                
    def reset(self, mode: str, channel: str, image_hash: str):
        """重置时域状态"""
        self.parameter_history.clear()
        self.distortion_history.clear()
        self.timestamp_history.clear()
        self.current_frame = 0
        self.total_frames = 0
        self.variance_reduction = 0.0
        self.smoothing_active = False
        self.last_mode = mode
        self.last_channel = channel
        self.last_image_hash = image_hash
        self.last_updated = datetime.now().isoformat()


class StateManager:
    """状态管理器 - 管理会话状态和时域状态的分离存储"""
    
    def __init__(self, state_dir: str = ".kiro_state"):
        """
        初始化状态管理器
        
        Args:
            state_dir: 状态文件存储目录
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)
        
        self.session_file = self.state_dir / "session_state.json"
        self.temporal_file = self.state_dir / "temporal_state.json"
        
        # 状态对象
        self._session_state: Optional[SessionState] = None
        self._temporal_state: Optional[TemporalStateData] = None
        
        # 自动保存设置
        self.auto_save_interval = 30.0  # 30秒
        self.last_save_time = 0.0
        
        # 状态变化监听器
        self.state_change_listeners: List[callable] = []
        
        # 日志设置
        self.logger = logging.getLogger(__name__)
        
    def get_session_state(self) -> SessionState:
        """获取会话状态"""
        if self._session_state is None:
            self._session_state = self.load_session_state()
        return self._session_state
        
    def get_temporal_state(self) -> TemporalStateData:
        """获取时域状态"""
        if self._temporal_state is None:
            self._temporal_state = self.load_temporal_state()
        return self._temporal_state
        
    def update_session_state(self, **kwargs) -> bool:
        """
        更新会话状态
        
        Args:
            **kwargs: 要更新的状态字段
            
        Returns:
            bool: 更新是否成功
        """
        try:
            session_state = self.get_session_state()
            
            # 更新字段
            for key, value in kwargs.items():
                if hasattr(session_state, key):
                    setattr(session_state, key, value)
                else:
                    self.logger.warning(f"未知的会话状态字段: {key}")
                    
            # 更新时间戳
            session_state.last_updated = datetime.now().isoformat()
            
            # 触发状态变化事件
            self._notify_state_change("session", kwargs)
            
            # 自动保存
            if session_state.auto_save_enabled:
                self._auto_save_if_needed()
                
            return True
            
        except Exception as e:
            self.logger.error(f"更新会话状态失败: {e}")
            return False
            
    def update_temporal_state(self, p: float, a: float, distortion: float,
                            mode: str, channel: str, image_hash: str = "") -> bool:
        """
        更新时域状态
        
        Args:
            p: Phoenix参数p
            a: Phoenix参数a
            distortion: 感知失真值
            mode: 当前模式
            channel: 当前亮度通道
            image_hash: 当前图像哈希
            
        Returns:
            bool: 更新是否成功
        """
        try:
            temporal_state = self.get_temporal_state()
            
            # 检查是否需要重置
            if temporal_state.should_reset(mode, channel, image_hash):
                temporal_state.reset(mode, channel, image_hash)
                self.logger.info("时域状态已重置")
                
            # 添加新数据
            current_time = time.time()
            temporal_state.parameter_history.append((p, a))
            temporal_state.distortion_history.append(distortion)
            temporal_state.timestamp_history.append(current_time)
            
            # 更新计数器
            temporal_state.current_frame += 1
            temporal_state.total_frames += 1
            
            # 限制历史长度
            max_history = self.get_session_state().window_size * 2
            if len(temporal_state.parameter_history) > max_history:
                temporal_state.parameter_history = temporal_state.parameter_history[-max_history:]
                temporal_state.distortion_history = temporal_state.distortion_history[-max_history:]
                temporal_state.timestamp_history = temporal_state.timestamp_history[-max_history:]
                
            # 计算方差降低
            if len(temporal_state.distortion_history) >= 2:
                recent_var = np.var(temporal_state.distortion_history[-5:])
                total_var = np.var(temporal_state.distortion_history)
                if total_var > 0:
                    temporal_state.variance_reduction = max(0, (total_var - recent_var) / total_var * 100)
                    
            # 更新平滑状态
            temporal_state.smoothing_active = len(temporal_state.parameter_history) >= 3
            temporal_state.last_updated = datetime.now().isoformat()
            
            # 触发状态变化事件
            self._notify_state_change("temporal", {
                "frame": temporal_state.current_frame,
                "variance_reduction": temporal_state.variance_reduction
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"更新时域状态失败: {e}")
            return False
            
    def clear_temporal_state(self, mode: str = "", channel: str = "", image_hash: str = "") -> bool:
        """
        清空时域状态
        
        Args:
            mode: 新模式
            channel: 新通道
            image_hash: 新图像哈希
            
        Returns:
            bool: 清空是否成功
        """
        try:
            temporal_state = self.get_temporal_state()
            temporal_state.reset(mode, channel, image_hash)
            
            self.logger.info("时域状态已清空")
            self._notify_state_change("temporal_reset", {})
            
            return True
            
        except Exception as e:
            self.logger.error(f"清空时域状态失败: {e}")
            return False
            
    def save_session_state(self) -> bool:
        """保存会话状态到文件"""
        try:
            session_state = self.get_session_state()
            
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(session_state.to_dict(), f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"会话状态已保存到 {self.session_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存会话状态失败: {e}")
            return False
            
    def save_temporal_state(self) -> bool:
        """保存时域状态到文件"""
        try:
            temporal_state = self.get_temporal_state()
            
            with open(self.temporal_file, 'w', encoding='utf-8') as f:
                json.dump(temporal_state.to_dict(), f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"时域状态已保存到 {self.temporal_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存时域状态失败: {e}")
            return False
            
    def load_session_state(self) -> SessionState:
        """从文件加载会话状态"""
        try:
            if self.session_file.exists():
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                session_state = SessionState.from_dict(data)
                self.logger.info(f"会话状态已从 {self.session_file} 加载")
                return session_state
                
        except Exception as e:
            self.logger.warning(f"加载会话状态失败，使用默认值: {e}")
            
        # 返回默认状态
        return SessionState()
        
    def load_temporal_state(self) -> TemporalStateData:
        """从文件加载时域状态"""
        try:
            if self.temporal_file.exists():
                with open(self.temporal_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                temporal_state = TemporalStateData.from_dict(data)
                self.logger.info(f"时域状态已从 {self.temporal_file} 加载")
                return temporal_state
                
        except Exception as e:
            self.logger.warning(f"加载时域状态失败，使用默认值: {e}")
            
        # 返回默认状态
        return TemporalStateData()
        
    def save_all_states(self) -> bool:
        """保存所有状态"""
        session_ok = self.save_session_state()
        temporal_ok = self.save_temporal_state()
        return session_ok and temporal_ok
        
    def validate_state_consistency(self) -> Tuple[bool, List[str]]:
        """
        验证状态一致性
        
        Returns:
            Tuple[bool, List[str]]: (是否一致, 错误信息列表)
        """
        errors = []
        
        try:
            session_state = self.get_session_state()
            temporal_state = self.get_temporal_state()
            
            # 检查参数范围
            if not (0.1 <= session_state.p <= 6.0):
                errors.append(f"会话状态中p值超出范围: {session_state.p}")
                
            if not (0.0 <= session_state.a <= 1.0):
                errors.append(f"会话状态中a值超出范围: {session_state.a}")
                
            # 检查样条节点
            if session_state.enable_spline:
                nodes = [session_state.th1, session_state.th2, session_state.th3]
                if not all(0.0 <= node <= 1.0 for node in nodes):
                    errors.append("样条节点超出[0,1]范围")
                    
                if not (nodes[0] < nodes[1] < nodes[2]):
                    errors.append("样条节点顺序不正确")
                    
            # 检查时域状态长度一致性
            hist_lens = [
                len(temporal_state.parameter_history),
                len(temporal_state.distortion_history),
                len(temporal_state.timestamp_history)
            ]
            
            if len(set(hist_lens)) > 1:
                errors.append(f"时域历史长度不一致: {hist_lens}")
                
            # 检查时间戳单调性
            if len(temporal_state.timestamp_history) > 1:
                timestamps = temporal_state.timestamp_history
                if not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
                    errors.append("时域时间戳非单调递增")
                    
        except Exception as e:
            errors.append(f"状态验证过程中出现异常: {e}")
            
        return len(errors) == 0, errors
        
    def get_state_summary(self) -> Dict[str, Any]:
        """获取状态摘要信息"""
        try:
            session_state = self.get_session_state()
            temporal_state = self.get_temporal_state()
            
            return {
                "session": {
                    "mode": session_state.mode,
                    "parameters": {
                        "p": session_state.p,
                        "a": session_state.a,
                        "enable_spline": session_state.enable_spline
                    },
                    "last_updated": session_state.last_updated,
                    "auto_save": session_state.auto_save_enabled
                },
                "temporal": {
                    "current_frame": temporal_state.current_frame,
                    "total_frames": temporal_state.total_frames,
                    "history_length": len(temporal_state.parameter_history),
                    "variance_reduction": temporal_state.variance_reduction,
                    "smoothing_active": temporal_state.smoothing_active,
                    "last_updated": temporal_state.last_updated
                },
                "files": {
                    "session_exists": self.session_file.exists(),
                    "temporal_exists": self.temporal_file.exists(),
                    "session_size": self.session_file.stat().st_size if self.session_file.exists() else 0,
                    "temporal_size": self.temporal_file.stat().st_size if self.temporal_file.exists() else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"获取状态摘要失败: {e}")
            return {"error": str(e)}
            
    def add_state_change_listener(self, listener: callable):
        """添加状态变化监听器"""
        self.state_change_listeners.append(listener)
        
    def remove_state_change_listener(self, listener: callable):
        """移除状态变化监听器"""
        if listener in self.state_change_listeners:
            self.state_change_listeners.remove(listener)
            
    def _notify_state_change(self, state_type: str, changes: Dict[str, Any]):
        """通知状态变化"""
        for listener in self.state_change_listeners:
            try:
                listener(state_type, changes)
            except Exception as e:
                self.logger.error(f"状态变化监听器执行失败: {e}")
                
    def _auto_save_if_needed(self):
        """根据需要自动保存"""
        current_time = time.time()
        if current_time - self.last_save_time >= self.auto_save_interval:
            self.save_all_states()
            self.last_save_time = current_time
            
    def reset_all_states(self) -> bool:
        """重置所有状态到默认值"""
        try:
            self._session_state = SessionState()
            self._temporal_state = TemporalStateData()
            
            # 删除状态文件
            if self.session_file.exists():
                self.session_file.unlink()
            if self.temporal_file.exists():
                self.temporal_file.unlink()
                
            self.logger.info("所有状态已重置")
            self._notify_state_change("reset_all", {})
            
            return True
            
        except Exception as e:
            self.logger.error(f"重置状态失败: {e}")
            return False
            
    def export_states(self, export_path: str) -> bool:
        """
        导出所有状态到指定路径
        
        Args:
            export_path: 导出文件路径
            
        Returns:
            bool: 导出是否成功
        """
        try:
            export_data = {
                "session_state": self.get_session_state().to_dict(),
                "temporal_state": self.get_temporal_state().to_dict(),
                "export_time": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"状态已导出到 {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出状态失败: {e}")
            return False
            
    def import_states(self, import_path: str) -> bool:
        """
        从指定路径导入状态
        
        Args:
            import_path: 导入文件路径
            
        Returns:
            bool: 导入是否成功
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
                
            # 导入会话状态
            if "session_state" in import_data:
                self._session_state = SessionState.from_dict(import_data["session_state"])
                
            # 导入时域状态
            if "temporal_state" in import_data:
                self._temporal_state = TemporalStateData.from_dict(import_data["temporal_state"])
                
            # 保存到文件
            self.save_all_states()
            
            self.logger.info(f"状态已从 {import_path} 导入")
            self._notify_state_change("import", {"path": import_path})
            
            return True
            
        except Exception as e:
            self.logger.error(f"导入状态失败: {e}")
            return False


# 全局状态管理器实例
_global_state_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """获取全局状态管理器实例"""
    global _global_state_manager
    if _global_state_manager is None:
        _global_state_manager = StateManager()
    return _global_state_manager


def reset_state_manager():
    """重置全局状态管理器"""
    global _global_state_manager
    _global_state_manager = None