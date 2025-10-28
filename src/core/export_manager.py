"""
数据导出和诊断功能管理器
实现1D LUT (.cube格式)导出、曲线数据CSV导出、完整诊断包生成和元数据记录
"""

import json
import csv
import zipfile
import io
import os
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
import logging

from .state_manager import SessionState, TemporalStateData
from .quality_metrics import QualityMetricsCalculator
from .image_processor import ImageStats


@dataclass
class CurveData:
    """曲线数据模型"""
    input_luminance: np.ndarray
    output_luminance: np.ndarray
    phoenix_curve: np.ndarray
    spline_curve: Optional[np.ndarray] = None
    identity_line: Optional[np.ndarray] = None
    
    # 验证和归一化数据
    validation_curve: Optional[np.ndarray] = None
    normalized_curve: Optional[np.ndarray] = None
    curve_parameters: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典"""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value
        return result


@dataclass
class QualityMetrics:
    """质量指标数据模型"""
    perceptual_distortion: float
    local_contrast: float
    variance_distortion: float
    recommended_mode: str
    computation_time: float
    
    # 扩展字段
    is_monotonic: bool = True
    endpoint_error: float = 0.0
    luminance_channel: str = "MaxRGB"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class ExportMetadata:
    """导出元数据"""
    export_time: str
    version: str
    source_system: str
    parameters: Dict[str, Any]
    image_stats: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    processing_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class LUTExporter:
    """1D LUT导出器 (.cube格式)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def export_lut_cube(self, curve_data: CurveData, session_state: SessionState,
                       samples: int = 4096, filename: Optional[str] = None) -> str:
        """
        导出1D LUT到.cube格式文件
        
        Args:
            curve_data: 曲线数据
            session_state: 会话状态
            samples: LUT采样点数
            filename: 输出文件名，None时返回内容字符串
            
        Returns:
            str: 文件路径或内容字符串
        """
        try:
            # 生成高密度采样
            L_input = np.linspace(0, 1, samples)
            L_output = np.interp(L_input, curve_data.input_luminance, curve_data.output_luminance)
            
            # 生成.cube文件内容
            content = self._generate_cube_content(L_output, session_state, samples)
            
            if filename:
                # 写入文件
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.logger.info(f"1D LUT已导出到: {filename}")
                return filename
            else:
                return content
                
        except Exception as e:
            self.logger.error(f"导出1D LUT失败: {e}")
            raise
            
    def _generate_cube_content(self, lut_data: np.ndarray, session_state: SessionState,
                              samples: int) -> str:
        """生成.cube文件内容"""
        timestamp = datetime.now().isoformat()
        
        # 构建头部注释
        header_lines = [
            "# HDR Tone Mapping 1D LUT",
            f"# Generated: {timestamp}",
            f"# System: HDR色调映射专利可视化工具 v1.0",
            "#",
            "# Phoenix Curve Parameters:",
            f"# p={session_state.p:.6f}, a={session_state.a:.6f}",
            f"# Mode: {session_state.mode}",
            "#",
            "# Quality Thresholds:",
            f"# D_T_low={session_state.dt_low:.6f}, D_T_high={session_state.dt_high:.6f}",
            f"# Luminance Channel: {session_state.luminance_channel}",
            "#",
            "# Temporal Smoothing:",
            f"# Window Size M={session_state.window_size}",
            f"# Lambda λ={session_state.lambda_smooth:.6f}",
            "#",
            "# Spline Parameters:",
            f"# Enabled: {session_state.enable_spline}",
            f"# TH_nodes: [{session_state.th1:.3f}, {session_state.th2:.3f}, {session_state.th3:.3f}]",
            f"# TH_strength: {session_state.th_strength:.6f}",
            "#",
            "# Auto Mode Parameters:",
            f"# Alpha α={session_state.auto_alpha:.6f}, Beta β={session_state.auto_beta:.6f}",
            f"# p0={session_state.auto_p0:.6f}, a0={session_state.auto_a0:.6f}",
            "#",
            f"LUT_1D_SIZE {samples}",
            ""
        ]
        
        # 生成LUT数据行
        lut_lines = []
        for value in lut_data:
            # .cube格式要求RGB三个通道相同的值
            lut_lines.append(f"{value:.6f} {value:.6f} {value:.6f}")
            
        return "\n".join(header_lines + lut_lines)
        
    def validate_lut_export(self, original_curve: np.ndarray, 
                           exported_lut: np.ndarray) -> Tuple[bool, float]:
        """
        验证导出LUT的一致性
        
        Args:
            original_curve: 原始曲线数据
            exported_lut: 导出的LUT数据
            
        Returns:
            Tuple[bool, float]: (是否一致, 最大绝对误差)
        """
        try:
            # 重采样到相同长度进行比较
            if len(original_curve) != len(exported_lut):
                x_orig = np.linspace(0, 1, len(original_curve))
                x_lut = np.linspace(0, 1, len(exported_lut))
                resampled_orig = np.interp(x_lut, x_orig, original_curve)
            else:
                resampled_orig = original_curve
                
            # 计算最大绝对误差
            max_error = np.max(np.abs(resampled_orig - exported_lut))
            
            # 根据需求15.4，重建曲线最大绝对误差应≤1e-4
            is_consistent = max_error <= 1e-4
            
            return is_consistent, float(max_error)
            
        except Exception as e:
            self.logger.error(f"LUT一致性验证失败: {e}")
            return False, float('inf')


class CSVExporter:
    """曲线数据CSV导出器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def export_curve_csv(self, curve_data: CurveData, session_state: SessionState,
                        metadata: Optional[ExportMetadata] = None,
                        filename: Optional[str] = None) -> str:
        """
        导出曲线数据到CSV格式
        
        Args:
            curve_data: 曲线数据
            session_state: 会话状态
            metadata: 导出元数据
            filename: 输出文件名，None时返回内容字符串
            
        Returns:
            str: 文件路径或内容字符串
        """
        try:
            content = self._generate_csv_content(curve_data, session_state, metadata)
            
            if filename:
                with open(filename, 'w', encoding='utf-8', newline='') as f:
                    f.write(content)
                self.logger.info(f"曲线数据已导出到: {filename}")
                return filename
            else:
                return content
                
        except Exception as e:
            self.logger.error(f"导出曲线CSV失败: {e}")
            raise
            
    def _generate_csv_content(self, curve_data: CurveData, session_state: SessionState,
                             metadata: Optional[ExportMetadata]) -> str:
        """生成CSV内容"""
        output = io.StringIO()
        
        # 写入元数据头部
        self._write_csv_header(output, session_state, metadata)
        
        # 准备数据
        data_dict = {
            'input_pq': curve_data.input_luminance,
            'output_pq': curve_data.output_luminance,
            'phoenix_curve': curve_data.phoenix_curve
        }
        
        # 添加可选数据
        if curve_data.spline_curve is not None:
            data_dict['spline_curve'] = curve_data.spline_curve
            
        if curve_data.identity_line is not None:
            data_dict['identity_line'] = curve_data.identity_line
            
        if curve_data.normalized_curve is not None:
            data_dict['normalized_curve'] = curve_data.normalized_curve
            
        # 写入CSV数据
        writer = csv.DictWriter(output, fieldnames=data_dict.keys())
        writer.writeheader()
        
        # 逐行写入数据
        for i in range(len(curve_data.input_luminance)):
            row = {}
            for key, values in data_dict.items():
                row[key] = f"{values[i]:.6f}"
            writer.writerow(row)
            
        return output.getvalue()
        
    def _write_csv_header(self, output: io.StringIO, session_state: SessionState,
                         metadata: Optional[ExportMetadata]):
        """写入CSV头部元数据"""
        timestamp = datetime.now().isoformat()
        
        header_lines = [
            "# HDR Tone Mapping Curve Data",
            f"# Generated: {timestamp}",
            f"# System: HDR色调映射专利可视化工具 v1.0",
            "#",
            "# Phoenix Parameters:",
            f"# p={session_state.p:.6f}, a={session_state.a:.6f}",
            f"# Mode: {session_state.mode}",
            f"# Luminance Channel: {session_state.luminance_channel}",
            "#",
            "# Quality Thresholds:",
            f"# D_T_low={session_state.dt_low:.6f}, D_T_high={session_state.dt_high:.6f}",
            "#",
            "# Spline Configuration:",
            f"# Enabled: {session_state.enable_spline}",
            f"# Nodes: [{session_state.th1:.3f}, {session_state.th2:.3f}, {session_state.th3:.3f}]",
            f"# Strength: {session_state.th_strength:.6f}",
            "#"
        ]
        
        # 添加额外元数据
        if metadata:
            if metadata.image_stats:
                header_lines.extend([
                    "# Image Statistics:",
                    f"# Min/Avg/Max PQ: {metadata.image_stats.get('min_pq', 0):.6f}/"
                    f"{metadata.image_stats.get('avg_pq', 0):.6f}/"
                    f"{metadata.image_stats.get('max_pq', 1):.6f}",
                    f"# Variance PQ: {metadata.image_stats.get('var_pq', 0):.6f}",
                    "#"
                ])
                
            if metadata.quality_metrics:
                header_lines.extend([
                    "# Quality Metrics:",
                    f"# Perceptual Distortion D': {metadata.quality_metrics.get('perceptual_distortion', 0):.6f}",
                    f"# Local Contrast: {metadata.quality_metrics.get('local_contrast', 0):.6f}",
                    f"# Recommended Mode: {metadata.quality_metrics.get('recommended_mode', 'N/A')}",
                    "#"
                ])
                
        for line in header_lines:
            output.write(line + "\n")


class DiagnosticPackageGenerator:
    """完整诊断包生成器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.lut_exporter = LUTExporter()
        self.csv_exporter = CSVExporter()
        
    def create_diagnostic_package(self, curve_data: CurveData, session_state: SessionState,
                                temporal_state: TemporalStateData, quality_metrics: QualityMetrics,
                                image_stats: Optional[ImageStats] = None,
                                output_dir: str = "exports") -> str:
        """
        创建完整的诊断包
        
        Args:
            curve_data: 曲线数据
            session_state: 会话状态
            temporal_state: 时域状态
            quality_metrics: 质量指标
            image_stats: 图像统计信息
            output_dir: 输出目录
            
        Returns:
            str: 诊断包文件路径
        """
        try:
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            package_name = f"hdr_diagnostic_{timestamp}.zip"
            package_path = output_path / package_name
            
            # 创建元数据
            metadata = self._create_export_metadata(session_state, temporal_state, 
                                                  quality_metrics, image_stats)
            
            # 创建ZIP包
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # 添加配置文件
                self._add_config_files(zf, session_state, temporal_state, metadata)
                
                # 添加质量指标
                self._add_quality_metrics(zf, quality_metrics)
                
                # 添加曲线数据CSV
                csv_content = self.csv_exporter.export_curve_csv(curve_data, session_state, metadata)
                zf.writestr("curve_data.csv", csv_content)
                
                # 添加1D LUT
                lut_content = self.lut_exporter.export_lut_cube(curve_data, session_state)
                zf.writestr("tone_mapping.cube", lut_content)
                
                # 添加图像统计信息
                if image_stats:
                    self._add_image_stats(zf, image_stats)
                    
                # 添加时域分析
                self._add_temporal_analysis(zf, temporal_state)
                
                # 添加系统信息
                self._add_system_info(zf, metadata)
                
                # 添加README
                self._add_readme(zf, metadata)
                
            self.logger.info(f"诊断包已创建: {package_path}")
            return str(package_path)
            
        except Exception as e:
            self.logger.error(f"创建诊断包失败: {e}")
            raise
            
    def _create_export_metadata(self, session_state: SessionState, temporal_state: TemporalStateData,
                               quality_metrics: QualityMetrics, image_stats: Optional[ImageStats]) -> ExportMetadata:
        """创建导出元数据"""
        metadata = ExportMetadata(
            export_time=datetime.now().isoformat(),
            version="1.0",
            source_system="HDR色调映射专利可视化工具",
            parameters=session_state.to_dict(),
            quality_metrics=quality_metrics.to_dict()
        )
        
        if image_stats:
            metadata.image_stats = asdict(image_stats)
            
        # 添加处理信息
        metadata.processing_info = {
            "temporal_frames": temporal_state.total_frames,
            "smoothing_active": temporal_state.smoothing_active,
            "variance_reduction": temporal_state.variance_reduction,
            "last_mode": temporal_state.last_mode,
            "last_channel": temporal_state.last_channel
        }
        
        return metadata
        
    def _add_config_files(self, zf: zipfile.ZipFile, session_state: SessionState,
                         temporal_state: TemporalStateData, metadata: ExportMetadata):
        """添加配置文件到ZIP包"""
        # 会话配置
        session_config = {
            "session_state": session_state.to_dict(),
            "export_metadata": metadata.to_dict()
        }
        zf.writestr("config/session_config.json", 
                   json.dumps(session_config, indent=2, ensure_ascii=False))
        
        # 时域配置
        temporal_config = {
            "temporal_state": temporal_state.to_dict(),
            "analysis_summary": {
                "total_frames": temporal_state.total_frames,
                "current_frame": temporal_state.current_frame,
                "history_length": len(temporal_state.parameter_history),
                "variance_reduction": temporal_state.variance_reduction
            }
        }
        zf.writestr("config/temporal_config.json",
                   json.dumps(temporal_config, indent=2, ensure_ascii=False))
                   
    def _add_quality_metrics(self, zf: zipfile.ZipFile, quality_metrics: QualityMetrics):
        """添加质量指标到ZIP包"""
        metrics_data = {
            "quality_metrics": quality_metrics.to_dict(),
            "analysis": {
                "distortion_level": "低" if quality_metrics.perceptual_distortion < 0.05 else 
                                   "中" if quality_metrics.perceptual_distortion < 0.10 else "高",
                "monotonic_status": "单调" if quality_metrics.is_monotonic else "非单调",
                "endpoint_accuracy": "精确" if quality_metrics.endpoint_error < 1e-4 else "误差较大"
            }
        }
        zf.writestr("analysis/quality_metrics.json",
                   json.dumps(metrics_data, indent=2, ensure_ascii=False))
                   
    def _add_image_stats(self, zf: zipfile.ZipFile, image_stats: ImageStats):
        """添加图像统计信息到ZIP包"""
        stats_data = {
            "image_statistics": asdict(image_stats),
            "analysis": {
                "dynamic_range": image_stats.max_pq - image_stats.min_pq,
                "brightness_level": "暗" if image_stats.avg_pq < 0.3 else 
                                   "中" if image_stats.avg_pq < 0.7 else "亮",
                "contrast_level": "低" if image_stats.var_pq < 0.01 else
                                 "中" if image_stats.var_pq < 0.05 else "高"
            }
        }
        zf.writestr("analysis/image_stats.json",
                   json.dumps(stats_data, indent=2, ensure_ascii=False))
                   
    def _add_temporal_analysis(self, zf: zipfile.ZipFile, temporal_state: TemporalStateData):
        """添加时域分析到ZIP包"""
        # 计算时域统计
        if len(temporal_state.parameter_history) > 1:
            p_values = [params[0] for params in temporal_state.parameter_history]
            a_values = [params[1] for params in temporal_state.parameter_history]
            
            temporal_analysis = {
                "parameter_statistics": {
                    "p_mean": float(np.mean(p_values)),
                    "p_std": float(np.std(p_values)),
                    "p_min": float(np.min(p_values)),
                    "p_max": float(np.max(p_values)),
                    "a_mean": float(np.mean(a_values)),
                    "a_std": float(np.std(a_values)),
                    "a_min": float(np.min(a_values)),
                    "a_max": float(np.max(a_values))
                },
                "distortion_statistics": {
                    "mean": float(np.mean(temporal_state.distortion_history)),
                    "std": float(np.std(temporal_state.distortion_history)),
                    "min": float(np.min(temporal_state.distortion_history)),
                    "max": float(np.max(temporal_state.distortion_history))
                },
                "smoothing_effectiveness": {
                    "variance_reduction": temporal_state.variance_reduction,
                    "frames_processed": len(temporal_state.parameter_history),
                    "smoothing_active": temporal_state.smoothing_active
                }
            }
        else:
            temporal_analysis = {
                "parameter_statistics": "insufficient_data",
                "distortion_statistics": "insufficient_data",
                "smoothing_effectiveness": {
                    "variance_reduction": 0.0,
                    "frames_processed": len(temporal_state.parameter_history),
                    "smoothing_active": False
                }
            }
            
        zf.writestr("analysis/temporal_analysis.json",
                   json.dumps(temporal_analysis, indent=2, ensure_ascii=False))
                   
    def _add_system_info(self, zf: zipfile.ZipFile, metadata: ExportMetadata):
        """添加系统信息到ZIP包"""
        system_info = {
            "export_metadata": metadata.to_dict(),
            "system_version": "1.0",
            "export_format_version": "1.0",
            "supported_formats": {
                "lut": ".cube (1D LUT)",
                "curve_data": ".csv",
                "config": ".json",
                "package": ".zip"
            },
            "precision_info": {
                "curve_precision": "float64",
                "image_precision": "float32",
                "export_precision": "6位小数",
                "lut_samples": "4096点默认"
            }
        }
        zf.writestr("system_info.json",
                   json.dumps(system_info, indent=2, ensure_ascii=False))
                   
    def _add_readme(self, zf: zipfile.ZipFile, metadata: ExportMetadata):
        """添加README文件到ZIP包"""
        readme_content = f"""# HDR色调映射诊断包

## 包信息
- 生成时间: {metadata.export_time}
- 系统版本: {metadata.version}
- 源系统: {metadata.source_system}

## 文件结构
```
├── README.md                    # 本文件
├── system_info.json            # 系统信息和元数据
├── config/
│   ├── session_config.json     # 会话配置和参数
│   └── temporal_config.json    # 时域状态和历史
├── analysis/
│   ├── quality_metrics.json    # 质量指标分析
│   ├── image_stats.json        # 图像统计信息
│   └── temporal_analysis.json  # 时域分析结果
├── curve_data.csv              # 曲线采样数据
└── tone_mapping.cube           # 1D LUT文件

```

## 使用说明

### 1D LUT文件 (tone_mapping.cube)
- 格式: Adobe .cube 1D LUT
- 采样点数: 4096
- 精度: 6位小数
- 用途: 可导入到支持.cube格式的图像/视频处理软件

### 曲线数据 (curve_data.csv)
- 格式: CSV with metadata header
- 包含: input_pq, output_pq, phoenix_curve, spline_curve (如启用)
- 精度: 6位小数
- 用途: 数据分析、可视化、算法验证

### 配置文件 (config/*.json)
- session_config.json: 完整的UI参数和设置
- temporal_config.json: 时域平滑历史和统计
- 用途: 状态恢复、参数复现

### 分析文件 (analysis/*.json)
- quality_metrics.json: 感知失真、局部对比度等指标
- image_stats.json: 图像亮度统计和动态范围
- temporal_analysis.json: 时域平滑效果和参数稳定性
- 用途: 质量评估、算法优化

## 数据精度和一致性
- 曲线重建误差: ≤1e-4 (符合需求15.4)
- 质量指标误差: ≤1e-6 (符合需求15.5)
- 所有计算统一使用PQ域
- 完整的元数据记录确保可复现性

## 技术支持
如有问题请参考系统文档或联系技术支持。
"""
        zf.writestr("README.md", readme_content)


class ExportManager:
    """导出管理器 - 统一的数据导出接口"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.lut_exporter = LUTExporter()
        self.csv_exporter = CSVExporter()
        self.diagnostic_generator = DiagnosticPackageGenerator()
        
    def export_lut(self, curve_data: CurveData, session_state: SessionState,
                   filename: str, samples: int = 4096) -> bool:
        """
        导出1D LUT文件
        
        Args:
            curve_data: 曲线数据
            session_state: 会话状态
            filename: 输出文件名
            samples: LUT采样点数
            
        Returns:
            bool: 导出是否成功
        """
        try:
            self.lut_exporter.export_lut_cube(curve_data, session_state, samples, filename)
            return True
        except Exception as e:
            self.logger.error(f"导出LUT失败: {e}")
            return False
            
    def export_csv(self, curve_data: CurveData, session_state: SessionState,
                   filename: str, metadata: Optional[ExportMetadata] = None) -> bool:
        """
        导出曲线数据CSV文件
        
        Args:
            curve_data: 曲线数据
            session_state: 会话状态
            filename: 输出文件名
            metadata: 导出元数据
            
        Returns:
            bool: 导出是否成功
        """
        try:
            self.csv_exporter.export_curve_csv(curve_data, session_state, metadata, filename)
            return True
        except Exception as e:
            self.logger.error(f"导出CSV失败: {e}")
            return False
            
    def create_diagnostic_package(self, curve_data: CurveData, session_state: SessionState,
                                temporal_state: TemporalStateData, quality_metrics: QualityMetrics,
                                image_stats: Optional[ImageStats] = None,
                                output_dir: str = "exports") -> Optional[str]:
        """
        创建诊断包
        
        Args:
            curve_data: 曲线数据
            session_state: 会话状态
            temporal_state: 时域状态
            quality_metrics: 质量指标
            image_stats: 图像统计信息
            output_dir: 输出目录
            
        Returns:
            Optional[str]: 诊断包路径，失败时返回None
        """
        try:
            return self.diagnostic_generator.create_diagnostic_package(
                curve_data, session_state, temporal_state, quality_metrics, image_stats, output_dir
            )
        except Exception as e:
            self.logger.error(f"创建诊断包失败: {e}")
            return None
            
    def validate_export_consistency(self, original_curve: np.ndarray,
                                  exported_file: str, file_type: str = "lut") -> Tuple[bool, float]:
        """
        验证导出文件的一致性
        
        Args:
            original_curve: 原始曲线数据
            exported_file: 导出文件路径
            file_type: 文件类型 ("lut" 或 "csv")
            
        Returns:
            Tuple[bool, float]: (是否一致, 最大误差)
        """
        try:
            if file_type == "lut":
                # 读取.cube文件并验证
                exported_data = self._read_cube_file(exported_file)
                return self.lut_exporter.validate_lut_export(original_curve, exported_data)
            elif file_type == "csv":
                # 读取CSV文件并验证
                exported_data = self._read_csv_file(exported_file)
                max_error = np.max(np.abs(original_curve - exported_data))
                return max_error <= 1e-4, float(max_error)
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
                
        except Exception as e:
            self.logger.error(f"验证导出一致性失败: {e}")
            return False, float('inf')
            
    def _read_cube_file(self, filename: str) -> np.ndarray:
        """读取.cube文件数据"""
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 跳过注释和头部，提取数据
        data_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('LUT_1D_SIZE'):
                parts = line.split()
                if len(parts) >= 3:
                    # 取第一个通道的值（RGB相同）
                    data_lines.append(float(parts[0]))
                    
        return np.array(data_lines)
        
    def _read_csv_file(self, filename: str) -> np.ndarray:
        """读取CSV文件的输出曲线数据"""
        with open(filename, 'r', encoding='utf-8') as f:
            # 跳过注释行
            lines = []
            for line in f:
                if not line.strip().startswith('#'):
                    lines.append(line)
                    
        # 解析CSV数据
        reader = csv.DictReader(lines)
        output_data = []
        for row in reader:
            output_data.append(float(row['output_pq']))
            
        return np.array(output_data)
        
    def get_export_summary(self, export_path: str) -> Dict[str, Any]:
        """
        获取导出文件摘要信息
        
        Args:
            export_path: 导出文件路径
            
        Returns:
            Dict[str, Any]: 摘要信息
        """
        try:
            path = Path(export_path)
            
            summary = {
                "file_path": str(path),
                "file_name": path.name,
                "file_size": path.stat().st_size,
                "file_type": path.suffix,
                "created_time": datetime.fromtimestamp(path.stat().st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            }
            
            # 计算文件哈希
            with open(path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            summary["file_hash"] = file_hash
            
            # 根据文件类型添加特定信息
            if path.suffix == '.cube':
                summary["format"] = "1D LUT (.cube)"
                summary["estimated_samples"] = self._estimate_cube_samples(path)
            elif path.suffix == '.csv':
                summary["format"] = "Curve Data (CSV)"
                summary["estimated_rows"] = self._estimate_csv_rows(path)
            elif path.suffix == '.zip':
                summary["format"] = "Diagnostic Package (ZIP)"
                summary["archive_contents"] = self._list_zip_contents(path)
                
            return summary
            
        except Exception as e:
            self.logger.error(f"获取导出摘要失败: {e}")
            return {"error": str(e)}
            
    def _estimate_cube_samples(self, path: Path) -> int:
        """估算.cube文件的采样点数"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            data_lines = 0
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('LUT_1D_SIZE'):
                    parts = line.split()
                    if len(parts) >= 3:
                        data_lines += 1
                        
            return data_lines
        except:
            return 0
            
    def _estimate_csv_rows(self, path: Path) -> int:
        """估算CSV文件的数据行数"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            data_lines = 0
            for line in lines:
                if not line.strip().startswith('#') and line.strip():
                    data_lines += 1
                    
            return max(0, data_lines - 1)  # 减去头部行
        except:
            return 0
            
    def _list_zip_contents(self, path: Path) -> List[str]:
        """列出ZIP文件内容"""
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                return zf.namelist()
        except:
            return []


# 全局导出管理器实例
_global_export_manager: Optional[ExportManager] = None


def get_export_manager() -> ExportManager:
    """获取全局导出管理器实例"""
    global _global_export_manager
    if _global_export_manager is None:
        _global_export_manager = ExportManager()
    return _global_export_manager


def reset_export_manager():
    """重置全局导出管理器"""
    global _global_export_manager
    _global_export_manager = None