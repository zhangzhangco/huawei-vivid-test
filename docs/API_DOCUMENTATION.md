# HDR色调映射专利可视化工具 - API文档

## 目录
1. [概述](#概述)
2. [核心模块](#核心模块)
3. [API参考](#api参考)
4. [使用示例](#使用示例)
5. [扩展开发](#扩展开发)

## 概述

本文档描述了HDR色调映射专利可视化工具的核心API接口，供开发者集成和扩展使用。

### 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    Gradio用户界面                            │
├─────────────────────────────────────────────────────────────┤
│                    核心计算模块                              │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐   │
│  │Phoenix计算器│质量指标计算器│图像处理器   │状态管理器   │   │
│  └─────────────┴─────────────┴─────────────┴─────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    支持模块                                  │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐   │
│  │PQ转换器     │参数验证器   │错误处理器   │导出管理器   │   │
│  └─────────────┴─────────────┴─────────────┴─────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 导入方式

```python
# 导入核心模块
from core import (
    PhoenixCurveCalculator,
    QualityMetricsCalculator,
    ImageProcessor,
    PQConverter,
    get_state_manager,
    get_export_manager
)

# 导入数据类型
from core import (
    SessionState,
    TemporalStateData,
    ImageStats,
    EstimationResult,
    CurveData
)
```

## 核心模块

### 1. PhoenixCurveCalculator

Phoenix曲线计算器，实现核心的色调映射算法。

#### 类定义
```python
class PhoenixCurveCalculator:
    def __init__(self, display_samples: int = 512)
```

#### 主要方法

##### compute_phoenix_curve()
```python
def compute_phoenix_curve(self, L: np.ndarray, p: float, a: float) -> np.ndarray:
    """
    计算Phoenix曲线
    
    Args:
        L: 输入亮度数组 (PQ域, 0-1)
        p: 亮度控制因子 (0.1-6.0)
        a: 缩放因子 (0.0-1.0)
        
    Returns:
        np.ndarray: 输出亮度数组 (PQ域, 0-1)
        
    Raises:
        ValueError: 参数超出有效范围
        RuntimeError: 计算过程中出现数值错误
    """
```

##### validate_monotonicity()
```python
def validate_monotonicity(self, L_out: np.ndarray) -> bool:
    """
    验证曲线单调性
    
    Args:
        L_out: 输出亮度数组
        
    Returns:
        bool: 是否单调递增
    """
```

##### normalize_endpoints()
```python
def normalize_endpoints(self, L_out: np.ndarray, 
                       start_val: float = 0.0, 
                       end_val: float = 1.0) -> np.ndarray:
    """
    端点归一化
    
    Args:
        L_out: 输入曲线
        start_val: 起点目标值
        end_val: 终点目标值
        
    Returns:
        np.ndarray: 归一化后的曲线
    """
```

#### 使用示例
```python
# 创建计算器
calc = PhoenixCurveCalculator(display_samples=1024)

# 计算Phoenix曲线
L = np.linspace(0, 1, 1024)
L_out = calc.compute_phoenix_curve(L, p=2.0, a=0.5)

# 验证单调性
is_monotonic = calc.validate_monotonicity(L_out)

# 端点归一化
normalized = calc.normalize_endpoints(L_out, 0.0, 1.0)
```

### 2. QualityMetricsCalculator

质量指标计算器，提供多种图像质量评估指标。

#### 类定义
```python
class QualityMetricsCalculator:
    def __init__(self, luminance_channel: str = "MaxRGB")
```

#### 主要方法

##### compute_perceptual_distortion()
```python
def compute_perceptual_distortion(self, L_in: np.ndarray, L_out: np.ndarray) -> float:
    """
    计算感知失真
    
    Args:
        L_in: 输入亮度数组
        L_out: 输出亮度数组
        
    Returns:
        float: 感知失真值 (0-1)
    """
```

##### compute_local_contrast()
```python
def compute_local_contrast(self, L: np.ndarray) -> float:
    """
    计算局部对比度
    
    Args:
        L: 亮度数组
        
    Returns:
        float: 局部对比度值
    """
```

##### recommend_mode_with_hysteresis()
```python
def recommend_mode_with_hysteresis(self, distortion: float) -> str:
    """
    带滞回特性的模式推荐
    
    Args:
        distortion: 感知失真值
        
    Returns:
        str: "自动模式" 或 "艺术模式"
    """
```

#### 使用示例
```python
# 创建质量指标计算器
quality_calc = QualityMetricsCalculator(luminance_channel="MaxRGB")

# 计算感知失真
distortion = quality_calc.compute_perceptual_distortion(L_in, L_out)

# 计算局部对比度
contrast = quality_calc.compute_local_contrast(L_out)

# 获取模式推荐
mode = quality_calc.recommend_mode_with_hysteresis(distortion)
```

### 3. ImageProcessor

图像处理器，处理多格式图像的加载、转换和色调映射。

#### 类定义
```python
class ImageProcessor:
    def __init__(self)
```

#### 主要方法

##### convert_to_pq_domain()
```python
def convert_to_pq_domain(self, image: np.ndarray, input_format: str) -> np.ndarray:
    """
    转换图像到PQ域
    
    Args:
        image: 输入图像数组
        input_format: 输入格式 ("sRGB", "Rec2020", "Linear")
        
    Returns:
        np.ndarray: PQ域图像
    """
```

##### apply_tone_mapping()
```python
def apply_tone_mapping(self, pq_image: np.ndarray, 
                      tone_curve_func: callable,
                      luminance_channel: str = "MaxRGB") -> np.ndarray:
    """
    应用色调映射
    
    Args:
        pq_image: PQ域图像
        tone_curve_func: 色调映射函数
        luminance_channel: 亮度通道类型
        
    Returns:
        np.ndarray: 映射后的PQ域图像
    """
```

##### get_image_stats()
```python
def get_image_stats(self, pq_image: np.ndarray, 
                   luminance_channel: str) -> ImageStats:
    """
    获取图像统计信息
    
    Args:
        pq_image: PQ域图像
        luminance_channel: 亮度通道类型
        
    Returns:
        ImageStats: 图像统计信息对象
    """
```

#### 使用示例
```python
# 创建图像处理器
processor = ImageProcessor()

# 转换到PQ域
pq_image = processor.convert_to_pq_domain(image, "sRGB")

# 定义色调映射函数
def tone_curve_func(L):
    calc = PhoenixCurveCalculator()
    return calc.compute_phoenix_curve(L, 2.0, 0.5)

# 应用色调映射
mapped_image = processor.apply_tone_mapping(pq_image, tone_curve_func)

# 获取统计信息
stats = processor.get_image_stats(pq_image, "MaxRGB")
```

### 4. PQConverter

PQ转换器，实现ST 2084标准的感知量化转换。

#### 类定义
```python
class PQConverter:
    def __init__(self)
```

#### 主要方法

##### linear_to_pq()
```python
def linear_to_pq(self, L: np.ndarray) -> np.ndarray:
    """
    线性光转PQ域
    
    Args:
        L: 线性光亮度 (nits)
        
    Returns:
        np.ndarray: PQ域值 (0-1)
    """
```

##### pq_to_linear()
```python
def pq_to_linear(self, pq: np.ndarray) -> np.ndarray:
    """
    PQ域转线性光
    
    Args:
        pq: PQ域值 (0-1)
        
    Returns:
        np.ndarray: 线性光亮度 (nits)
    """
```

##### srgb_to_linear()
```python
def srgb_to_linear(self, srgb: np.ndarray) -> np.ndarray:
    """
    sRGB转线性光
    
    Args:
        srgb: sRGB值 (0-1)
        
    Returns:
        np.ndarray: 线性光值 (0-1)
    """
```

#### 使用示例
```python
# 创建PQ转换器
pq_converter = PQConverter()

# 线性光转PQ域
pq_values = pq_converter.linear_to_pq(linear_nits)

# PQ域转线性光
linear_nits = pq_converter.pq_to_linear(pq_values)

# sRGB转线性光
linear_rgb = pq_converter.srgb_to_linear(srgb_values)
```

### 5. StateManager

状态管理器，管理会话状态和时域状态的持久化。

#### 获取实例
```python
from core import get_state_manager
state_manager = get_state_manager()
```

#### 主要方法

##### update_session_state()
```python
def update_session_state(self, **kwargs) -> bool:
    """
    更新会话状态
    
    Args:
        **kwargs: 状态参数
        
    Returns:
        bool: 更新是否成功
    """
```

##### save_all_states()
```python
def save_all_states(self) -> bool:
    """
    保存所有状态到文件
    
    Returns:
        bool: 保存是否成功
    """
```

##### get_state_summary()
```python
def get_state_summary(self) -> Dict[str, Any]:
    """
    获取状态摘要
    
    Returns:
        Dict: 状态摘要信息
    """
```

#### 使用示例
```python
# 获取状态管理器
state_manager = get_state_manager()

# 更新会话状态
state_manager.update_session_state(p=2.0, a=0.5, mode="艺术模式")

# 保存状态
success = state_manager.save_all_states()

# 获取状态摘要
summary = state_manager.get_state_summary()
```

## 数据类型

### SessionState
```python
@dataclass
class SessionState:
    p: float = 2.0
    a: float = 0.5
    mode: str = "艺术模式"
    dt_low: float = 0.05
    dt_high: float = 0.10
    # ... 其他参数
    
    def to_dict(self) -> Dict[str, Any]
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState'
```

### ImageStats
```python
@dataclass
class ImageStats:
    min_pq: float
    max_pq: float
    avg_pq: float
    var_pq: float
    input_format: str
    processing_path: str
    pixel_count: int
```

### EstimationResult
```python
@dataclass
class EstimationResult:
    p_estimated: float
    a_estimated: float
    min_pq: float
    max_pq: float
    avg_pq: float
    var_pq: float
    p_raw: float
    a_raw: float
    statistics: str
```

### CurveData
```python
@dataclass
class CurveData:
    input_luminance: np.ndarray
    output_luminance: np.ndarray
    phoenix_curve: np.ndarray
```

## 使用示例

### 完整的处理流程

```python
import numpy as np
from core import (
    PhoenixCurveCalculator,
    QualityMetricsCalculator,
    ImageProcessor,
    PQConverter,
    get_state_manager,
    get_export_manager
)

def process_hdr_image(image_path: str, p: float = 2.0, a: float = 0.5):
    """完整的HDR图像处理流程"""
    
    # 1. 初始化组件
    phoenix_calc = PhoenixCurveCalculator()
    quality_calc = QualityMetricsCalculator()
    image_processor = ImageProcessor()
    pq_converter = PQConverter()
    state_manager = get_state_manager()
    export_manager = get_export_manager()
    
    # 2. 加载和转换图像
    # (假设已有图像加载逻辑)
    image = load_image(image_path)  # 用户实现
    pq_image = image_processor.convert_to_pq_domain(image, "sRGB")
    
    # 3. 计算Phoenix曲线
    L = np.linspace(0, 1, 1024)
    L_out = phoenix_calc.compute_phoenix_curve(L, p, a)
    
    # 4. 验证曲线质量
    is_monotonic = phoenix_calc.validate_monotonicity(L_out)
    if not is_monotonic:
        print("警告: 曲线非单调，建议调整参数")
    
    # 5. 应用色调映射
    def tone_curve_func(x):
        return phoenix_calc.compute_phoenix_curve(x, p, a)
    
    mapped_image = image_processor.apply_tone_mapping(
        pq_image, tone_curve_func, "MaxRGB"
    )
    
    # 6. 计算质量指标
    stats_before = image_processor.get_image_stats(pq_image, "MaxRGB")
    stats_after = image_processor.get_image_stats(mapped_image, "MaxRGB")
    
    L_in = quality_calc.extract_luminance(pq_image)
    L_mapped = quality_calc.extract_luminance(mapped_image)
    
    distortion = quality_calc.compute_perceptual_distortion(L_in, L_mapped)
    contrast = quality_calc.compute_local_contrast(L_mapped)
    recommendation = quality_calc.recommend_mode_with_hysteresis(distortion)
    
    # 7. 更新状态
    state_manager.update_session_state(p=p, a=a, mode="艺术模式")
    state_manager.save_all_states()
    
    # 8. 导出结果
    from core import CurveData, SessionState
    curve_data = CurveData(
        input_luminance=L,
        output_luminance=L_out,
        phoenix_curve=L_out
    )
    session_state = SessionState(p=p, a=a)
    
    lut_file = "output.cube"
    export_success = export_manager.export_lut(
        curve_data, session_state, lut_file
    )
    
    # 9. 返回结果
    return {
        'mapped_image': mapped_image,
        'distortion': distortion,
        'contrast': contrast,
        'recommendation': recommendation,
        'export_success': export_success,
        'stats_before': stats_before,
        'stats_after': stats_after
    }

# 使用示例
result = process_hdr_image("input.exr", p=2.2, a=0.6)
print(f"处理完成，感知失真: {result['distortion']:.6f}")
```

### 批量处理示例

```python
def batch_process_images(image_list: List[str], 
                        output_dir: str,
                        p: float = 2.0, 
                        a: float = 0.5):
    """批量处理HDR图像"""
    
    # 初始化组件
    phoenix_calc = PhoenixCurveCalculator()
    image_processor = ImageProcessor()
    export_manager = get_export_manager()
    
    results = []
    
    for i, image_path in enumerate(image_list):
        print(f"处理图像 {i+1}/{len(image_list)}: {image_path}")
        
        try:
            # 处理单张图像
            result = process_hdr_image(image_path, p, a)
            
            # 保存结果
            output_path = os.path.join(output_dir, f"processed_{i:03d}.exr")
            save_image(result['mapped_image'], output_path)  # 用户实现
            
            # 导出LUT
            lut_path = os.path.join(output_dir, f"lut_{i:03d}.cube")
            # ... 导出逻辑
            
            results.append({
                'input': image_path,
                'output': output_path,
                'lut': lut_path,
                'distortion': result['distortion'],
                'success': True
            })
            
        except Exception as e:
            print(f"处理失败: {e}")
            results.append({
                'input': image_path,
                'success': False,
                'error': str(e)
            })
    
    return results

# 使用示例
image_list = ["image1.exr", "image2.exr", "image3.exr"]
results = batch_process_images(image_list, "output/", p=2.2, a=0.6)
```

### 自定义扩展示例

```python
class CustomToneMappingOperator:
    """自定义色调映射算子"""
    
    def __init__(self):
        self.phoenix_calc = PhoenixCurveCalculator()
        self.quality_calc = QualityMetricsCalculator()
    
    def custom_curve(self, L: np.ndarray, 
                    p: float, a: float, 
                    gamma: float = 1.0) -> np.ndarray:
        """自定义曲线：Phoenix + Gamma校正"""
        
        # 应用Phoenix曲线
        phoenix_out = self.phoenix_calc.compute_phoenix_curve(L, p, a)
        
        # 应用Gamma校正
        gamma_out = np.power(phoenix_out, 1.0/gamma)
        
        return gamma_out
    
    def evaluate_quality(self, L_in: np.ndarray, 
                        L_out: np.ndarray) -> Dict[str, float]:
        """评估质量指标"""
        
        distortion = self.quality_calc.compute_perceptual_distortion(L_in, L_out)
        contrast = self.quality_calc.compute_local_contrast(L_out)
        
        # 自定义指标
        dynamic_range = np.max(L_out) - np.min(L_out)
        
        return {
            'distortion': distortion,
            'contrast': contrast,
            'dynamic_range': dynamic_range
        }

# 使用自定义算子
custom_op = CustomToneMappingOperator()
L = np.linspace(0, 1, 1024)
custom_curve = custom_op.custom_curve(L, p=2.0, a=0.5, gamma=2.2)
quality_metrics = custom_op.evaluate_quality(L, custom_curve)
```

## 扩展开发

### 添加新的色调映射算法

1. **继承基础接口**
```python
from abc import ABC, abstractmethod

class ToneMappingOperator(ABC):
    @abstractmethod
    def compute_curve(self, L: np.ndarray, **params) -> np.ndarray:
        pass
    
    @abstractmethod
    def validate_parameters(self, **params) -> Tuple[bool, str]:
        pass
```

2. **实现具体算法**
```python
class MyToneMappingOperator(ToneMappingOperator):
    def compute_curve(self, L: np.ndarray, **params) -> np.ndarray:
        # 实现自定义算法
        pass
    
    def validate_parameters(self, **params) -> Tuple[bool, str]:
        # 实现参数验证
        pass
```

### 添加新的质量指标

```python
class CustomQualityMetric:
    def compute_metric(self, L_in: np.ndarray, L_out: np.ndarray) -> float:
        """计算自定义质量指标"""
        # 实现自定义指标计算
        pass
    
    def get_metric_name(self) -> str:
        return "Custom Metric"
    
    def get_metric_range(self) -> Tuple[float, float]:
        return (0.0, 1.0)  # 指标范围
```

### 集成到主界面

```python
# 在gradio_app.py中集成自定义组件
from my_extensions import MyToneMappingOperator, CustomQualityMetric

class ExtendedGradioInterface(GradioInterface):
    def __init__(self):
        super().__init__()
        self.custom_operator = MyToneMappingOperator()
        self.custom_metric = CustomQualityMetric()
    
    def create_extended_interface(self):
        # 扩展界面逻辑
        pass
```

---

**版本**: 1.0  
**更新日期**: 2025-10-27  
**维护者**: HDR Tone Mapping Team