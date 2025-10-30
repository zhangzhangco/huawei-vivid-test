# HDR质量评估扩展模块 API 文档

## 概述

HDR质量评估扩展模块为HDR色调映射专利可视化工具提供了高级质量指标计算、自动质量评估和艺术家友好的语义化提示功能。本文档详细介绍了模块的API接口、使用方法和配置选项。

## 模块架构

```
src/core/
├── metrics_extension.py      # 核心质量指标计算模块
├── config_manager.py         # 配置管理模块
└── ui_integration.py         # UI集成模块
```

## 核心类

### ExtendedMetrics

主要的质量评估类，提供所有质量指标的计算功能。

#### 初始化

```python
from src.core.metrics_extension import ExtendedMetrics

# 使用默认配置
metrics = ExtendedMetrics()

# 使用自定义配置文件
metrics = ExtendedMetrics(config_path="custom/metrics.json")
```

#### 主要方法

##### get_all_metrics(lin, lout)

计算所有质量指标的主要接口。

**参数:**
- `lin` (np.ndarray): 输入亮度数据，PQ域，范围0-1
- `lout` (np.ndarray): 输出亮度数据，PQ域，映射后，范围0-1

**返回值:**
```python
{
    # 基础统计数据
    'Lmin_in': float,      # 输入最小亮度
    'Lmax_in': float,      # 输入最大亮度
    'Lmin_out': float,     # 输出最小亮度
    'Lmax_out': float,     # 输出最大亮度
    
    # 曝光相关指标
    'S_ratio': float,      # 高光饱和比例 (0-1)
    'C_shadow': float,     # 暗部压缩比例 (0-1)
    'R_DR': float,         # 动态范围保持率
    'ΔL_mean_norm': float, # 归一化平均亮度漂移
    
    # 直方图分析
    'Hist_overlap': float, # 直方图重叠度 (0-1)
    
    # 状态评估
    'Exposure_status': str,    # 质量状态 ("正常", "过曝", "过暗", "动态范围异常")
    'Status_display': str      # 格式化状态显示 (包含emoji)
}
```

**使用示例:**
```python
import numpy as np
from src.core.metrics_extension import ExtendedMetrics

# 创建测试数据
lin = np.random.rand(1000, 1000).astype(np.float32)
lout = np.random.rand(1000, 1000).astype(np.float32)

# 计算质量指标
metrics = ExtendedMetrics()
results = metrics.get_all_metrics(lin, lout)

print(f"高光饱和比例: {results['S_ratio']:.3f}")
print(f"质量状态: {results['Exposure_status']}")
```

##### calculate_basic_stats(lin, lout)

计算基础统计数据。

**返回值:**
```python
{
    'Lmin_in': float,   # 输入最小亮度值
    'Lmax_in': float,   # 输入最大亮度值
    'Lmin_out': float,  # 输出最小亮度值
    'Lmax_out': float   # 输出最大亮度值
}
```

##### calculate_exposure_metrics(lin, lout)

计算曝光相关指标。

**返回值:**
```python
{
    'S_ratio': float,      # 高光饱和比例
    'C_shadow': float,     # 暗部压缩比例
    'R_DR': float,         # 动态范围保持率
    'ΔL_mean_norm': float  # 归一化平均亮度漂移
}
```

##### calculate_histogram_overlap(lin, lout)

计算直方图重叠度。

**返回值:** `float` - 重叠度值 (0-1之间)

##### evaluate_quality_status(metrics)

基于指标自动判断质量状态。

**参数:**
- `metrics` (dict): 质量指标字典

**返回值:** `str` - 质量状态 ("正常", "过曝", "过暗", "动态范围异常", "评估失败")

##### to_json(metrics, indent=2)

将指标转换为JSON格式字符串。

**参数:**
- `metrics` (dict): 指标字典
- `indent` (int): JSON缩进级别

**返回值:** `str` - JSON格式字符串

#### 配置管理方法

##### reload_thresholds()

重新加载阈值配置，支持热更新。

##### get_current_thresholds()

获取当前使用的阈值配置。

**返回值:**
```python
{
    'S_ratio': float,        # 高光饱和比例阈值
    'C_shadow': float,       # 暗部压缩比例阈值
    'R_DR_tolerance': float, # 动态范围容差
    'Dprime': float          # D'指标阈值
}
```

##### update_threshold(key, value)

更新单个阈值。

**参数:**
- `key` (str): 阈值配置项名称
- `value` (float): 新的阈值

**返回值:** `bool` - 是否更新成功

### ConfigManager

配置管理类，处理阈值配置的加载、验证和热更新。

#### 初始化

```python
from src.core.config_manager import ConfigManager

config_manager = ConfigManager("config/metrics.json")
```

#### 主要方法

##### load_thresholds()

加载质量判定阈值，支持热更新。

##### get_default_thresholds()

获取默认阈值配置。

##### validate_thresholds(thresholds)

验证阈值配置的有效性。

##### create_default_config_file()

创建默认配置文件。

##### update_threshold(key, value)

更新单个阈值配置。

##### reset_to_defaults()

重置配置为默认值。

## 配置文件

### config/metrics.json

质量评估阈值配置文件。

```json
{
  "S_ratio": 0.05,        // 高光饱和比例阈值 (0-1)
  "C_shadow": 0.1,        // 暗部压缩比例阈值 (0-1)
  "R_DR_tolerance": 0.2,  // 动态范围保持率容差 (0-1)
  "Dprime": 0.25          // D'指标阈值 (0-1)
}
```

#### 配置说明

- **S_ratio**: 高光饱和比例阈值，超过此值判定为过曝
- **C_shadow**: 暗部压缩比例阈值，超过此值判定为过暗
- **R_DR_tolerance**: 动态范围保持率容差，偏离1.0超过此值判定为异常
- **Dprime**: 感知失真阈值，超过此值可能判定为过曝

## 性能特性

### 性能目标

- **处理时间**: 30ms内完成1MP图像处理
- **内存优化**: 使用float32减少内存占用
- **向量化计算**: 利用numpy向量化操作提升性能

### 性能优化特性

1. **预先展平数组**: 避免重复flatten操作
2. **批量计算**: 一次性计算多个指标
3. **向量化操作**: 使用numpy的高效函数
4. **内存管理**: 就地操作减少内存分配

### 性能监控

模块内置性能监控，当处理时间超过30ms时会记录警告日志。

```python
# 性能监控示例
metrics = ExtendedMetrics()
results = metrics.get_all_metrics(lin, lout)
# 如果处理时间超过30ms，会在日志中看到警告信息
```

## 错误处理

### 异常类型

1. **数值异常**: 除零错误，使用1e-6作为保护阈值
2. **配置异常**: 配置文件损坏或缺失，自动回退到默认值
3. **数据异常**: 输入数据格式错误，返回错误信息
4. **计算异常**: 计算过程中的其他错误，记录日志并返回失败状态

### 错误恢复

```python
# 示例：处理计算错误
try:
    results = metrics.get_all_metrics(lin, lout)
    if 'error' in results:
        print(f"计算失败: {results['error']}")
        # 使用默认值或重试
except Exception as e:
    print(f"发生异常: {e}")
```

## 集成指南

### 与主系统集成

```python
# 在主处理流程中集成质量评估
from src.core.metrics_extension import ExtendedMetrics

def process_image_with_quality_assessment(lin, lout):
    # 执行原有的Phoenix曲线处理
    processed_result = phoenix_curve_processing(lin, lout)
    
    # 添加质量评估
    metrics = ExtendedMetrics()
    quality_results = metrics.get_all_metrics(lin, lout)
    
    # 合并结果
    processed_result.update(quality_results)
    
    return processed_result
```

### UI集成

```python
# 更新Gradio界面显示
def update_quality_display(metrics):
    status = metrics.get('Exposure_status', '未知')
    status_display = metrics.get('Status_display', '❓ 未知')
    
    # 更新质量摘要区
    quality_summary = f"""
    状态: {status_display}
    高光饱和: {metrics.get('S_ratio', 0) * 100:.1f}%
    暗部压缩: {metrics.get('C_shadow', 0) * 100:.1f}%
    动态范围保持率: {metrics.get('R_DR', 1.0):.2f}
    """
    
    return quality_summary
```

## 测试

### 单元测试

```python
# 运行质量评估模块测试
python -m pytest tests/test_metrics_extension.py -v

# 运行配置管理测试
python -m pytest tests/test_config_manager.py -v
```

### 性能测试

```python
import time
import numpy as np
from src.core.metrics_extension import ExtendedMetrics

# 性能测试示例
def performance_test():
    metrics = ExtendedMetrics()
    
    # 创建1MP测试数据
    lin = np.random.rand(1000, 1000).astype(np.float32)
    lout = np.random.rand(1000, 1000).astype(np.float32)
    
    start_time = time.time()
    results = metrics.get_all_metrics(lin, lout)
    elapsed_time = (time.time() - start_time) * 1000
    
    print(f"处理时间: {elapsed_time:.1f}ms")
    print(f"性能目标: {'✓' if elapsed_time <= 30 else '✗'}")
    
    return elapsed_time <= 30

# 运行性能测试
performance_test()
```

## 故障排除

### 常见问题

1. **配置文件加载失败**
   - 检查config/metrics.json文件是否存在
   - 验证JSON格式是否正确
   - 查看日志中的错误信息

2. **性能不达标**
   - 检查输入数据类型（建议使用float32）
   - 确认数据大小是否合理
   - 查看性能监控日志

3. **质量评估结果异常**
   - 验证输入数据范围（应在0-1之间）
   - 检查阈值配置是否合理
   - 查看计算过程中的警告信息

### 调试模式

```python
import logging

# 启用调试日志
logging.basicConfig(level=logging.DEBUG)

# 创建质量评估实例
metrics = ExtendedMetrics()

# 查看详细的计算过程
results = metrics.get_all_metrics(lin, lout)
```

## 版本信息

- **模块版本**: 1.0.0
- **兼容性**: Python 3.7+
- **依赖**: numpy, json, logging, pathlib
- **最后更新**: 2024年

## 许可证

本模块遵循与主项目相同的许可证条款。