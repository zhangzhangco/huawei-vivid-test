# HDR质量评估扩展模块设计文档

## 概述

HDR质量评估扩展模块是对现有HDR色调映射专利可视化工具的增强，旨在提供客观的质量指标计算、自动质量评估、直方图可视化修复和艺术家友好的语义化提示功能。该模块采用非侵入式设计，确保与现有Phoenix曲线核心算法的兼容性。

## 架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    HDR主系统                                │
├─────────────────────────────────────────────────────────────┤
│  Phoenix曲线处理  │  图像处理器  │  Gradio界面              │
├─────────────────────────────────────────────────────────────┤
│                质量评估扩展模块                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ExtendedMetrics│  │ConfigManager│  │UIIntegration│         │
│  │质量指标计算  │  │配置管理     │  │界面集成     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### 模块分层设计

1. **核心计算层**: ExtendedMetrics类负责所有质量指标的计算
2. **配置管理层**: ConfigManager处理阈值配置和热更新
3. **界面集成层**: UIIntegration负责Gradio界面的更新和显示
4. **数据流层**: 处理Lin/Lout数据的传递和JSON格式化

## 组件和接口

### 1. ExtendedMetrics类

**设计决策**: 采用单一职责原则，将所有质量指标计算集中在一个类中，便于维护和测试。

```python
class ExtendedMetrics:
    def __init__(self, config_path="config/metrics.json"):
        """初始化质量评估模块"""
        
    def calculate_basic_stats(self, lin: np.ndarray, lout: np.ndarray) -> dict:
        """计算基础统计数据 (Lmin_in, Lmax_in, Lmin_out, Lmax_out)"""
        
    def calculate_exposure_metrics(self, lin: np.ndarray, lout: np.ndarray) -> dict:
        """计算曝光相关指标 (S_ratio, C_shadow, R_DR, ΔL_mean_norm)"""
        
    def calculate_histogram_overlap(self, lin: np.ndarray, lout: np.ndarray) -> float:
        """计算直方图重叠度"""
        
    def evaluate_quality_status(self, metrics: dict) -> str:
        """基于指标自动判断质量状态"""
        
    def get_all_metrics(self, lin: np.ndarray, lout: np.ndarray) -> dict:
        """一次性计算所有质量指标"""
```

**性能考虑**: 
- 使用numpy向量化操作确保30ms内完成1MP图像处理
- 实现除零保护机制，使用1e-6作为安全阈值
- 采用lazy evaluation，只在需要时计算特定指标

### 2. ConfigManager类

**设计决策**: 独立的配置管理确保阈值的灵活性和可维护性。

```python
class ConfigManager:
    def __init__(self, config_path="config/metrics.json"):
        """初始化配置管理器"""
        
    def load_thresholds(self) -> dict:
        """加载质量判定阈值"""
        
    def get_default_thresholds(self) -> dict:
        """获取默认阈值配置"""
        
    def validate_thresholds(self, thresholds: dict) -> bool:
        """验证阈值配置的有效性"""
```

**默认阈值配置**:
```json
{
    "S_ratio": 0.05,
    "C_shadow": 0.10, 
    "R_DR_tolerance": 0.2,
    "Dprime": 0.25
}
```

### 3. UIIntegration类

**设计决策**: 分离界面逻辑，确保与Gradio组件的松耦合。

```python
class UIIntegration:
    def update_quality_summary(self, metrics: dict, status: str) -> dict:
        """更新质量摘要显示"""
        
    def update_pq_histogram(self, lin: np.ndarray, lout: np.ndarray) -> plt.Figure:
        """更新PQ直方图显示"""
        
    def generate_artist_tips(self, metrics: dict, status: str) -> str:
        """生成艺术家模式提示"""
        
    def format_percentage_display(self, value: float) -> str:
        """格式化百分比显示 (0.078 -> 7.8%)"""
```

## 数据模型

### 质量指标数据结构

```python
@dataclass
class QualityMetrics:
    # 基础统计
    Lmin_in: float
    Lmax_in: float  
    Lmin_out: float
    Lmax_out: float
    
    # 曝光指标
    S_ratio: float          # 高光饱和比例
    C_shadow: float         # 暗部压缩比例  
    R_DR: float            # 动态范围保持率
    ΔL_mean_norm: float    # 归一化平均亮度漂移
    
    # 直方图分析
    Hist_overlap: float     # 直方图重叠度
    
    # 状态评估
    Exposure_status: str    # 质量状态标识
```

### JSON输出格式

**设计决策**: 保持与现有系统的兼容性，采用扁平化JSON结构便于解析。

```json
{
  "Lmin_in": 0.02,
  "Lmax_in": 0.98,
  "Lmin_out": 0.01, 
  "Lmax_out": 0.94,
  "S_ratio": 0.078,
  "C_shadow": 0.021,
  "R_DR": 1.21,
  "ΔL_mean_norm": 1.38,
  "Hist_overlap": 0.41,
  "Exposure_status": "过曝"
}
```

## 错误处理

### 异常处理策略

1. **数值异常**: 使用1e-6除零保护，确保计算稳定性
2. **配置异常**: 配置文件损坏时回退到默认值，记录警告日志
3. **数据异常**: Lin/Lout数据格式错误时返回空指标，不中断主流程
4. **界面异常**: UI更新失败时静默处理，不影响核心功能

### 错误恢复机制

```python
def safe_divide(numerator: float, denominator: float, fallback: float = 1e-6) -> float:
    """安全除法操作"""
    return numerator / max(denominator, fallback)

def safe_log(value: float, fallback: float = 1e-6) -> float:
    """安全对数操作"""  
    return np.log(max(value, fallback))
```

## 测试策略

### 单元测试

**设计决策**: 采用数据驱动测试，使用预定义的测试数据集验证指标计算的准确性。

1. **指标计算测试**: 验证每个质量指标的计算精度
2. **边界条件测试**: 测试极值输入下的系统稳定性
3. **配置管理测试**: 验证配置加载和验证逻辑
4. **性能测试**: 确保30ms性能要求

### 集成测试

1. **主流程集成**: 验证与Phoenix曲线处理的无缝集成
2. **界面集成**: 测试Gradio组件的正确更新
3. **数据流测试**: 验证Lin/Lout数据的正确传递

### 测试数据集

```python
# 测试用例数据
TEST_CASES = {
    "normal": {"lin": normal_hdr_data, "expected_status": "正常"},
    "overexposed": {"lin": bright_hdr_data, "expected_status": "过曝"},
    "underexposed": {"lin": dark_hdr_data, "expected_status": "过暗"},
    "abnormal_dr": {"lin": compressed_hdr_data, "expected_status": "动态范围异常"}
}
```

## 实现细节

### 1. PQ直方图修复

**设计决策**: 使用matplotlib重新实现直方图显示，确保数据准确性和视觉效果。

- 使用256个bins进行精确分析
- 采用density归一化确保可比性
- 双曲线对比显示(Input/Output)
- 动态更新机制，仅在数据变化时重绘

### 2. 艺术家模式实现

**设计决策**: 基于规则的语义化提示系统，提供直观的参数调整建议。

```python
def generate_artist_tips(self, metrics: dict, status: str) -> str:
    """生成艺术家友好的调整建议"""
    tips = []
    
    if status == "过曝":
        tips.append("建议减小p参数(0.1-0.3)或增大a参数")
        tips.append(f"当前高光饱和: {metrics['S_ratio']*100:.1f}%, 建议<5%")
    
    elif status == "过暗": 
        tips.append("建议增大p参数或减小a参数")
        tips.append(f"当前暗部压缩: {metrics['C_shadow']*100:.1f}%, 建议<10%")
        
    return "\n".join(tips)
```

### 3. 性能优化

**设计决策**: 采用批量计算和缓存机制提升性能。

- **向量化计算**: 使用numpy广播避免循环
- **内存优化**: 就地操作减少内存分配
- **计算缓存**: 相同参数下复用计算结果
- **异步更新**: UI更新与计算分离，避免阻塞

### 4. 集成策略

**设计决策**: 采用装饰器模式，最小化对现有代码的修改。

```python
def with_quality_assessment(func):
    """质量评估装饰器"""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)  # 原始处理
        
        # 提取Lin/Lout数据
        lin, lout = extract_luminance_data(result)
        
        # 计算质量指标
        metrics = ExtendedMetrics().get_all_metrics(lin, lout)
        
        # 更新界面
        update_quality_display(metrics)
        
        return result
    return wrapper
```

## 部署考虑

### 文件结构

```
src/
├── core/
│   ├── metrics_extension.py      # ExtendedMetrics类
│   ├── config_manager.py         # ConfigManager类
│   └── ui_integration.py         # UIIntegration类
├── config/
│   └── metrics.json              # 阈值配置文件
└── tests/
    ├── test_metrics_extension.py
    ├── test_config_manager.py
    └── test_ui_integration.py
```

### 依赖管理

**设计决策**: 复用现有依赖，避免引入新的外部库。

- numpy: 数值计算
- matplotlib: 直方图绘制  
- json: 配置管理
- gradio: 界面集成

### 向后兼容性

- 保持现有API接口不变
- 新功能通过可选参数启用
- 配置文件缺失时使用默认值
- 渐进式部署，支持功能开关