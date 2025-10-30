# HDR质量评估扩展模块部署和集成指南

## 概述

本指南详细介绍了如何部署和集成HDR质量评估扩展模块到现有的HDR色调映射专利可视化工具中。该模块采用非侵入式设计，确保与现有Phoenix曲线核心算法的完全兼容性。

## 系统要求

### 软件要求

- **Python**: 3.7 或更高版本
- **操作系统**: Linux, macOS, Windows
- **内存**: 建议4GB以上（处理大尺寸HDR图像）
- **存储**: 额外需要约10MB空间

### 依赖库

模块复用现有项目依赖，无需安装额外库：

```
numpy >= 1.19.0
matplotlib >= 3.3.0
gradio >= 3.0.0
```

## 安装部署

### 1. 文件结构检查

确认以下文件已正确放置：

```
project_root/
├── src/
│   └── core/
│       ├── metrics_extension.py      # 核心质量指标计算模块
│       ├── config_manager.py         # 配置管理模块
│       └── ui_integration.py         # UI集成模块
├── config/
│   └── metrics.json                  # 阈值配置文件
├── tests/
│   ├── test_metrics_extension.py     # 质量评估测试
│   └── test_config_manager.py        # 配置管理测试
└── docs/
    ├── HDR_QUALITY_ASSESSMENT_API.md
    └── HDR_QUALITY_ASSESSMENT_DEPLOYMENT.md
```

### 2. 配置文件初始化

检查并创建默认配置文件：

```python
from src.core.config_manager import ConfigManager

# 创建配置管理器
config_manager = ConfigManager("config/metrics.json")

# 如果配置文件不存在，创建默认配置
if not config_manager.create_default_config_file():
    print("配置文件创建失败，请检查权限")
else:
    print("配置文件创建成功")
```

### 3. 模块导入测试

验证模块能够正常导入：

```python
try:
    from src.core.metrics_extension import ExtendedMetrics
    from src.core.config_manager import ConfigManager
    from src.core.ui_integration import UIIntegration
    print("✓ 所有模块导入成功")
except ImportError as e:
    print(f"✗ 模块导入失败: {e}")
```

## 集成方案

### 方案1: 装饰器集成（推荐）

最小化对现有代码的修改，使用装饰器模式：

```python
from functools import wraps
from src.core.metrics_extension import ExtendedMetrics
from src.core.ui_integration import UIIntegration

def with_quality_assessment(func):
    """质量评估装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 执行原始处理函数
        result = func(*args, **kwargs)
        
        try:
            # 提取Lin/Lout数据（需要根据实际数据结构调整）
            lin = result.get('lin_data')  # 根据实际情况调整
            lout = result.get('lout_data')  # 根据实际情况调整
            
            if lin is not None and lout is not None:
                # 计算质量指标
                metrics = ExtendedMetrics()
                quality_results = metrics.get_all_metrics(lin, lout)
                
                # 更新UI显示
                ui_integration = UIIntegration()
                ui_updates = ui_integration.update_quality_summary(
                    quality_results, 
                    quality_results.get('Exposure_status', '未知')
                )
                
                # 合并结果
                result.update(quality_results)
                result.update(ui_updates)
                
        except Exception as e:
            print(f"质量评估过程中发生错误: {e}")
            # 不影响主流程，继续返回原始结果
        
        return result
    
    return wrapper

# 应用装饰器到主处理函数
@with_quality_assessment
def process_image_with_progress(image_data, params):
    # 原有的Phoenix曲线处理逻辑
    # ...
    return processing_result
```

### 方案2: 直接集成

直接在主处理流程中添加质量评估：

```python
from src.core.metrics_extension import ExtendedMetrics
from src.core.ui_integration import UIIntegration

def enhanced_process_image_with_progress(image_data, params):
    # 1. 执行原有的Phoenix曲线处理
    phoenix_result = original_phoenix_processing(image_data, params)
    
    # 2. 提取Lin/Lout数据
    lin = extract_lin_data(phoenix_result)
    lout = extract_lout_data(phoenix_result)
    
    # 3. 计算质量指标
    metrics = ExtendedMetrics()
    quality_results = metrics.get_all_metrics(lin, lout)
    
    # 4. 更新UI显示
    ui_integration = UIIntegration()
    ui_updates = ui_integration.update_quality_summary(
        quality_results, 
        quality_results.get('Exposure_status', '未知')
    )
    
    # 5. 合并所有结果
    final_result = {
        **phoenix_result,
        **quality_results,
        **ui_updates
    }
    
    return final_result
```

### 方案3: 事件驱动集成

使用事件系统进行松耦合集成：

```python
import threading
from typing import Callable, Dict, Any

class QualityAssessmentEventHandler:
    """质量评估事件处理器"""
    
    def __init__(self):
        self.metrics = ExtendedMetrics()
        self.ui_integration = UIIntegration()
        self.callbacks = []
    
    def register_callback(self, callback: Callable):
        """注册回调函数"""
        self.callbacks.append(callback)
    
    def handle_processing_complete(self, event_data: Dict[str, Any]):
        """处理图像处理完成事件"""
        try:
            lin = event_data.get('lin')
            lout = event_data.get('lout')
            
            if lin is not None and lout is not None:
                # 异步计算质量指标
                threading.Thread(
                    target=self._async_quality_assessment,
                    args=(lin, lout)
                ).start()
                
        except Exception as e:
            print(f"质量评估事件处理失败: {e}")
    
    def _async_quality_assessment(self, lin, lout):
        """异步质量评估"""
        quality_results = self.metrics.get_all_metrics(lin, lout)
        ui_updates = self.ui_integration.update_quality_summary(
            quality_results,
            quality_results.get('Exposure_status', '未知')
        )
        
        # 通知所有回调函数
        for callback in self.callbacks:
            callback({**quality_results, **ui_updates})

# 使用事件处理器
event_handler = QualityAssessmentEventHandler()

# 注册UI更新回调
def update_gradio_interface(results):
    # 更新Gradio界面
    pass

event_handler.register_callback(update_gradio_interface)

# 在主处理流程中触发事件
def process_image_with_events(image_data, params):
    result = phoenix_processing(image_data, params)
    
    # 触发质量评估事件
    event_handler.handle_processing_complete({
        'lin': result.get('lin'),
        'lout': result.get('lout')
    })
    
    return result
```

## Gradio界面集成

### 1. 添加质量摘要组件

```python
import gradio as gr
from src.core.ui_integration import UIIntegration

def create_quality_assessment_interface():
    """创建质量评估界面组件"""
    
    with gr.Column():
        # 质量状态显示
        quality_status = gr.HTML(
            value="<div id='quality-status'>等待处理...</div>",
            label="质量状态"
        )
        
        # 质量指标摘要
        quality_summary = gr.HTML(
            value="<div id='quality-summary'>暂无数据</div>",
            label="质量摘要"
        )
        
        # PQ直方图显示
        pq_histogram = gr.Plot(
            label="PQ直方图对比",
            value=None
        )
        
        # 艺术家模式提示
        artist_tips = gr.HTML(
            value="<div id='artist-tips'>暂无建议</div>",
            label="调整建议"
        )
    
    return quality_status, quality_summary, pq_histogram, artist_tips

def update_quality_interface(lin, lout):
    """更新质量评估界面"""
    ui_integration = UIIntegration()
    
    # 计算质量指标
    metrics = ExtendedMetrics()
    quality_results = metrics.get_all_metrics(lin, lout)
    
    # 更新各个组件
    status_html = ui_integration.format_status_display(
        quality_results.get('Exposure_status', '未知')
    )
    
    summary_html = ui_integration.format_quality_summary(quality_results)
    
    histogram_plot = ui_integration.update_pq_histogram(lin, lout)
    
    tips_html = ui_integration.generate_artist_tips(
        quality_results,
        quality_results.get('Exposure_status', '未知')
    )
    
    return status_html, summary_html, histogram_plot, tips_html
```

### 2. 集成到主界面

```python
def create_main_interface():
    """创建主界面，集成质量评估功能"""
    
    with gr.Blocks() as interface:
        # 原有的参数控制组件
        with gr.Row():
            # Phoenix曲线参数
            p_param = gr.Slider(0.1, 2.0, value=1.0, label="p参数")
            a_param = gr.Slider(0.1, 2.0, value=1.0, label="a参数")
        
        # 图像处理按钮
        process_btn = gr.Button("处理图像")
        
        # 结果显示区域
        with gr.Row():
            # 原有的图像显示
            with gr.Column():
                input_image = gr.Image(label="输入图像")
                output_image = gr.Image(label="输出图像")
            
            # 新增的质量评估区域
            with gr.Column():
                quality_components = create_quality_assessment_interface()
        
        # 绑定处理函数
        process_btn.click(
            fn=process_with_quality_assessment,
            inputs=[input_image, p_param, a_param],
            outputs=[output_image] + list(quality_components)
        )
    
    return interface

def process_with_quality_assessment(image, p_param, a_param):
    """集成质量评估的图像处理函数"""
    # 执行Phoenix曲线处理
    result = phoenix_curve_processing(image, p_param, a_param)
    
    # 提取Lin/Lout数据
    lin = result['lin_data']
    lout = result['lout_data']
    
    # 更新质量评估界面
    quality_updates = update_quality_interface(lin, lout)
    
    return [result['output_image']] + list(quality_updates)
```

## 性能优化配置

### 1. 内存优化

```python
# 在主程序启动时设置
import numpy as np

# 设置numpy使用单精度浮点数
np.seterr(all='warn')

# 配置内存映射阈值
import os
os.environ['NUMPY_MMAP_THRESHOLD'] = '1048576'  # 1MB
```

### 2. 并行处理配置

```python
# 配置numpy多线程
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

# 使用线程池处理质量评估
from concurrent.futures import ThreadPoolExecutor

class AsyncQualityAssessment:
    def __init__(self, max_workers=2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.metrics = ExtendedMetrics()
    
    def submit_assessment(self, lin, lout, callback=None):
        """提交异步质量评估任务"""
        future = self.executor.submit(self.metrics.get_all_metrics, lin, lout)
        
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        
        return future
```

### 3. 缓存配置

```python
from functools import lru_cache
import hashlib

class CachedQualityAssessment:
    def __init__(self):
        self.metrics = ExtendedMetrics()
    
    @lru_cache(maxsize=128)
    def get_cached_metrics(self, data_hash):
        """缓存质量指标计算结果"""
        # 注意：这里需要从缓存中恢复实际数据
        # 实际实现需要更复杂的缓存策略
        pass
    
    def get_data_hash(self, lin, lout):
        """计算数据哈希值"""
        combined = np.concatenate([lin.flatten(), lout.flatten()])
        return hashlib.md5(combined.tobytes()).hexdigest()
```

## 配置管理

### 1. 环境变量配置

```bash
# 设置配置文件路径
export HDR_METRICS_CONFIG="/path/to/custom/metrics.json"

# 设置日志级别
export HDR_METRICS_LOG_LEVEL="INFO"

# 设置性能监控
export HDR_METRICS_PERFORMANCE_MONITORING="true"
```

### 2. 运行时配置

```python
# 动态配置示例
from src.core.config_manager import ConfigManager

def setup_runtime_config():
    """设置运行时配置"""
    config_manager = ConfigManager()
    
    # 根据图像类型调整阈值
    image_type = detect_image_type()  # 自定义函数
    
    if image_type == "portrait":
        config_manager.update_threshold("S_ratio", 0.03)  # 人像更严格
        config_manager.update_threshold("C_shadow", 0.08)
    elif image_type == "landscape":
        config_manager.update_threshold("S_ratio", 0.07)  # 风景更宽松
        config_manager.update_threshold("C_shadow", 0.12)
```

## 测试和验证

### 1. 功能测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_metrics_extension.py -v
python -m pytest tests/test_config_manager.py -v

# 运行性能测试
python -m pytest tests/test_performance.py -v
```

### 2. 集成测试

```python
def integration_test():
    """集成测试示例"""
    import numpy as np
    from src.core.metrics_extension import ExtendedMetrics
    
    # 创建测试数据
    lin = np.random.rand(100, 100).astype(np.float32)
    lout = np.random.rand(100, 100).astype(np.float32)
    
    # 测试完整流程
    metrics = ExtendedMetrics()
    results = metrics.get_all_metrics(lin, lout)
    
    # 验证结果
    assert 'S_ratio' in results
    assert 'Exposure_status' in results
    assert isinstance(results['S_ratio'], float)
    
    print("✓ 集成测试通过")

if __name__ == "__main__":
    integration_test()
```

### 3. 性能基准测试

```python
import time
import numpy as np
from src.core.metrics_extension import ExtendedMetrics

def benchmark_performance():
    """性能基准测试"""
    metrics = ExtendedMetrics()
    
    # 测试不同尺寸的图像
    sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    
    for width, height in sizes:
        lin = np.random.rand(height, width).astype(np.float32)
        lout = np.random.rand(height, width).astype(np.float32)
        
        start_time = time.time()
        results = metrics.get_all_metrics(lin, lout)
        elapsed_time = (time.time() - start_time) * 1000
        
        pixels = width * height
        print(f"{width}x{height} ({pixels/1e6:.1f}MP): {elapsed_time:.1f}ms")
        
        # 检查是否满足性能要求
        if pixels >= 1e6:  # 1MP或以上
            assert elapsed_time <= 30, f"性能不达标: {elapsed_time:.1f}ms > 30ms"

if __name__ == "__main__":
    benchmark_performance()
```

## 故障排除

### 常见问题及解决方案

1. **模块导入失败**
   ```python
   # 检查Python路径
   import sys
   print(sys.path)
   
   # 添加项目根目录到路径
   sys.path.insert(0, '/path/to/project/root')
   ```

2. **配置文件加载失败**
   ```python
   # 检查文件权限
   import os
   config_path = "config/metrics.json"
   print(f"文件存在: {os.path.exists(config_path)}")
   print(f"可读权限: {os.access(config_path, os.R_OK)}")
   ```

3. **性能不达标**
   ```python
   # 检查数据类型
   print(f"Lin数据类型: {lin.dtype}")
   print(f"Lout数据类型: {lout.dtype}")
   
   # 转换为float32
   lin = lin.astype(np.float32)
   lout = lout.astype(np.float32)
   ```

4. **内存不足**
   ```python
   # 分块处理大图像
   def process_large_image_in_chunks(lin, lout, chunk_size=1000000):
       if lin.size <= chunk_size:
           return metrics.get_all_metrics(lin, lout)
       
       # 分块处理逻辑
       # ...
   ```

### 日志配置

```python
import logging

# 配置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hdr_quality_assessment.log'),
        logging.StreamHandler()
    ]
)

# 设置特定模块的日志级别
logging.getLogger('src.core.metrics_extension').setLevel(logging.INFO)
logging.getLogger('src.core.config_manager').setLevel(logging.WARNING)
```

## 维护和更新

### 1. 版本管理

```python
# 在metrics_extension.py中添加版本信息
__version__ = "1.0.0"
__author__ = "HDR Quality Assessment Team"
__email__ = "support@hdr-tools.com"

def get_version_info():
    """获取版本信息"""
    return {
        "version": __version__,
        "author": __author__,
        "email": __email__
    }
```

### 2. 配置迁移

```python
def migrate_config_v1_to_v2(old_config_path, new_config_path):
    """配置文件版本迁移"""
    # 读取旧版本配置
    with open(old_config_path, 'r') as f:
        old_config = json.load(f)
    
    # 转换为新版本格式
    new_config = {
        "version": "2.0",
        "thresholds": old_config,
        "performance": {
            "target_time_ms": 30,
            "use_float32": True
        }
    }
    
    # 保存新版本配置
    with open(new_config_path, 'w') as f:
        json.dump(new_config, f, indent=2)
```

### 3. 监控和报告

```python
class QualityAssessmentMonitor:
    """质量评估监控器"""
    
    def __init__(self):
        self.stats = {
            "total_assessments": 0,
            "average_time_ms": 0,
            "error_count": 0
        }
    
    def record_assessment(self, elapsed_time_ms, success=True):
        """记录评估统计"""
        self.stats["total_assessments"] += 1
        
        if success:
            # 更新平均时间
            current_avg = self.stats["average_time_ms"]
            total = self.stats["total_assessments"]
            self.stats["average_time_ms"] = (
                (current_avg * (total - 1) + elapsed_time_ms) / total
            )
        else:
            self.stats["error_count"] += 1
    
    def get_report(self):
        """生成监控报告"""
        return {
            "总评估次数": self.stats["total_assessments"],
            "平均处理时间": f"{self.stats['average_time_ms']:.1f}ms",
            "错误率": f"{self.stats['error_count'] / max(1, self.stats['total_assessments']) * 100:.1f}%"
        }
```

## 总结

HDR质量评估扩展模块的部署和集成需要注意以下关键点：

1. **兼容性**: 确保与现有系统的完全兼容
2. **性能**: 满足30ms处理时间要求
3. **稳定性**: 提供完善的错误处理和恢复机制
4. **可维护性**: 支持配置热更新和版本迁移
5. **可扩展性**: 预留接口支持未来功能扩展

通过遵循本指南，可以确保质量评估模块的成功部署和稳定运行。