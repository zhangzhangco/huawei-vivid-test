# HDR质量评估扩展模块使用说明

## 快速开始

### 基本使用

```python
import numpy as np
from src.core.metrics_extension import ExtendedMetrics

# 创建质量评估实例
metrics = ExtendedMetrics()

# 准备测试数据（PQ域，范围0-1）
lin = np.random.rand(1000, 1000).astype(np.float32)  # 输入亮度数据
lout = np.random.rand(1000, 1000).astype(np.float32)  # 输出亮度数据

# 计算所有质量指标
results = metrics.get_all_metrics(lin, lout)

# 查看结果
print(f"质量状态: {results['Exposure_status']}")
print(f"高光饱和比例: {results['S_ratio']:.3f}")
print(f"暗部压缩比例: {results['C_shadow']:.3f}")
print(f"动态范围保持率: {results['R_DR']:.3f}")
```

### 性能监控

```python
# 启用性能监控
metrics = ExtendedMetrics(enable_performance_monitoring=True)

# 处理多个图像
for i in range(10):
    lin = np.random.rand(1000, 1000).astype(np.float32)
    lout = np.random.rand(1000, 1000).astype(np.float32)
    results = metrics.get_all_metrics(lin, lout)

# 查看性能报告
performance_report = metrics.get_performance_report()
print(f"平均处理时间: {performance_report['average_time_ms']:.1f}ms")
print(f"成功率: {performance_report['success_rate_percent']:.1f}%")
```

## 配置管理

### 使用默认配置

```python
from src.core.config_manager import ConfigManager

# 创建配置管理器
config_manager = ConfigManager()

# 查看当前阈值
thresholds = config_manager.load_thresholds()
print("当前阈值配置:", thresholds)
```

### 自定义配置

```python
# 使用自定义配置文件
metrics = ExtendedMetrics(config_path="custom/my_metrics.json")

# 动态更新阈值
metrics.update_threshold("S_ratio", 0.03)  # 更严格的高光阈值
metrics.update_threshold("C_shadow", 0.08)  # 更严格的暗部阈值

# 重新加载配置
metrics.reload_thresholds()
```

### 创建自定义配置文件

```python
import json

# 自定义阈值配置
custom_thresholds = {
    "S_ratio": 0.03,        # 人像摄影用更严格的高光阈值
    "C_shadow": 0.08,       # 更严格的暗部阈值
    "R_DR_tolerance": 0.15, # 更严格的动态范围容差
    "Dprime": 0.20          # 更严格的感知失真阈值
}

# 保存自定义配置
with open("config/portrait_metrics.json", "w") as f:
    json.dump(custom_thresholds, f, indent=2)

# 使用自定义配置
metrics = ExtendedMetrics(config_path="config/portrait_metrics.json")
```

## 高级用法

### 批量处理

```python
def batch_quality_assessment(image_pairs, batch_size=10):
    """批量处理图像质量评估"""
    metrics = ExtendedMetrics()
    results = []
    
    for i in range(0, len(image_pairs), batch_size):
        batch = image_pairs[i:i+batch_size]
        batch_results = []
        
        for lin, lout in batch:
            result = metrics.get_all_metrics(lin, lout)
            batch_results.append(result)
        
        results.extend(batch_results)
        
        # 批量处理后清理内存
        if i % (batch_size * 5) == 0:
            import gc
            gc.collect()
    
    return results

# 使用示例
image_pairs = [(lin1, lout1), (lin2, lout2), ...]  # 图像对列表
results = batch_quality_assessment(image_pairs)
```

### 大图像优化

```python
# 为大图像处理启用优化
metrics = ExtendedMetrics()
metrics.optimize_for_large_images(enable=True)

# 处理4K图像
large_lin = np.random.rand(4000, 4000).astype(np.float32)
large_lout = np.random.rand(4000, 4000).astype(np.float32)

results = metrics.get_all_metrics(large_lin, large_lout)
print(f"4K图像处理完成，状态: {results['Exposure_status']}")
```

### 异步处理

```python
import asyncio
import concurrent.futures
from typing import List, Tuple

class AsyncQualityAssessment:
    def __init__(self, max_workers=4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.metrics = ExtendedMetrics()
    
    async def assess_quality_async(self, lin: np.ndarray, lout: np.ndarray) -> dict:
        """异步质量评估"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.metrics.get_all_metrics, 
            lin, lout
        )
    
    async def batch_assess_async(self, image_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[dict]:
        """批量异步质量评估"""
        tasks = [
            self.assess_quality_async(lin, lout) 
            for lin, lout in image_pairs
        ]
        return await asyncio.gather(*tasks)

# 使用异步处理
async def main():
    async_assessor = AsyncQualityAssessment()
    
    # 准备测试数据
    image_pairs = [
        (np.random.rand(1000, 1000).astype(np.float32), 
         np.random.rand(1000, 1000).astype(np.float32))
        for _ in range(5)
    ]
    
    # 异步批量处理
    results = await async_assessor.batch_assess_async(image_pairs)
    
    for i, result in enumerate(results):
        print(f"图像 {i+1}: {result['Exposure_status']}")

# 运行异步处理
# asyncio.run(main())
```

## 与Gradio集成

### 基本集成

```python
import gradio as gr
from src.core.metrics_extension import ExtendedMetrics
from src.core.ui_integration import UIIntegration

def create_quality_assessment_interface():
    """创建质量评估Gradio界面"""
    
    metrics = ExtendedMetrics()
    ui_integration = UIIntegration()
    
    def process_image_with_quality(image, p_param, a_param):
        """处理图像并返回质量评估结果"""
        # 这里应该是实际的Phoenix曲线处理
        # 为演示目的，我们使用随机数据
        lin = np.random.rand(*image.shape[:2]).astype(np.float32)
        lout = np.random.rand(*image.shape[:2]).astype(np.float32)
        
        # 计算质量指标
        quality_results = metrics.get_all_metrics(lin, lout)
        
        # 更新UI组件
        status_html = ui_integration.format_status_display(
            quality_results.get('Exposure_status', '未知')
        )
        
        summary_html = ui_integration.format_quality_summary(quality_results)
        
        histogram_plot = ui_integration.update_pq_histogram(lin, lout)
        
        tips_html = ui_integration.generate_artist_tips(
            quality_results,
            quality_results.get('Exposure_status', '未知')
        )
        
        return image, status_html, summary_html, histogram_plot, tips_html
    
    # 创建界面
    with gr.Blocks(title="HDR质量评估") as interface:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="输入图像")
                p_param = gr.Slider(0.1, 2.0, value=1.0, label="p参数")
                a_param = gr.Slider(0.1, 2.0, value=1.0, label="a参数")
                process_btn = gr.Button("处理图像")
            
            with gr.Column():
                output_image = gr.Image(label="输出图像")
                quality_status = gr.HTML(label="质量状态")
                quality_summary = gr.HTML(label="质量摘要")
        
        with gr.Row():
            pq_histogram = gr.Plot(label="PQ直方图对比")
            artist_tips = gr.HTML(label="调整建议")
        
        # 绑定处理函数
        process_btn.click(
            fn=process_image_with_quality,
            inputs=[input_image, p_param, a_param],
            outputs=[output_image, quality_status, quality_summary, pq_histogram, artist_tips]
        )
    
    return interface

# 启动界面
# interface = create_quality_assessment_interface()
# interface.launch()
```

### 实时更新

```python
def create_realtime_interface():
    """创建实时更新的质量评估界面"""
    
    metrics = ExtendedMetrics()
    
    def update_quality_realtime(p_param, a_param):
        """参数变化时实时更新质量评估"""
        # 使用缓存的图像数据（实际应用中从全局状态获取）
        if hasattr(update_quality_realtime, 'cached_lin') and hasattr(update_quality_realtime, 'cached_lout'):
            lin = update_quality_realtime.cached_lin
            lout = update_quality_realtime.cached_lout
            
            # 根据新参数重新计算lout（这里简化处理）
            lout_new = lout * p_param * a_param  # 简化的参数影响
            
            # 重新计算质量指标
            quality_results = metrics.get_all_metrics(lin, lout_new)
            
            return (
                f"状态: {quality_results['Exposure_status']}",
                f"高光饱和: {quality_results['S_ratio']*100:.1f}%"
            )
        
        return "等待图像处理...", ""
    
    with gr.Blocks() as interface:
        # 参数控制
        p_param = gr.Slider(0.1, 2.0, value=1.0, label="p参数")
        a_param = gr.Slider(0.1, 2.0, value=1.0, label="a参数")
        
        # 实时显示
        status_display = gr.Textbox(label="质量状态", interactive=False)
        metrics_display = gr.Textbox(label="关键指标", interactive=False)
        
        # 参数变化时自动更新
        for param in [p_param, a_param]:
            param.change(
                fn=update_quality_realtime,
                inputs=[p_param, a_param],
                outputs=[status_display, metrics_display]
            )
    
    return interface
```

## 性能优化技巧

### 1. 数据类型优化

```python
# 推荐：使用float32减少内存占用和计算时间
lin = image_data.astype(np.float32)
lout = processed_data.astype(np.float32)

# 避免：使用float64（除非必要）
# lin = image_data.astype(np.float64)  # 不推荐
```

### 2. 内存管理

```python
def process_large_dataset(image_list):
    """处理大数据集的内存优化示例"""
    metrics = ExtendedMetrics()
    results = []
    
    for i, (lin, lout) in enumerate(image_list):
        # 确保数据类型正确
        if lin.dtype != np.float32:
            lin = lin.astype(np.float32)
        if lout.dtype != np.float32:
            lout = lout.astype(np.float32)
        
        # 计算质量指标
        result = metrics.get_all_metrics(lin, lout)
        results.append(result)
        
        # 定期清理内存
        if i % 10 == 0:
            import gc
            gc.collect()
        
        # 删除不再需要的引用
        del lin, lout
    
    return results
```

### 3. 缓存优化

```python
from functools import lru_cache
import hashlib

class CachedExtendedMetrics(ExtendedMetrics):
    """带缓存的质量评估类"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_data_hash(self, lin: np.ndarray, lout: np.ndarray) -> str:
        """计算数据哈希值"""
        # 使用数据的统计特征作为哈希键（避免对大数组直接哈希）
        lin_stats = (lin.mean(), lin.std(), lin.min(), lin.max())
        lout_stats = (lout.mean(), lout.std(), lout.min(), lout.max())
        combined_stats = lin_stats + lout_stats
        
        return hashlib.md5(str(combined_stats).encode()).hexdigest()
    
    @lru_cache(maxsize=128)
    def _cached_metrics_calculation(self, data_hash: str, lin_tuple: tuple, lout_tuple: tuple) -> dict:
        """缓存的指标计算"""
        # 从元组重建数组
        lin = np.array(lin_tuple, dtype=np.float32)
        lout = np.array(lout_tuple, dtype=np.float32)
        
        return super().get_all_metrics(lin, lout)
    
    def get_all_metrics(self, lin: np.ndarray, lout: np.ndarray) -> dict:
        """带缓存的指标计算"""
        # 对于小数组，使用缓存
        if lin.size <= 10000:  # 100x100像素以下
            data_hash = self._get_data_hash(lin, lout)
            lin_tuple = tuple(lin.flatten())
            lout_tuple = tuple(lout.flatten())
            
            try:
                result = self._cached_metrics_calculation(data_hash, lin_tuple, lout_tuple)
                self.cache_hits += 1
                return result
            except:
                self.cache_misses += 1
                return super().get_all_metrics(lin, lout)
        else:
            # 大数组直接计算
            self.cache_misses += 1
            return super().get_all_metrics(lin, lout)
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total * 100 if total > 0 else 0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': round(hit_rate, 1)
        }
```

## 错误处理和调试

### 错误处理示例

```python
def robust_quality_assessment(lin, lout):
    """健壮的质量评估函数"""
    try:
        metrics = ExtendedMetrics()
        results = metrics.get_all_metrics(lin, lout)
        
        # 检查结果有效性
        if 'error' in results:
            print(f"计算错误: {results['error']}")
            return None
        
        return results
        
    except ValueError as e:
        print(f"数据错误: {e}")
        return None
    except Exception as e:
        print(f"未知错误: {e}")
        return None

# 使用示例
lin = np.random.rand(1000, 1000).astype(np.float32)
lout = np.random.rand(1000, 1000).astype(np.float32)

results = robust_quality_assessment(lin, lout)
if results:
    print(f"评估成功: {results['Exposure_status']}")
else:
    print("评估失败，使用默认值")
```

### 调试模式

```python
import logging

# 启用详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 创建调试实例
metrics = ExtendedMetrics()

# 查看详细计算过程
results = metrics.get_all_metrics(lin, lout)

# 查看性能报告
if metrics.performance_monitor:
    report = metrics.get_performance_report()
    print("性能报告:", report)
```

## 最佳实践

### 1. 数据预处理

```python
def preprocess_hdr_data(lin, lout):
    """HDR数据预处理最佳实践"""
    # 确保数据类型
    lin = np.asarray(lin, dtype=np.float32)
    lout = np.asarray(lout, dtype=np.float32)
    
    # 检查数据范围
    if np.any(lin < 0) or np.any(lin > 1):
        print("警告: Lin数据超出PQ域范围[0,1]")
        lin = np.clip(lin, 0, 1)
    
    if np.any(lout < 0) or np.any(lout > 1):
        print("警告: Lout数据超出PQ域范围[0,1]")
        lout = np.clip(lout, 0, 1)
    
    # 检查数据完整性
    if np.any(np.isnan(lin)) or np.any(np.isnan(lout)):
        print("错误: 数据包含NaN值")
        return None, None
    
    return lin, lout
```

### 2. 配置管理

```python
def setup_quality_assessment_config(image_type="general"):
    """根据图像类型设置最佳配置"""
    config_manager = ConfigManager()
    
    if image_type == "portrait":
        # 人像摄影：更严格的高光控制
        config_manager.update_threshold("S_ratio", 0.03)
        config_manager.update_threshold("C_shadow", 0.08)
    elif image_type == "landscape":
        # 风景摄影：允许更大的动态范围
        config_manager.update_threshold("S_ratio", 0.07)
        config_manager.update_threshold("R_DR_tolerance", 0.25)
    elif image_type == "night":
        # 夜景摄影：更宽松的暗部处理
        config_manager.update_threshold("C_shadow", 0.15)
        config_manager.update_threshold("Dprime", 0.30)
    
    return ExtendedMetrics()
```

### 3. 性能监控

```python
def monitor_system_performance():
    """系统性能监控示例"""
    metrics = ExtendedMetrics(enable_performance_monitoring=True)
    
    # 测试不同尺寸的图像
    test_sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    
    for width, height in test_sizes:
        lin = np.random.rand(height, width).astype(np.float32)
        lout = np.random.rand(height, width).astype(np.float32)
        
        # 多次测试取平均
        for _ in range(5):
            results = metrics.get_all_metrics(lin, lout)
        
        # 查看当前性能
        report = metrics.get_performance_report()
        print(f"{width}x{height}: 平均 {report['average_time_ms']:.1f}ms")
        
        # 重置统计
        metrics.reset_performance_stats()
```

这个使用说明涵盖了从基本使用到高级优化的各个方面，帮助用户充分利用HDR质量评估扩展模块的功能。