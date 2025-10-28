# HDR色调映射专利可视化工具 - 演示示例

## 目录
1. [快速演示](#快速演示)
2. [基础使用示例](#基础使用示例)
3. [高级功能演示](#高级功能演示)
4. [API使用示例](#api使用示例)
5. [批量处理示例](#批量处理示例)
6. [自定义扩展示例](#自定义扩展示例)

## 快速演示

### 5分钟快速体验

1. **启动应用**
   ```bash
   python src/gradio_app.py
   ```

2. **基础参数调节**
   - 打开浏览器访问 `http://localhost:7860`
   - 调节"亮度控制因子 p"滑块从1.0到3.0
   - 观察Phoenix曲线的实时变化
   - 注意曲线的单调性和端点归一化

3. **模式切换演示**
   - 切换到"自动模式"
   - 观察系统自动估算的参数值
   - 切换回"艺术模式"进行手动调节

4. **质量指标观察**
   - 调节参数时观察"感知失真"和"局部对比度"的变化
   - 注意"模式建议"的滞回特性

## 基础使用示例

### 示例1: Phoenix曲线基础计算

```python
# examples/basic_phoenix_demo.py
import numpy as np
import matplotlib.pyplot as plt
from core import PhoenixCurveCalculator

def basic_phoenix_demo():
    """Phoenix曲线基础演示"""
    
    # 创建计算器
    calc = PhoenixCurveCalculator(display_samples=1024)
    
    # 输入亮度范围
    L = np.linspace(0, 1, 1024)
    
    # 不同参数的Phoenix曲线
    params = [
        (1.0, 0.3, "低对比度"),
        (2.0, 0.5, "标准设置"),
        (3.0, 0.7, "高对比度")
    ]
    
    plt.figure(figsize=(12, 8))
    
    # 绘制恒等线
    plt.plot(L, L, 'k--', alpha=0.5, linewidth=1, label='恒等线')
    
    for p, a, label in params:
        # 计算Phoenix曲线
        L_out = calc.compute_phoenix_curve(L, p, a)
        
        # 验证单调性
        is_monotonic = calc.validate_monotonicity(L_out)
        
        # 端点归一化
        L_normalized = calc.normalize_endpoints(L_out, 0.0, 1.0)
        
        # 绘制曲线
        plt.plot(L, L_normalized, linewidth=2, 
                label=f'{label} (p={p}, a={a}) - 单调性: {is_monotonic}')
    
    plt.xlabel('输入亮度 (PQ域)')
    plt.ylabel('输出亮度 (PQ域)')
    plt.title('Phoenix曲线基础演示')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('examples/outputs/basic_phoenix_demo.png', dpi=150)
    plt.show()
    
    print("基础Phoenix曲线演示完成")
    print("图像已保存到: examples/outputs/basic_phoenix_demo.png")

if __name__ == "__main__":
    basic_phoenix_demo()
```

### 示例2: 质量指标计算演示

```python
# examples/quality_metrics_demo.py
import numpy as np
import matplotlib.pyplot as plt
from core import QualityMetricsCalculator, PhoenixCurveCalculator

def quality_metrics_demo():
    """质量指标计算演示"""
    
    # 创建计算器
    phoenix_calc = PhoenixCurveCalculator()
    quality_calc = QualityMetricsCalculator(luminance_channel="MaxRGB")
    
    # 创建测试图像
    np.random.seed(42)
    test_image = np.random.exponential(0.3, (128, 128)).astype(np.float32)
    test_image = np.clip(test_image, 0, 1)
    
    # 提取亮度
    L_in = quality_calc.extract_luminance(test_image)
    
    # 不同参数下的质量指标
    p_values = np.linspace(0.5, 4.0, 20)
    distortions = []
    contrasts = []
    recommendations = []
    
    print("计算不同参数下的质量指标...")
    
    for p in p_values:
        # 应用Phoenix曲线
        L_out = phoenix_calc.compute_phoenix_curve(L_in, p, 0.5)
        
        # 计算质量指标
        distortion = quality_calc.compute_perceptual_distortion(L_in, L_out)
        contrast = quality_calc.compute_local_contrast(L_out)
        recommendation = quality_calc.recommend_mode_with_hysteresis(distortion)
        
        distortions.append(distortion)
        contrasts.append(contrast)
        recommendations.append(recommendation)
    
    # 可视化结果
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # 感知失真
    ax1.plot(p_values, distortions, 'b-', linewidth=2, marker='o')
    ax1.axhline(y=0.05, color='g', linestyle='--', alpha=0.7, label='下阈值 (0.05)')
    ax1.axhline(y=0.10, color='r', linestyle='--', alpha=0.7, label='上阈值 (0.10)')
    ax1.set_xlabel('亮度控制因子 p')
    ax1.set_ylabel('感知失真 D\'')
    ax1.set_title('感知失真随参数变化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 局部对比度
    ax2.plot(p_values, contrasts, 'g-', linewidth=2, marker='s')
    ax2.set_xlabel('亮度控制因子 p')
    ax2.set_ylabel('局部对比度')
    ax2.set_title('局部对比度随参数变化')
    ax2.grid(True, alpha=0.3)
    
    # 模式推荐
    mode_colors = ['blue' if r == '自动模式' else 'red' for r in recommendations]
    ax3.scatter(p_values, [1 if r == '自动模式' else 0 for r in recommendations], 
                c=mode_colors, s=50, alpha=0.7)
    ax3.set_xlabel('亮度控制因子 p')
    ax3.set_ylabel('推荐模式')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['艺术模式', '自动模式'])
    ax3.set_title('模式推荐随参数变化')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/outputs/quality_metrics_demo.png', dpi=150)
    plt.show()
    
    # 滞回特性演示
    demonstrate_hysteresis(quality_calc)
    
    print("质量指标演示完成")

def demonstrate_hysteresis(quality_calc):
    """演示滞回特性"""
    
    print("\n演示滞回特性...")
    
    # 重置滞回状态
    quality_calc.reset_hysteresis()
    
    # 测试序列：低 -> 高 -> 中间 -> 低 -> 中间
    test_sequence = [0.03, 0.12, 0.07, 0.04, 0.08]
    sequence_labels = ["低失真", "高失真", "中等失真", "低失真", "中等失真"]
    
    print("失真值 -> 推荐模式")
    print("-" * 30)
    
    for i, (distortion, label) in enumerate(zip(test_sequence, sequence_labels)):
        mode = quality_calc.recommend_mode_with_hysteresis(distortion)
        print(f"{distortion:.3f} ({label}) -> {mode}")
    
    print("\n注意：中等失真时保持上次决策（滞回效应）")

if __name__ == "__main__":
    quality_metrics_demo()
```

### 示例3: 图像处理完整流程

```python
# examples/image_processing_demo.py
import numpy as np
import matplotlib.pyplot as plt
from core import ImageProcessor, PhoenixCurveCalculator, QualityMetricsCalculator

def create_synthetic_hdr_image():
    """创建合成HDR图像"""
    
    # 创建具有不同亮度区域的图像
    height, width = 256, 256
    image = np.zeros((height, width, 3), dtype=np.float32)
    
    # 天空区域（高亮度）
    image[:height//3, :, :] = [0.8, 0.9, 1.0]
    
    # 建筑区域（中等亮度）
    image[height//3:2*height//3, :, :] = [0.4, 0.4, 0.3]
    
    # 阴影区域（低亮度）
    image[2*height//3:, :, :] = [0.1, 0.1, 0.15]
    
    # 添加一些细节和噪声
    noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
    image = np.clip(image + noise, 0, 1)
    
    return image

def image_processing_demo():
    """图像处理完整流程演示"""
    
    # 创建处理器
    image_processor = ImageProcessor()
    phoenix_calc = PhoenixCurveCalculator()
    quality_calc = QualityMetricsCalculator()
    
    # 创建测试图像
    print("创建合成HDR图像...")
    original_image = create_synthetic_hdr_image()
    
    # 转换到PQ域
    print("转换到PQ域...")
    pq_image = image_processor.convert_to_pq_domain(original_image, "sRGB")
    
    # 获取图像统计
    print("计算图像统计...")
    stats = image_processor.get_image_stats(pq_image, "MaxRGB")
    
    print(f"图像统计信息:")
    print(f"  最小PQ值: {stats.min_pq:.6f}")
    print(f"  最大PQ值: {stats.max_pq:.6f}")
    print(f"  平均PQ值: {stats.avg_pq:.6f}")
    print(f"  方差: {stats.var_pq:.6f}")
    print(f"  像素总数: {stats.pixel_count:,}")
    
    # 定义不同的色调映射参数
    tone_mapping_configs = [
        (1.5, 0.3, "保守映射"),
        (2.0, 0.5, "标准映射"),
        (2.5, 0.7, "激进映射")
    ]
    
    # 处理图像
    results = []
    
    for p, a, label in tone_mapping_configs:
        print(f"\n应用{label} (p={p}, a={a})...")
        
        # 定义色调映射函数
        def tone_curve_func(L):
            return phoenix_calc.compute_phoenix_curve(L, p, a)
        
        # 应用色调映射
        mapped_image = image_processor.apply_tone_mapping(
            pq_image, tone_curve_func, "MaxRGB"
        )
        
        # 计算处理后统计
        mapped_stats = image_processor.get_image_stats(mapped_image, "MaxRGB")
        
        # 计算质量指标
        L_in = quality_calc.extract_luminance(pq_image)
        L_out = quality_calc.extract_luminance(mapped_image)
        
        distortion = quality_calc.compute_perceptual_distortion(L_in, L_out)
        contrast = quality_calc.compute_local_contrast(L_out)
        recommendation = quality_calc.recommend_mode_with_hysteresis(distortion)
        
        results.append({
            'label': label,
            'p': p,
            'a': a,
            'mapped_image': mapped_image,
            'mapped_stats': mapped_stats,
            'distortion': distortion,
            'contrast': contrast,
            'recommendation': recommendation
        })
        
        print(f"  感知失真: {distortion:.6f}")
        print(f"  局部对比度: {contrast:.6f}")
        print(f"  模式推荐: {recommendation}")
    
    # 可视化结果
    visualize_processing_results(original_image, pq_image, results)
    
    print("\n图像处理演示完成")

def visualize_processing_results(original_image, pq_image, results):
    """可视化处理结果"""
    
    fig, axes = plt.subplots(2, len(results) + 1, figsize=(16, 8))
    
    # 显示原始图像
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(pq_image)
    axes[1, 0].set_title('PQ域图像')
    axes[1, 0].axis('off')
    
    # 显示处理结果
    for i, result in enumerate(results):
        col = i + 1
        
        # 转换为显示格式
        display_image = np.clip(result['mapped_image'], 0, 1)
        
        axes[0, col].imshow(display_image)
        axes[0, col].set_title(f"{result['label']}\n(p={result['p']}, a={result['a']})")
        axes[0, col].axis('off')
        
        # 显示统计信息
        stats_text = f"失真: {result['distortion']:.4f}\n"
        stats_text += f"对比度: {result['contrast']:.4f}\n"
        stats_text += f"推荐: {result['recommendation']}"
        
        axes[1, col].text(0.1, 0.5, stats_text, transform=axes[1, col].transAxes,
                         fontsize=10, verticalalignment='center')
        axes[1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('examples/outputs/image_processing_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 创建输出目录
    import os
    os.makedirs('examples/outputs', exist_ok=True)
    
    image_processing_demo()
```

## 高级功能演示

### 示例4: 样条曲线功能演示

```python
# examples/spline_curve_demo.py
import numpy as np
import matplotlib.pyplot as plt
from core import PhoenixCurveCalculator, SplineCurveCalculator

def spline_curve_demo():
    """样条曲线功能演示"""
    
    # 创建计算器
    phoenix_calc = PhoenixCurveCalculator()
    spline_calc = SplineCurveCalculator()
    
    # 基础Phoenix曲线
    L = np.linspace(0, 1, 1000)
    phoenix_curve = phoenix_calc.compute_phoenix_curve(L, 2.0, 0.5)
    
    # 样条节点配置
    spline_configs = [
        ([0.2, 0.5, 0.8], 0.3, "轻微调整"),
        ([0.1, 0.4, 0.7], 0.6, "中等调整"),
        ([0.15, 0.45, 0.75], 0.9, "强烈调整")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 原始Phoenix曲线
    axes[0, 0].plot(L, L, 'k--', alpha=0.5, label='恒等线')
    axes[0, 0].plot(L, phoenix_curve, 'b-', linewidth=2, label='Phoenix曲线')
    axes[0, 0].set_title('原始Phoenix曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 样条曲线对比
    for i, (nodes, strength, label) in enumerate(spline_configs):
        row = (i + 1) // 2
        col = (i + 1) % 2
        
        # 计算样条曲线
        final_curve, used_spline, status = spline_calc.compute_spline_with_fallback(
            phoenix_curve, L, nodes, strength
        )
        
        # 检查单调性
        is_monotonic = spline_calc.check_monotonicity(final_curve)
        
        # 绘制结果
        axes[row, col].plot(L, L, 'k--', alpha=0.5, label='恒等线')
        axes[row, col].plot(L, phoenix_curve, 'b-', alpha=0.7, label='Phoenix曲线')
        
        if used_spline:
            axes[row, col].plot(L, final_curve, 'r-', linewidth=2, label='样条混合曲线')
            
            # 标记节点位置
            for node in nodes:
                node_idx = int(node * len(L))
                axes[row, col].axvline(x=node, color='orange', linestyle=':', alpha=0.7)
                axes[row, col].plot(node, final_curve[node_idx], 'ro', markersize=6)
        
        title = f"{label} (强度={strength})\n"
        title += f"使用样条: {used_spline}, 单调性: {is_monotonic}"
        axes[row, col].set_title(title)
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
        
        print(f"{label}:")
        print(f"  节点: {nodes}")
        print(f"  强度: {strength}")
        print(f"  使用样条: {used_spline}")
        print(f"  单调性: {is_monotonic}")
        print(f"  状态: {status}")
        print()
    
    plt.tight_layout()
    plt.savefig('examples/outputs/spline_curve_demo.png', dpi=150)
    plt.show()
    
    # 演示零强度回退
    demonstrate_zero_strength_fallback(phoenix_calc, spline_calc, L, phoenix_curve)

def demonstrate_zero_strength_fallback(phoenix_calc, spline_calc, L, phoenix_curve):
    """演示零强度回退机制"""
    
    print("演示零强度回退机制...")
    
    nodes = [0.2, 0.5, 0.8]
    
    # 零强度应该回退到Phoenix曲线
    zero_strength_curve, used_spline, status = spline_calc.compute_spline_with_fallback(
        phoenix_curve, L, nodes, 0.0
    )
    
    # 验证是否与Phoenix曲线相同
    is_identical = np.allclose(zero_strength_curve, phoenix_curve, atol=1e-10)
    
    print(f"零强度测试:")
    print(f"  使用样条: {used_spline}")
    print(f"  与Phoenix曲线相同: {is_identical}")
    print(f"  最大差异: {np.max(np.abs(zero_strength_curve - phoenix_curve)):.2e}")
    print(f"  状态: {status}")

if __name__ == "__main__":
    spline_curve_demo()
```

### 示例5: 时域平滑演示

```python
# examples/temporal_smoothing_demo.py
import numpy as np
import matplotlib.pyplot as plt
from core import TemporalSmoothingProcessor

def temporal_smoothing_demo():
    """时域平滑功能演示"""
    
    # 创建时域平滑处理器
    processor = TemporalSmoothingProcessor(window_size=9)
    
    # 模拟参数变化序列（带噪声）
    np.random.seed(42)
    frames = 50
    base_p = 2.0
    base_a = 0.5
    
    # 生成带噪声的参数序列
    p_sequence = base_p + 0.3 * np.sin(np.linspace(0, 4*np.pi, frames)) + 0.1 * np.random.randn(frames)
    a_sequence = base_a + 0.2 * np.cos(np.linspace(0, 3*np.pi, frames)) + 0.05 * np.random.randn(frames)
    
    # 确保参数在有效范围内
    p_sequence = np.clip(p_sequence, 0.1, 6.0)
    a_sequence = np.clip(a_sequence, 0.0, 1.0)
    
    # 模拟失真序列
    distortion_sequence = 0.07 + 0.03 * np.sin(np.linspace(0, 2*np.pi, frames)) + 0.01 * np.random.randn(frames)
    distortion_sequence = np.clip(distortion_sequence, 0.0, 0.2)
    
    # 存储结果
    smoothed_p = []
    smoothed_a = []
    variance_reductions = []
    
    print("模拟时域平滑过程...")
    
    for i in range(frames):
        # 添加当前帧参数
        frame_params = {"p": p_sequence[i], "a": a_sequence[i]}
        processor.add_frame_parameters(frame_params, distortion_sequence[i])
        
        # 计算平滑结果
        smoothed = processor.compute_weighted_average(lambda_smooth=0.3)
        stats = processor.get_smoothing_stats()
        
        smoothed_p.append(smoothed.get("p", p_sequence[i]))
        smoothed_a.append(smoothed.get("a", a_sequence[i]))
        variance_reductions.append(stats.variance_reduction)
        
        if i % 10 == 0:
            print(f"帧 {i:2d}: p={p_sequence[i]:.3f}->{smoothed_p[-1]:.3f}, "
                  f"a={a_sequence[i]:.3f}->{smoothed_a[-1]:.3f}, "
                  f"方差降低={variance_reductions[-1]:.1%}")
    
    # 可视化结果
    visualize_temporal_smoothing(
        frames, p_sequence, a_sequence, smoothed_p, smoothed_a, 
        variance_reductions, distortion_sequence
    )
    
    # 演示冷启动机制
    demonstrate_cold_start(processor)

def visualize_temporal_smoothing(frames, p_orig, a_orig, p_smooth, a_smooth, 
                                variance_reductions, distortions):
    """可视化时域平滑结果"""
    
    frame_indices = np.arange(frames)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # p参数平滑
    axes[0, 0].plot(frame_indices, p_orig, 'b-', alpha=0.7, label='原始p值', marker='o', markersize=3)
    axes[0, 0].plot(frame_indices, p_smooth, 'r-', linewidth=2, label='平滑p值')
    axes[0, 0].set_title('p参数时域平滑')
    axes[0, 0].set_xlabel('帧数')
    axes[0, 0].set_ylabel('p值')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # a参数平滑
    axes[0, 1].plot(frame_indices, a_orig, 'b-', alpha=0.7, label='原始a值', marker='o', markersize=3)
    axes[0, 1].plot(frame_indices, a_smooth, 'r-', linewidth=2, label='平滑a值')
    axes[0, 1].set_title('a参数时域平滑')
    axes[0, 1].set_xlabel('帧数')
    axes[0, 1].set_ylabel('a值')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 方差降低
    axes[1, 0].plot(frame_indices, np.array(variance_reductions) * 100, 'g-', linewidth=2, marker='s', markersize=4)
    axes[1, 0].set_title('方差降低效果')
    axes[1, 0].set_xlabel('帧数')
    axes[1, 0].set_ylabel('方差降低 (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 失真变化
    axes[1, 1].plot(frame_indices, distortions, 'purple', linewidth=2, label='感知失真')
    axes[1, 1].axhline(y=0.05, color='g', linestyle='--', alpha=0.7, label='下阈值')
    axes[1, 1].axhline(y=0.10, color='r', linestyle='--', alpha=0.7, label='上阈值')
    axes[1, 1].set_title('感知失真变化')
    axes[1, 1].set_xlabel('帧数')
    axes[1, 1].set_ylabel('感知失真')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/outputs/temporal_smoothing_demo.png', dpi=150)
    plt.show()

def demonstrate_cold_start(processor):
    """演示冷启动机制"""
    
    print("\n演示冷启动机制...")
    
    # 获取当前状态
    stats_before = processor.get_smoothing_stats()
    print(f"冷启动前 - 历史帧数: {stats_before.frame_count}")
    
    # 执行冷启动
    processor.cold_start()
    
    # 获取冷启动后状态
    stats_after = processor.get_smoothing_stats()
    smoothed_after = processor.compute_weighted_average()
    
    print(f"冷启动后 - 历史帧数: {stats_after.frame_count}")
    print(f"冷启动后 - 平滑结果: {smoothed_after}")
    print("冷启动机制验证完成")

if __name__ == "__main__":
    temporal_smoothing_demo()
```

## API使用示例

### 示例6: 完整API集成示例

```python
# examples/complete_api_demo.py
import numpy as np
from typing import Dict, Any
from core import (
    PhoenixCurveCalculator,
    QualityMetricsCalculator,
    ImageProcessor,
    AutoModeParameterEstimator,
    get_state_manager,
    get_export_manager,
    SessionState,
    CurveData
)

class HDRToneMappingAPI:
    """HDR色调映射API封装类"""
    
    def __init__(self):
        """初始化API"""
        self.phoenix_calc = PhoenixCurveCalculator()
        self.quality_calc = QualityMetricsCalculator()
        self.image_processor = ImageProcessor()
        self.auto_estimator = AutoModeParameterEstimator()
        self.state_manager = get_state_manager()
        self.export_manager = get_export_manager()
    
    def process_image_auto(self, image: np.ndarray, 
                          input_format: str = "sRGB") -> Dict[str, Any]:
        """自动模式图像处理"""
        
        try:
            # 转换到PQ域
            pq_image = self.image_processor.convert_to_pq_domain(image, input_format)
            
            # 获取图像统计
            stats = self.image_processor.get_image_stats(pq_image, "MaxRGB")
            
            # 自动估算参数
            estimation = self.auto_estimator.estimate_parameters(stats)
            
            # 应用色调映射
            def tone_curve_func(L):
                return self.phoenix_calc.compute_phoenix_curve(
                    L, estimation.p_estimated, estimation.a_estimated
                )
            
            mapped_image = self.image_processor.apply_tone_mapping(
                pq_image, tone_curve_func, "MaxRGB"
            )
            
            # 计算质量指标
            L_in = self.quality_calc.extract_luminance(pq_image)
            L_out = self.quality_calc.extract_luminance(mapped_image)
            
            distortion = self.quality_calc.compute_perceptual_distortion(L_in, L_out)
            contrast = self.quality_calc.compute_local_contrast(L_out)
            recommendation = self.quality_calc.recommend_mode_with_hysteresis(distortion)
            
            # 更新状态
            self.state_manager.update_session_state(
                p=estimation.p_estimated,
                a=estimation.a_estimated,
                mode="自动模式"
            )
            
            return {
                'success': True,
                'mapped_image': mapped_image,
                'parameters': {
                    'p': estimation.p_estimated,
                    'a': estimation.a_estimated
                },
                'quality_metrics': {
                    'distortion': distortion,
                    'contrast': contrast,
                    'recommendation': recommendation
                },
                'image_stats': {
                    'original': stats,
                    'processed': self.image_processor.get_image_stats(mapped_image, "MaxRGB")
                },
                'estimation_info': estimation.statistics
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_image_manual(self, image: np.ndarray, 
                           p: float, a: float,
                           input_format: str = "sRGB",
                           enable_spline: bool = False,
                           spline_config: Dict = None) -> Dict[str, Any]:
        """手动模式图像处理"""
        
        try:
            # 参数验证
            if not (0.1 <= p <= 6.0):
                raise ValueError(f"参数p={p}超出范围[0.1, 6.0]")
            if not (0.0 <= a <= 1.0):
                raise ValueError(f"参数a={a}超出范围[0.0, 1.0]")
            
            # 转换到PQ域
            pq_image = self.image_processor.convert_to_pq_domain(image, input_format)
            
            # 定义色调映射函数
            def tone_curve_func(L):
                phoenix_curve = self.phoenix_calc.compute_phoenix_curve(L, p, a)
                
                if enable_spline and spline_config:
                    from core import SplineCurveCalculator
                    spline_calc = SplineCurveCalculator()
                    
                    final_curve, used_spline, status = spline_calc.compute_spline_with_fallback(
                        phoenix_curve, L, 
                        spline_config.get('nodes', [0.2, 0.5, 0.8]),
                        spline_config.get('strength', 0.5)
                    )
                    return final_curve
                else:
                    return phoenix_curve
            
            # 应用色调映射
            mapped_image = self.image_processor.apply_tone_mapping(
                pq_image, tone_curve_func, "MaxRGB"
            )
            
            # 计算质量指标
            L_in = self.quality_calc.extract_luminance(pq_image)
            L_out = self.quality_calc.extract_luminance(mapped_image)
            
            distortion = self.quality_calc.compute_perceptual_distortion(L_in, L_out)
            contrast = self.quality_calc.compute_local_contrast(L_out)
            recommendation = self.quality_calc.recommend_mode_with_hysteresis(distortion)
            
            # 验证曲线质量
            L_test = np.linspace(0, 1, 1024)
            test_curve = tone_curve_func(L_test)
            is_monotonic = self.phoenix_calc.validate_monotonicity(test_curve)
            
            # 更新状态
            self.state_manager.update_session_state(
                p=p, a=a, mode="艺术模式",
                enable_spline=enable_spline
            )
            
            return {
                'success': True,
                'mapped_image': mapped_image,
                'parameters': {'p': p, 'a': a},
                'quality_metrics': {
                    'distortion': distortion,
                    'contrast': contrast,
                    'recommendation': recommendation
                },
                'curve_validation': {
                    'is_monotonic': is_monotonic
                },
                'spline_info': {
                    'enabled': enable_spline,
                    'config': spline_config if enable_spline else None
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def export_lut(self, p: float, a: float, 
                   output_file: str,
                   samples: int = 1024) -> Dict[str, Any]:
        """导出LUT文件"""
        
        try:
            # 生成曲线数据
            L = np.linspace(0, 1, samples)
            L_out = self.phoenix_calc.compute_phoenix_curve(L, p, a)
            
            curve_data = CurveData(
                input_luminance=L,
                output_luminance=L_out,
                phoenix_curve=L_out
            )
            
            session_state = SessionState(p=p, a=a)
            
            # 导出LUT
            success = self.export_manager.export_lut(
                curve_data, session_state, output_file, samples
            )
            
            if success:
                # 验证导出一致性
                is_consistent, max_error = self.export_manager.validate_export_consistency(
                    L_out, output_file, "lut"
                )
                
                return {
                    'success': True,
                    'output_file': output_file,
                    'samples': samples,
                    'validation': {
                        'is_consistent': is_consistent,
                        'max_error': max_error
                    }
                }
            else:
                return {
                    'success': False,
                    'error': 'LUT导出失败'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_curve_data(self, p: float, a: float, 
                      samples: int = 1024) -> Dict[str, Any]:
        """获取曲线数据"""
        
        try:
            # 生成曲线
            L = np.linspace(0, 1, samples)
            L_out = self.phoenix_calc.compute_phoenix_curve(L, p, a)
            
            # 验证曲线
            is_monotonic = self.phoenix_calc.validate_monotonicity(L_out)
            
            # 端点归一化
            L_normalized = self.phoenix_calc.normalize_endpoints(L_out, 0.0, 1.0)
            
            return {
                'success': True,
                'input_luminance': L.tolist(),
                'output_luminance': L_out.tolist(),
                'normalized_output': L_normalized.tolist(),
                'validation': {
                    'is_monotonic': is_monotonic,
                    'endpoint_error': max(abs(L_normalized[0]), abs(L_normalized[-1] - 1.0))
                },
                'parameters': {'p': p, 'a': a},
                'samples': samples
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

def api_demo():
    """API使用演示"""
    
    print("HDR色调映射API演示")
    print("=" * 50)
    
    # 创建API实例
    api = HDRToneMappingAPI()
    
    # 创建测试图像
    test_image = np.random.exponential(0.3, (128, 128, 3)).astype(np.float32)
    test_image = np.clip(test_image, 0, 1)
    
    print("1. 自动模式处理...")
    auto_result = api.process_image_auto(test_image)
    
    if auto_result['success']:
        print(f"   自动估算参数: p={auto_result['parameters']['p']:.3f}, a={auto_result['parameters']['a']:.3f}")
        print(f"   感知失真: {auto_result['quality_metrics']['distortion']:.6f}")
        print(f"   模式推荐: {auto_result['quality_metrics']['recommendation']}")
    else:
        print(f"   处理失败: {auto_result['error']}")
    
    print("\n2. 手动模式处理...")
    manual_result = api.process_image_manual(
        test_image, p=2.2, a=0.6,
        enable_spline=True,
        spline_config={'nodes': [0.2, 0.5, 0.8], 'strength': 0.5}
    )
    
    if manual_result['success']:
        print(f"   处理参数: p={manual_result['parameters']['p']}, a={manual_result['parameters']['a']}")
        print(f"   曲线单调性: {manual_result['curve_validation']['is_monotonic']}")
        print(f"   样条配置: {manual_result['spline_info']['config']}")
    else:
        print(f"   处理失败: {manual_result['error']}")
    
    print("\n3. 导出LUT...")
    lut_result = api.export_lut(2.0, 0.5, "examples/outputs/api_demo.cube", samples=1024)
    
    if lut_result['success']:
        print(f"   导出文件: {lut_result['output_file']}")
        print(f"   采样点数: {lut_result['samples']}")
        print(f"   一致性验证: {lut_result['validation']['is_consistent']}")
        print(f"   最大误差: {lut_result['validation']['max_error']:.2e}")
    else:
        print(f"   导出失败: {lut_result['error']}")
    
    print("\n4. 获取曲线数据...")
    curve_result = api.get_curve_data(2.5, 0.7, samples=512)
    
    if curve_result['success']:
        print(f"   采样点数: {curve_result['samples']}")
        print(f"   单调性: {curve_result['validation']['is_monotonic']}")
        print(f"   端点误差: {curve_result['validation']['endpoint_error']:.2e}")
    else:
        print(f"   获取失败: {curve_result['error']}")
    
    print("\nAPI演示完成")

if __name__ == "__main__":
    # 创建输出目录
    import os
    os.makedirs('examples/outputs', exist_ok=True)
    
    api_demo()
```

## 批量处理示例

### 示例7: 批量图像处理

```python
# examples/batch_processing_demo.py
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from core import (
    PhoenixCurveCalculator,
    ImageProcessor,
    QualityMetricsCalculator,
    get_export_manager,
    SessionState,
    CurveData
)

class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.phoenix_calc = PhoenixCurveCalculator()
        self.image_processor = ImageProcessor()
        self.quality_calc = QualityMetricsCalculator()
        self.export_manager = get_export_manager()
    
    def create_synthetic_dataset(self, output_dir: str, count: int = 10):
        """创建合成数据集"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"创建{count}张合成图像...")
        
        np.random.seed(42)
        
        for i in range(count):
            # 创建不同特征的图像
            if i % 3 == 0:
                # 高动态范围图像
                image = np.random.exponential(0.5, (128, 128, 3)).astype(np.float32)
            elif i % 3 == 1:
                # 低动态范围图像
                image = np.random.beta(2, 5, (128, 128, 3)).astype(np.float32)
            else:
                # 混合动态范围图像
                image = np.random.gamma(2, 0.3, (128, 128, 3)).astype(np.float32)
            
            image = np.clip(image, 0, 1)
            
            # 保存图像（这里用numpy格式，实际应用中可能是EXR等格式）
            image_path = os.path.join(output_dir, f"synthetic_{i:03d}.npy")
            np.save(image_path, image)
        
        print(f"合成数据集已创建: {output_dir}")
        return [os.path.join(output_dir, f"synthetic_{i:03d}.npy") for i in range(count)]
    
    def process_single_image(self, image_path: str, 
                           output_dir: str,
                           p: float, a: float,
                           export_lut: bool = True) -> Dict[str, Any]:
        """处理单张图像"""
        
        try:
            start_time = time.time()
            
            # 加载图像
            image = np.load(image_path)
            
            # 转换到PQ域
            pq_image = self.image_processor.convert_to_pq_domain(image, "sRGB")
            
            # 应用色调映射
            def tone_curve_func(L):
                return self.phoenix_calc.compute_phoenix_curve(L, p, a)
            
            mapped_image = self.image_processor.apply_tone_mapping(
                pq_image, tone_curve_func, "MaxRGB"
            )
            
            # 计算质量指标
            L_in = self.quality_calc.extract_luminance(pq_image)
            L_out = self.quality_calc.extract_luminance(mapped_image)
            
            distortion = self.quality_calc.compute_perceptual_distortion(L_in, L_out)
            contrast = self.quality_calc.compute_local_contrast(L_out)
            
            # 保存处理结果
            base_name = Path(image_path).stem
            output_image_path = os.path.join(output_dir, f"{base_name}_processed.npy")
            np.save(output_image_path, mapped_image)
            
            # 导出LUT（可选）
            lut_path = None
            if export_lut:
                lut_path = os.path.join(output_dir, f"{base_name}_lut.cube")
                
                L = np.linspace(0, 1, 1024)
                L_out_curve = self.phoenix_calc.compute_phoenix_curve(L, p, a)
                
                curve_data = CurveData(
                    input_luminance=L,
                    output_luminance=L_out_curve,
                    phoenix_curve=L_out_curve
                )
                
                session_state = SessionState(p=p, a=a)
                self.export_manager.export_lut(curve_data, session_state, lut_path)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'input_path': image_path,
                'output_path': output_image_path,
                'lut_path': lut_path,
                'parameters': {'p': p, 'a': a},
                'quality_metrics': {
                    'distortion': distortion,
                    'contrast': contrast
                },
                'processing_time': processing_time,
                'image_shape': image.shape
            }
            
        except Exception as e:
            return {
                'success': False,
                'input_path': image_path,
                'error': str(e)
            }
    
    def batch_process(self, image_paths: List[str], 
                     output_dir: str,
                     processing_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量处理图像"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        total_images = len(image_paths)
        total_configs = len(processing_configs)
        total_tasks = total_images * total_configs
        
        print(f"开始批量处理: {total_images}张图像 × {total_configs}种配置 = {total_tasks}个任务")
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_info = {}
            
            for config_idx, config in enumerate(processing_configs):
                config_output_dir = os.path.join(output_dir, f"config_{config_idx:02d}")
                os.makedirs(config_output_dir, exist_ok=True)
                
                for image_path in image_paths:
                    future = executor.submit(
                        self.process_single_image,
                        image_path,
                        config_output_dir,
                        config['p'],
                        config['a'],
                        config.get('export_lut', True)
                    )
                    
                    future_to_info[future] = {
                        'config_idx': config_idx,
                        'config': config,
                        'image_path': image_path
                    }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_info):
                info = future_to_info[future]
                result = future.result()
                
                result['config_idx'] = info['config_idx']
                result['config'] = info['config']
                results.append(result)
                
                completed += 1
                if completed % 10 == 0 or completed == total_tasks:
                    print(f"进度: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%)")
        
        total_time = time.time() - start_time
        
        # 生成统计报告
        summary = self._generate_batch_summary(results, total_time)
        
        # 保存结果
        results_file = os.path.join(output_dir, "batch_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': summary,
                'results': results
            }, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"批量处理完成，结果已保存到: {results_file}")
        
        return {
            'summary': summary,
            'results': results,
            'results_file': results_file
        }
    
    def _generate_batch_summary(self, results: List[Dict], total_time: float) -> Dict[str, Any]:
        """生成批量处理摘要"""
        
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        if successful_results:
            processing_times = [r['processing_time'] for r in successful_results]
            distortions = [r['quality_metrics']['distortion'] for r in successful_results]
            contrasts = [r['quality_metrics']['contrast'] for r in successful_results]
            
            summary = {
                'total_tasks': len(results),
                'successful_tasks': len(successful_results),
                'failed_tasks': len(failed_results),
                'success_rate': len(successful_results) / len(results) * 100,
                'total_time': total_time,
                'average_processing_time': np.mean(processing_times),
                'total_processing_time': np.sum(processing_times),
                'quality_metrics': {
                    'distortion': {
                        'mean': np.mean(distortions),
                        'std': np.std(distortions),
                        'min': np.min(distortions),
                        'max': np.max(distortions)
                    },
                    'contrast': {
                        'mean': np.mean(contrasts),
                        'std': np.std(contrasts),
                        'min': np.min(contrasts),
                        'max': np.max(contrasts)
                    }
                }
            }
        else:
            summary = {
                'total_tasks': len(results),
                'successful_tasks': 0,
                'failed_tasks': len(failed_results),
                'success_rate': 0.0,
                'total_time': total_time
            }
        
        return summary

def batch_processing_demo():
    """批量处理演示"""
    
    print("批量处理演示")
    print("=" * 50)
    
    # 创建批量处理器
    processor = BatchProcessor(max_workers=4)
    
    # 创建合成数据集
    dataset_dir = "examples/outputs/synthetic_dataset"
    image_paths = processor.create_synthetic_dataset(dataset_dir, count=20)
    
    # 定义处理配置
    processing_configs = [
        {'p': 1.5, 'a': 0.3, 'name': '保守映射', 'export_lut': True},
        {'p': 2.0, 'a': 0.5, 'name': '标准映射', 'export_lut': True},
        {'p': 2.5, 'a': 0.7, 'name': '激进映射', 'export_lut': True},
        {'p': 3.0, 'a': 0.8, 'name': '极端映射', 'export_lut': False}
    ]
    
    print(f"处理配置:")
    for i, config in enumerate(processing_configs):
        print(f"  {i}: {config['name']} (p={config['p']}, a={config['a']})")
    
    # 执行批量处理
    output_dir = "examples/outputs/batch_results"
    batch_result = processor.batch_process(image_paths, output_dir, processing_configs)
    
    # 显示摘要
    summary = batch_result['summary']
    print(f"\n批量处理摘要:")
    print(f"  总任务数: {summary['total_tasks']}")
    print(f"  成功任务: {summary['successful_tasks']}")
    print(f"  失败任务: {summary['failed_tasks']}")
    print(f"  成功率: {summary['success_rate']:.1f}%")
    print(f"  总耗时: {summary['total_time']:.1f}秒")
    
    if summary['successful_tasks'] > 0:
        print(f"  平均处理时间: {summary['average_processing_time']:.3f}秒/图像")
        print(f"  感知失真范围: {summary['quality_metrics']['distortion']['min']:.4f} - {summary['quality_metrics']['distortion']['max']:.4f}")
        print(f"  局部对比度范围: {summary['quality_metrics']['contrast']['min']:.4f} - {summary['quality_metrics']['contrast']['max']:.4f}")

if __name__ == "__main__":
    batch_processing_demo()
```

---

**版本**: 1.0  
**更新日期**: 2025-10-27  
**作者**: HDR Tone Mapping Team