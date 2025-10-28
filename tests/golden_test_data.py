#!/usr/bin/env python3
"""
金标测试数据生成器
生成标准测试图像、参考曲线和验证数据
"""

import numpy as np
import os
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass
import json
import sys

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import PhoenixCurveCalculator, PQConverter


@dataclass
class GoldenTestImage:
    """金标测试图像数据结构"""
    name: str
    description: str
    image_data: np.ndarray
    expected_stats: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class GoldenCurveData:
    """金标曲线数据结构"""
    name: str
    parameters: Dict[str, float]
    input_values: np.ndarray
    expected_output: np.ndarray
    tolerance: float
    properties: Dict[str, Any]


class GoldenTestDataGenerator:
    """金标测试数据生成器"""
    
    def __init__(self):
        self.phoenix_calc = PhoenixCurveCalculator()
        self.pq_converter = PQConverter()
        
    def generate_golden_images(self) -> Dict[str, GoldenTestImage]:
        """生成金标测试图像"""
        images = {}
        
        # 1. 均匀灰度图像系列
        for gray_level in [0.0, 0.18, 0.5, 0.9, 1.0]:
            name = f"uniform_gray_{int(gray_level*100):02d}"
            image = np.full((128, 128), gray_level, dtype=np.float32)
            
            images[name] = GoldenTestImage(
                name=name,
                description=f"均匀灰度图像 (灰度值: {gray_level})",
                image_data=image,
                expected_stats={
                    "min_pq": gray_level,
                    "max_pq": gray_level,
                    "avg_pq": gray_level,
                    "var_pq": 0.0,
                    "local_contrast": 0.0
                },
                metadata={
                    "type": "uniform",
                    "gray_level": gray_level,
                    "size": (128, 128),
                    "dtype": "float32"
                }
            )
            
        # 2. 线性渐变图像
        x = np.linspace(0, 1, 256)
        y = np.linspace(0, 1, 256)
        X, Y = np.meshgrid(x, y)
        
        # 水平渐变
        images["horizontal_gradient"] = GoldenTestImage(
            name="horizontal_gradient",
            description="水平线性渐变 (0到1)",
            image_data=X.astype(np.float32),
            expected_stats={
                "min_pq": 0.0,
                "max_pq": 1.0,
                "avg_pq": 0.5,
                "var_pq": 1.0/12,  # 均匀分布方差
                "local_contrast": 1.0/255  # 相邻像素差
            },
            metadata={
                "type": "gradient",
                "direction": "horizontal",
                "size": (256, 256)
            }
        )
        
        # 垂直渐变
        images["vertical_gradient"] = GoldenTestImage(
            name="vertical_gradient", 
            description="垂直线性渐变 (0到1)",
            image_data=Y.astype(np.float32),
            expected_stats={
                "min_pq": 0.0,
                "max_pq": 1.0,
                "avg_pq": 0.5,
                "var_pq": 1.0/12,
                "local_contrast": 1.0/255
            },
            metadata={
                "type": "gradient",
                "direction": "vertical",
                "size": (256, 256)
            }
        )
        
        # 3. 棋盘图案
        checker_size = 64
        checker = np.zeros((checker_size, checker_size), dtype=np.float32)
        block_size = 8
        for i in range(0, checker_size, block_size):
            for j in range(0, checker_size, block_size):
                if (i // block_size + j // block_size) % 2 == 0:
                    checker[i:i+block_size, j:j+block_size] = 1.0
                    
        images["checkerboard_8x8"] = GoldenTestImage(
            name="checkerboard_8x8",
            description="8x8棋盘图案",
            image_data=checker,
            expected_stats={
                "min_pq": 0.0,
                "max_pq": 1.0,
                "avg_pq": 0.5,
                "var_pq": 0.25,  # 二值图像方差
                "local_contrast": 1.0  # 边界处最大对比度
            },
            metadata={
                "type": "checkerboard",
                "block_size": block_size,
                "size": (checker_size, checker_size)
            }
        )
        
        # 4. 高斯噪声图像
        np.random.seed(42)  # 固定随机种子
        noise_image = np.random.normal(0.5, 0.1, (64, 64)).astype(np.float32)
        noise_image = np.clip(noise_image, 0, 1)
        
        images["gaussian_noise"] = GoldenTestImage(
            name="gaussian_noise",
            description="高斯噪声图像 (均值0.5, 标准差0.1)",
            image_data=noise_image,
            expected_stats={
                "min_pq": float(np.min(noise_image)),
                "max_pq": float(np.max(noise_image)),
                "avg_pq": float(np.mean(noise_image)),
                "var_pq": float(np.var(noise_image)),
                "local_contrast": float(np.mean(np.abs(np.diff(noise_image, axis=1))))
            },
            metadata={
                "type": "noise",
                "distribution": "gaussian",
                "mean": 0.5,
                "std": 0.1,
                "seed": 42
            }
        )
        
        # 5. RGB彩色测试图像
        rgb_image = np.zeros((64, 64, 3), dtype=np.float32)
        
        # R通道: 水平渐变
        rgb_image[:, :, 0] = np.linspace(0, 1, 64).reshape(1, -1)
        # G通道: 垂直渐变  
        rgb_image[:, :, 1] = np.linspace(0, 1, 64).reshape(-1, 1)
        # B通道: 恒定值
        rgb_image[:, :, 2] = 0.5
        
        images["rgb_gradient"] = GoldenTestImage(
            name="rgb_gradient",
            description="RGB渐变图像 (R水平, G垂直, B恒定)",
            image_data=rgb_image,
            expected_stats={
                "min_pq_r": 0.0,
                "max_pq_r": 1.0,
                "avg_pq_r": 0.5,
                "min_pq_g": 0.0,
                "max_pq_g": 1.0,
                "avg_pq_g": 0.5,
                "min_pq_b": 0.5,
                "max_pq_b": 0.5,
                "avg_pq_b": 0.5
            },
            metadata={
                "type": "rgb_gradient",
                "channels": 3,
                "size": (64, 64, 3)
            }
        )
        
        # 6. HDR模拟图像
        hdr_image = self._generate_hdr_test_image()
        images["hdr_synthetic"] = GoldenTestImage(
            name="hdr_synthetic",
            description="合成HDR测试图像",
            image_data=hdr_image,
            expected_stats={
                "min_pq": float(np.min(hdr_image)),
                "max_pq": float(np.max(hdr_image)),
                "avg_pq": float(np.mean(hdr_image)),
                "var_pq": float(np.var(hdr_image)),
                "dynamic_range": float(np.max(hdr_image) / (np.min(hdr_image) + 1e-8))
            },
            metadata={
                "type": "hdr_synthetic",
                "generation_method": "multi_exposure_simulation"
            }
        )
        
        return images
        
    def _generate_hdr_test_image(self) -> np.ndarray:
        """生成HDR测试图像"""
        size = 128
        image = np.zeros((size, size), dtype=np.float32)
        
        # 创建多个亮度区域模拟HDR场景
        center = size // 2
        
        # 背景: 低亮度
        image[:, :] = 0.1
        
        # 中等亮度区域
        y, x = np.ogrid[:size, :size]
        mask1 = (x - center)**2 + (y - center)**2 < (size//4)**2
        image[mask1] = 0.5
        
        # 高亮度区域 (模拟光源)
        mask2 = (x - center)**2 + (y - center)**2 < (size//8)**2
        image[mask2] = 0.9
        
        # 添加一些细节
        for i in range(5):
            cx = np.random.randint(size//4, 3*size//4)
            cy = np.random.randint(size//4, 3*size//4)
            radius = np.random.randint(5, 15)
            brightness = np.random.uniform(0.3, 0.8)
            
            mask = (x - cx)**2 + (y - cy)**2 < radius**2
            image[mask] = brightness
            
        return image
        
    def generate_golden_curves(self) -> Dict[str, GoldenCurveData]:
        """生成金标曲线数据"""
        curves = {}
        
        # 标准测试输入
        L_input = np.linspace(0, 1, 1000)
        
        # 1. 恒等映射 (p=1, a=0)
        curves["identity"] = GoldenCurveData(
            name="identity",
            parameters={"p": 1.0, "a": 0.0},
            input_values=L_input,
            expected_output=L_input.copy(),
            tolerance=1e-10,
            properties={
                "monotonic": True,
                "endpoint_0": 0.0,
                "endpoint_1": 1.0,
                "midpoint": 0.5,
                "curve_type": "linear"
            }
        )
        
        # 2. 标准Gamma 2.2
        gamma_22_output = np.power(L_input, 2.2)
        curves["gamma_22"] = GoldenCurveData(
            name="gamma_22",
            parameters={"p": 2.2, "a": 0.0},
            input_values=L_input,
            expected_output=gamma_22_output,
            tolerance=1e-6,
            properties={
                "monotonic": True,
                "endpoint_0": 0.0,
                "endpoint_1": 1.0,
                "midpoint": np.power(0.5, 2.2),
                "curve_type": "power"
            }
        )
        
        # 3. 逆Gamma (p=1/2.2)
        inv_gamma_output = np.power(L_input, 1.0/2.2)
        curves["inv_gamma_22"] = GoldenCurveData(
            name="inv_gamma_22",
            parameters={"p": 1.0/2.2, "a": 0.0},
            input_values=L_input,
            expected_output=inv_gamma_output,
            tolerance=1e-6,
            properties={
                "monotonic": True,
                "endpoint_0": 0.0,
                "endpoint_1": 1.0,
                "midpoint": np.power(0.5, 1.0/2.2),
                "curve_type": "inverse_power"
            }
        )
        
        # 4. 典型HDR色调映射 (p=2.0, a=0.5)
        hdr_output = self.phoenix_calc.compute_phoenix_curve(L_input, 2.0, 0.5)
        curves["hdr_typical"] = GoldenCurveData(
            name="hdr_typical",
            parameters={"p": 2.0, "a": 0.5},
            input_values=L_input,
            expected_output=hdr_output,
            tolerance=1e-6,
            properties={
                "monotonic": True,
                "endpoint_0": 0.0,
                "endpoint_1": 1.0,
                "curve_type": "phoenix"
            }
        )
        
        # 5. 极端参数测试
        extreme_cases = [
            ("extreme_low_p", {"p": 0.1, "a": 0.5}),
            ("extreme_high_p", {"p": 6.0, "a": 0.1}),
            ("extreme_high_a", {"p": 2.0, "a": 1.0}),
            ("extreme_low_a", {"p": 2.0, "a": 0.0})
        ]
        
        for name, params in extreme_cases:
            try:
                output = self.phoenix_calc.compute_phoenix_curve(L_input, params["p"], params["a"])
                is_monotonic = self.phoenix_calc.validate_monotonicity(output)
                
                curves[name] = GoldenCurveData(
                    name=name,
                    parameters=params,
                    input_values=L_input,
                    expected_output=output,
                    tolerance=1e-6,
                    properties={
                        "monotonic": is_monotonic,
                        "endpoint_0": float(output[0]),
                        "endpoint_1": float(output[-1]),
                        "curve_type": "phoenix_extreme"
                    }
                )
            except Exception as e:
                print(f"警告: 极端参数 {name} 计算失败: {e}")
                
        return curves
        
    def generate_reference_quality_metrics(self) -> Dict[str, Dict[str, float]]:
        """生成参考质量指标"""
        from core import QualityMetricsCalculator
        
        calc = QualityMetricsCalculator()
        references = {}
        
        # 1. 恒等映射的质量指标
        uniform_image = np.full((100, 100), 0.5, dtype=np.float32)
        L_in = calc.extract_luminance(uniform_image)
        L_out = L_in.copy()  # 恒等映射
        
        references["identity_uniform"] = {
            "perceptual_distortion": calc.compute_perceptual_distortion(L_in, L_out),
            "local_contrast": calc.compute_local_contrast(L_out),
            "variance_distortion": calc.compute_variance_distortion(L_in, L_out)
        }
        
        # 2. 线性渐变的质量指标
        gradient_image = np.linspace(0, 1, 10000).reshape(100, 100).astype(np.float32)
        L_in = calc.extract_luminance(gradient_image)
        
        # Gamma 2.2映射
        L_out_gamma = np.power(L_in, 2.2)
        references["gamma22_gradient"] = {
            "perceptual_distortion": calc.compute_perceptual_distortion(L_in, L_out_gamma),
            "local_contrast": calc.compute_local_contrast(L_out_gamma),
            "variance_distortion": calc.compute_variance_distortion(L_in, L_out_gamma)
        }
        
        # 3. 棋盘图案的质量指标
        checker = np.zeros((64, 64), dtype=np.float32)
        checker[::2, ::2] = 1.0
        checker[1::2, 1::2] = 1.0
        L_in = calc.extract_luminance(checker)
        L_out = L_in.copy()
        
        references["checkerboard_identity"] = {
            "perceptual_distortion": calc.compute_perceptual_distortion(L_in, L_out),
            "local_contrast": calc.compute_local_contrast(L_out),
            "variance_distortion": calc.compute_variance_distortion(L_in, L_out)
        }
        
        return references
        
    def save_golden_data(self, output_dir: str):
        """保存金标数据到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存图像数据
        images = self.generate_golden_images()
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        for name, image_data in images.items():
            # 保存图像数组
            np.save(os.path.join(images_dir, f"{name}.npy"), image_data.image_data)
            
            # 保存元数据
            metadata = {
                "name": image_data.name,
                "description": image_data.description,
                "expected_stats": image_data.expected_stats,
                "metadata": image_data.metadata
            }
            with open(os.path.join(images_dir, f"{name}_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
                
        # 保存曲线数据
        curves = self.generate_golden_curves()
        curves_dir = os.path.join(output_dir, "curves")
        os.makedirs(curves_dir, exist_ok=True)
        
        for name, curve_data in curves.items():
            # 保存曲线数组
            np.save(os.path.join(curves_dir, f"{name}_input.npy"), curve_data.input_values)
            np.save(os.path.join(curves_dir, f"{name}_output.npy"), curve_data.expected_output)
            
            # 保存元数据
            metadata = {
                "name": curve_data.name,
                "parameters": curve_data.parameters,
                "tolerance": curve_data.tolerance,
                "properties": curve_data.properties
            }
            with open(os.path.join(curves_dir, f"{name}_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
                
        # 保存质量指标参考
        quality_refs = self.generate_reference_quality_metrics()
        with open(os.path.join(output_dir, "quality_references.json"), 'w') as f:
            json.dump(quality_refs, f, indent=2)
            
        print(f"金标数据已保存到: {output_dir}")
        print(f"- 图像数据: {len(images)} 个")
        print(f"- 曲线数据: {len(curves)} 个")
        print(f"- 质量指标参考: {len(quality_refs)} 个")
        
    def load_golden_data(self, data_dir: str) -> Tuple[Dict[str, GoldenTestImage], 
                                                      Dict[str, GoldenCurveData],
                                                      Dict[str, Dict[str, float]]]:
        """从文件加载金标数据"""
        images = {}
        curves = {}
        quality_refs = {}
        
        # 加载图像数据
        images_dir = os.path.join(data_dir, "images")
        if os.path.exists(images_dir):
            for file in os.listdir(images_dir):
                if file.endswith("_metadata.json"):
                    name = file.replace("_metadata.json", "")
                    
                    # 加载元数据
                    with open(os.path.join(images_dir, file), 'r') as f:
                        metadata = json.load(f)
                        
                    # 加载图像数组
                    image_file = os.path.join(images_dir, f"{name}.npy")
                    if os.path.exists(image_file):
                        image_data = np.load(image_file)
                        
                        images[name] = GoldenTestImage(
                            name=metadata["name"],
                            description=metadata["description"],
                            image_data=image_data,
                            expected_stats=metadata["expected_stats"],
                            metadata=metadata["metadata"]
                        )
                        
        # 加载曲线数据
        curves_dir = os.path.join(data_dir, "curves")
        if os.path.exists(curves_dir):
            for file in os.listdir(curves_dir):
                if file.endswith("_metadata.json"):
                    name = file.replace("_metadata.json", "")
                    
                    # 加载元数据
                    with open(os.path.join(curves_dir, file), 'r') as f:
                        metadata = json.load(f)
                        
                    # 加载曲线数组
                    input_file = os.path.join(curves_dir, f"{name}_input.npy")
                    output_file = os.path.join(curves_dir, f"{name}_output.npy")
                    
                    if os.path.exists(input_file) and os.path.exists(output_file):
                        input_values = np.load(input_file)
                        expected_output = np.load(output_file)
                        
                        curves[name] = GoldenCurveData(
                            name=metadata["name"],
                            parameters=metadata["parameters"],
                            input_values=input_values,
                            expected_output=expected_output,
                            tolerance=metadata["tolerance"],
                            properties=metadata["properties"]
                        )
                        
        # 加载质量指标参考
        quality_file = os.path.join(data_dir, "quality_references.json")
        if os.path.exists(quality_file):
            with open(quality_file, 'r') as f:
                quality_refs = json.load(f)
                
        return images, curves, quality_refs


def create_golden_test_data():
    """创建并保存金标测试数据"""
    generator = GoldenTestDataGenerator()
    
    # 创建测试数据目录
    test_data_dir = os.path.join(os.path.dirname(__file__), "golden_test_data")
    
    # 生成并保存数据
    generator.save_golden_data(test_data_dir)
    
    return test_data_dir


if __name__ == "__main__":
    # 生成金标测试数据
    data_dir = create_golden_test_data()
    print(f"金标测试数据已生成: {data_dir}")