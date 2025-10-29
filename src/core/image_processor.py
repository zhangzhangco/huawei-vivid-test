"""
图像处理器模块
实现多格式图像加载、颜色空间转换、色调映射应用和显示优化
"""

import os
import numpy as np
import cv2
from typing import Tuple, Dict, Callable, Optional, Union, Any
from dataclasses import dataclass
import logging

from .pq_converter import PQConverter

# BT.2100 Y 通道权重常量，避免重复创建
BT2100_Y_WEIGHTS = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)


def extract_luminance(image: np.ndarray, mode: str = "Y") -> np.ndarray:
    """
    提取亮度通道

    Args:
        image: 输入图像 (H, W), (H, W, 1), (H, W, 3) 或 (H, W, 4)
        mode: 提取模式 ("Y" 使用BT.2100权重 或 "MaxRGB")

    Returns:
        亮度通道 (H, W)
    """
    if image.ndim == 2:
        # 已经是单通道
        return image

    if image.ndim == 3:
        if image.shape[2] == 1:
            # 单通道图像，直接压缩维度
            return np.squeeze(image, axis=2)
        elif image.shape[2] == 3:
            # 3通道图像
            if mode == "Y":
                # 使用 BT.2100 Y 权重
                return np.tensordot(image, BT2100_Y_WEIGHTS, axes=([-1], [0]))
            else:  # MaxRGB
                return np.max(image, axis=-1)
        elif image.shape[2] == 4:
            # 4通道图像，丢弃Alpha通道后处理RGB
            rgb_image = image[:, :, :3]
            if mode == "Y":
                # 使用 BT.2100 Y 权重
                return np.tensordot(rgb_image, BT2100_Y_WEIGHTS, axes=([-1], [0]))
            else:  # MaxRGB
                return np.max(rgb_image, axis=-1)
        else:
            raise ValueError(f"不支持的通道数: {image.shape[2]}")
    else:
        raise ValueError(f"不支持的图像维度: {image.shape}")


@dataclass
class ImageStats:
    """图像统计信息"""
    min_pq: float
    max_pq: float
    avg_pq: float
    var_pq: float
    input_format: str           # 输入格式路径
    processing_path: str        # 处理路径记录
    pixel_count: int           # 像素总数


class ImageProcessingError(Exception):
    """图像处理错误"""
    pass


class ImageProcessor:
    """图像处理器
    
    实现多格式图像加载、颜色空间转换、色调映射应用和显示优化功能
    支持EXR、PNG、标准格式的图像处理
    """
    
    def __init__(self):
        self.supported_formats = ['.hdr', '.exr', '.tiff', '.tif', '.jpg', '.jpeg', '.png', '.bmp']
        self.max_image_size = 1280  # 最长边像素限制
        self.pq_converter = PQConverter()
        
    def detect_input_format(self, file_path: str) -> str:
        """检测输入格式和色彩空间
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            格式标识字符串
        """
        if not os.path.exists(file_path):
            raise ImageProcessingError(f"文件不存在: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        filename_lower = os.path.basename(file_path).lower()
        
        if ext == '.exr':
            return 'openexr_linear'
        elif 'pq' in filename_lower or 'st2084' in filename_lower:
            return 'pq_encoded'
        elif ext in ['.hdr']:
            return 'hdr_linear'
        else:
            return 'srgb_standard'
            
    def load_hdr_image(self, file_path: str) -> Tuple[np.ndarray, str]:
        """加载HDR图像并返回处理路径
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            (image_array, processing_path): 图像数组和处理路径描述
            
        Raises:
            ImageProcessingError: 图像加载失败
        """
        if not os.path.exists(file_path):
            raise ImageProcessingError(f"文件不存在: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext not in self.supported_formats:
            raise ImageProcessingError(f"不支持的文件格式: {ext}")
            
        try:
            if ext == '.exr':
                # 尝试使用OpenCV读取EXR
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
                if img is None:
                    raise ImageProcessingError("无法读取EXR文件，请检查OpenCV是否支持EXR格式")
                    
                # OpenCV读取的是BGR格式，转换为RGB
                if img.ndim == 3:
                    img = img[..., ::-1]  # BGR->RGB
                    
                return img.astype(np.float32), 'OpenEXR(linear)'
                
            elif ext == '.png':
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ImageProcessingError("无法读取PNG文件")
                    
                # BGR->RGB转换
                if img.ndim == 3:
                    img = img[..., ::-1]
                    
                if img.dtype == np.uint16:
                    img = img.astype(np.float32) / 65535.0
                    return img, 'PNG16(sRGB)'
                else:
                    img = img.astype(np.float32) / 255.0
                    return img, 'PNG8(sRGB)'
                    
            elif ext in ['.hdr']:
                # HDR格式通常是线性光
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
                if img is None:
                    raise ImageProcessingError("无法读取HDR文件")
                    
                if img.ndim == 3:
                    img = img[..., ::-1]  # BGR->RGB
                    
                return img.astype(np.float32), 'HDR(linear)'
                
            else:
                # 标准格式 (JPEG, BMP等)
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ImageProcessingError("无法读取图像文件")
                    
                if img.ndim == 3:
                    img = img[..., ::-1]  # BGR->RGB
                    
                img = img.astype(np.float32) / 255.0
                return img, 'Standard(sRGB)'
                
        except Exception as e:
            if isinstance(e, ImageProcessingError):
                raise
            else:
                raise ImageProcessingError(f"图像加载失败: {str(e)}")
                
    def convert_to_pq_domain(self, image: np.ndarray, input_format: str,
                            reference_white_nits: float = 1000.0) -> np.ndarray:
        """转换到PQ域

        Args:
            image: 输入图像数组
            input_format: 输入格式标识
            reference_white_nits: 参考白点亮度 (nits), 默认1000, 可选2000/10000

        Returns:
            PQ域图像数组
        """
        # 确保输入在合理范围内
        image = np.clip(image, 0, None)  # 不限制上限，但确保非负

        if input_format.lower().startswith('openexr') or 'linear' in input_format.lower():
            # 线性光 → PQ
            max_val = np.max(image)
            if max_val > 1.0:
                # 如果有超过1的值，假设这是绝对亮度值（nits）
                # 使用参考白点做合理限制，避免丢失高光细节
                max_safe = max_val if max_val <= reference_white_nits * 10 else reference_white_nits * 10
                linear_nits = np.clip(image, 0, max_safe)
            else:
                # 归一化值，映射到参考白点亮度
                linear_nits = image * reference_white_nits

            return self.pq_converter.linear_to_pq(linear_nits)

        elif 'pq' in input_format.lower():
            # 已经是PQ编码，直接使用
            return np.clip(image, 0, 1)

        else:  # sRGB或标准格式
            # sRGB → 线性光 → PQ
            linear = self.pq_converter.srgb_to_linear(np.clip(image, 0, 1))
            # 假设线性[0,1] → 0..reference_white_nits
            linear_nits = linear * reference_white_nits
            return self.pq_converter.linear_to_pq(linear_nits)
            
    def apply_tone_mapping(self, image: np.ndarray, 
                          tone_curve_func: Callable[[np.ndarray], np.ndarray],
                          luminance_channel: str = "MaxRGB") -> np.ndarray:
        """应用色调映射 (保持色度策略)
        
        Args:
            image: PQ域输入图像
            tone_curve_func: 色调映射函数
            luminance_channel: 亮度通道类型 ("MaxRGB" 或 "Y")
            
        Returns:
            色调映射后的图像
        """
        if image.ndim == 3:
            # 提取亮度通道
            L_in = extract_luminance(image, luminance_channel)
                
            # 应用色调映射到亮度通道
            L_in_safe = np.clip(L_in, 1e-8, 1.0)  # 避免除零
            L_out = tone_curve_func(L_in_safe)
            
            # 计算缩放比例，保持色度
            ratio = np.divide(L_out, L_in_safe, 
                            out=np.ones_like(L_out), 
                            where=L_in_safe > 1e-8)
            
            # 应用比例到所有通道
            ratio_expanded = ratio.reshape(L_in.shape + (1,))
            mapped_image = image * ratio_expanded
            
            return np.clip(mapped_image, 0, 1)
        else:
            # 灰度图像，直接应用色调映射
            return tone_curve_func(np.clip(image, 1e-8, 1.0))
            
    def resize_for_display(self, image: np.ndarray, use_auto_downsampler: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """调整显示尺寸 (等比缩放到最长边≤max_image_size)
        
        Args:
            image: 输入图像
            use_auto_downsampler: 是否使用自动降采样器
            
        Returns:
            (调整尺寸后的图像, 处理信息)
        """
        if image.ndim < 2:
            return image, {'downsampled': False, 'reason': '无效图像维度'}
            
        original_shape = image.shape
        processing_info = {
            'original_shape': original_shape,
            'downsampled': False,
            'scale_factor': 1.0,
            'reason': '无需调整',
            'method': 'none'
        }
        
        if use_auto_downsampler:
            try:
                # 使用自动降采样器
                from .performance_monitor import get_auto_downsampler
                downsampler = get_auto_downsampler()
                
                should_downsample, scale, reason = downsampler.should_downsample(image.shape)
                
                if should_downsample:
                    resized_image = downsampler.downsample_image(image, scale)
                    processing_info.update({
                        'downsampled': True,
                        'scale_factor': scale,
                        'reason': reason,
                        'method': 'auto_downsampler',
                        'final_shape': resized_image.shape
                    })
                    return resized_image, processing_info
                else:
                    processing_info['reason'] = reason
                    return image, processing_info
                    
            except ImportError:
                # 如果自动降采样器不可用，使用传统方法
                pass
                
        # 传统的尺寸调整方法
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        if max_dim > self.max_image_size:
            scale = self.max_image_size / max_dim
            new_h, new_w = int(h * scale), int(w * scale)
            
            # 确保新尺寸至少为1
            new_h = max(1, new_h)
            new_w = max(1, new_w)
            
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            processing_info.update({
                'downsampled': True,
                'scale_factor': scale,
                'reason': f'图像过大 ({max_dim} > {self.max_image_size})',
                'method': 'traditional',
                'final_shape': resized_image.shape
            })
            return resized_image, processing_info
            
        return image, processing_info
        
    def get_image_stats(self, image: np.ndarray, luminance_channel: str = "MaxRGB") -> ImageStats:
        """获取图像统计信息 (PQ域)
        
        Args:
            image: PQ域图像
            luminance_channel: 亮度通道类型
            
        Returns:
            图像统计信息
        """
        # 提取亮度通道
        L = extract_luminance(image, luminance_channel)
            
        return ImageStats(
            min_pq=float(np.min(L)),
            max_pq=float(np.max(L)),
            avg_pq=float(np.mean(L)),
            var_pq=float(np.var(L)),
            input_format="",  # 将在调用时设置
            processing_path="",  # 将在调用时设置
            pixel_count=int(L.size)
        )
        
    def process_image_pipeline(self, file_path: str, 
                             tone_curve_func: Callable[[np.ndarray], np.ndarray],
                             luminance_channel: str = "MaxRGB") -> Dict[str, Union[np.ndarray, ImageStats, str]]:
        """完整的图像处理管线
        
        Args:
            file_path: 图像文件路径
            tone_curve_func: 色调映射函数
            luminance_channel: 亮度通道类型
            
        Returns:
            包含处理结果的字典
        """
        try:
            # 1. 检测输入格式
            input_format = self.detect_input_format(file_path)
            
            # 2. 加载图像
            original_image, processing_path = self.load_hdr_image(file_path)
            
            # 3. 转换到PQ域
            pq_image = self.convert_to_pq_domain(original_image, input_format)
            
            # 4. 调整显示尺寸
            pq_image_resized, resize_info = self.resize_for_display(pq_image)
            
            # 5. 应用色调映射
            mapped_image = self.apply_tone_mapping(pq_image_resized, tone_curve_func, luminance_channel)
            
            # 6. 获取统计信息
            stats_before = self.get_image_stats(pq_image_resized, luminance_channel)
            stats_after = self.get_image_stats(mapped_image, luminance_channel)
            
            # 设置统计信息的格式和路径
            stats_before.input_format = input_format
            stats_before.processing_path = processing_path
            stats_after.input_format = input_format + " -> Tone Mapped"
            stats_after.processing_path = processing_path + " -> Phoenix Curve"
            
            return {
                'original_image': pq_image_resized,
                'mapped_image': mapped_image,
                'stats_before': stats_before,
                'stats_after': stats_after,
                'input_format': input_format,
                'processing_path': processing_path,
                'resize_info': resize_info,
                'success': True,
                'message': f"成功处理图像: {processing_path}"
            }
            
        except Exception as e:
            logging.error(f"图像处理管线失败: {e}")
            return {
                'original_image': None,
                'mapped_image': None,
                'stats_before': None,
                'stats_after': None,
                'input_format': '',
                'processing_path': '',
                'success': False,
                'message': f"图像处理失败: {str(e)}"
            }
            
    def validate_image_upload(self, image: Optional[np.ndarray], file_path: Optional[str] = None) -> Tuple[bool, str]:
        """验证上传图像
        
        Args:
            image: 图像数组
            file_path: 文件路径（可选）
            
        Returns:
            (is_valid, message): 验证结果和消息
        """
        if image is None:
            return False, "未检测到有效图像"
            
        if not isinstance(image, np.ndarray):
            return False, "图像格式无效"
            
        if image.size == 0:
            return False, "图像为空"
            
        if len(image.shape) not in [2, 3]:
            return False, "不支持的图像维度，仅支持2D或3D图像"
            
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            return False, "不支持的通道数，仅支持1、3或4通道图像"
            
        # 检查图像尺寸
        total_pixels = image.shape[0] * image.shape[1]
        if total_pixels > 10 * 1024 * 1024:  # 10MP限制
            return False, "图像过大，请上传小于10MP的图像"
            
        # 检查文件格式（如果提供了路径）
        if file_path:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in self.supported_formats:
                return False, f"不支持的文件格式: {ext}"
                
        return True, "图像验证通过"
        
    def convert_for_display(self, pq_image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """将PQ域图像转换为适合显示的sRGB图像
        
        Args:
            pq_image: PQ域图像
            gamma: 显示gamma值
            
        Returns:
            sRGB显示图像 (0-1范围)
        """
        # PQ -> 线性光
        linear_nits = self.pq_converter.pq_to_linear(np.clip(pq_image, 0, 1))
        
        # 归一化到显示范围 (假设100 nits为白点)
        linear_normalized = np.clip(linear_nits / 100.0, 0, 1)
        
        # 应用显示gamma
        display_image = np.power(linear_normalized, 1.0 / gamma)
        
        return np.clip(display_image, 0, 1)