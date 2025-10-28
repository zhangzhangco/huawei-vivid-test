"""
图像处理器测试
测试多格式图像加载、颜色空间转换、色调映射应用和显示优化功能
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import ImageProcessor, ImageStats, ImageProcessingError, PhoenixCurveCalculator


class TestImageProcessor:
    """图像处理器测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.processor = ImageProcessor()
        self.calculator = PhoenixCurveCalculator()
        
    def test_initialization(self):
        """测试初始化"""
        assert self.processor.max_image_size == 1280
        assert '.png' in self.processor.supported_formats
        assert '.exr' in self.processor.supported_formats
        assert '.jpg' in self.processor.supported_formats
        
    def test_detect_input_format(self):
        """测试输入格式检测"""
        # 测试不存在的文件
        with pytest.raises(ImageProcessingError, match="文件不存在"):
            self.processor.detect_input_format("nonexistent.png")
            
        # 创建临时文件测试
        with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as f:
            temp_path = f.name
            
        try:
            format_type = self.processor.detect_input_format(temp_path)
            assert format_type == 'openexr_linear'
        finally:
            os.unlink(temp_path)
            
        # 测试PQ编码文件名检测
        with tempfile.NamedTemporaryFile(suffix='_pq.png', delete=False) as f:
            temp_path = f.name
            
        try:
            format_type = self.processor.detect_input_format(temp_path)
            assert format_type == 'pq_encoded'
        finally:
            os.unlink(temp_path)
            
    def test_convert_to_pq_domain(self):
        """测试PQ域转换"""
        test_image = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        
        # sRGB转换
        pq_srgb = self.processor.convert_to_pq_domain(test_image, 'srgb_standard')
        assert pq_srgb.shape == test_image.shape
        assert np.all(pq_srgb >= 0)
        assert np.all(pq_srgb <= 1)
        
        # 线性光转换
        pq_linear = self.processor.convert_to_pq_domain(test_image, 'openexr_linear')
        assert pq_linear.shape == test_image.shape
        assert np.all(pq_linear >= 0)
        assert np.all(pq_linear <= 1)
        
        # PQ直接使用
        pq_direct = self.processor.convert_to_pq_domain(test_image, 'pq_encoded')
        np.testing.assert_array_equal(pq_direct, test_image)
        
        # 测试超范围值处理
        high_values = np.array([0.0, 5.0, 15000.0], dtype=np.float32)
        pq_high = self.processor.convert_to_pq_domain(high_values, 'openexr_linear')
        assert np.all(pq_high >= 0)
        assert np.all(pq_high <= 1)
        
    def test_apply_tone_mapping_3d(self):
        """测试3D图像色调映射"""
        # 创建测试图像
        test_image = np.random.rand(32, 32, 3).astype(np.float32)
        
        # 创建简单的色调映射函数
        def simple_tone_curve(L):
            return self.calculator.compute_phoenix_curve(L, p=2.0, a=0.5)
            
        # MaxRGB通道测试
        mapped_maxrgb = self.processor.apply_tone_mapping(
            test_image, simple_tone_curve, "MaxRGB"
        )
        assert mapped_maxrgb.shape == test_image.shape
        assert np.all(mapped_maxrgb >= 0)
        assert np.all(mapped_maxrgb <= 1)
        
        # Y通道测试
        mapped_y = self.processor.apply_tone_mapping(
            test_image, simple_tone_curve, "Y"
        )
        assert mapped_y.shape == test_image.shape
        assert np.all(mapped_y >= 0)
        assert np.all(mapped_y <= 1)
        
        # 验证色调映射确实改变了图像
        assert not np.array_equal(mapped_maxrgb, test_image)
        assert not np.array_equal(mapped_y, test_image)
        
    def test_apply_tone_mapping_2d(self):
        """测试2D图像色调映射"""
        test_image = np.random.rand(32, 32).astype(np.float32)
        
        def simple_tone_curve(L):
            return self.calculator.compute_phoenix_curve(L, p=2.0, a=0.5)
            
        mapped = self.processor.apply_tone_mapping(test_image, simple_tone_curve)
        assert mapped.shape == test_image.shape
        assert np.all(mapped >= 0)
        assert np.all(mapped <= 1)
        
    def test_resize_for_display(self):
        """测试显示尺寸调整"""
        # 小图像不变
        small_image = np.random.rand(100, 100, 3).astype(np.float32)
        resized_small = self.processor.resize_for_display(small_image)
        assert resized_small.shape == small_image.shape
        
        # 大图像缩放
        large_image = np.random.rand(2000, 3000, 3).astype(np.float32)
        resized_large = self.processor.resize_for_display(large_image)
        assert max(resized_large.shape[:2]) <= self.processor.max_image_size
        
        # 验证等比缩放
        original_ratio = large_image.shape[0] / large_image.shape[1]
        resized_ratio = resized_large.shape[0] / resized_large.shape[1]
        assert abs(original_ratio - resized_ratio) < 0.01
        
        # 测试1D图像
        image_1d = np.random.rand(100).astype(np.float32)
        resized_1d = self.processor.resize_for_display(image_1d)
        assert resized_1d.shape == image_1d.shape
        
    def test_get_image_stats(self):
        """测试图像统计信息"""
        test_image = np.random.rand(64, 64, 3).astype(np.float32)
        
        # MaxRGB统计
        stats_maxrgb = self.processor.get_image_stats(test_image, "MaxRGB")
        assert isinstance(stats_maxrgb, ImageStats)
        assert 0 <= stats_maxrgb.min_pq <= 1
        assert 0 <= stats_maxrgb.max_pq <= 1
        assert stats_maxrgb.min_pq <= stats_maxrgb.avg_pq <= stats_maxrgb.max_pq
        assert stats_maxrgb.pixel_count == 64 * 64
        
        # Y通道统计
        stats_y = self.processor.get_image_stats(test_image, "Y")
        assert isinstance(stats_y, ImageStats)
        assert stats_y.pixel_count == 64 * 64
        
        # 2D图像统计
        test_image_2d = np.random.rand(64, 64).astype(np.float32)
        stats_2d = self.processor.get_image_stats(test_image_2d)
        assert stats_2d.pixel_count == 64 * 64
        
    def test_validate_image_upload(self):
        """测试图像上传验证"""
        # 空图像
        is_valid, msg = self.processor.validate_image_upload(None)
        assert not is_valid
        assert "未检测到有效图像" in msg
        
        # 无效类型
        is_valid, msg = self.processor.validate_image_upload("not_an_array")
        assert not is_valid
        assert "图像格式无效" in msg
        
        # 空数组
        is_valid, msg = self.processor.validate_image_upload(np.array([]))
        assert not is_valid
        assert "图像为空" in msg
        
        # 有效2D图像
        valid_2d = np.random.rand(100, 100).astype(np.float32)
        is_valid, msg = self.processor.validate_image_upload(valid_2d)
        assert is_valid
        assert "图像验证通过" in msg
        
        # 有效3D图像
        valid_3d = np.random.rand(100, 100, 3).astype(np.float32)
        is_valid, msg = self.processor.validate_image_upload(valid_3d)
        assert is_valid
        
        # 无效维度
        invalid_dims = np.random.rand(100, 100, 100, 100).astype(np.float32)
        is_valid, msg = self.processor.validate_image_upload(invalid_dims)
        assert not is_valid
        assert "不支持的图像维度" in msg
        
        # 无效通道数
        invalid_channels = np.random.rand(100, 100, 5).astype(np.float32)
        is_valid, msg = self.processor.validate_image_upload(invalid_channels)
        assert not is_valid
        assert "不支持的通道数" in msg
        
        # 过大图像
        too_large = np.random.rand(5000, 5000, 3).astype(np.float32)
        is_valid, msg = self.processor.validate_image_upload(too_large)
        assert not is_valid
        assert "图像过大" in msg
        
    def test_convert_for_display(self):
        """测试显示转换"""
        pq_image = np.random.rand(64, 64, 3).astype(np.float32)
        
        display_image = self.processor.convert_for_display(pq_image)
        assert display_image.shape == pq_image.shape
        assert np.all(display_image >= 0)
        assert np.all(display_image <= 1)
        
        # 测试不同gamma值
        display_gamma_1 = self.processor.convert_for_display(pq_image, gamma=1.0)
        display_gamma_2 = self.processor.convert_for_display(pq_image, gamma=2.2)
        
        # 不同gamma应该产生不同结果
        assert not np.array_equal(display_gamma_1, display_gamma_2)
        
    @patch('cv2.imread')
    def test_load_hdr_image_png(self, mock_imread):
        """测试PNG图像加载"""
        # 模拟8位PNG
        mock_image_8bit = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image_8bit
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            
        try:
            image, path_desc = self.processor.load_hdr_image(temp_path)
            assert image.dtype == np.float32
            assert image.shape == (100, 100, 3)
            assert np.all(image >= 0) and np.all(image <= 1)
            assert "PNG8(sRGB)" in path_desc
        finally:
            os.unlink(temp_path)
            
        # 模拟16位PNG
        mock_image_16bit = np.random.randint(0, 65536, (100, 100, 3), dtype=np.uint16)
        mock_imread.return_value = mock_image_16bit
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            
        try:
            image, path_desc = self.processor.load_hdr_image(temp_path)
            assert image.dtype == np.float32
            assert "PNG16(sRGB)" in path_desc
        finally:
            os.unlink(temp_path)
            
    @patch('cv2.imread')
    def test_load_hdr_image_errors(self, mock_imread):
        """测试图像加载错误处理"""
        # 模拟读取失败
        mock_imread.return_value = None
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            
        try:
            with pytest.raises(ImageProcessingError, match="无法读取PNG文件"):
                self.processor.load_hdr_image(temp_path)
        finally:
            os.unlink(temp_path)
            
        # 测试不支持的格式
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name
            
        try:
            with pytest.raises(ImageProcessingError, match="不支持的文件格式"):
                self.processor.load_hdr_image(temp_path)
        finally:
            os.unlink(temp_path)
            
        # 测试文件不存在
        with pytest.raises(ImageProcessingError, match="文件不存在"):
            self.processor.load_hdr_image("nonexistent.png")
            
    def test_process_image_pipeline_mock(self):
        """测试完整图像处理管线（模拟）"""
        # 创建模拟图像
        mock_image = np.random.rand(100, 100, 3).astype(np.float32)
        
        def mock_tone_curve(L):
            return self.calculator.compute_phoenix_curve(L, p=2.0, a=0.5)
            
        # 模拟成功的管线处理
        with patch.object(self.processor, 'load_hdr_image') as mock_load:
            mock_load.return_value = (mock_image, 'Test(sRGB)')
            
            with patch.object(self.processor, 'detect_input_format') as mock_detect:
                mock_detect.return_value = 'srgb_standard'
                
                result = self.processor.process_image_pipeline(
                    'test.png', mock_tone_curve, 'MaxRGB'
                )
                
                assert result['success'] is True
                assert result['original_image'] is not None
                assert result['mapped_image'] is not None
                assert result['stats_before'] is not None
                assert result['stats_after'] is not None
                assert result['input_format'] == 'srgb_standard'
                assert 'Test(sRGB)' in result['processing_path']
                
    def test_process_image_pipeline_error(self):
        """测试图像处理管线错误处理"""
        def mock_tone_curve(L):
            return L
            
        # 测试文件不存在的情况
        result = self.processor.process_image_pipeline(
            'nonexistent.png', mock_tone_curve
        )
        
        assert result['success'] is False
        assert '图像处理失败' in result['message']
        assert result['original_image'] is None
        assert result['mapped_image'] is None
        
    def test_edge_cases(self):
        """测试边界情况"""
        # 全零图像
        zero_image = np.zeros((64, 64, 3), dtype=np.float32)
        
        def identity_curve(L):
            return L
            
        mapped_zero = self.processor.apply_tone_mapping(zero_image, identity_curve)
        assert np.all(mapped_zero == 0)
        
        # 全一图像
        ones_image = np.ones((64, 64, 3), dtype=np.float32)
        mapped_ones = self.processor.apply_tone_mapping(ones_image, identity_curve)
        assert np.all(mapped_ones <= 1)
        
        # 单像素图像
        single_pixel = np.array([[[0.5, 0.6, 0.7]]], dtype=np.float32)
        mapped_single = self.processor.apply_tone_mapping(single_pixel, identity_curve)
        assert mapped_single.shape == single_pixel.shape
        
    def test_numerical_stability(self):
        """测试数值稳定性"""
        # 极小值测试
        tiny_image = np.full((32, 32, 3), 1e-10, dtype=np.float32)
        
        def phoenix_curve(L):
            return self.calculator.compute_phoenix_curve(L, p=2.0, a=0.5)
            
        mapped_tiny = self.processor.apply_tone_mapping(tiny_image, phoenix_curve)
        assert np.all(np.isfinite(mapped_tiny))
        assert np.all(mapped_tiny >= 0)
        
        # 接近1的值测试
        near_one = np.full((32, 32, 3), 0.999999, dtype=np.float32)
        mapped_near_one = self.processor.apply_tone_mapping(near_one, phoenix_curve)
        assert np.all(np.isfinite(mapped_near_one))
        assert np.all(mapped_near_one <= 1)


class TestImageStats:
    """图像统计信息测试"""
    
    def test_image_stats_creation(self):
        """测试ImageStats创建"""
        stats = ImageStats(
            min_pq=0.0,
            max_pq=1.0,
            avg_pq=0.5,
            var_pq=0.1,
            input_format="test_format",
            processing_path="test_path",
            pixel_count=1000
        )
        
        assert stats.min_pq == 0.0
        assert stats.max_pq == 1.0
        assert stats.avg_pq == 0.5
        assert stats.var_pq == 0.1
        assert stats.input_format == "test_format"
        assert stats.processing_path == "test_path"
        assert stats.pixel_count == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])