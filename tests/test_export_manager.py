#!/usr/bin/env python3
"""
数据导出和诊断功能测试
测试1D LUT导出、CSV导出、诊断包生成和一致性验证
"""

import unittest
import tempfile
import os
import json
import zipfile
import numpy as np
from pathlib import Path
import sys

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import (
    ExportManager, LUTExporter, CSVExporter, DiagnosticPackageGenerator,
    CurveData, QualityMetrics, ExportMetadata,
    SessionState, TemporalStateData, ImageStats,
    get_export_manager, reset_export_manager
)


class TestLUTExporter(unittest.TestCase):
    """测试1D LUT导出器"""
    
    def setUp(self):
        """设置测试环境"""
        self.exporter = LUTExporter()
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试数据
        self.L_input = np.linspace(0, 1, 256)
        self.L_output = self.L_input ** 2.0  # 简单的平方映射
        self.curve_data = CurveData(
            input_luminance=self.L_input,
            output_luminance=self.L_output,
            phoenix_curve=self.L_output
        )
        
        self.session_state = SessionState(
            p=2.0,
            a=0.5,
            mode="艺术模式",
            luminance_channel="MaxRGB"
        )
        
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_export_lut_cube_content(self):
        """测试LUT内容生成"""
        content = self.exporter.export_lut_cube(self.curve_data, self.session_state, samples=64)
        
        # 检查内容格式
        self.assertIn("# HDR Tone Mapping 1D LUT", content)
        self.assertIn("LUT_1D_SIZE 64", content)
        self.assertIn(f"# p={self.session_state.p:.6f}", content)
        self.assertIn(f"# a={self.session_state.a:.6f}", content)
        
        # 检查数据行数
        lines = content.split('\n')
        data_lines = [line for line in lines if not line.startswith('#') and 
                     not line.startswith('LUT_1D_SIZE') and line.strip()]
        self.assertEqual(len(data_lines), 64)
        
        # 检查数据格式
        for line in data_lines[:5]:  # 检查前5行
            parts = line.split()
            self.assertEqual(len(parts), 3)  # RGB三个值
            # 检查是否为有效数字
            for part in parts:
                self.assertIsInstance(float(part), float)
                
    def test_export_lut_cube_file(self):
        """测试LUT文件导出"""
        filename = os.path.join(self.temp_dir, "test.cube")
        result = self.exporter.export_lut_cube(self.curve_data, self.session_state, 
                                              samples=128, filename=filename)
        
        self.assertEqual(result, filename)
        self.assertTrue(os.path.exists(filename))
        
        # 验证文件内容
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
        self.assertIn("LUT_1D_SIZE 128", content)
        
    def test_validate_lut_export(self):
        """测试LUT导出一致性验证"""
        # 创建测试LUT数据
        original = np.linspace(0, 1, 100) ** 2
        exported = original.copy()
        
        # 测试完全一致的情况
        is_consistent, max_error = self.exporter.validate_lut_export(original, exported)
        self.assertTrue(is_consistent)
        self.assertLess(max_error, 1e-10)
        
        # 测试有小误差的情况
        exported_with_error = original + 1e-5
        is_consistent, max_error = self.exporter.validate_lut_export(original, exported_with_error)
        self.assertTrue(is_consistent)  # 1e-5 < 1e-4，应该通过
        self.assertAlmostEqual(max_error, 1e-5, places=6)
        
        # 测试误差过大的情况
        exported_large_error = original + 1e-3
        is_consistent, max_error = self.exporter.validate_lut_export(original, exported_large_error)
        self.assertFalse(is_consistent)  # 1e-3 > 1e-4，应该失败
        self.assertAlmostEqual(max_error, 1e-3, places=4)


class TestCSVExporter(unittest.TestCase):
    """测试CSV导出器"""
    
    def setUp(self):
        """设置测试环境"""
        self.exporter = CSVExporter()
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试数据
        self.L_input = np.linspace(0, 1, 100)
        self.L_output = self.L_input ** 1.5
        self.curve_data = CurveData(
            input_luminance=self.L_input,
            output_luminance=self.L_output,
            phoenix_curve=self.L_output,
            identity_line=self.L_input
        )
        
        self.session_state = SessionState(p=1.5, a=0.3)
        
        self.metadata = ExportMetadata(
            export_time="2024-01-01T12:00:00",
            version="1.0",
            source_system="Test System",
            parameters={"test": "value"}
        )
        
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_export_csv_content(self):
        """测试CSV内容生成"""
        content = self.exporter.export_curve_csv(self.curve_data, self.session_state, self.metadata)
        
        # 检查头部信息
        self.assertIn("# HDR Tone Mapping Curve Data", content)
        self.assertIn(f"# p={self.session_state.p:.6f}", content)
        self.assertIn(f"# a={self.session_state.a:.6f}", content)
        
        # 检查CSV结构
        lines = content.split('\n')
        data_lines = [line for line in lines if not line.startswith('#') and line.strip()]
        
        # 应该有头部行和数据行
        self.assertGreater(len(data_lines), 1)
        
        # 检查头部行
        header = data_lines[0]
        self.assertIn("input_pq", header)
        self.assertIn("output_pq", header)
        self.assertIn("phoenix_curve", header)
        
        # 检查数据行数量
        self.assertEqual(len(data_lines) - 1, len(self.L_input))  # 减去头部行
        
    def test_export_csv_file(self):
        """测试CSV文件导出"""
        filename = os.path.join(self.temp_dir, "test.csv")
        result = self.exporter.export_curve_csv(self.curve_data, self.session_state, 
                                               self.metadata, filename)
        
        self.assertEqual(result, filename)
        self.assertTrue(os.path.exists(filename))
        
        # 验证文件可读性
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
        self.assertIn("input_pq,output_pq", content)


class TestDiagnosticPackageGenerator(unittest.TestCase):
    """测试诊断包生成器"""
    
    def setUp(self):
        """设置测试环境"""
        self.generator = DiagnosticPackageGenerator()
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建完整的测试数据
        self.curve_data = CurveData(
            input_luminance=np.linspace(0, 1, 50),
            output_luminance=np.linspace(0, 1, 50) ** 2,
            phoenix_curve=np.linspace(0, 1, 50) ** 2
        )
        
        self.session_state = SessionState(p=2.0, a=0.5)
        
        self.temporal_state = TemporalStateData()
        self.temporal_state.parameter_history = [(2.0, 0.5), (2.1, 0.52)]
        self.temporal_state.distortion_history = [0.03, 0.035]
        self.temporal_state.total_frames = 2
        
        self.quality_metrics = QualityMetrics(
            perceptual_distortion=0.03,
            local_contrast=0.15,
            variance_distortion=0.08,
            recommended_mode="自动模式",
            computation_time=0.12
        )
        
        self.image_stats = ImageStats(
            min_pq=0.01,
            max_pq=0.95,
            avg_pq=0.45,
            var_pq=0.08,
            input_format="Test",
            processing_path="Test",
            pixel_count=1000
        )
        
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_create_diagnostic_package(self):
        """测试诊断包创建"""
        package_path = self.generator.create_diagnostic_package(
            self.curve_data, self.session_state, self.temporal_state,
            self.quality_metrics, self.image_stats, self.temp_dir
        )
        
        self.assertIsNotNone(package_path)
        self.assertTrue(os.path.exists(package_path))
        self.assertTrue(package_path.endswith('.zip'))
        
        # 验证ZIP文件内容
        with zipfile.ZipFile(package_path, 'r') as zf:
            file_list = zf.namelist()
            
            # 检查必需文件
            required_files = [
                'README.md',
                'system_info.json',
                'config/session_config.json',
                'config/temporal_config.json',
                'analysis/quality_metrics.json',
                'curve_data.csv',
                'tone_mapping.cube'
            ]
            
            for required_file in required_files:
                self.assertIn(required_file, file_list, f"缺少必需文件: {required_file}")
                
            # 验证JSON文件格式
            for json_file in ['system_info.json', 'config/session_config.json']:
                with zf.open(json_file) as f:
                    data = json.load(f)
                    self.assertIsInstance(data, dict)
                    
    def test_diagnostic_package_content_validity(self):
        """测试诊断包内容有效性"""
        package_path = self.generator.create_diagnostic_package(
            self.curve_data, self.session_state, self.temporal_state,
            self.quality_metrics, self.image_stats, self.temp_dir
        )
        
        with zipfile.ZipFile(package_path, 'r') as zf:
            # 检查会话配置
            with zf.open('config/session_config.json') as f:
                session_config = json.load(f)
                self.assertEqual(session_config['session_state']['p'], 2.0)
                self.assertEqual(session_config['session_state']['a'], 0.5)
                
            # 检查质量指标
            with zf.open('analysis/quality_metrics.json') as f:
                metrics_data = json.load(f)
                self.assertEqual(metrics_data['quality_metrics']['perceptual_distortion'], 0.03)
                
            # 检查README存在且非空
            with zf.open('README.md') as f:
                readme_content = f.read().decode('utf-8')
                self.assertIn("HDR色调映射诊断包", readme_content)
                self.assertGreater(len(readme_content), 100)


class TestExportManager(unittest.TestCase):
    """测试导出管理器"""
    
    def setUp(self):
        """设置测试环境"""
        reset_export_manager()  # 重置全局实例
        self.manager = get_export_manager()
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试数据
        self.curve_data = CurveData(
            input_luminance=np.linspace(0, 1, 64),
            output_luminance=np.linspace(0, 1, 64) ** 1.8,
            phoenix_curve=np.linspace(0, 1, 64) ** 1.8
        )
        
        self.session_state = SessionState(p=1.8, a=0.4)
        
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        reset_export_manager()
        
    def test_export_lut(self):
        """测试LUT导出"""
        filename = os.path.join(self.temp_dir, "test.cube")
        success = self.manager.export_lut(self.curve_data, self.session_state, filename, samples=128)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(filename))
        
    def test_export_csv(self):
        """测试CSV导出"""
        filename = os.path.join(self.temp_dir, "test.csv")
        success = self.manager.export_csv(self.curve_data, self.session_state, filename)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(filename))
        
    def test_get_export_summary(self):
        """测试导出摘要功能"""
        # 创建测试文件
        filename = os.path.join(self.temp_dir, "test.cube")
        self.manager.export_lut(self.curve_data, self.session_state, filename, samples=64)
        
        # 获取摘要
        summary = self.manager.get_export_summary(filename)
        
        self.assertIn('file_path', summary)
        self.assertIn('file_size', summary)
        self.assertIn('file_hash', summary)
        self.assertEqual(summary['file_type'], '.cube')
        self.assertIn('estimated_samples', summary)
        
    def test_validate_export_consistency(self):
        """测试导出一致性验证"""
        # 创建LUT文件
        lut_filename = os.path.join(self.temp_dir, "test.cube")
        self.manager.export_lut(self.curve_data, self.session_state, lut_filename, samples=64)
        
        # 验证一致性
        is_consistent, max_error = self.manager.validate_export_consistency(
            self.curve_data.output_luminance, lut_filename, "lut"
        )
        
        self.assertTrue(is_consistent)
        self.assertLess(max_error, 1e-4)
        
    def test_global_export_manager(self):
        """测试全局导出管理器"""
        manager1 = get_export_manager()
        manager2 = get_export_manager()
        
        # 应该返回同一个实例
        self.assertIs(manager1, manager2)
        
        # 重置后应该是新实例
        reset_export_manager()
        manager3 = get_export_manager()
        self.assertIsNot(manager1, manager3)


class TestExportConsistency(unittest.TestCase):
    """测试导出一致性和精度"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ExportManager()
        
        # 创建高精度测试数据
        self.L_input = np.linspace(0, 1, 1024)
        self.L_output = self.L_input ** 2.2  # 标准gamma曲线
        self.curve_data = CurveData(
            input_luminance=self.L_input,
            output_luminance=self.L_output,
            phoenix_curve=self.L_output
        )
        
        self.session_state = SessionState(p=2.2, a=0.0)
        
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_lut_precision_requirement(self):
        """测试LUT精度要求 (需求15.4: 重建曲线最大绝对误差≤1e-4)"""
        filename = os.path.join(self.temp_dir, "precision_test.cube")
        
        # 导出高采样率LUT
        self.manager.export_lut(self.curve_data, self.session_state, filename, samples=4096)
        
        # 验证精度
        is_consistent, max_error = self.manager.validate_export_consistency(
            self.curve_data.output_luminance, filename, "lut"
        )
        
        self.assertTrue(is_consistent, f"LUT一致性验证失败，最大误差: {max_error}")
        self.assertLessEqual(max_error, 1e-4, f"LUT精度不满足要求，最大误差: {max_error}")
        
    def test_csv_precision_requirement(self):
        """测试CSV精度要求"""
        filename = os.path.join(self.temp_dir, "precision_test.csv")
        
        # 导出CSV
        self.manager.export_csv(self.curve_data, self.session_state, filename)
        
        # 验证精度
        is_consistent, max_error = self.manager.validate_export_consistency(
            self.curve_data.output_luminance, filename, "csv"
        )
        
        self.assertTrue(is_consistent, f"CSV一致性验证失败，最大误差: {max_error}")
        self.assertLessEqual(max_error, 1e-4, f"CSV精度不满足要求，最大误差: {max_error}")
        
    def test_metadata_completeness(self):
        """测试元数据完整性 (需求15.2, 20.5)"""
        # 创建诊断包
        temporal_state = TemporalStateData()
        quality_metrics = QualityMetrics(
            perceptual_distortion=0.05,
            local_contrast=0.1,
            variance_distortion=0.02,
            recommended_mode="自动模式",
            computation_time=0.1
        )
        
        package_path = self.manager.create_diagnostic_package(
            self.curve_data, self.session_state, temporal_state, quality_metrics,
            output_dir=self.temp_dir
        )
        
        self.assertIsNotNone(package_path)
        
        # 验证元数据完整性
        with zipfile.ZipFile(package_path, 'r') as zf:
            # 检查LUT头部元数据
            with zf.open('tone_mapping.cube') as f:
                lut_content = f.read().decode('utf-8')
                
            required_metadata = [
                f"# p={self.session_state.p:.6f}",
                f"# a={self.session_state.a:.6f}",
                f"# D_T_low={self.session_state.dt_low:.6f}",
                f"# D_T_high={self.session_state.dt_high:.6f}",
                f"# Window Size M={self.session_state.window_size}",
                f"# Lambda λ={self.session_state.lambda_smooth:.6f}"
            ]
            
            for metadata in required_metadata:
                self.assertIn(metadata, lut_content, f"LUT缺少元数据: {metadata}")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_classes = [
        TestLUTExporter,
        TestCSVExporter,
        TestDiagnosticPackageGenerator,
        TestExportManager,
        TestExportConsistency
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)