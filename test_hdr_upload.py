#!/usr/bin/env python3
"""
测试HDR文件上传功能
"""

import sys
import os
sys.path.insert(0, 'src')

def test_hdr_support():
    """测试HDR文件支持"""
    
    print("🔍 测试HDR文件上传支持")
    print("=" * 40)
    
    try:
        from gradio_app import GradioInterface
        
        # 创建界面
        interface = GradioInterface()
        
        # 测试文件列表
        test_files = [
            'test_images/synthetic_hdr/hdr_gradient.hdr',
            'test_images/synthetic_hdr/high_contrast_scene.hdr',
            'test_images/synthetic_hdr/indoor_outdoor.hdr'
        ]
        
        print("📁 测试HDR文件加载...")
        
        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"\n🔍 测试文件: {test_file}")
                
                # 测试加载
                image, info = interface.load_hdr_image(test_file)
                
                if image is not None:
                    print(f"✅ {info}")
                    print(f"   📊 形状: {image.shape}")
                    print(f"   📊 类型: {image.dtype}")
                    print(f"   📊 范围: [{image.min():.6f}, {image.max():.6f}]")
                    
                    # 测试图像上传处理
                    try:
                        upload_info, stats = interface.handle_image_upload(test_file, "Y")
                        print(f"   ✅ 上传处理成功")
                        print(f"   📋 统计信息: {len(stats)} 项")
                    except Exception as e:
                        print(f"   ❌ 上传处理失败: {e}")
                        
                else:
                    print(f"❌ {info}")
            else:
                print(f"⚠️ 文件不存在: {test_file}")
        
        print(f"\n🎯 支持的文件格式:")
        print("   - .hdr (Radiance HDR)")
        print("   - .exr (OpenEXR)")
        print("   - .jpg/.jpeg (JPEG)")
        print("   - .png (PNG)")
        print("   - .tiff/.tif (TIFF)")
        
        print(f"\n✅ HDR文件上传功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hdr_support()