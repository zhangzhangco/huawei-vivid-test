#!/usr/bin/env python3
"""
状态管理演示
展示会话状态和时域状态的分离存储、自动保存和状态一致性验证功能
"""

import sys
import os
import time
import tempfile
import shutil
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.state_manager import StateManager, get_state_manager, reset_state_manager
import numpy as np


def demo_basic_state_operations():
    """演示基本状态操作"""
    
    print("=" * 60)
    print("演示1: 基本状态操作")
    print("=" * 60)
    
    # 创建临时目录用于演示
    temp_dir = tempfile.mkdtemp()
    print(f"使用临时目录: {temp_dir}")
    
    try:
        # 创建状态管理器
        state_manager = StateManager(temp_dir)
        
        # 获取默认状态
        print("\n1. 获取默认状态:")
        session_state = state_manager.get_session_state()
        temporal_state = state_manager.get_temporal_state()
        
        print(f"   默认Phoenix参数: p={session_state.p}, a={session_state.a}")
        print(f"   默认模式: {session_state.mode}")
        print(f"   时域帧数: {temporal_state.current_frame}")
        
        # 更新会话状态
        print("\n2. 更新会话状态:")
        success = state_manager.update_session_state(
            p=3.0, a=0.8, mode="自动模式", 
            enable_spline=True, th1=0.2, th2=0.5, th3=0.8
        )
        print(f"   更新成功: {success}")
        
        session_state = state_manager.get_session_state()
        print(f"   新Phoenix参数: p={session_state.p}, a={session_state.a}")
        print(f"   新模式: {session_state.mode}")
        print(f"   样条曲线启用: {session_state.enable_spline}")
        
        # 更新时域状态
        print("\n3. 更新时域状态:")
        for i in range(5):
            p_val = 3.0 + i * 0.1
            distortion = 0.05 + i * 0.01
            
            success = state_manager.update_temporal_state(
                p=p_val, a=0.8, distortion=distortion,
                mode="自动模式", channel="MaxRGB", image_hash="demo_image"
            )
            print(f"   帧 {i+1}: p={p_val:.1f}, D'={distortion:.3f}, 成功={success}")
            
        temporal_state = state_manager.get_temporal_state()
        print(f"   总帧数: {temporal_state.current_frame}")
        print(f"   历史长度: {len(temporal_state.parameter_history)}")
        print(f"   方差降低: {temporal_state.variance_reduction:.1f}%")
        print(f"   平滑激活: {temporal_state.smoothing_active}")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_state_persistence():
    """演示状态持久化"""
    
    print("\n" + "=" * 60)
    print("演示2: 状态持久化")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    print(f"使用临时目录: {temp_dir}")
    
    try:
        # 第一阶段：创建和保存状态
        print("\n1. 创建和保存状态:")
        state_manager1 = StateManager(temp_dir)
        
        # 设置会话状态
        state_manager1.update_session_state(
            p=2.5, a=0.7, mode="艺术模式",
            dt_low=0.06, dt_high=0.12,
            window_size=7, lambda_smooth=0.4
        )
        
        # 添加时域数据
        for i in range(3):
            state_manager1.update_temporal_state(
                p=2.5 + i * 0.1, a=0.7, distortion=0.06 + i * 0.005,
                mode="艺术模式", channel="MaxRGB", image_hash="persistent_demo"
            )
            
        # 保存状态
        session_saved = state_manager1.save_session_state()
        temporal_saved = state_manager1.save_temporal_state()
        
        print(f"   会话状态保存: {session_saved}")
        print(f"   时域状态保存: {temporal_saved}")
        
        session_file = Path(temp_dir) / "session_state.json"
        temporal_file = Path(temp_dir) / "temporal_state.json"
        print(f"   会话状态文件存在: {session_file.exists()}")
        print(f"   时域状态文件存在: {temporal_file.exists()}")
        
        # 第二阶段：加载状态
        print("\n2. 加载保存的状态:")
        state_manager2 = StateManager(temp_dir)
        
        session_state = state_manager2.get_session_state()
        temporal_state = state_manager2.get_temporal_state()
        
        print(f"   加载的Phoenix参数: p={session_state.p}, a={session_state.a}")
        print(f"   加载的模式: {session_state.mode}")
        print(f"   加载的质量参数: dt_low={session_state.dt_low}, dt_high={session_state.dt_high}")
        print(f"   加载的时域帧数: {temporal_state.current_frame}")
        print(f"   加载的历史长度: {len(temporal_state.parameter_history)}")
        
        if temporal_state.parameter_history:
            print(f"   第一个历史参数: {temporal_state.parameter_history[0]}")
            print(f"   最后一个历史参数: {temporal_state.parameter_history[-1]}")
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_temporal_state_reset():
    """演示时域状态重置机制"""
    
    print("\n" + "=" * 60)
    print("演示3: 时域状态重置机制")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        state_manager = StateManager(temp_dir)
        
        # 初始状态
        print("\n1. 建立初始时域状态:")
        for i in range(4):
            state_manager.update_temporal_state(
                p=2.0 + i * 0.1, a=0.5, distortion=0.05 + i * 0.01,
                mode="艺术模式", channel="MaxRGB", image_hash="image1"
            )
            
        temporal_state = state_manager.get_temporal_state()
        print(f"   初始帧数: {temporal_state.current_frame}")
        print(f"   初始历史长度: {len(temporal_state.parameter_history)}")
        
        # 切换模式触发重置
        print("\n2. 切换模式触发重置:")
        state_manager.update_temporal_state(
            p=1.8, a=0.4, distortion=0.04,
            mode="自动模式", channel="MaxRGB", image_hash="image1"  # 模式改变
        )
        
        temporal_state = state_manager.get_temporal_state()
        print(f"   重置后帧数: {temporal_state.current_frame}")
        print(f"   重置后历史长度: {len(temporal_state.parameter_history)}")
        print(f"   新模式: {temporal_state.last_mode}")
        
        # 继续添加数据
        print("\n3. 继续添加数据:")
        for i in range(3):
            state_manager.update_temporal_state(
                p=1.8 + i * 0.05, a=0.4, distortion=0.04 + i * 0.002,
                mode="自动模式", channel="MaxRGB", image_hash="image1"
            )
            
        temporal_state = state_manager.get_temporal_state()
        print(f"   最终帧数: {temporal_state.current_frame}")
        print(f"   最终历史长度: {len(temporal_state.parameter_history)}")
        
        # 切换图像触发重置
        print("\n4. 切换图像触发重置:")
        state_manager.update_temporal_state(
            p=2.2, a=0.6, distortion=0.06,
            mode="自动模式", channel="MaxRGB", image_hash="image2"  # 图像改变
        )
        
        temporal_state = state_manager.get_temporal_state()
        print(f"   图像切换后帧数: {temporal_state.current_frame}")
        print(f"   新图像哈希: {temporal_state.last_image_hash}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_state_validation():
    """演示状态一致性验证"""
    
    print("\n" + "=" * 60)
    print("演示4: 状态一致性验证")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        state_manager = StateManager(temp_dir)
        
        # 测试有效状态
        print("\n1. 测试有效状态:")
        state_manager.update_session_state(
            p=2.0, a=0.5, enable_spline=True,
            th1=0.2, th2=0.5, th3=0.8
        )
        
        state_manager.update_temporal_state(
            p=2.0, a=0.5, distortion=0.05,
            mode="艺术模式", channel="MaxRGB", image_hash="valid_test"
        )
        
        is_consistent, errors = state_manager.validate_state_consistency()
        print(f"   状态一致性: {is_consistent}")
        print(f"   错误数量: {len(errors)}")
        
        # 测试无效参数
        print("\n2. 测试无效参数:")
        state_manager.update_session_state(p=10.0, a=1.5)  # 超出范围
        
        is_consistent, errors = state_manager.validate_state_consistency()
        print(f"   状态一致性: {is_consistent}")
        print(f"   错误数量: {len(errors)}")
        for i, error in enumerate(errors, 1):
            print(f"   错误 {i}: {error}")
            
        # 测试无效样条节点
        print("\n3. 测试无效样条节点:")
        state_manager.update_session_state(
            p=2.0, a=0.5, enable_spline=True,
            th1=0.8, th2=0.5, th3=0.2  # 顺序错误
        )
        
        is_consistent, errors = state_manager.validate_state_consistency()
        print(f"   状态一致性: {is_consistent}")
        print(f"   错误数量: {len(errors)}")
        for i, error in enumerate(errors, 1):
            print(f"   错误 {i}: {error}")
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_state_export_import():
    """演示状态导出导入"""
    
    print("\n" + "=" * 60)
    print("演示5: 状态导出导入")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 创建原始状态
        print("\n1. 创建原始状态:")
        state_manager1 = StateManager(temp_dir)
        
        state_manager1.update_session_state(
            p=2.8, a=0.9, mode="自动模式",
            enable_spline=True, th_strength=0.3,
            window_size=11, lambda_smooth=0.6
        )
        
        for i in range(6):
            state_manager1.update_temporal_state(
                p=2.8 + i * 0.02, a=0.9, distortion=0.07 + i * 0.001,
                mode="自动模式", channel="Y", image_hash="export_demo"
            )
            
        session_state = state_manager1.get_session_state()
        temporal_state = state_manager1.get_temporal_state()
        
        print(f"   原始Phoenix参数: p={session_state.p}, a={session_state.a}")
        print(f"   原始模式: {session_state.mode}")
        print(f"   原始时域帧数: {temporal_state.current_frame}")
        print(f"   原始样条强度: {session_state.th_strength}")
        
        # 导出状态
        print("\n2. 导出状态:")
        export_path = Path(temp_dir) / "exported_states.json"
        success = state_manager1.export_states(str(export_path))
        print(f"   导出成功: {success}")
        print(f"   导出文件: {export_path}")
        print(f"   文件大小: {export_path.stat().st_size} 字节")
        
        # 重置状态
        print("\n3. 重置状态:")
        state_manager1.reset_all_states()
        
        session_state = state_manager1.get_session_state()
        temporal_state = state_manager1.get_temporal_state()
        
        print(f"   重置后Phoenix参数: p={session_state.p}, a={session_state.a}")
        print(f"   重置后模式: {session_state.mode}")
        print(f"   重置后时域帧数: {temporal_state.current_frame}")
        
        # 导入状态
        print("\n4. 导入状态:")
        success = state_manager1.import_states(str(export_path))
        print(f"   导入成功: {success}")
        
        session_state = state_manager1.get_session_state()
        temporal_state = state_manager1.get_temporal_state()
        
        print(f"   导入后Phoenix参数: p={session_state.p}, a={session_state.a}")
        print(f"   导入后模式: {session_state.mode}")
        print(f"   导入后时域帧数: {temporal_state.current_frame}")
        print(f"   导入后样条强度: {session_state.th_strength}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_state_summary_and_listeners():
    """演示状态摘要和监听器"""
    
    print("\n" + "=" * 60)
    print("演示6: 状态摘要和监听器")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        state_manager = StateManager(temp_dir)
        
        # 设置状态变化监听器
        print("\n1. 设置状态变化监听器:")
        changes_log = []
        
        def state_change_listener(state_type, changes):
            changes_log.append((state_type, changes))
            print(f"   状态变化: {state_type} -> {changes}")
            
        state_manager.add_state_change_listener(state_change_listener)
        
        # 触发状态变化
        print("\n2. 触发状态变化:")
        state_manager.update_session_state(p=3.2, a=0.85)
        state_manager.update_temporal_state(
            p=3.2, a=0.85, distortion=0.08,
            mode="艺术模式", channel="MaxRGB", image_hash="listener_demo"
        )
        state_manager.clear_temporal_state("自动模式", "Y", "new_image")
        
        print(f"   总共捕获 {len(changes_log)} 个状态变化事件")
        
        # 获取状态摘要
        print("\n3. 获取状态摘要:")
        summary = state_manager.get_state_summary()
        
        print("   会话状态摘要:")
        print(f"     模式: {summary['session']['mode']}")
        print(f"     参数: {summary['session']['parameters']}")
        print(f"     自动保存: {summary['session']['auto_save']}")
        
        print("   时域状态摘要:")
        print(f"     当前帧: {summary['temporal']['current_frame']}")
        print(f"     总帧数: {summary['temporal']['total_frames']}")
        print(f"     历史长度: {summary['temporal']['history_length']}")
        print(f"     方差降低: {summary['temporal']['variance_reduction']:.1f}%")
        
        print("   文件状态摘要:")
        print(f"     会话文件存在: {summary['files']['session_exists']}")
        print(f"     时域文件存在: {summary['files']['temporal_exists']}")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demo_global_state_manager():
    """演示全局状态管理器"""
    
    print("\n" + "=" * 60)
    print("演示7: 全局状态管理器")
    print("=" * 60)
    
    # 重置全局状态管理器
    reset_state_manager()
    
    print("\n1. 获取全局状态管理器:")
    manager1 = get_state_manager()
    manager2 = get_state_manager()
    
    print(f"   管理器1 ID: {id(manager1)}")
    print(f"   管理器2 ID: {id(manager2)}")
    print(f"   是同一实例: {manager1 is manager2}")
    
    # 使用全局管理器
    print("\n2. 使用全局管理器:")
    manager1.update_session_state(p=4.0, a=0.3, mode="全局测试模式")
    
    session_state = manager2.get_session_state()
    print(f"   通过manager2获取的参数: p={session_state.p}, a={session_state.a}")
    print(f"   模式: {session_state.mode}")
    
    # 重置全局管理器
    print("\n3. 重置全局管理器:")
    reset_state_manager()
    manager3 = get_state_manager()
    
    print(f"   新管理器 ID: {id(manager3)}")
    print(f"   与原管理器相同: {manager1 is manager3}")
    
    session_state = manager3.get_session_state()
    print(f"   重置后参数: p={session_state.p}, a={session_state.a}")


def main():
    """主演示函数"""
    
    print("HDR色调映射专利可视化工具 - 状态管理演示")
    print("展示会话状态和时域状态的分离存储、持久化和一致性验证功能")
    
    try:
        demo_basic_state_operations()
        demo_state_persistence()
        demo_temporal_state_reset()
        demo_state_validation()
        demo_state_export_import()
        demo_state_summary_and_listeners()
        demo_global_state_manager()
        
        print("\n" + "=" * 60)
        print("✓ 所有状态管理演示完成！")
        print("\n主要功能:")
        print("- 会话状态和时域状态分离存储")
        print("- JSON序列化和反序列化")
        print("- 自动保存和加载机制")
        print("- 状态一致性验证")
        print("- 时域状态自动重置")
        print("- 状态导出导入")
        print("- 状态变化监听器")
        print("- 全局状态管理器")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()