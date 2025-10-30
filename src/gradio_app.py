"""
HDR色调映射专利可视化工具 - Gradio用户界面
实现参数控制面板、实时曲线可视化、图像处理和质量指标显示
"""

import gradio as gr
import numpy as np
# 在导入 pyplot 之前设置后端，避免在无显示环境中触发警告
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager

def _configure_matplotlib_fonts():
    """确保Matplotlib具备可用的中文字体，避免渲染乱码"""
    import os
    import glob

    preferred_keywords = [
        "pingfang", "heiti", "song", "kaiti", "fangsong",
        "yahei", "simsun", "simhei", "sourcehansans",
        "noto sans cjk", "wqy", "wenquanyi", "sarasa"
    ]

    font_paths = set(font_manager.findSystemFonts())
    search_roots = [
        "/System/Library/Fonts",
        "/Library/Fonts",
        os.path.expanduser("~/Library/Fonts"),
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.local/share/fonts"),
        "C:/Windows/Fonts",
    ]
    for root in search_roots:
        if os.path.isdir(root):
            for pattern in ("*.ttf", "*.ttc", "*.otf"):
                font_paths.update(glob.glob(os.path.join(root, pattern)))

    selected_font_name = None
    for path in font_paths:
        name = os.path.basename(path).lower()
        if any(keyword in name for keyword in preferred_keywords):
            try:
                font_manager.fontManager.addfont(path)
                font_name = font_manager.FontProperties(fname=path).get_name()
                if font_name:
                    selected_font_name = font_name
                    break
            except Exception:
                continue

    fallback_families = [
        selected_font_name,
        'PingFang SC', 'Microsoft YaHei', 'Source Han Sans SC',
        'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei',
        'Arial Unicode MS', 'DejaVu Sans'
    ]
    fallback_families = [name for name in fallback_families if name]

    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = fallback_families
    rcParams['axes.unicode_minus'] = False

_configure_matplotlib_fonts()

import io
import base64
from typing import Dict, List, Tuple, Optional, Any
import json
import traceback
from dataclasses import dataclass, asdict
import time

from core import (
    PhoenixCurveCalculator, PQConverter, QualityMetricsCalculator,
    ImageProcessor, TemporalSmoothingProcessor, SplineCurveCalculator,
    AutoModeParameterEstimator, ParameterValidator, SafeCalculator,
    ImageStats, TemporalStats, EstimationResult, get_state_manager,
    UIErrorHandler, ErrorRecoverySystem, BoundaryChecker, ErrorSeverity,
    get_performance_monitor, get_progress_handler, create_gradio_progress_callback
)


@dataclass
class UIState:
    """UI状态管理"""
    current_mode: str = "艺术模式"
    current_image_path: Optional[str] = None  # 存储文件路径而非图像数组
    current_image_stats: Optional[ImageStats] = None
    last_curve_update: float = 0.0
    processing_time: float = 0.0


class GradioInterface:
    """Gradio界面主类"""
    
    def __init__(self):
        # 初始化核心组件
        self.phoenix_calc = PhoenixCurveCalculator()
        self.pq_converter = PQConverter()
        self.quality_calc = QualityMetricsCalculator()
        self.image_processor = ImageProcessor()
        self.temporal_processor = TemporalSmoothingProcessor()
        self.spline_calc = SplineCurveCalculator()
        self.auto_estimator = AutoModeParameterEstimator()
        self.validator = ParameterValidator()
        self.safe_calc = SafeCalculator()
        
        # 状态管理器
        self.state_manager = get_state_manager()
        
        # 错误处理系统
        self.ui_error_handler = UIErrorHandler()
        self.error_recovery = ErrorRecoverySystem()
        self.boundary_checker = BoundaryChecker()
        
        # 性能监控和进度处理
        self.performance_monitor = get_performance_monitor()
        self.progress_handler = get_progress_handler()
        
        # UI状态
        self.ui_state = UIState()
        
        # 质量评估扩展模块
        try:
            from src.core.metrics_extension import ExtendedMetrics
            from src.core.ui_integration import UIIntegration
            
            self.extended_metrics = ExtendedMetrics()
            self.ui_integration = UIIntegration()
            self.quality_assessment_enabled = True
            
        except ImportError as e:
            logging.warning(f"质量评估模块导入失败: {e}")
            self.extended_metrics = None
            self.ui_integration = None
            self.quality_assessment_enabled = False
        
        # 验证质量评估模块集成
        self._verify_quality_assessment_integration()
        
        # 设置默认参数
        self.default_params = {
            'p': 2.0,
            'a': 0.5,
            'dt_low': 0.05,
            'dt_high': 0.15,
            'window_size': 10,
            'lambda_smooth': 0.35,
            'th1': 0.25,
            'th2': 0.5,
            'th3': 0.75,
            'th_strength': 0.3,
            'luminance_channel': 'MaxRGB'
        }
        
    def create_interface(self) -> gr.Blocks:
        """创建Gradio界面"""
        
        with gr.Blocks(
            title="HDR色调映射专利可视化工具",
            theme=gr.themes.Soft(),
            css=self._get_custom_css()
        ) as interface:
            
            # 标题和说明
            gr.Markdown("""
            # HDR色调映射专利可视化工具
            
            基于Phoenix曲线算法的HDR色调映射可视化系统，支持实时参数调节、质量指标分析和图像处理。
            """)
            
            with gr.Row():
                # 左侧：参数控制面板
                with gr.Column(scale=1):
                    self._create_parameter_panel()
                    
                # 右侧：可视化和结果显示
                with gr.Column(scale=2):
                    self._create_visualization_panel()
                    
            # 底部：图像处理界面
            with gr.Row():
                self._create_image_interface()
                
            # 设置事件处理
            self._setup_event_handlers()
            
            # 设置状态管理监听器
            self._setup_state_listeners()
            
            # 设置性能监控定时更新
            self._setup_performance_monitoring()
            
        return interface
        
    def _create_parameter_panel(self):
        """创建参数控制面板"""
        
        gr.Markdown("## 参数控制")
        
        # 工作模式选择
        self.mode_radio = gr.Radio(
            choices=["自动模式", "艺术模式"],
            value="艺术模式",
            label="工作模式",
            info="自动模式：系统自动计算最优参数；艺术模式：手动调节参数"
        )
        
        # Phoenix曲线参数
        with gr.Group():
            gr.Markdown("### Phoenix曲线参数")
            
            self.p_slider = gr.Slider(
                minimum=0.1,
                maximum=6.0,
                value=self.default_params['p'],
                step=0.1,
                label="亮度控制因子 p",
                info="控制曲线的整体形状，值越大对比度越强"
            )
            
            self.a_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=self.default_params['a'],
                step=0.01,
                label="缩放因子 a",
                info="控制曲线的缩放程度，影响亮度映射范围"
            )
            
        # 质量指标参数
        with gr.Group():
            gr.Markdown("### 质量指标参数")
            
            self.dt_low_slider = gr.Slider(
                minimum=0.01,
                maximum=0.15,
                value=self.default_params['dt_low'],
                step=0.01,
                label="失真下阈值 D_T_low",
                info="模式推荐的下阈值"
            )
            
            self.dt_high_slider = gr.Slider(
                minimum=0.05,
                maximum=0.20,
                value=self.default_params['dt_high'],
                step=0.01,
                label="失真上阈值 D_T_high",
                info="模式推荐的上阈值"
            )
            
            self.channel_radio = gr.Radio(
                choices=["MaxRGB", "Y"],
                value=self.default_params['luminance_channel'],
                label="亮度通道",
                info="选择用于计算的亮度通道"
            )
            
        # 时域平滑参数
        with gr.Group():
            gr.Markdown("### 时域平滑参数")
            
            self.window_slider = gr.Slider(
                minimum=5,
                maximum=15,
                value=self.default_params['window_size'],
                step=1,
                label="时域窗口大小 M",
                info="时域平滑的窗口长度（帧数）"
            )
            
            self.lambda_slider = gr.Slider(
                minimum=0.2,
                maximum=0.5,
                value=self.default_params['lambda_smooth'],
                step=0.05,
                label="平滑强度 λ"
            )
            
        # 样条曲线参数
        with gr.Group():
            gr.Markdown("### 样条曲线参数（可选）")
            
            self.enable_spline = gr.Checkbox(
                value=False,
                label="启用样条曲线",
                info="启用多段样条曲线进行局部优化"
            )
            
            with gr.Row():
                self.th1_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.4,
                    value=self.default_params['th1'],
                    step=0.01,
                    label="节点1 (TH1)"
                )
                
                self.th2_slider = gr.Slider(
                    minimum=0.4,
                    maximum=0.6,
                    value=self.default_params['th2'],
                    step=0.01,
                    label="节点2 (TH2)"
                )
                
                self.th3_slider = gr.Slider(
                    minimum=0.6,
                    maximum=0.9,
                    value=self.default_params['th3'],
                    step=0.01,
                    label="节点3 (TH3)"
                )
                
            self.th_strength_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=self.default_params['th_strength'],
                step=0.1,
                label="样条强度",
                info="样条曲线与Phoenix曲线的混合比例"
            )
            
        # 控制按钮
        with gr.Row():
            self.reset_btn = gr.Button("重置参数", variant="secondary")
            self.apply_auto_btn = gr.Button("应用自动参数", variant="primary")
            
    def _create_visualization_panel(self):
        """创建可视化面板"""
        
        # 曲线可视化
        with gr.Group():
            gr.Markdown("## 曲线可视化")
            self.curve_plot = gr.Plot(label="Phoenix曲线")
            
        # 质量指标显示
        with gr.Group():
            gr.Markdown("## 质量指标")
            
            with gr.Row():
                self.distortion_number = gr.Number(
                    label="感知失真 D'",
                    precision=6,
                    interactive=False
                )
                
                self.contrast_number = gr.Number(
                    label="局部对比度",
                    precision=6,
                    interactive=False
                )
                
            with gr.Row():
                self.mode_recommendation = gr.Textbox(
                    label="模式建议",
                    interactive=False,
                    max_lines=1
                )
                
                self.processing_time = gr.Number(
                    label="处理时间 (ms)",
                    precision=1,
                    interactive=False
                )
                
        # 时域平滑统计
        with gr.Group():
            gr.Markdown("## 时域平滑统计")

            with gr.Row():
                self.frame_count = gr.Number(
                    label="历史帧数",
                    precision=0,
                    interactive=False
                )

                self.variance_reduction = gr.Number(
                    label="方差降低 (%)",
                    precision=1,
                    interactive=False
                )

            with gr.Row():
                self.delta_p_raw = gr.Number(
                    label="Δp_raw",
                    precision=4,
                    interactive=False
                )

                self.delta_p_filtered = gr.Number(
                    label="Δp_filtered",
                    precision=4,
                    interactive=False
                )
                
        # HDR质量评估扩展
        with gr.Group():
            gr.Markdown("## HDR质量评估")
            
            # 质量状态显示
            self.quality_status_html = gr.HTML(
                value="<div id='quality-status'>等待处理...</div>",
                label="质量状态"
            )
            
            # PQ直方图显示
            self.pq_histogram_plot = gr.Plot(
                label="PQ直方图对比",
                value=None
            )
            
            # 艺术家模式提示
            self.artist_tips_html = gr.HTML(
                value="<div id='artist-tips'>暂无建议</div>",
                label="调整建议"
            )
        
        # 系统状态和错误反馈
        with gr.Group():
            gr.Markdown("## 系统状态")
            
            with gr.Row():
                self.system_status = gr.Textbox(
                    label="系统状态",
                    value="正常",
                    interactive=False,
                    max_lines=1
                )
                
                self.error_count = gr.Number(
                    label="错误计数",
                    value=0,
                    precision=0,
                    interactive=False
                )
                
            # 性能监控显示
            with gr.Row():
                self.performance_status = gr.Textbox(
                    label="性能状态",
                    value="监控中...",
                    interactive=False,
                    max_lines=2
                )
                
                self.acceleration_status = gr.Textbox(
                    label="加速状态",
                    value="检测中...",
                    interactive=False,
                    max_lines=2
                )
                
            with gr.Row():
                self.auto_recovery_status = gr.Textbox(
                    label="自动恢复",
                    value="启用",
                    interactive=False,
                    max_lines=1
                )
                
                self.last_error = gr.Textbox(
                    label="最近错误",
                    value="无",
                    interactive=False,
                    max_lines=2
                )
                
            with gr.Row():
                self.reset_errors_btn = gr.Button("重置错误", variant="secondary", size="sm")
                self.system_diagnostic_btn = gr.Button("系统诊断", variant="secondary", size="sm")
                self.performance_reset_btn = gr.Button("重置性能", variant="secondary", size="sm")
                
        # Auto模式信息显示
        with gr.Group():
            gr.Markdown("## Auto模式信息")
            
            with gr.Row():
                self.estimated_p = gr.Number(
                    label="估算 p 值",
                    precision=3,
                    interactive=False
                )
                
                self.estimated_a = gr.Number(
                    label="估算 a 值",
                    precision=3,
                    interactive=False
                )
                
            self.estimation_info = gr.Textbox(
                label="估算信息",
                interactive=False,
                max_lines=3
            )
            
        # 状态管理信息显示
        with gr.Group():
            gr.Markdown("## 状态管理信息")
            
            with gr.Row():
                self.temporal_frames = gr.Number(
                    label="时域帧数",
                    precision=0,
                    interactive=False
                )
                
                self.state_variance_reduction = gr.Number(
                    label="方差降低 (%)",
                    precision=1,
                    interactive=False
                )
                
            with gr.Row():
                self.save_state_btn = gr.Button("保存状态", variant="secondary")
                self.load_state_btn = gr.Button("加载状态", variant="secondary")
            
    def _create_image_interface(self):
        """创建图像处理界面"""
        
        gr.Markdown("## 图像处理")
        
        with gr.Row():
            # 图像上传
            with gr.Column():
                self.image_input = gr.File(
                    label="上传HDR图像 (.hdr, .exr, .jpg, .png)",
                    file_types=[".hdr", ".exr", ".jpg", ".jpeg", ".png", ".tiff", ".tif"],
                    type="filepath"
                )
                
                self.image_info = gr.Textbox(
                    label="图像信息",
                    interactive=False,
                    max_lines=4
                )
                
            # 原图显示
            with gr.Column():
                self.original_image_display = gr.Image(
                    label="原始图像"
                )

        # 处理结果
        with gr.Column():
            self.image_output = gr.Image(
                label="色调映射结果"
            )

            with gr.Row():
                self.process_btn = gr.Button("处理图像", variant="primary")

                with gr.Row():
                    self.export_format = gr.Dropdown(
                        choices=["json", "lut", "csv", "diagnostic"],
                        value="json",
                        label="导出格式"
                    )
                    self.export_btn = gr.Button("导出数据", variant="secondary")
                    
        # 图像统计对比
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 原始图像统计")
                self.orig_stats = gr.JSON(label="统计信息")

            with gr.Column():
                gr.Markdown("### 处理后统计")
                self.processed_stats = gr.JSON(label="统计信息")

        # PQ直方图对比视图
        with gr.Row():
            with gr.Column():
                gr.Markdown("### PQ直方图对比")
                self.histogram_plot = gr.Plot(
                    label="原始/处理后PQ直方图对比"
                )
                
    def _compute_core_tone_mapping(self, p: float, a: float, channel: str = "MaxRGB",
                                   use_real_image: bool = True) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        核心色调映射计算函数，统一曲线和指标计算流程

        Args:
            p: Phoenix曲线参数p
            a: Phoenix曲线参数a
            channel: 亮度通道类型
            use_real_image: 是否使用真实图像数据

        Returns:
            (L_in, L_out, success): 输入亮度、输出亮度、是否成功
        """
        try:
            if use_real_image and self.ui_state.current_image_path is not None:
                # 使用真实图像
                image, processing_path = self.load_hdr_image(self.ui_state.current_image_path)
                if image is not None:
                    # 检测输入格式并转换到PQ域（符合需求12.4）
                    input_format = self.image_processor.detect_input_format(self.ui_state.current_image_path)
                    pq_image = self.image_processor.convert_to_pq_domain(image, input_format)

                    # 从PQ域图像提取亮度
                    from src.core.image_processor import extract_luminance
                    L_in = extract_luminance(pq_image, channel)

                    # 应用色调映射
                    tone_curve_func = lambda x: self.phoenix_calc.compute_phoenix_curve(x, p, a)
                    L_out = tone_curve_func(L_in)
                    return L_in, L_out, True

            # 使用合成数据（回退方案）
            L_in = np.linspace(0, 1, 1000)
            L_out = self.phoenix_calc.compute_phoenix_curve(L_in, p, a)
            return L_in, L_out, True

        except Exception as e:
            # 失败时返回空数组
            return np.array([]), np.array([]), False

    def _setup_event_handlers(self):
        """设置事件处理器"""
        
        # 参数变化时更新曲线
        param_inputs = [
            self.p_slider, self.a_slider, self.enable_spline,
            self.th1_slider, self.th2_slider, self.th3_slider, self.th_strength_slider
        ]
        
        for param_input in param_inputs:
            param_input.change(
                fn=self.update_curve_visualization,
                inputs=param_inputs + [self.mode_radio],
                outputs=[
                    self.curve_plot, self.distortion_number, self.contrast_number,
                    self.mode_recommendation, self.processing_time
                ]
            )
            
        # 质量指标参数变化
        quality_inputs = [self.dt_low_slider, self.dt_high_slider, self.channel_radio]
        for quality_input in quality_inputs:
            quality_input.change(
                fn=self.update_quality_metrics,
                inputs=param_inputs + quality_inputs + [self.mode_radio],
                outputs=[
                    self.distortion_number, self.contrast_number,
                    self.mode_recommendation
                ]
            )
            
        # 模式切换
        self.mode_radio.change(
            fn=self.handle_mode_change,
            inputs=[self.mode_radio] + param_inputs,
            outputs=[
                self.estimated_p, self.estimated_a, self.estimation_info,
                self.p_slider, self.a_slider
            ]
        )
        
        # 按钮事件
        self.reset_btn.click(
            fn=self.reset_parameters,
            outputs=param_inputs + quality_inputs + [self.window_slider, self.lambda_slider]
        )
        
        self.apply_auto_btn.click(
            fn=self.apply_auto_parameters,
            inputs=[self.mode_radio],
            outputs=[self.p_slider, self.a_slider, self.estimation_info]
        )
        
        # 图像处理事件
        self.image_input.change(
            fn=self.handle_image_upload,
            inputs=[self.image_input, self.channel_radio],
            outputs=[self.image_info, self.orig_stats, self.original_image_display]
        )
        
        self.process_btn.click(
            fn=self.process_image_with_progress,
            inputs=param_inputs + [self.image_input, self.channel_radio],
            outputs=[
                self.image_output, self.processed_stats,
                self.distortion_number, self.contrast_number,
                self.mode_recommendation, self.performance_status,
                self.pq_histogram_plot, self.quality_status_html, self.artist_tips_html
            ]
        )
        
        self.export_btn.click(
            fn=self.export_data,
            inputs=param_inputs + quality_inputs + [self.window_slider, self.lambda_slider, self.export_format],
            outputs=[gr.File()]
        )
        
        # 状态管理按钮事件
        self.save_state_btn.click(
            fn=self.save_state,
            outputs=[]
        )
        
        self.load_state_btn.click(
            fn=self.load_state,
            outputs=param_inputs + quality_inputs + [self.window_slider, self.lambda_slider]
        )
        
        # 性能监控按钮事件
        self.performance_reset_btn.click(
            fn=self.reset_performance_metrics,
            outputs=[self.performance_status, self.acceleration_status]
        )
        
    @get_performance_monitor().measure_operation("curve_visualization")
    def update_curve_visualization(self, p: float, a: float, enable_spline: bool,
                                 th1: float, th2: float, th3: float, th_strength: float,
                                 mode: str) -> Tuple[plt.Figure, float, float, str, float]:
        """更新曲线可视化"""
        
        start_time = time.time()
        
        try:
            # 使用增强的安全计算器进行参数验证和计算
            parameters = {
                'p': p, 'a': a, 'th1': th1, 'th2': th2, 'th3': th3, 
                'th_strength': th_strength, 'mode': mode
            }
            
            # 全面参数验证
            is_valid, corrected_params, validation_errors = self.safe_calc.comprehensive_parameter_validation(parameters)
            
            # 使用修正后的参数
            p_safe = corrected_params.get('p', p)
            a_safe = corrected_params.get('a', a)
            th_strength_safe = corrected_params.get('th_strength', th_strength)
            
            # 获取显示曲线
            L, L_out, success, status_msg, detailed_status = self.safe_calc.safe_phoenix_calculation_enhanced(
                np.linspace(0, 1, self.phoenix_calc.display_samples), p_safe, a_safe
            )
            
            if not success:
                error_plot = self.ui_error_handler.create_error_plot(status_msg, "curve")
                return error_plot, 0.0, 0.0, f"Calculation failed: {status_msg}", 0.0
                
            # 样条曲线处理
            final_curve = L_out.copy()
            spline_status = ""
            
            if enable_spline and th_strength_safe > 0:
                try:
                    nodes = [corrected_params.get('th1', th1), 
                            corrected_params.get('th2', th2), 
                            corrected_params.get('th3', th3)]
                    
                    # 使用安全的样条计算
                    final_curve, spline_success, spline_msg = self.safe_calc.safe_spline_calculation(
                        L_out, L, nodes, th_strength_safe
                    )
                    
                    spline_status = f" | 样条: {spline_msg}"
                    
                    if not spline_success:
                        # 创建警告但继续使用Phoenix曲线
                        self.ui_error_handler.add_error(
                            ErrorSeverity.WARNING,
                            "样条计算警告",
                            spline_msg,
                            "已回退到Phoenix曲线"
                        )
                        
                except Exception as e:
                    spline_status = f" | 样条错误: {str(e)}"
                    self.ui_error_handler.create_calculation_error("样条曲线", str(e))
                    
            # 创建图表
            try:
                fig = self._create_enhanced_curve_plot(L, L_out, final_curve, enable_spline, 
                                                     p_safe, a_safe, detailed_status)
            except Exception as plot_error:
                self.ui_error_handler.create_system_error("plot_creation_failed", reason=str(plot_error))
                fig = self.ui_error_handler.create_error_plot(f"图表创建失败: {str(plot_error)}", "curve")
                
            # 计算质量指标（基于当前曲线）
            try:
                L_in_curve = np.asarray(L, dtype=np.float64)
                L_out_curve = np.asarray(final_curve, dtype=np.float64)

                distortion = self.quality_calc.compute_perceptual_distortion(L_in_curve, L_out_curve)
                contrast = self.quality_calc.compute_local_contrast(L_out_curve)

                # 模式推荐
                recommendation = self.quality_calc.recommend_mode_with_hysteresis(distortion)
                
            except Exception as metrics_error:
                self.ui_error_handler.create_calculation_error("质量指标", str(metrics_error))
                distortion, contrast, recommendation = 0.0, 0.0, "计算错误"
                
            # 更新时域状态
            try:
                self._update_temporal_state_from_ui(p_safe, a_safe, distortion)
            except Exception as temporal_error:
                self.ui_error_handler.add_error(ErrorSeverity.WARNING, "时域状态更新", str(temporal_error))
                
            # 更新会话状态
            try:
                self._update_session_state_from_ui(p=p_safe, a=a_safe, enable_spline=enable_spline)
            except Exception as session_error:
                self.ui_error_handler.add_error(ErrorSeverity.WARNING, "会话状态更新", str(session_error))
                
            processing_time = (time.time() - start_time) * 1000
            
            # 组合状态消息
            full_status = status_msg + spline_status
            if validation_errors:
                full_status += f" | 参数修正: {len(validation_errors)}项"
                
            return fig, distortion, contrast, recommendation, processing_time
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.ui_error_handler.create_calculation_error("曲线更新", str(e))
            return self.ui_error_handler.create_error_plot(str(e), "curve"), 0.0, 0.0, "Calculation failed", processing_time
            
    def _compute_simple_spline(self, L: np.ndarray, nodes: List[float]) -> np.ndarray:
        """计算简化的样条曲线"""
        # 简化的样条实现，实际应该使用SplineCurveCalculator
        spline_curve = L.copy()
        
        # 在节点处添加一些变化
        for i, node in enumerate(nodes):
            idx = int(node * len(L))
            if 0 < idx < len(L) - 1:
                # 简单的局部调整
                adjustment = 0.1 * np.sin(np.pi * (i + 1))
                start_idx = max(0, idx - 20)
                end_idx = min(len(L), idx + 20)
                
                # 高斯权重
                x = np.arange(start_idx, end_idx)
                weights = np.exp(-0.5 * ((x - idx) / 10) ** 2)
                spline_curve[start_idx:end_idx] += adjustment * weights
                
        return np.clip(spline_curve, 0, 1)
        
    def _create_curve_plot(self, L: np.ndarray, phoenix_curve: np.ndarray,
                          final_curve: np.ndarray, enable_spline: bool,
                          p: float, a: float) -> plt.Figure:
        """创建曲线图表"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 恒等线
        ax.plot(L, L, 'k--', alpha=0.5, linewidth=1, label='恒等线')
        
        # Phoenix曲线
        ax.plot(L, phoenix_curve, 'b-', linewidth=2, label=f'Phoenix曲线 (p={p:.1f}, a={a:.2f})')
        
        # 样条曲线（如果启用）
        if enable_spline and not np.array_equal(phoenix_curve, final_curve):
            ax.plot(L, final_curve, 'r-', linewidth=2, label='样条混合曲线')
            
        ax.set_xlabel('输入亮度 (PQ域)', fontsize=12)
        ax.set_ylabel('输出亮度 (PQ域)', fontsize=12)
        ax.set_title('HDR色调映射曲线', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        ax.set_xlabel('Input Luminance (PQ Domain)', fontsize=12)
        ax.set_ylabel('Output Luminance (PQ Domain)', fontsize=12)
        ax.set_title('HDR Tone Mapping Curve', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
        
    def _create_enhanced_curve_plot(self, L: np.ndarray, phoenix_curve: np.ndarray, 
                                  final_curve: np.ndarray, enable_spline: bool,
                                  p: float, a: float, detailed_status: Dict[str, bool]) -> plt.Figure:
        """创建增强的曲线图表，包含状态指示"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 设置背景色基于系统状态
        if all(detailed_status.values()):
            fig.patch.set_facecolor('#f8fff8')  # 浅绿色 - 一切正常
        elif detailed_status.get('recovery_applied', False):
            fig.patch.set_facecolor('#fff8e1')  # 浅黄色 - 已恢复
        else:
            fig.patch.set_facecolor('#ffffff')  # 白色 - 默认
            
        # 绘制恒等线
        ax.plot(L, L, 'k--', alpha=0.3, linewidth=1, label='恒等线')
        
        # 绘制Phoenix曲线
        phoenix_color = '#2196F3' if detailed_status.get('computation_success', False) else '#FF9800'
        ax.plot(L, phoenix_curve, color=phoenix_color, linewidth=2, 
                label=f'Phoenix曲线 (p={p:.2f}, a={a:.2f})')
        
        # 绘制最终曲线（如果与Phoenix不同）
        if enable_spline and not np.array_equal(phoenix_curve, final_curve):
            final_color = '#4CAF50' if detailed_status.get('monotonicity_check', False) else '#F44336'
            ax.plot(L, final_curve, color=final_color, linewidth=2, 
                    label='最终曲线 (含样条)', linestyle='-')
            
        # 添加状态指示器
        status_text = []
        if detailed_status.get('parameter_validation', False):
            status_text.append("✓ 参数验证")
        else:
            status_text.append("✗ 参数验证")
            
        if detailed_status.get('computation_success', False):
            status_text.append("✓ 计算成功")
        else:
            status_text.append("✗ 计算失败")
            
        if detailed_status.get('monotonicity_check', False):
            status_text.append("✓ 单调性")
        else:
            status_text.append("✗ 单调性")
            
        if detailed_status.get('numerical_stability', False):
            status_text.append("✓ 数值稳定")
        else:
            status_text.append("✗ 数值稳定")
            
        if detailed_status.get('recovery_applied', False):
            status_text.append("🔧 已恢复")
            
        # 在图表上显示状态
        status_str = " | ".join(status_text)
        ax.text(0.02, 0.98, status_str, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='white', alpha=0.8))
        
        # 设置坐标轴
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Input Luminance (PQ Domain)', fontsize=12)
        ax.set_ylabel('Output Luminance (PQ Domain)', fontsize=12)
        ax.set_title('Phoenix Tone-Mapping Curve', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')

        plt.tight_layout()
        return fig
        
    def _create_error_plot(self, error_msg: str) -> plt.Figure:
        """创建错误显示图表（保持向后兼容）"""
        return self.ui_error_handler.create_error_plot(error_msg, "curve")
        
    def _create_fallback_error_plot(self, error_msg: str) -> plt.Figure:
        """创建最简单的错误图表"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, error_msg, 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='blue')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('HDR色调映射曲线')
            ax.set_xlabel('输入亮度')
            ax.set_ylabel('输出亮度')
            return fig
        except:
            return self._create_minimal_plot()
    
    def _create_minimal_plot(self) -> plt.Figure:
        """创建最基础的图表"""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 绘制简单的恒等线
            x = np.linspace(0, 1, 100)
            y = x
            ax.plot(x, y, 'k--', alpha=0.5, label='恒等线')
            
            # 绘制默认Phoenix曲线
            p_default = 2.0
            y_phoenix = np.power(x + 1e-8, 1.0/p_default) * 0.5 + x * 0.5
            ax.plot(x, y_phoenix, 'b-', linewidth=2, label='Phoenix曲线')
            
            ax.set_xlabel('输入亮度')
            ax.set_ylabel('输出亮度')
            ax.set_title('HDR色调映射曲线')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            return fig
        except:
            # 最后的备用方案
            fig = plt.figure(figsize=(8, 6))
            return fig
        
    def update_quality_metrics(self, p: float, a: float, enable_spline: bool,
                             th1: float, th2: float, th3: float, th_strength: float,
                             dt_low: float, dt_high: float, channel: str,
                             mode: str) -> Tuple[float, float, str]:
        """更新质量指标"""
        
        try:
            # 更新质量计算器参数
            self.quality_calc.dt_low = dt_low
            self.quality_calc.dt_high = dt_high
            self.quality_calc.luminance_channel = channel
            
            # 使用统一的核心计算函数
            L_in, L_out, success = self._compute_core_tone_mapping(p, a, channel, use_real_image=True)

            if not success:
                return 0.0, 0.0, "计算失败"
                
            # 计算指标
            distortion = self.quality_calc.compute_perceptual_distortion(L_in, L_out)
            contrast = self.quality_calc.compute_local_contrast(L_out)
            recommendation = self.quality_calc.recommend_mode_with_hysteresis(distortion)
            
            return distortion, contrast, recommendation
            
        except Exception as e:
            return 0.0, 0.0, f"计算错误: {str(e)}"
            
    def handle_mode_change(self, mode: str, p: float, a: float, enable_spline: bool,
                          th1: float, th2: float, th3: float, th_strength: float
                          ) -> Tuple[float, float, str, float, float]:
        """处理模式切换"""
        
        self.ui_state.current_mode = mode
        
        if mode == "自动模式":
            # 自动估算参数
            if self.ui_state.current_image_stats is not None:
                # 使用真实图像统计
                result = self.auto_estimator.estimate_parameters(self.ui_state.current_image_stats)
                estimated_p = result.p_estimated
                estimated_a = result.a_estimated
                info = f"基于图像统计自动估算\n置信度: {result.confidence_score:.2f}"
            else:
                # 使用默认估算
                estimated_p = 1.8
                estimated_a = 0.4
                info = "使用默认自动参数\n（上传图像后将基于图像统计估算）"
                
            return estimated_p, estimated_a, info, estimated_p, estimated_a
            
        else:
            # 艺术模式，保持当前参数
            return 0.0, 0.0, "手动调节模式", p, a
            
    def reset_parameters(self) -> Tuple:
        """重置所有参数到默认值"""
        
        # 重置状态管理器中的会话状态
        self.state_manager.reset_all_states()
        
        # 获取重置后的状态
        session_state = self.state_manager.get_session_state()
        
        return (
            session_state.p,                    # p_slider
            session_state.a,                    # a_slider
            session_state.enable_spline,        # enable_spline
            session_state.th1,                  # th1_slider
            session_state.th2,                  # th2_slider
            session_state.th3,                  # th3_slider
            session_state.th_strength,          # th_strength_slider
            session_state.dt_low,               # dt_low_slider
            session_state.dt_high,              # dt_high_slider
            session_state.luminance_channel,    # channel_radio
            session_state.window_size,          # window_slider
            session_state.lambda_smooth         # lambda_slider
        )
        
    def apply_auto_parameters(self, mode: str) -> Tuple[float, float, str]:
        """应用自动参数"""
        
        if mode != "自动模式":
            return 2.0, 0.5, "请先切换到自动模式"
            
        try:
            if self.ui_state.current_image_stats is not None:
                result = self.auto_estimator.estimate_parameters(self.ui_state.current_image_stats)
                info = f"参数已应用\n置信度: {result.confidence_score:.2f}"
                return result.p_estimated, result.a_estimated, info
            else:
                return 1.8, 0.4, "已应用默认自动参数"
                
        except Exception as e:
            return 2.0, 0.5, f"参数估算失败: {str(e)}"
            
    def handle_image_upload(self, image_file: str, channel: str):
        """处理图像上传"""

        if image_file is None:
            self.ui_state.current_image_path = None
            self.ui_state.current_image_stats = None
            return "未上传图像", {}, None

        # 检测输入格式
        input_format = self.image_processor.detect_input_format(image_file)

        # 加载HDR图像
        image, processing_path = self.load_hdr_image(image_file)
        if image is None:
            self.ui_state.current_image_path = None
            self.ui_state.current_image_stats = None
            return processing_path, {}, None  # 此时 processing_path 包含错误信息

        try:
            # 存储文件路径
            self.ui_state.current_image_path = image_file

            # 转换到PQ域，使用检测到的格式
            pq_image = self.image_processor.convert_to_pq_domain(image, input_format)
            
            # 计算图像统计
            stats = self.image_processor.get_image_stats(pq_image, channel)

            # 设置统计信息的格式和路径
            stats.input_format = input_format
            stats.processing_path = processing_path

            self.ui_state.current_image_stats = stats  # stats is already an ImageStats object

            # 生成信息文本
            from pathlib import Path
            file_name = Path(image_file).name if image_file else "未知文件"

            info_text = f"""已加载HDR图像
尺寸: {image.shape[0]} x {image.shape[1]} x {image.shape[2] if len(image.shape) > 2 else 1}
文件: {file_name}
格式: {stats.input_format}
处理路径: {stats.processing_path}
亮度通道: {channel}
像素总数: {stats.pixel_count:,}"""
            
            # 统计信息字典
            stats_dict = {
                "最小PQ值": f"{stats.min_pq:.6f}",
                "最大PQ值": f"{stats.max_pq:.6f}",
                "平均PQ值": f"{stats.avg_pq:.6f}",
                "方差": f"{stats.var_pq:.6f}",
                "动态范围": f"{stats.max_pq - stats.min_pq:.6f}"
            }

            # 转换图像用于显示（PQ域转sRGB显示域）
            display_image = self.image_processor.convert_for_display(pq_image)

            return info_text, stats_dict, display_image
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"图像处理失败: {str(e)}", {}, None
            
    def load_hdr_image(self, file_path: str) -> Tuple[np.ndarray, str]:
        """加载HDR图像文件"""
        
        if file_path is None:
            return None, "未选择文件"
            
        try:
            import cv2
            from pathlib import Path
            
            file_path = Path(file_path)
            
            if not file_path.exists():
                return None, f"文件不存在: {file_path}"
            
            # 根据文件扩展名选择加载方式
            ext = file_path.suffix.lower()
            
            if ext in ['.hdr', '.pic']:
                # 加载HDR文件
                image = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                if image is not None:
                    # OpenCV加载的是BGR格式，转换为RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    info = f"HDR文件加载成功: {file_path.name}"
                else:
                    return None, f"无法加载HDR文件: {file_path.name}"
                    
            elif ext in ['.exr']:
                # 加载EXR文件
                try:
                    image = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        info = f"EXR文件加载成功: {file_path.name}"
                    else:
                        return None, f"无法加载EXR文件: {file_path.name} (可能需要OpenEXR支持)"
                except Exception as e:
                    return None, f"EXR加载错误: {str(e)}"
                    
            elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                # 加载常规图像文件
                image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # 转换为浮点数并归一化
                    image = image.astype(np.float32) / 255.0
                    info = f"图像文件加载成功: {file_path.name}"
                else:
                    return None, f"无法加载图像文件: {file_path.name}"
            else:
                return None, f"不支持的文件格式: {ext}"
            
            # 确保图像是浮点数格式
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            
            # 检查图像尺寸
            if image.size > 10_000_000:  # 10M像素限制
                return None, f"图像过大: {image.shape}, 请使用较小的图像"
            
            return image, info
            
        except Exception as e:
            return None, f"文件加载失败: {str(e)}"

    def export_data(self, p: float, a: float, enable_spline: bool,
                   th1: float, th2: float, th3: float, th_strength: float,
                   dt_low: float, dt_high: float, channel: str,
                   window_size: int, lambda_smooth: float,
                   export_format: str = "json"):
        """导出数据 - 支持多种格式"""

        try:
            # 生成曲线数据
            L = np.linspace(0, 1, 1024)
            L_out = self.phoenix_calc.compute_phoenix_curve(L, p, a)

            # 基本导出数据
            base_data = {
                "parameters": {
                    "p": p,
                    "a": a,
                    "enable_spline": enable_spline,
                    "th_nodes": [th1, th2, th3],
                    "th_strength": th_strength,
                    "dt_low": dt_low,
                    "dt_high": dt_high,
                    "luminance_channel": channel,
                    "window_size": window_size,
                    "lambda_smooth": lambda_smooth
                },
                "metadata": {
                    "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "version": "1.0",
                    "tool": "HDR色调映射专利可视化工具"
                }
            }

            # 使用ExportManager进行高级导出
            from core import get_export_manager
            export_manager = get_export_manager()

            timestamp = int(time.time())
            base_filename = f"hdr_tone_mapping_{timestamp}"

            if export_format == "lut":
                # 导出LUT文件 (.cube)
                try:
                    lut_filename = export_manager.export_lut(
                        p=p, a=a,
                        filename=f"{base_filename}.cube"
                    )
                    return lut_filename
                except Exception as e:
                    return f"LUT导出失败: {str(e)}"

            elif export_format == "csv":
                # 导出CSV文件
                try:
                    csv_data = {
                        "input_luminance": L.tolist(),
                        "output_luminance": L_out.tolist(),
                        **base_data
                    }
                    csv_filename = export_manager.export_csv(
                        data=csv_data,
                        filename=f"{base_filename}.csv"
                    )
                    return csv_filename
                except Exception as e:
                    return f"CSV导出失败: {str(e)}"

            elif export_format == "diagnostic":
                # 导出诊断包
                try:
                    # 包含当前图像信息（如果有）
                    image_data = None
                    if self.ui_state.current_image_path is not None:
                        image_data = {
                            "image_path": self.ui_state.current_image_path,
                            "image_stats": self.ui_state.current_image_stats.__dict__ if self.ui_state.current_image_stats else None
                        }

                    diagnostic_data = {
                        **base_data,
                        "curve_data": {
                            "input_luminance": L.tolist(),
                            "output_luminance": L_out.tolist()
                        },
                        "image_data": image_data,
                        "quality_metrics": {
                            "distortion": 0.0,  # 可以从当前UI状态获取
                            "contrast": 0.0,   # 可以从当前UI状态获取
                            "recommendation": "",
                            # 集成扩展质量指标
                            "extended_metrics": getattr(self.ui_state, 'current_quality_metrics', {})
                        }
                    }

                    diagnostic_filename = export_manager.export_diagnostic_package(
                        data=diagnostic_data,
                        filename=f"{base_filename}_diagnostic.zip"
                    )
                    return diagnostic_filename
                except Exception as e:
                    return f"诊断包导出失败: {str(e)}"

            else:
                # 默认JSON导出
                export_data = {
                    **base_data,
                    "curve_data": {
                        "input_luminance": L.tolist(),
                        "output_luminance": L_out.tolist()
                    },
                    "quality_metrics": {
                        # 集成扩展质量指标到JSON输出
                        "extended_metrics": getattr(self.ui_state, 'current_quality_metrics', {})
                    }
                }

                filename = f"{base_filename}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)

                return filename

        except Exception as e:
            return f"导出失败: {str(e)}"

    def _get_custom_css(self) -> str:
        """获取自定义CSS样式"""
        
        return """
        .gradio-container {
            max-width: 1400px !important;
        }
        
        .gr-group {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        
        .gr-form {
            background: #f8f9fa;
        }
        
        .gr-button {
            margin: 5px;
        }
        
        .gr-plot {
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        """
        
    def _create_histogram_comparison(self, L_in: np.ndarray, L_out: np.ndarray,
                                    title: str = "PQ直方图对比") -> plt.Figure:
        """
        创建PQ直方图对比视图

        Args:
            L_in: 输入亮度数组 (PQ域)
            L_out: 输出亮度数组 (PQ域)
            title: 图表标题

        Returns:
            matplotlib Figure对象
        """
        try:
            # 计算直方图
            hist_in, bin_edges = self.quality_calc.compute_histogram(L_in, bins=256)
            hist_out, _ = self.quality_calc.compute_histogram(L_out, bins=256)

            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # 原始直方图
            ax1.bar(bin_edges[:-1], hist_in, width=bin_edges[1]-bin_edges[0],
                   alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
            ax1.set_title('原始图像PQ直方图')
            ax1.set_xlabel('PQ值')
            ax1.set_ylabel('像素数量')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 1)

            # 处理后直方图
            ax2.bar(bin_edges[:-1], hist_out, width=bin_edges[1]-bin_edges[0],
                   alpha=0.7, color='red', edgecolor='black', linewidth=0.5)
            ax2.set_title('处理后图像PQ直方图')
            ax2.set_xlabel('PQ值')
            ax2.set_ylabel('像素数量')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 1)

            plt.suptitle(title)
            plt.tight_layout()

            return fig

        except Exception as e:
            # 错误处理
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f'直方图生成失败: {str(e)}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('直方图错误')
            return fig

    def _setup_state_listeners(self):
        """设置状态管理监听器"""
        
        def state_change_listener(state_type: str, changes: dict):
            """状态变化监听器"""
            if state_type == "session":
                print(f"会话状态更新: {changes}")
            elif state_type == "temporal":
                print(f"时域状态更新: 帧={changes.get('frame', 0)}")
                
        self.state_manager.add_state_change_listener(state_change_listener)
        
    def _setup_performance_monitoring(self):
        """设置性能监控"""
        
        # 初始化加速状态检测
        try:
            acceleration_status = self.performance_monitor.get_acceleration_summary()
            print(f"加速状态: {acceleration_status}")
        except Exception as e:
            print(f"加速状态检测失败: {e}")
            
        # 设置定期性能检查（如果需要的话）
        # 注意：在Gradio中，我们通常不使用定时器，而是在用户交互时更新
        
    def _update_session_state_from_ui(self, **kwargs):
        """从UI更新会话状态"""
        return self.state_manager.update_session_state(**kwargs)
        
    def _update_temporal_state_from_ui(self, p: float, a: float, distortion: float):
        """从UI更新时域状态"""
        # 计算图像哈希
        image_hash = ""
        if self.ui_state.current_image_path is not None:
            # 使用文件路径和修改时间作为哈希基础
            import os
            try:
                stat_info = os.stat(self.ui_state.current_image_path)
                image_hash = f"{self.ui_state.current_image_path}_{stat_info.st_mtime}_{stat_info.st_size}"
            except OSError:
                image_hash = self.ui_state.current_image_path or ""
            
        return self.state_manager.update_temporal_state(
            p=p, a=a, distortion=distortion,
            mode=self.ui_state.current_mode,
            channel=self.state_manager.get_session_state().luminance_channel,
            image_hash=image_hash
        )
        
    def _get_state_summary_for_ui(self) -> dict:
        """获取用于UI显示的状态摘要"""
        summary = self.state_manager.get_state_summary()
        temporal_state = self.state_manager.get_temporal_state()
        
        return {
            "temporal_frames": temporal_state.current_frame,
            "temporal_variance_reduction": temporal_state.variance_reduction,
            "temporal_smoothing_active": temporal_state.smoothing_active,
            "session_auto_save": summary["session"]["auto_save"]
        }
        
    def save_state(self) -> str:
        """保存当前状态"""
        try:
            success = self.state_manager.save_all_states()
            if success:
                return "✓ 状态保存成功"
            else:
                return "✗ 状态保存失败"
        except Exception as e:
            return f"✗ 状态保存错误: {str(e)}"
            
    def load_state(self) -> Tuple:
        """加载保存的状态"""
        try:
            # 重新加载状态
            session_state = self.state_manager.load_session_state()
            self.state_manager._session_state = session_state
            
            return (
                session_state.p,                    # p_slider
                session_state.a,                    # a_slider
                session_state.enable_spline,        # enable_spline
                session_state.th1,                  # th1_slider
                session_state.th2,                  # th2_slider
                session_state.th3,                  # th3_slider
                session_state.th_strength,          # th_strength_slider
                session_state.dt_low,               # dt_low_slider
                session_state.dt_high,              # dt_high_slider
                session_state.luminance_channel,    # channel_radio
                session_state.window_size,          # window_slider
                session_state.lambda_smooth         # lambda_slider
            )
            
        except Exception as e:
            print(f"加载状态失败: {e}")
            self.ui_error_handler.create_system_error("state_load_failed", reason=str(e))
            # 返回当前状态作为回退
            session_state = self.state_manager.get_session_state()
            return (
                session_state.p, session_state.a, session_state.enable_spline,
                session_state.th1, session_state.th2, session_state.th3, session_state.th_strength,
                session_state.dt_low, session_state.dt_high, session_state.luminance_channel,
                session_state.window_size, session_state.lambda_smooth
            )
            
    def update_system_status(self) -> Tuple[str, int, str, str]:
        """更新系统状态显示"""
        try:
            # 获取系统状态
            status = self.safe_calc.get_comprehensive_system_status()
            error_summary = self.ui_error_handler.get_error_summary()
            
            # 系统状态
            if not status['system_stable']:
                system_status = "❌ 不稳定"
            elif error_summary['warning_count'] > 0:
                system_status = "⚠️ 有警告"
            else:
                system_status = "✅ 正常"
                
            # 错误计数
            error_count = error_summary['total_errors']
            
            # 自动恢复状态
            if status['auto_recovery_enabled']:
                recovery_attempts = status['error_recovery']['recovery_attempts']
                max_attempts = status['error_recovery']['max_recovery_attempts']
                auto_recovery_status = f"✅ 启用 ({recovery_attempts}/{max_attempts})"
            else:
                auto_recovery_status = "❌ 禁用"
                
            # 最近错误
            recent_errors = self.ui_error_handler.get_recent_errors(1)
            if recent_errors:
                last_error = recent_errors[-1].message[:100] + "..." if len(recent_errors[-1].message) > 100 else recent_errors[-1].message
            else:
                last_error = "无"
                
            return system_status, error_count, auto_recovery_status, last_error
            
        except Exception as e:
            return f"❌ 状态更新失败: {str(e)}", 0, "未知", str(e)
            
    def reset_error_system(self) -> Tuple[str, int, str, str]:
        """重置错误处理系统"""
        try:
            self.safe_calc.reset_error_handling_system()
            return self.update_system_status()
        except Exception as e:
            error_msg = f"重置失败: {str(e)}"
            return f"❌ {error_msg}", 0, "未知", error_msg
            
    def generate_system_diagnostic(self) -> str:
        """生成系统诊断报告"""
        try:
            # 获取全面的诊断报告
            diagnostic_report = self.safe_calc.create_system_diagnostic_report()
            
            # 添加错误历史
            recent_errors = self.ui_error_handler.get_recent_errors(10)
            if recent_errors:
                diagnostic_report += "\n## 最近错误历史\n"
                for i, error in enumerate(recent_errors[-5:], 1):  # 只显示最近5个
                    diagnostic_report += f"{i}. {self.ui_error_handler.format_error_for_display(error)}\n"
                    
            # 添加边界检查报告
            test_params = {
                'p': 2.0, 'a': 0.5, 'dt_low': 0.05, 'dt_high': 0.10,
                'window_size': 9, 'lambda_smooth': 0.3
            }
            is_valid, violations = self.boundary_checker.check_all_boundaries(test_params)
            boundary_report = self.boundary_checker.create_violation_report(violations)
            diagnostic_report += f"\n## 边界条件检查\n{boundary_report}\n"
            
            return diagnostic_report
            
        except Exception as e:
            return f"诊断报告生成失败: {str(e)}"
            
    def process_image_with_progress(self, p: float, a: float, enable_spline: bool,
                                  th1: float, th2: float, th3: float, th_strength: float,
                                  image_file: str, channel: str) -> Tuple[np.ndarray, Dict, float, float, str, str, object, str, str]:
        """带进度指示的图像处理"""
        
        if image_file is None:
            return None, {}, 0.0, 0.0, "未上传图像", "无图像", None, "等待处理...", "暂无建议"
        
        # 加载HDR图像
        image, load_info = self.load_hdr_image(image_file)
        if image is None:
            return None, {}, 0.0, 0.0, load_info, "加载失败", None, "加载失败", "暂无建议"
            
        try:
            # 创建色调映射函数
            def tone_curve_func(L):
                return self.phoenix_calc.compute_phoenix_curve(L, p, a)
                
            # 使用进度处理器处理图像
            result = self.progress_handler.process_image_with_progress(
                image=image,
                tone_curve_func=tone_curve_func,
                luminance_channel=channel
            )
            
            if not result['success']:
                return None, {}, 0.0, 0.0, f"处理失败: {result['error']}", "处理失败", None, "处理失败", "暂无建议"
                
            # 转换为显示格式
            display_image = result['display_image']
            
            # 构建统计信息 - 集成质量评估指标
            stats_after = result['stats_after']
            
            # 如果有质量指标，显示格式化的质量数据
            if quality_metrics:
                stats_dict = {
                    "高光饱和比例": f"{quality_metrics.get('S_ratio', 0) * 100:.1f}%",
                    "暗部压缩比例": f"{quality_metrics.get('C_shadow', 0) * 100:.1f}%", 
                    "动态范围保持率": f"{quality_metrics.get('R_DR', 1.0):.2f}",
                    "亮度漂移": f"{quality_metrics.get('ΔL_mean_norm', 0) * 100:.1f}%",
                    "直方图重叠度": f"{quality_metrics.get('Hist_overlap', 0) * 100:.1f}%"
                }
            else:
                # 回退到原有显示格式
                stats_dict = {
                    "最小PQ值": f"{stats_after.min_pq:.6f}",
                    "最大PQ值": f"{stats_after.max_pq:.6f}",
                    "平均PQ值": f"{stats_after.avg_pq:.6f}",
                    "方差": f"{stats_after.var_pq:.6f}",
                    "动态范围": f"{stats_after.max_pq - stats_after.min_pq:.6f}"
                }
            
            # 获取质量指标（从新的质量评估模块）
            quality_metrics = result.get('quality_metrics', {})
            
            # 保持向后兼容性，计算原有的质量指标
            stats_before = result['stats_before']
            L_in = np.full(1000, stats_before.avg_pq)  # 简化的输入
            L_out = np.full(1000, stats_after.avg_pq)  # 简化的输出
            
            distortion = self.quality_calc.compute_perceptual_distortion(L_in, L_out)
            contrast = self.quality_calc.compute_local_contrast(L_out)
            recommendation = self.quality_calc.recommend_mode_with_hysteresis(distortion)
            
            # 存储质量指标供导出使用
            self.ui_state.current_quality_metrics = quality_metrics
            
            # 构建性能状态信息
            processing_info = result['processing_info']
            performance_status = f"处理时间: {processing_info['processing_time_ms']:.1f}ms"
            
            if processing_info.get('downsampled', False):
                performance_status += f" | 已降采样: {processing_info['downsample_reason']}"
            
            # 生成质量评估UI内容
            pq_histogram_plot = None
            quality_status_html = "等待处理..."
            artist_tips_html = "暂无建议"
            
            if quality_metrics:
                # 获取Lin和Lout数据用于直方图
                lin_lout_data = result.get('lin_lout_data')
                
                if lin_lout_data is not None:
                    lin_data = lin_lout_data.get('lin')
                    lout_data = lin_lout_data.get('lout')
                    
                    if lin_data is not None and lout_data is not None:
                        # 创建PQ直方图
                        pq_histogram_plot = self.ui_integration.update_pq_histogram(lin_data, lout_data)
                
                # 创建质量状态显示
                status = quality_metrics.get('Exposure_status', '未知')
                quality_status_html = self.ui_integration.create_quality_status_display(quality_metrics, status)
                
                # 创建艺术家提示
                artist_tips_html = self.ui_integration.create_artist_tips_display(quality_metrics, status)
                
            return display_image, stats_dict, distortion, contrast, recommendation, performance_status, pq_histogram_plot, quality_status_html, artist_tips_html
            
        except Exception as e:
            error_msg = f"图像处理失败: {str(e)}"
            return None, {}, 0.0, 0.0, error_msg, error_msg, None, error_msg, "暂无建议"
            
    def update_performance_display(self) -> Tuple[str, str]:
        """更新性能显示"""
        try:
            # 获取性能摘要
            perf_summary = self.performance_monitor.get_performance_summary()
            performance_status = (
                f"操作数: {perf_summary['total_operations']} | "
                f"平均时间: {perf_summary['average_duration_ms']:.1f}ms | "
                f"成功率: {perf_summary['success_rate']:.1f}%"
            )
            
            # 获取加速状态
            acceleration_status = self.performance_monitor.get_acceleration_summary()
            
            return performance_status, acceleration_status
            
        except Exception as e:
            error_msg = f"性能状态更新失败: {str(e)}"
            return error_msg, error_msg
    
    def _verify_quality_assessment_integration(self):
        """验证质量评估模块集成状态"""
        try:
            if self.quality_assessment_enabled:
                # 测试基本功能
                test_lin = np.linspace(0.1, 0.9, 100)
                test_lout = test_lin * 0.8  # 简单的测试映射
                
                # 测试指标计算
                metrics = self.extended_metrics.get_all_metrics(test_lin, test_lout)
                
                # 测试UI集成
                status = metrics.get('Exposure_status', '未知')
                ui_summary = self.ui_integration.update_quality_summary(metrics, status)
                
                logging.info("✅ 质量评估模块集成验证成功")
                logging.info(f"   - 计算指标数量: {len(metrics)}")
                logging.info(f"   - 状态评估: {status}")
                logging.info(f"   - UI组件: {len(ui_summary)} 项")
                
            else:
                logging.warning("⚠️ 质量评估模块未启用")
                
        except Exception as e:
            logging.error(f"❌ 质量评估模块集成验证失败: {e}")
            self.quality_assessment_enabled = False
            
    def reset_performance_metrics(self) -> Tuple[str, str]:
        """重置性能指标"""
        try:
            self.performance_monitor.reset_metrics()
            return self.update_performance_display()
        except Exception as e:
            error_msg = f"性能重置失败: {str(e)}"
            return error_msg, error_msg


def create_app() -> gr.Blocks:
    """创建Gradio应用"""
    
    interface = GradioInterface()
    return interface.create_interface()


def main():
    """主函数"""
    
    print("启动HDR色调映射专利可视化工具...")
    
    try:
        app = create_app()
        app.queue()  # 启用队列以支持并发处理
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            debug=False,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"启动失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
