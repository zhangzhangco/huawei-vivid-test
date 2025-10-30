"""
UI集成模块 - 处理HDR质量评估扩展模块的界面更新和显示
实现质量摘要区显示、PQ直方图可视化修复和艺术家模式语义提示功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import logging
from typing import Dict, Tuple, Optional, Any, Union
import json

from .metrics_extension import ExtendedMetrics
from .config_manager import ConfigManager


class UIIntegration:
    """
    UI集成类，负责处理界面更新和显示功能
    分离界面逻辑，确保与Gradio组件的松耦合
    """
    
    def __init__(self, config_path: str = "config/metrics.json"):
        """
        初始化UI集成模块
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # 初始化依赖组件
        self.extended_metrics = ExtendedMetrics(config_path)
        self.config_manager = ConfigManager(config_path)
        
        # 配置matplotlib中文字体支持
        self._configure_matplotlib_fonts()
        
    def _configure_matplotlib_fonts(self):
        """配置matplotlib中文字体支持"""
        try:
            # 设置中文字体
            rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
            rcParams['axes.unicode_minus'] = False
        except Exception as e:
            self.logger.warning(f"配置中文字体失败: {e}")
    
    def update_quality_summary(self, metrics: Dict[str, Union[float, str]], status: str) -> Dict[str, str]:
        """
        更新质量摘要显示
        实现质量摘要区显示功能，包含百分比格式化
        
        Args:
            metrics: 质量指标字典
            status: 质量状态字符串
            
        Returns:
            格式化后的质量摘要显示字典
        """
        try:
            # 获取状态显示信息
            status_info = self.extended_metrics.get_status_display_info(status)
            
            # 格式化百分比显示
            s_ratio_percent = self.format_percentage_display(metrics.get('S_ratio', 0.0))
            c_shadow_percent = self.format_percentage_display(metrics.get('C_shadow', 0.0))
            delta_l_percent = self.format_percentage_display(metrics.get('ΔL_mean_norm', 1.0))
            
            # 格式化动态范围保持率（显示为小数）
            r_dr_display = f"{metrics.get('R_DR', 1.0):.2f}"
            
            # 构建质量摘要显示字典
            quality_summary = {
                "quality_status": f"{status_info['emoji']} {status_info['text']}",
                "highlight_saturation": f"高光饱和: {s_ratio_percent}",
                "shadow_compression": f"暗部压缩: {c_shadow_percent}",
                "dynamic_range_retention": f"动态范围保持: {r_dr_display}",
                "luminance_drift": f"亮度漂移: {delta_l_percent}",
                "status_description": status_info['description']
            }
            
            return quality_summary
            
        except Exception as e:
            self.logger.error(f"更新质量摘要显示时发生错误: {e}")
            return {
                "quality_status": "❓ 未知",
                "highlight_saturation": "高光饱和: --",
                "shadow_compression": "暗部压缩: --", 
                "dynamic_range_retention": "动态范围保持: --",
                "luminance_drift": "亮度漂移: --",
                "status_description": f"显示更新失败: {str(e)}"
            }
    
    def format_percentage_display(self, value: float) -> str:
        """
        格式化百分比显示 (0.078 -> 7.8%)
        
        Args:
            value: 原始数值 (0-1范围)
            
        Returns:
            格式化的百分比字符串
        """
        try:
            if isinstance(value, (int, float)) and not np.isnan(value):
                percentage = value * 100
                return f"{percentage:.1f}%"
            else:
                return "N/A"
        except Exception:
            return "N/A"
    
    def update_pq_histogram(self, lin: np.ndarray, lout: np.ndarray) -> plt.Figure:
        """
        更新PQ直方图显示
        修复PQ直方图显示，使用Lin和Lout数据重新绘制
        实现双曲线对比显示(Input/Output)，包含图例说明
        使用256个bins和(0,1)范围进行归一化处理
        
        Args:
            lin: 输入亮度数据（PQ域，范围0-1）
            lout: 输出亮度数据（PQ域，映射后，范围0-1）
            
        Returns:
            matplotlib Figure对象
        """
        try:
            # 验证输入数据
            if lin is None or lout is None:
                return self._create_error_histogram("输入数据为空")
            
            lin_array = np.asarray(lin, dtype=np.float32)
            lout_array = np.asarray(lout, dtype=np.float32)
            
            if lin_array.size == 0 or lout_array.size == 0:
                return self._create_error_histogram("输入数组为空")
            
            # 展平数组
            lin_flat = lin_array.flatten()
            lout_flat = lout_array.flatten()
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 使用256个bins和(0,1)范围进行归一化处理
            bins = 256
            range_pq = (0.0, 1.0)
            
            # 计算归一化直方图（使用density参数）
            hist_in, bin_edges = np.histogram(lin_flat, bins=bins, range=range_pq, density=True)
            hist_out, _ = np.histogram(lout_flat, bins=bins, range=range_pq, density=True)
            
            # 计算bin中心点用于绘制
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # 绘制输入和输出两条曲线
            ax.plot(bin_centers, hist_in, 'b-', linewidth=2, label='Input', alpha=0.8)
            ax.plot(bin_centers, hist_out, 'r-', linewidth=2, label='Output', alpha=0.8)
            
            # 设置图表属性
            ax.set_xlabel('PQ值', fontsize=12)
            ax.set_ylabel('密度', fontsize=12)
            ax.set_title('原始/处理后PQ直方图对比', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_xlim(0, 1)
            
            # 设置y轴范围，避免过大的峰值影响显示
            y_max = max(np.max(hist_in), np.max(hist_out))
            if y_max > 0:
                ax.set_ylim(0, y_max * 1.1)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.error(f"更新PQ直方图时发生错误: {e}")
            return self._create_error_histogram(f"直方图生成失败: {str(e)}")
    
    def _create_error_histogram(self, error_msg: str) -> plt.Figure:
        """
        创建错误显示的直方图
        
        Args:
            error_msg: 错误信息
            
        Returns:
            显示错误信息的matplotlib Figure对象
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, error_msg, 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            ax.set_xlabel('PQ值', fontsize=12)
            ax.set_ylabel('密度', fontsize=12)
            ax.set_title('PQ直方图对比 - 错误', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
        except Exception:
            # 最后的备用方案
            fig = plt.figure(figsize=(10, 6))
            return fig
    
    def generate_artist_tips(self, metrics: Dict[str, Union[float, str]], status: str) -> str:
        """
        生成艺术家模式语义化建议
        实现generate_artist_tips方法生成中文语义化建议
        创建基于D'≈0.10～0.20目标区间的参数调整指导
        开发针对过曝、过暗问题的具体参数建议(p、a参数调整)
        
        Args:
            metrics: 质量指标字典
            status: 质量状态字符串
            
        Returns:
            中文语义化的调整建议文本
        """
        try:
            tips = []
            
            # 获取关键指标
            s_ratio = metrics.get('S_ratio', 0.0)
            c_shadow = metrics.get('C_shadow', 0.0)
            r_dr = metrics.get('R_DR', 1.0)
            delta_l_mean_norm = metrics.get('ΔL_mean_norm', 1.0)
            
            # 计算简化的D'指标（基于亮度漂移）
            dprime = abs(delta_l_mean_norm - 1.0)
            
            # 根据状态提供具体的参数调整建议
            if status == "过曝":
                tips.append("🔴 检测到过曝问题")
                tips.append(f"当前高光饱和: {s_ratio*100:.1f}% (建议<5%)")
                tips.append(f"当前D'指标: {dprime:.3f} (目标区间: 0.10～0.20)")
                tips.append("")
                tips.append("📝 参数调整建议:")
                tips.append("• 减小p参数 (建议范围: 0.8-1.5) - 降低对比度增强")
                tips.append("• 或增大a参数 (建议范围: 0.6-0.8) - 增加整体压缩")
                tips.append("• 优先调整p参数，效果更直接")
                
            elif status == "过暗":
                tips.append("🟣 检测到过暗问题")
                tips.append(f"当前暗部压缩: {c_shadow*100:.1f}% (建议<10%)")
                tips.append(f"当前D'指标: {dprime:.3f} (目标区间: 0.10～0.20)")
                tips.append("")
                tips.append("📝 参数调整建议:")
                tips.append("• 增大p参数 (建议范围: 2.5-4.0) - 增强暗部细节")
                tips.append("• 或减小a参数 (建议范围: 0.2-0.4) - 减少整体压缩")
                tips.append("• 建议先微调p参数，观察暗部细节变化")
                
            elif status == "动态范围异常":
                tips.append("⚪ 检测到动态范围异常")
                tips.append(f"当前动态范围保持率: {r_dr:.2f} (理想值接近1.0)")
                tips.append(f"当前D'指标: {dprime:.3f} (目标区间: 0.10～0.20)")
                tips.append("")
                tips.append("📝 参数调整建议:")
                if r_dr > 1.2:
                    tips.append("• 动态范围过度扩展，建议增大a参数")
                    tips.append("• 或适当减小p参数，避免过度增强")
                elif r_dr < 0.8:
                    tips.append("• 动态范围压缩过度，建议减小a参数")
                    tips.append("• 或适当增大p参数，保持更多细节")
                else:
                    tips.append("• 动态范围轻微异常，微调参数即可")
                
            elif status == "正常":
                tips.append("🟢 图像质量良好")
                tips.append(f"当前D'指标: {dprime:.3f} (目标区间: 0.10～0.20)")
                tips.append("")
                tips.append("✨ 优化建议:")
                if dprime < 0.10:
                    tips.append("• D'指标偏低，可适当增强对比度")
                    tips.append("• 建议微调p参数 (+0.1～+0.3)")
                elif dprime > 0.20:
                    tips.append("• D'指标偏高，建议适当降低增强强度")
                    tips.append("• 建议微调p参数 (-0.1～-0.3)")
                else:
                    tips.append("• 各项指标均在理想范围内")
                    tips.append("• 可根据艺术需求进行微调")
                    
            else:
                tips.append("❓ 状态未知或评估失败")
                tips.append("建议检查输入数据或重新处理")
            
            # 添加通用提示
            tips.append("")
            tips.append("💡 调参小贴士:")
            tips.append("• p参数主要影响对比度和细节增强")
            tips.append("• a参数主要影响整体亮度映射范围")
            tips.append("• 建议每次调整幅度不超过0.2，观察效果后再继续")
            tips.append("• 可结合直方图变化判断调整效果")
            
            return "\n".join(tips)
            
        except Exception as e:
            self.logger.error(f"生成艺术家提示时发生错误: {e}")
            return f"❌ 提示生成失败: {str(e)}\n\n请检查输入数据或联系技术支持。"
    
    def update_dom_element(self, element_id: str, content: str) -> Dict[str, str]:
        """
        更新DOM元素内容
        开发DOM元素更新方法(quality-status元素)
        
        Args:
            element_id: DOM元素ID
            content: 要更新的内容
            
        Returns:
            包含更新信息的字典
        """
        try:
            # 在实际的Gradio环境中，这里会更新对应的组件
            # 目前返回更新信息供调用者使用
            update_info = {
                "element_id": element_id,
                "content": content,
                "timestamp": str(np.datetime64('now')),
                "status": "success"
            }
            
            self.logger.info(f"DOM元素更新: {element_id}")
            return update_info
            
        except Exception as e:
            self.logger.error(f"更新DOM元素时发生错误: {e}")
            return {
                "element_id": element_id,
                "content": content,
                "timestamp": str(np.datetime64('now')),
                "status": "error",
                "error": str(e)
            }
    
    def create_quality_status_display(self, metrics: Dict[str, Union[float, str]], status: str) -> str:
        """
        创建质量状态显示内容（用于DOM元素id="quality-status"）
        
        Args:
            metrics: 质量指标字典
            status: 质量状态字符串
            
        Returns:
            格式化的质量状态显示文本
        """
        try:
            status_info = self.extended_metrics.get_status_display_info(status)
            quality_summary = self.update_quality_summary(metrics, status)
            
            # 构建HTML格式的状态显示
            status_html = f"""
            <div style="padding: 10px; border-radius: 5px; background-color: #f8f9fa;">
                <h4 style="margin: 0 0 10px 0; color: {status_info['color']};">
                    {quality_summary['quality_status']}
                </h4>
                <div style="font-size: 14px; line-height: 1.5;">
                    <p style="margin: 5px 0;">{quality_summary['highlight_saturation']}</p>
                    <p style="margin: 5px 0;">{quality_summary['shadow_compression']}</p>
                    <p style="margin: 5px 0;">{quality_summary['dynamic_range_retention']}</p>
                    <p style="margin: 5px 0;">{quality_summary['luminance_drift']}</p>
                </div>
                <div style="margin-top: 10px; font-size: 12px; color: #666;">
                    {quality_summary['status_description']}
                </div>
            </div>
            """
            
            return status_html
            
        except Exception as e:
            self.logger.error(f"创建质量状态显示时发生错误: {e}")
            return f"<div style='color: red;'>状态显示生成失败: {str(e)}</div>"
    
    def create_artist_tips_display(self, metrics: Dict[str, Union[float, str]], status: str) -> str:
        """
        创建艺术家提示显示内容（用于DOM元素id="artist-tips"）
        
        Args:
            metrics: 质量指标字典
            status: 质量状态字符串
            
        Returns:
            格式化的艺术家提示显示文本
        """
        try:
            tips_content = self.generate_artist_tips(metrics, status)
            
            # 将文本转换为HTML格式，保持换行和格式
            tips_html = tips_content.replace('\n', '<br>')
            
            # 添加样式
            styled_tips = f"""
            <div style="padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #fafafa; font-family: monospace;">
                <div style="font-size: 14px; line-height: 1.6;">
                    {tips_html}
                </div>
            </div>
            """
            
            return styled_tips
            
        except Exception as e:
            self.logger.error(f"创建艺术家提示显示时发生错误: {e}")
            return f"<div style='color: red;'>艺术家提示生成失败: {str(e)}</div>"
    
    def get_integration_status(self) -> Dict[str, Any]:
        """
        获取UI集成模块的状态信息
        
        Returns:
            包含模块状态的字典
        """
        try:
            return {
                "module": "UIIntegration",
                "version": "1.0.0",
                "config_path": self.config_path,
                "extended_metrics_available": self.extended_metrics is not None,
                "config_manager_available": self.config_manager is not None,
                "matplotlib_configured": True,
                "status": "ready"
            }
        except Exception as e:
            return {
                "module": "UIIntegration", 
                "status": "error",
                "error": str(e)
            }