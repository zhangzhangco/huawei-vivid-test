---
title: HDR色调映射专利可视化工具
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# HDR色调映射专利可视化工具

基于Phoenix曲线算法的HDR色调映射专利技术可视化系统，提供实时参数调节、质量评估和图像处理功能。

## 🚀 功能特性

- **实时Phoenix曲线可视化** - 参数调节即时生效
- **智能参数控制** - 自动模式和艺术模式切换
- **图像处理功能** - 支持HDR图像上传和处理
- **质量指标评估** - 实时感知失真和对比度计算
- **时域平滑处理** - 视频序列时间一致性优化
- **样条曲线扩展** - 局部精细化调节
- **GPU加速支持** - CUDA并行计算优化

## 🎯 核心技术

### Phoenix曲线算法
- 基于感知量化器(PQ)的非线性变换
- 参数p控制曲线形状和对比度
- 参数a控制亮度映射范围
- 保持色调映射的单调性和连续性

### 质量指标系统
- **感知失真 (D')**: 基于Weber-Fechner定律的视觉质量评估
- **局部对比度**: 图像细节保持程度评估
- **智能模式推荐**: 基于阈值的自动模式建议

### 性能优化
- **GPU加速**: Numba CUDA并行计算
- **内存优化**: 智能内存管理和缓存策略
- **实时处理**: 2K图像<100ms处理速度

## 📊 界面功能

### 参数控制面板
- 工作模式选择（自动/艺术模式）
- Phoenix曲线参数调节
- 质量指标参数设置
- 时域平滑参数控制
- 样条曲线扩展选项

### 可视化显示
- 实时曲线图表更新
- 质量指标数值显示
- 系统状态监控
- 性能指标展示

### 图像处理
- 多格式HDR图像支持
- 实时处理结果对比
- 详细统计信息显示
- 多格式数据导出

## 🛠️ 技术规格

### 支持格式
- **输入**: EXR, HDR, TIFF, PNG, JPEG
- **输出**: PNG, JPEG, TIFF, EXR
- **色彩空间**: sRGB, Rec.2020, DCI-P3
- **位深度**: 8位, 16位, 32位浮点

### 参数范围
- **亮度控制因子 p**: 0.1 - 6.0
- **缩放因子 a**: 0.0 - 1.0
- **失真阈值**: 0.01 - 0.20
- **时域窗口**: 5 - 15帧

### 性能指标
- **最大图像**: 8K (7680×4320)
- **实时处理**: 2K图像 < 100ms
- **内存占用**: < 4GB (典型场景)
- **数值精度**: 32位浮点

## � 使用方法

1. **参数调节**: 使用滑块调节Phoenix曲线参数
2. **模式切换**: 选择自动模式或艺术模式
3. **图像上传**: 拖拽HDR图像到上传区域
4. **实时预览**: 观察曲线变化和质量指标
5. **结果导出**: 保存处理结果和参数配置

## 📚 技术文档

- [完整技术说明书](HDR色调映射专利可视化技术说明书.md)
- [部署指南](DEPLOYMENT_GUIDE.md)
- [用户手册](docs/USER_MANUAL.md)
- [API文档](docs/API_DOCUMENTATION.md)
- [开发文档](docs/DEVELOPER_DOCUMENTATION.md)

## 🔧 本地运行

```bash
# 克隆仓库
git clone https://huggingface.co/spaces/zhangzhangco/HuaweiHDR
cd HuaweiHDR

# 安装依赖
pip install -r requirements.txt

# 启动应用
python launch_gradio.py
```

## 🚀 GPU加速

系统支持NVIDIA GPU加速，可显著提升处理性能：

```bash
# 安装GPU支持
pip install cupy-cuda12x numba

# 检查GPU状态
python check_gpu_support.py
```

## 📈 系统架构

```
HDR色调映射可视化系统
├── 用户界面层 (Gradio Frontend)
├── 业务逻辑层 (Core Logic)
├── 数据处理层 (Data Processing)
└── 基础设施层 (Infrastructure)
```

## 🎯 应用场景

- **HDR图像处理**: 专业HDR内容创作
- **算法研究**: 色调映射算法开发和验证
- **教学演示**: 图像处理课程教学工具
- **质量评估**: HDR显示设备校准和测试
- **批量处理**: 大规模HDR内容处理

## 📄 许可证

本项目采用 MIT 许可证。Phoenix曲线算法受专利保护，商业使用需要相应授权。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📞 联系方式

- **技术支持**: 通过GitHub Issues
- **功能建议**: 欢迎在Discussions中讨论
- **商业合作**: 请通过私信联系

---

*基于Phoenix曲线算法的专业HDR色调映射可视化工具 - 让HDR图像处理更直观、更高效！*api.star-history.com/svg?repos=example/hdr-tone-mapping-gradio&type=Date)](https://star-history.com/#example/hdr-tone-mapping-gradio&Date)

---

**版本**: 1.0.0  
**更新日期**: 2025-10-27  
**开发团队**: HDR Tone Mapping Team