# HDR色调映射专利可视化工具需求文档

## 简介

本文档定义了基于Gradio框架的HDR色调映射专利可视化工具的需求。该工具用于实现和验证基于CN115428006A、CN115361510A和CN114648447A专利的核心算法，提供交互式的Phoenix曲线展示、参数控制、质量指标反馈和时域平滑演示功能。

## 术语表

- **HDR_Visualization_System**: HDR色调映射专利可视化系统
- **Phoenix_Curve**: Phoenix色调映射曲线，基于公式 L' = L^p / (L^p + a^p)
- **PQ_Domain**: 感知量化域，亮度值范围0-1
- **Tone_Mapping_Function**: 色调映射函数，将HDR亮度映射到显示设备范围
- **Perceptual_Distortion**: 感知失真D'，衡量映射前后的亮度差异
- **Local_Contrast**: 局部对比度，衡量相邻像素间的亮度梯度
- **Temporal_Smoothing**: 时域平滑，用于减少多帧间参数变化的闪烁
- **Auto_Mode**: 自动模式，系统自动计算最优参数
- **Art_Mode**: 艺术模式，用户手动调节参数
- **Spline_Curve**: 样条曲线，多段三次样条函数用于局部映射优化

## 需求

### 需求1

**用户故事:** 作为HDR算法研究员，我希望能够实时可视化Phoenix曲线，以便理解不同参数对色调映射效果的影响

#### 验收标准

1. WHEN 用户调整参数p或a时，THE HDR_Visualization_System SHALL 实时更新Phoenix_Curve显示
2. THE HDR_Visualization_System SHALL 在同一图表中显示Phoenix_Curve和恒等线作为对比
3. THE HDR_Visualization_System SHALL 确保曲线在PQ_Domain范围内单调递增
4. THE HDR_Visualization_System SHALL 显示当前参数值在曲线图例中
5. THE HDR_Visualization_System SHALL 在500毫秒内完成曲线更新和重绘

### 需求2

**用户故事:** 作为算法工程师，我希望能够通过滑块控制Phoenix曲线参数，以便进行精确的参数调优

#### 验收标准

1. THE HDR_Visualization_System SHALL 提供参数p的滑块控制，范围为0.1到6.0
2. THE HDR_Visualization_System SHALL 提供参数a的滑块控制，范围为0.0到1.0
3. THE HDR_Visualization_System SHALL 设置参数p的默认值为2.0
4. THE HDR_Visualization_System SHALL 设置参数a的默认值为0.5
5. WHEN 用户拖动滑块时，THE HDR_Visualization_System SHALL 实时响应参数变化

### 需求3

**用户故事:** 作为研发人员，我希望系统能够计算和显示质量指标，以便评估色调映射的效果

#### 验收标准

1. THE HDR_Visualization_System SHALL 计算感知失真D'作为映射前后亮度均值的绝对差
2. THE HDR_Visualization_System SHALL 计算局部对比度作为相邻像素亮度梯度的平均值
3. THE HDR_Visualization_System SHALL 实时更新质量指标数值显示
4. THE HDR_Visualization_System SHALL 提供失真门限D_T的可调节范围0.05到0.10
5. WHEN D'小于等于D_T时，THE HDR_Visualization_System SHALL 推荐Auto_Mode

### 需求4

**用户故事:** 作为用户，我希望能够在自动模式和艺术模式之间切换，以便适应不同的使用场景

#### 验收标准

1. THE HDR_Visualization_System SHALL 提供Auto_Mode和Art_Mode的单选按钮
2. WHEN 选择Auto_Mode时，THE HDR_Visualization_System SHALL 自动计算最优参数p和a
3. WHEN 选择Art_Mode时，THE HDR_Visualization_System SHALL 允许用户手动调节所有参数
4. THE HDR_Visualization_System SHALL 根据当前D'值自动推荐合适的模式
5. THE HDR_Visualization_System SHALL 在模式切换时保持界面响应性

### 需求5

**用户故事:** 作为算法验证人员，我希望能够模拟时域平滑效果，以便验证多帧视频处理的稳定性

#### 验收标准

1. THE HDR_Visualization_System SHALL 提供时域窗口长度M的控制，范围为5到15帧
2. THE HDR_Visualization_System SHALL 提供平滑强度λ的控制，范围为0.2到0.5
3. THE HDR_Visualization_System SHALL 模拟多帧参数的加权平均计算
4. THE HDR_Visualization_System SHALL 显示平滑前后的参数对比
5. THE HDR_Visualization_System SHALL 设置M的默认值为9，λ的默认值为0.3

### 需求6

**用户故事:** 作为研究人员，我希望能够上传HDR图像并应用色调映射，以便验证算法在真实图像上的效果

#### 验收标准

1. THE HDR_Visualization_System SHALL 支持常见HDR图像格式的上传
2. THE HDR_Visualization_System SHALL 将上传的图像转换到PQ_Domain
3. THE HDR_Visualization_System SHALL 应用当前Phoenix_Curve参数到图像
4. THE HDR_Visualization_System SHALL 显示原始图像和映射后图像的对比
5. THE HDR_Visualization_System SHALL 在图像处理完成后更新质量指标

### 需求7

**用户故事:** 作为高级用户，我希望能够使用多段样条曲线功能，以便进行更精细的局部映射控制

#### 验收标准

1. WHERE 启用样条模式时，THE HDR_Visualization_System SHALL 提供TH1、TH2、TH3节点控制
2. THE HDR_Visualization_System SHALL 确保样条段在连接点处满足C¹连续性
3. THE HDR_Visualization_System SHALL 提供样条强度TH_strength的控制，范围为0到1
4. THE HDR_Visualization_System SHALL 在同一图表中显示Phoenix_Curve和Spline_Curve对比
5. THE HDR_Visualization_System SHALL 设置TH1、TH2、TH3的默认值分别为0.2、0.5、0.8

### 需求8

**用户故事:** 作为用户，我希望界面响应迅速且操作直观，以便高效地进行算法研究和验证

#### 验收标准

1. THE HDR_Visualization_System SHALL 在用户交互后300毫秒内更新所有相关显示（基于1MP像素以内图像）
2. THE HDR_Visualization_System SHALL 提供清晰的参数标签和数值显示，固定3位小数精度
3. THE HDR_Visualization_System SHALL 使用网格布局合理组织所有控件，提供参数重置功能
4. THE HDR_Visualization_System SHALL 在处理大图像时显示进度指示且不阻塞UI
5. THE HDR_Visualization_System SHALL 对非法输入值进行校验与友好提示

### 需求9

**用户故事:** 作为算法工程师，我希望系统具备数值稳定性和端点归一化，以便确保计算结果的可靠性

#### 验收标准

1. THE HDR_Visualization_System SHALL 接受MinDisplay_PQ与MaxDisplay_PQ参数，对Phoenix_Curve输出进行线性归一化
2. THE HDR_Visualization_System SHALL 使用ε=1e-6的安全夹取避免除零或NaN错误
3. THE HDR_Visualization_System SHALL 提供采样点数N的内部参数，默认512，范围256-2048
4. THE HDR_Visualization_System SHALL 确保在所有参数组合下曲线严格单调递增
5. THE HDR_Visualization_System SHALL 保证端点严格匹配：L=0→L'=MinDisplay_PQ，L=1→L'=MaxDisplay_PQ

### 需求10

**用户故事:** 作为研究人员，我希望模式推荐具备滞回特性，以便避免在临界值附近的抖动

#### 验收标准

1. THE HDR_Visualization_System SHALL 使用双阈值策略：D_T_low=0.05，D_T_high=0.10
2. WHEN D'≤D_T_low时，THE HDR_Visualization_System SHALL 推荐Auto_Mode
3. WHEN D'≥D_T_high时，THE HDR_Visualization_System SHALL 推荐Art_Mode
4. WHILE D'在[D_T_low, D_T_high]区间时，THE HDR_Visualization_System SHALL 保持上次决策
5. THE HDR_Visualization_System SHALL 在D'小幅摆动时避免模式建议跳变

### 需求11

**用户故事:** 作为算法验证人员，我希望时域平滑具备完整的权重计算和冷启动机制，以便准确模拟视频处理效果

#### 验收标准

1. THE HDR_Visualization_System SHALL 采用权重w_k = 1/(D[N-k]+ε)进行加权平均计算
2. WHEN 历史帧数不足M时，THE HDR_Visualization_System SHALL 以已有帧数计算平滑
3. WHEN 切换模式或载入新图像时，THE HDR_Visualization_System SHALL 清空时域缓冲
4. THE HDR_Visualization_System SHALL 展示Δp_raw与Δp_filtered的数值对比
5. THE HDR_Visualization_System SHALL 确保Δp_filtered的方差比Δp_raw降低50%以上

### 需求12

**用户故事:** 作为图像处理专家，我希望系统明确定义颜色空间转换和亮度通道，以便确保处理结果的准确性

#### 验收标准

1. THE HDR_Visualization_System SHALL 明确亮度统计通道采用MaxRGB(PQ)或Y(PQ)，默认MaxRGB
2. THE HDR_Visualization_System SHALL 实现ST 2084(PQ)的正反变换用于尼特值显示
3. THE HDR_Visualization_System SHALL 支持OpenEXR(float)与PNG 16-bit输入格式
4. THE HDR_Visualization_System SHALL 在映射前将线性光转换至PQ域
5. THE HDR_Visualization_System SHALL 提供PQ域直方图的映射前后对比视图

### 需求13

**用户故事:** 作为高级用户，我希望样条曲线实现具备严格的数学约束，以便获得平滑可靠的映射效果

#### 验收标准

1. THE HDR_Visualization_System SHALL 使用分段三次Hermite样条(PCHIP)实现
2. THE HDR_Visualization_System SHALL 确保样条段在连接处导数连续，误差≤1e-3
3. THE HDR_Visualization_System SHALL 采用凸组合公式：L'_final = (1-TH_strength)*L'_phoenix + TH_strength*L'_spline
4. THE HDR_Visualization_System SHALL 锚定样条两端y值到Phoenix_Curve
5. THE HDR_Visualization_System SHALL 避免样条曲线出现反齿或回卷现象

### 需求14

**用户故事:** 作为研发团队成员，我希望Auto模式具备确定性的参数估算，以便实现可复现的自动优化

#### 验收标准

1. THE HDR_Visualization_System SHALL 提供线性估参公式：p = p0 + α*(max_pq-avg_pq)
2. THE HDR_Visualization_System SHALL 提供线性估参公式：a = a0 + β*(avg_pq-min_pq)
3. THE HDR_Visualization_System SHALL 将p0,a0,α,β作为可配置的高级参数
4. THE HDR_Visualization_System SHALL 确保固定输入时估参结果稳定可复现
5. THE HDR_Visualization_System SHALL 在导出数据中包含最终p,a及估参超参数

### 需求15

**用户故事:** 作为工程师，我希望系统支持数据导出和会话管理，以便实现结果复现和流程对接

#### 验收标准

1. THE HDR_Visualization_System SHALL 导出1D LUT时明确位深与采样数，默认4096点float格式
2. THE HDR_Visualization_System SHALL 在LUT头注释包含p,a,Min/MaxDisplay_PQ,Channel,D_T_low/high,M,λ,TH_*参数
3. THE HDR_Visualization_System SHALL 分离存储temporal_state.json和session_state.json
4. THE HDR_Visualization_System SHALL 确保导入会话后重建曲线最大绝对误差≤1e-4
5. THE HDR_Visualization_System SHALL 确保D'与Local_Contrast重建误差≤1e-6

### 需求16

**用户故事:** 作为算法工程师，我希望系统保持计算域和统计的一致性，以便确保结果的可靠性

#### 验收标准

1. THE HDR_Visualization_System SHALL 在所有曲线计算、质量指标与直方图统计中统一使用PQ_Domain
2. THE HDR_Visualization_System SHALL 确保D'与Local_Contrast计算使用与直方图相同的亮度通道
3. WHEN 前端显示尼特值时，THE HDR_Visualization_System SHALL 仅作显示用途，不改变内部计算域
4. THE HDR_Visualization_System SHALL 明确显示当前使用的亮度通道类型
5. THE HDR_Visualization_System SHALL 在切换亮度通道时同步更新所有相关计算

### 需求17

**用户故事:** 作为质量保证工程师，我希望系统具备严格的单调性验证机制，以便确保曲线质量

#### 验收标准

1. THE HDR_Visualization_System SHALL 使用N≥1024的内部高密度采样进行曲线单调性验收
2. THE HDR_Visualization_System SHALL 以UI目标N进行显示绘制，以内部采样为验收标准
3. WHEN 样条模式开启时，THE HDR_Visualization_System SHALL 对L'_final进行单调性验证
4. IF 检测到非单调时，THE HDR_Visualization_System SHALL 自动回退到Phoenix_Curve
5. THE HDR_Visualization_System SHALL 在UI给出非单调警示提示

### 需求18

**用户故事:** 作为系统管理员，我希望明确性能基线和硬件要求，以便确保系统稳定运行

#### 验收标准

1. THE HDR_Visualization_System SHALL 在CPU i7/16GB、无GPU加速、1MP以内图像的基线上进行性能验收
2. THE HDR_Visualization_System SHALL 在启用GPU/Numpy MKL加速时在"关于"面板显示加速状态
3. THE HDR_Visualization_System SHALL 提供降采样开关与图像尺寸上限设置
4. WHEN 图像超过尺寸上限时，THE HDR_Visualization_System SHALL 自动降采样以满足300ms响应目标
5. THE HDR_Visualization_System SHALL 在处理大图像时显示当前加速状态

### 需求19

**用户故事:** 作为研究人员，我希望时域缓冲与会话状态分离管理，以便精确控制系统行为

#### 验收标准

1. THE HDR_Visualization_System SHALL 将时域缓冲存储在temporal_state.json中
2. THE HDR_Visualization_System SHALL 将UI参数存储在session_state.json中
3. WHEN 切换模式、切换图像或调整通道时，THE HDR_Visualization_System SHALL 清空temporal_state
4. THE HDR_Visualization_System SHALL 在冷启动时保留session_state
5. THE HDR_Visualization_System SHALL 提供temporal_state的可视化状态指示

### 需求20

**用户故事:** 作为图像处理专家，我希望系统明确处理不同输入格式的色度和OETF，以便确保转换准确性

#### 验收标准

1. THE HDR_Visualization_System SHALL 对OpenEXR(linear)执行linear→PQ转换
2. THE HDR_Visualization_System SHALL 对PQ PNG跳过OETF直接处理
3. THE HDR_Visualization_System SHALL 对sRGB输入先逆EOTF到线性再转PQ
4. THE HDR_Visualization_System SHALL 在UI明确显示输入解释路径
5. THE HDR_Visualization_System SHALL 在导出数据中记录输入解释路径

### 需求21

**用户故事:** 作为用户，我希望系统具备完善的错误处理和边界条件检查，以便避免无效操作

#### 验收标准

1. THE HDR_Visualization_System SHALL 对TH1≈TH2、TH2≈TH3、TH1≥TH3等非法样条配置进行约束
2. THE HDR_Visualization_System SHALL 自动排序样条节点并保持最小间隔δ=0.01
3. THE HDR_Visualization_System SHALL 对a=0、p<0.1、MaxDisplay_PQ≤MinDisplay_PQ等非法输入给出即时校验
4. THE HDR_Visualization_System SHALL 拒绝执行非法参数的计算
5. THE HDR_Visualization_System SHALL 提供友好的错误提示和修正建议

### 需求22

**用户故事:** 作为算法研究员，我希望Auto模式的估参过程可观测和可调节，以便理解和优化算法

#### 验收标准

1. THE HDR_Visualization_System SHALL 在Auto_Mode面板显示估参中间量p0,a0,α,β
2. THE HDR_Visualization_System SHALL 显示min/avg/var/max_pq统计值
3. THE HDR_Visualization_System SHALL 显示估算出的最终p,a值
4. THE HDR_Visualization_System SHALL 提供"一键应用到滑块"按钮
5. THE HDR_Visualization_System SHALL 提供"恢复默认"按钮重置估参参数