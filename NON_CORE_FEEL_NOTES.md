# canva-test 非核心手感内容梳理

本文档只总结 `canva-test` 测试壳中偏交互手感、观察和调试的内容。真正的画布核心能力在 `canvas_core`：画布创建、图层、笔触命令、合成、GPU 纹理输出、历史记录、稳定器算法等都由核心库承担。

## 范围判断

`canva-test` 的核心职责是把 `canvas_core` 跑起来，并提供一个可以手绘测试的桌面 UI。这里的非核心手感内容主要包括：

- 输入采样和拖拽生命周期的衔接。
- 鼠标滚轮缩放、右键平移、视口坐标换算。
- 稳定器参数的实时调节入口。
- 预测尾巴、原始输入点覆盖层、帧率和采样间隔审计。
- 画笔大小、颜色、橡皮/绘制模式等测试 UI。

这些内容对“画起来顺不顺”“延迟感强不强”“输入是否断点”很重要，但不应该被视为核心引擎 API 的一部分。

## 核心自带 vs test 补上

核心库已经自带了一些和手感有关、但仍然属于 headless 引擎层的能力：

- 流式笔触生命周期：`begin_stroke`、`append_stroke_points`、`end_stroke`，以及对应的 `CanvasCommand::BeginStroke`、`AppendStroke`、`EndStroke`。这让高频输入可以增量喂给 GPU，不需要每帧重传整条笔迹。
- 稳定器管线：`StabilizerConfig`、`StabilizerMode::Adaptive`、`StabilizerMode::PulledString`、`smooth_strength`、`spacing_ratio`。核心负责 One Euro 自适应平滑、Pulled-String 拉绳手感、Catmull-Rom 样条和弧长重采样。
- 采样合并：`streaming_sample_min_distance` 可以在核心层合并过密输入点；默认 `0.0` 表示保留调用方传入的有限坐标点。
- 笔压参与笔刷：`StrokePoint.pressure` 会进入笔刷半径/覆盖路径，核心会清洗非法压力值。
- 稳定后笔尖查询：`active_stroke_tip(stroke)` 暴露当前活动笔触的稳定后末端，供外壳做低延迟视觉补偿。
- 单笔 undo 合并：流式 append 期间产生的历史会在 `end_stroke` 后合并成一个撤销事务，避免一笔被拆成很多撤销步骤。
- 绘制/擦除/湿混模式：`BrushMode::Paint`、`Erase`、`WetMix` 是核心笔刷行为，其中擦除和湿混也会影响实际绘画手感。
- 输入健壮性：核心会丢弃非有限坐标、清洗半径/颜色/压力，防止坏输入导致 shader 或历史状态异常。

`canva-test` 这边补的是外壳层体验，不是核心算法：

- 鼠标/数位笔事件收集：从 egui 的本帧事件里抓取所有 `PointerMoved`，再批量喂给核心。
- 输入缺省补点：本帧没有移动事件时，用 `interact_pointer_pos()` 补一个当前点，避免拖拽中完全没样本。
- 屏幕到画布坐标换算：把 UI rect、`zoom`、`pan` 转成核心需要的画布坐标。
- 视口缩放和平移：滚轮向光标缩放、右键拖动画布、`Fit`/`4x`/`16x` 按钮和合法 pan 夹取。
- 实时调参 UI：平滑强度、稳定器模式、画笔大小、颜色、绘制/擦除模式、预测尾巴开关。
- 预测尾巴：利用核心的 `active_stroke_tip` 加上外壳估算的屏幕速度，画一段半透明临时线来缓解视觉滞后；这段不会进入核心历史。
- 输入审计：统计 raw events、events/frame、点间距、帧耗时，并把原始输入点画成红点。
- 连续重绘：`request_repaint()` 让预测尾巴和审计覆盖层持续刷新。

简化理解：核心负责“笔迹实际怎么被过滤、重采样、盖到画布上，并且如何进历史”；`canva-test` 负责“用户输入怎么被收集、如何看见延迟/采样问题、如何调参和补视觉反馈”。

## 启动配置中的手感项

`main.rs` 创建 `CanvasConfig` 时配置了几项直接影响绘制体验的参数：

- `streaming_sample_min_distance: 0.0`：测试壳不在输入层做最小距离过滤，尽量把高频原始点交给引擎，方便观察稳定器和重采样效果。
- `stabilizer.enabled: true`：默认开启稳定器，让测试启动后直接处于更平滑的绘制状态。
- `stabilizer.mode: Adaptive`：默认使用自适应稳定器。
- `smooth_strength: 0.5`：默认平滑强度居中，便于继续向两侧调参。
- `spacing_ratio: 0.05`：笔触点间距相关参数，影响曲线重采样密度和线条连续感。

这些配置属于测试入口的体验预设；核心能力来自 `canvas_core`，这里负责给出默认手感。

## 侧边栏调参 UI

左侧控制面板提供了一组非核心但便于试手感的控件：

- 画笔大小 `brush_radius`：范围 `1.0..=50.0`，直接影响 stroke 半径，也影响预测尾巴的可视宽度。
- 平滑强度 `smooth_strength`：范围 `0.0..=1.0`，变化后调用 `core.set_stabilizer_strength(...)`，用于实时试稳定器手感。
- 稳定器模式 `stab_mode`：在 `Adaptive` 和 `PulledString` 之间切换，变化后调用 `core.set_stabilizer_mode(...)`。
- 颜色编辑器：修改 `brush_color`，用于测试不同颜色笔触。
- `Prediction tail`：开关预测尾巴，只影响临时预览，不提交到引擎历史。
- 绘制/擦除模式：在 `BrushMode::Paint` 和 `BrushMode::Erase` 之间切换。
- `Clear Canvas`、`Undo`、`Redo`：测试核心命令，但按钮本身属于测试 UI。

这些控件的价值是快速暴露参数变化对手感的影响，而不是定义正式产品交互。

## 视口缩放和平移

画布中央区域不是直接显示完整纹理，而是通过 `zoom`、`pan` 和 UV 映射模拟视口：

- `zoom` 默认 `1.0`，滑块范围 `1.0..=32.0`。
- `Fit` 重置为 `1.0` 并清空 `pan`。
- `4x`、`16x` 是便捷倍率按钮。
- 鼠标滚轮缩放时会以光标所在画布点为中心重新计算 `pan`，让缩放焦点尽量不漂移。
- 右键拖拽平移，屏幕拖拽距离会除以 `display_scale` 转成画布坐标偏移。
- `pan` 会按当前倍率夹在合法范围内，避免视口拖出画布外。

这部分主要提升检查细节和手绘定位的体验，不改变底层画布内容。

## 坐标换算

测试壳里维护了屏幕坐标到画布坐标的换算：

- `display_scale = canvas_display_size / canvas_size * zoom`
- 画布坐标 `x = pan.x + (screen_x - rect.min.x) / display_scale`
- 画布坐标 `y = pan.y + (screen_y - rect.min.y) / display_scale`

所有输入点都会先通过这个换算变成 `StrokePoint`。这属于 UI 视口层逻辑，核心引擎只接收画布坐标。

## 输入采样和笔触生命周期

左键拖拽控制笔触生命周期：

- `drag_started_by(Primary)`：重置审计数据，记录初始光标，调用 `BeginStroke`，保存 `active_stroke`。
- `dragged_by(Primary)`：读取本帧所有 `PointerMoved` 事件，转成画布坐标，批量调用 `AppendStroke`。
- 如果本帧没有 `PointerMoved`，会回退到 `interact_pointer_pos()` 追加当前点，避免拖动期间完全没有采样。
- `drag_stopped()`：调用 `EndStroke`，清空预测状态。

这里比较偏手感的点是：测试壳尝试收集一帧内的全部高频移动事件，而不是只取当前光标位置。这样能减少快速移动时的断点和折线感，也方便验证核心 streaming API。

## 预测尾巴

`prediction` 开启后，绘制过程中会渲染一段临时预览线：

- 从核心引擎返回的稳定后笔尖 `active_stroke_tip(id)` 画到当前光标。
- 根据最近帧的屏幕速度 `pred_vel` 再外推一小段。
- 速度用指数平滑，系数为 `0.4`，用来压制抖动。
- 外推时间窗 `horizon = 0.024s`，约等于 60fps 下 1.5 帧。
- 最大外推距离为 `max(brush_radius * 4.0, 24.0)`，防止高速甩笔时过冲。
- 速度低于 `5.0` 时不外推，只画到当前光标。
- 颜色使用当前画笔色但半透明，宽度按画笔直径和当前缩放计算。

这段线只通过 egui painter 临时画在界面上，不会提交 `AppendStroke`，也不会进入历史记录。它的目标是降低稳定器带来的视觉滞后感。

## 输入审计覆盖层

`StrokeAudit` 和 `debug_overlay` 提供了观察输入质量的功能：

- 记录拖拽帧数 `frames`。
- 统计原始移动事件数量 `raw_events`。
- 计算每帧事件数 `events/frame`。
- 统计相邻原始点距离的最大值和平均值。
- 记录当前帧耗时和最大帧耗时。
- 在画布上用红点显示原始输入点。

这些数据不参与绘制结果，只用于判断输入设备、egui 事件采样、帧率和稳定器之间的关系。

## 连续刷新

`ui.ctx().request_repaint()` 每帧请求重绘，让 UI、预测尾巴和审计信息持续更新。对测试工具来说这很直接，但正式产品中可能需要按活动状态节流，否则会增加空闲 CPU/GPU 占用。

## 可以视为非核心的字段

`CanvasApp` 中以下字段主要服务手感或测试 UI：

- `brush_radius`
- `brush_color`
- `brush_mode`
- `smooth_strength`
- `stab_mode`
- `last_cursor`
- `pred_vel`
- `prediction`
- `zoom`
- `pan`
- `debug_overlay`
- `audit`

其中 `active_stroke`、`texture_id`、`canvas_size` 更偏测试壳和核心引擎之间的桥接状态。

## 后续整理建议

如果这个测试壳继续增长，可以把非核心手感逻辑拆成几个小模块：

- `ViewportState`：管理 `zoom`、`pan`、UV 和坐标换算。
- `StrokeInputState`：管理拖拽开始、追加、结束，以及高频事件收集。
- `PredictionTail`：管理速度估计和临时预览渲染。
- `StrokeAudit`：保留为独立调试模块。
- `ControlsPanel`：只负责调参 UI，不直接混在绘制区域逻辑里。

这样能让 `canva-test` 继续作为测试壳使用，同时避免手感实验逻辑和核心调用路径混在一个 `ui` 函数里。
