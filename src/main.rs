use canva_input::{
    DeviceId, InputBackend, PressureCalibration, StrokeSampleAdapter, StylusButtons, StylusPhase,
    StylusSample,
};
use canvas_core::{
    AppendStrokeCommand, BeginStrokeCommand, BlendMode, BrushCommand, BrushDynamics, BrushMode,
    BrushStamp, BrushStampKind, BrushStampTextureId, CanvasCommand, CanvasConfig, CanvasCore,
    CanvasError, CaptureLevel, Color, CommandOutput, DataCaptureConfig, InputDeviceCapabilities,
    LayerGroupMode, LayerId, LayerSnapshot, LayerSpec, PressureCurve, SelectedPixels,
    SelectionCombineMode, SelectionPoint, SelectionPolygon, SelectionRect, SessionTrace,
    StabilizerConfig, StabilizerMode, StrokeId, StrokePoint, StrokePredictionConfig, TraceReport,
};
use eframe::egui;
use eframe::egui_wgpu::wgpu;
use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, Mutex};

fn main() -> eframe::Result<()> {
    let config = CanvasConfig {
        width: 1024,
        height: 1024,
        tile_size: 256,
        history_budget_bytes: 512 * 1024 * 1024,
        streaming_sample_min_distance: 0.0,
        stabilizer: StabilizerConfig {
            enabled: true,
            mode: StabilizerMode::Adaptive,
            smooth_strength: 0.5,
            spacing_ratio: 0.05,
            ..StabilizerConfig::default()
        },
        prediction: StrokePredictionConfig {
            enabled: false,
            time_horizon_seconds: 1.0 / 120.0,
            max_distance: 12.0,
            min_samples: 2,
        },
        data_capture: initial_capture_config(),
        ..CanvasConfig::default()
    };

    let mut core = pollster::block_on(CanvasCore::new(config)).expect("创建 CanvasCore 失败");

    let layer_id = LayerId(1);
    core.execute(&CanvasCommand::AddLayer(LayerSpec::new(layer_id)))
        .expect("添加图层失败");
    core.composite().expect("合成画布失败");

    // Clone wgpu resources so eframe shares the same device/queue as the engine.
    // This enables zero-copy GPU texture interop without CPU readback.
    let instance = core.wgpu_instance();
    let adapter = core.wgpu_adapter();
    let device = core.wgpu_device();
    let queue = core.wgpu_queue();

    let core = Arc::new(Mutex::new(core));

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 900.0])
            .with_title("画布引擎测试"),
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            wgpu_setup: eframe::egui_wgpu::WgpuSetup::Existing(
                eframe::egui_wgpu::WgpuSetupExisting {
                    instance,
                    adapter,
                    device,
                    queue,
                },
            ),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "画布引擎测试",
        options,
        Box::new(|cc| Ok(Box::new(CanvasApp::new(core, layer_id, cc)))),
    )
}

// Input sampling diagnostics.
#[derive(Default)]
struct StrokeAudit {
    frames: u32,
    raw_events: u32,
    ignored_touch_starts: u32,
    last_touch_events: usize,
    last_touch_starts: usize,
    last_touch_moves: usize,
    last_touch_ends: usize,
    last_touch_cancels: usize,
    last_raw_force_min: Option<f32>,
    last_raw_force_max: Option<f32>,
    last_raw_force_missing: usize,
    last_adapted_pressure_min: Option<f32>,
    last_adapted_pressure_max: Option<f32>,
    max_event_gap: f32,
    sum_event_gap: f64,
    event_gap_count: u32,
    max_frame_dt: f32,
    last_frame_dt: f32,
    begin_point: Option<StrokePoint>,
    begin_pressure_from_device: bool,
    begin_input_source: Option<StrokeInputSource>,
    reanchored_touch_begins: u32,
    append_batches: u32,
    last_append_len: usize,
    last_append_first: Option<StrokePoint>,
    last_append_second: Option<StrokePoint>,
    last_core_uploaded_points: usize,
    last_core_dispatches: usize,
    raw_pts: Vec<StrokePoint>,
}

impl StrokeAudit {
    fn reset(&mut self) {
        *self = StrokeAudit::default();
    }
    fn avg_event_gap(&self) -> f64 {
        if self.event_gap_count == 0 {
            0.0
        } else {
            self.sum_event_gap / f64::from(self.event_gap_count)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ToolMode {
    Brush,
    RectSelection,
    LassoSelection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StrokeInputSource {
    Touch,
    PointerFallback,
}

#[derive(Clone, Copy)]
struct InputPoint {
    point: StrokePoint,
    source: StrokeInputSource,
    pressure_from_device: bool,
}

struct InputBatch {
    points: Vec<StrokePoint>,
    source: StrokeInputSource,
    first_pressure_from_device: bool,
}

#[derive(Clone, Copy, Debug)]
struct TapCandidate {
    anchor: StrokePoint,
    max_pressure: f32,
    max_distance: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BrushPreset {
    SoftRound,
    HardRound,
    SoftRoundPressureSize,
    HardRoundPressureSize,
    SoftRoundPressureOpacity,
    HardRoundPressureOpacity,
    SoftRoundPressureOpacityFlow,
    HardRoundPressureOpacityFlow,
}

impl BrushPreset {
    const ALL: [Self; 8] = [
        Self::SoftRound,
        Self::HardRound,
        Self::SoftRoundPressureSize,
        Self::HardRoundPressureSize,
        Self::SoftRoundPressureOpacity,
        Self::HardRoundPressureOpacity,
        Self::SoftRoundPressureOpacityFlow,
        Self::HardRoundPressureOpacityFlow,
    ];

    fn is_soft_round(self) -> bool {
        matches!(
            self,
            Self::SoftRound
                | Self::SoftRoundPressureSize
                | Self::SoftRoundPressureOpacity
                | Self::SoftRoundPressureOpacityFlow
        )
    }

    fn pressure_controls_size(self) -> bool {
        matches!(
            self,
            Self::SoftRoundPressureSize | Self::HardRoundPressureSize
        )
    }

    fn pressure_controls_opacity(self) -> bool {
        matches!(
            self,
            Self::SoftRoundPressureOpacity
                | Self::HardRoundPressureOpacity
                | Self::SoftRoundPressureOpacityFlow
                | Self::HardRoundPressureOpacityFlow
        )
    }

    fn pressure_controls_flow(self) -> bool {
        matches!(
            self,
            Self::SoftRoundPressureOpacityFlow | Self::HardRoundPressureOpacityFlow
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SelectionDrag {
    Rect {
        start: SelectionPoint,
        current: SelectionPoint,
    },
    Lasso,
}

#[derive(Clone, Copy, Debug)]
struct CanvasViewport {
    origin: egui::Pos2,
    display_scale: f32,
    pan: egui::Vec2,
}

impl CanvasViewport {
    fn to_stroke_point(self, pos: egui::Pos2, pressure: f32) -> StrokePoint {
        StrokePoint {
            x: self.pan.x + (pos.x - self.origin.x) / self.display_scale,
            y: self.pan.y + (pos.y - self.origin.y) / self.display_scale,
            pressure,
        }
    }
}

struct CanvasApp {
    core: Arc<Mutex<CanvasCore>>,
    layer_id: LayerId,
    brush_radius: f32,
    brush_color: [f32; 4],
    brush_mode: BrushMode,
    canvas_width: f32,
    canvas_height: f32,
    active_stroke: Option<StrokeId>,
    active_stroke_input_source: Option<StrokeInputSource>,
    active_stroke_dispatched: bool,
    active_tap_candidate: Option<TapCandidate>,
    smooth_strength: f32,
    stab_mode: StabilizerMode,
    /// Last cursor position used for preview prediction and input audit deltas.
    last_cursor: Option<egui::Pos2>,
    pred_vel: egui::Vec2,
    prediction: bool,
    texture_id: Option<egui::TextureId>,
    zoom: f32,
    pan: egui::Vec2,
    debug_overlay: bool,
    audit: StrokeAudit,
    fallback_pressure: f32,
    last_pressure: f32,
    pressure_from_device: bool,
    pressure_dynamics: bool,
    velocity_dynamics: bool,
    texture_grain: bool,
    brush_preset: BrushPreset,
    core_prediction: bool,
    stamp_kind: BrushStampKind,
    stamp_hardness: f32,
    stamp_aspect: f32,
    stamp_angle: f32,
    stamp_texture: bool,
    stamp_texture_id: Option<BrushStampTextureId>,
    tool_mode: ToolMode,
    selection_mode: SelectionCombineMode,
    selection_drag: Option<SelectionDrag>,
    lasso_points: Vec<SelectionPoint>,
    clipboard: Option<SelectedPixels>,
    next_layer_id: u64,
    last_error: Option<String>,
    new_canvas_width: u32,
    new_canvas_height: u32,
    new_canvas_tile_size: u32,
    trace_enabled: bool,
    trace_level: CaptureLevel,
    trace_budget_mb: u64,
    trace_rebuild_result: Option<String>,
    last_trace: Option<SessionTrace>,
    trace_base_layers: Vec<LayerSnapshot>,
}

fn stamp_kind_label(kind: BrushStampKind) -> &'static str {
    match kind {
        BrushStampKind::Circle => "圆形",
        BrushStampKind::Square => "方形",
        BrushStampKind::Diamond => "菱形",
        BrushStampKind::Stripe => "条纹",
    }
}

fn brush_preset_label(preset: BrushPreset) -> &'static str {
    match preset {
        BrushPreset::SoftRound => "柔边圆形",
        BrushPreset::HardRound => "硬边圆形",
        BrushPreset::SoftRoundPressureSize => "柔边圆形（压力控制大小）",
        BrushPreset::HardRoundPressureSize => "硬边圆形（压力控制大小）",
        BrushPreset::SoftRoundPressureOpacity => "柔边圆形（压力控制不透明度）",
        BrushPreset::HardRoundPressureOpacity => "硬边圆形（压力控制不透明度）",
        BrushPreset::SoftRoundPressureOpacityFlow => "柔边圆形（压力控制不透明度和流量）",
        BrushPreset::HardRoundPressureOpacityFlow => "硬边圆形（压力控制不透明度和流量）",
    }
}

fn selection_mode_label(mode: SelectionCombineMode) -> &'static str {
    match mode {
        SelectionCombineMode::Replace => "替换",
        SelectionCombineMode::Add => "添加",
        SelectionCombineMode::Subtract => "减去",
        SelectionCombineMode::Intersect => "相交",
    }
}

fn touch_phase_appends_to_active_stroke(phase: egui::TouchPhase) -> bool {
    matches!(phase, egui::TouchPhase::Move)
}

fn touch_phase_can_begin_stroke(phase: egui::TouchPhase) -> bool {
    matches!(phase, egui::TouchPhase::Start | egui::TouchPhase::Move)
}

fn input_source_label(source: Option<StrokeInputSource>) -> &'static str {
    match source {
        Some(StrokeInputSource::Touch) => "touch",
        Some(StrokeInputSource::PointerFallback) => "pointer fallback",
        None => "-",
    }
}

fn stroke_point_distance(a: StrokePoint, b: StrokePoint) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

fn tap_jitter_threshold(brush_radius: f32) -> f32 {
    (brush_radius * 0.2).clamp(2.5, 5.0)
}

fn points_within_tap_jitter(
    candidate: TapCandidate,
    points: &[StrokePoint],
    brush_radius: f32,
) -> bool {
    let threshold = tap_jitter_threshold(brush_radius);
    candidate.max_distance <= threshold
        && points
            .iter()
            .copied()
            .all(|point| stroke_point_distance(candidate.anchor, point) <= threshold)
}

fn should_reanchor_pointer_begin_to_touch(
    source: StrokeInputSource,
    active_source: Option<StrokeInputSource>,
    active_dispatched: bool,
    begin: Option<StrokePoint>,
    first: StrokePoint,
    brush_radius: f32,
) -> bool {
    if source != StrokeInputSource::Touch
        || active_source != Some(StrokeInputSource::PointerFallback)
        || active_dispatched
    {
        return false;
    }
    let Some(begin) = begin else {
        return false;
    };
    let threshold = (brush_radius * 2.0).max(16.0);
    stroke_point_distance(begin, first) > threshold
}

fn should_reanchor_touch_begin_pressure(
    source: StrokeInputSource,
    active_source: Option<StrokeInputSource>,
    active_dispatched: bool,
    begin_pressure_from_device: bool,
    first_pressure_from_device: bool,
) -> bool {
    source == StrokeInputSource::Touch
        && active_source == Some(StrokeInputSource::Touch)
        && !active_dispatched
        && !begin_pressure_from_device
        && first_pressure_from_device
}

fn stroke_point_debug(point: Option<StrokePoint>) -> String {
    point.map_or_else(
        || "-".to_owned(),
        |point| {
            format!(
                "x={:.2}, y={:.2}, p={:.3}",
                point.x, point.y, point.pressure
            )
        },
    )
}

fn optional_f32_debug(value: Option<f32>) -> String {
    value.map_or_else(|| "-".to_owned(), |value| format!("{value:.3}"))
}

fn selection_rect_from_points(a: SelectionPoint, b: SelectionPoint) -> Option<SelectionRect> {
    let min_x = a.x.min(b.x).floor().max(0.0) as u32;
    let min_y = a.y.min(b.y).floor().max(0.0) as u32;
    let max_x = a.x.max(b.x).ceil().max(0.0) as u32;
    let max_y = a.y.max(b.y).ceil().max(0.0) as u32;
    (max_x > min_x && max_y > min_y)
        .then(|| SelectionRect::new(min_x, min_y, max_x - min_x, max_y - min_y))
}

const CAPTURE_LEVELS: [CaptureLevel; 4] = [
    CaptureLevel::L0Structure,
    CaptureLevel::L1PointStream,
    CaptureLevel::L2PixelDelta,
    CaptureLevel::L3SparseCheckpoint,
];

fn capture_level_label(level: CaptureLevel) -> &'static str {
    match level {
        CaptureLevel::L0Structure => "L0 结构事件",
        CaptureLevel::L1PointStream => "L1 笔触点流",
        CaptureLevel::L2PixelDelta => "L2 像素变化（当前降级）",
        CaptureLevel::L3SparseCheckpoint => "L3 稀疏存档点（当前降级）",
    }
}

fn capture_config(enabled: bool, level: CaptureLevel, budget_mb: u64) -> DataCaptureConfig {
    DataCaptureConfig {
        enabled,
        level,
        max_bytes_per_session: budget_mb.max(1).saturating_mul(1024 * 1024),
        allow_gpu_readback: false,
        ..DataCaptureConfig::default()
    }
}

fn initial_capture_config() -> DataCaptureConfig {
    capture_config(true, CaptureLevel::L1PointStream, 8)
}

fn trace_report_summary(report: &TraceReport) -> String {
    format!(
        "事件={}，命令={}，笔触={}，追加={}，点={}，图层事件={}，选区事件={}，dirty 摘要={}，资源引用={}，降级={}，估算={} KB",
        report.event_count,
        report.command_count,
        report.stroke_count,
        report.stroke_append_count,
        report.point_count,
        report.layer_event_count,
        report.selection_event_count,
        report.dirty_summary_count,
        report.resource_reference_count,
        report.degradation_count,
        report.estimated_bytes / 1024
    )
}

fn max_pixel_abs_diff(a: &[[f32; 4]], b: &[[f32; 4]]) -> Option<f32> {
    (a.len() == b.len()).then(|| {
        a.iter()
            .zip(b)
            .flat_map(|(left, right)| {
                left.iter()
                    .zip(right)
                    .map(|(left, right)| (left - right).abs())
            })
            .fold(0.0_f32, f32::max)
    })
}

fn trace_rebuild_summary(trace: &SessionTrace, max_diff: f32) -> String {
    format!(
        "已重构到画布：命令={}，点={}，资源引用={}，降级={}，重构前最大像素误差={max_diff:.6}",
        trace.report.command_count,
        trace.report.point_count,
        trace.report.resource_reference_count,
        trace.report.degradation_count
    )
}

fn trace_rebuild_error_message(error: &CanvasError) -> String {
    match error {
        CanvasError::MissingLayer(layer) => format!("重构失败：图层 {layer:?} 不存在"),
        CanvasError::MissingStroke(stroke) => format!("重构失败：笔触 {stroke:?} 不存在"),
        CanvasError::MissingTraceResource(id) if id.starts_with("测试程序") => {
            format!("重构失败：{id}")
        }
        CanvasError::MissingTraceResource(_) | CanvasError::InvalidTraceResource(_) => {
            format!("重构失败：{}", error.diagnostic_message_zh())
        }
        _ => format!("重构失败：{}", error.diagnostic_message_zh()),
    }
}

fn trace_rebuild_unavailable_message(
    enabled: bool,
    has_current_trace: bool,
    has_cached_trace: bool,
) -> &'static str {
    match (enabled, has_current_trace, has_cached_trace) {
        (_, true, _) | (_, _, true) => "已有可重构记录",
        (false, false, false) => "记录未启用，无法重构",
        (true, false, false) => "记录为空，无法重构",
    }
}

fn trace_texture_registration_error_message(error: impl std::fmt::Display) -> String {
    format!("重构已完成，但显示纹理注册失败：{error}")
}

fn trace_status_label(
    enabled: bool,
    has_current_trace: bool,
    has_cached_trace: bool,
) -> &'static str {
    match (enabled, has_current_trace, has_cached_trace) {
        (_, true, _) => "正在记录当前绘画",
        (false, false, true) => "记录已停止，可重构上次记录",
        (true, false, true) => "当前记录未创建，仍可重构上次记录",
        _ => "还没有可用记录",
    }
}

fn trace_base_layer_commands(layers: &[LayerSnapshot]) -> Vec<CanvasCommand> {
    let mut commands = Vec::with_capacity(layers.len().saturating_mul(3));
    for layer in layers {
        commands.push(CanvasCommand::AddLayer(layer.spec.clone()));
    }
    for layer in layers {
        if let Some(parent) = layer.parent {
            commands.push(CanvasCommand::SetLayerParent {
                id: layer.spec.id,
                parent: Some(parent),
            });
        }
        if layer.group_mode != LayerGroupMode::PassThrough {
            commands.push(CanvasCommand::SetLayerGroupMode {
                id: layer.spec.id,
                mode: layer.group_mode,
            });
        }
    }
    commands
}

fn trace_replay_command_payload(
    command: &canvas_core::TraceCommandEvent,
) -> Result<&CanvasCommand, CanvasError> {
    match &command.replay {
        Some(canvas_core::TraceReplayCommand::Command(command)) => Ok(command),
        Some(canvas_core::TraceReplayCommand::MissingResource { .. }) => {
            Err(CanvasError::MissingTraceResource(format!(
                "测试程序重构暂不解析外部资源引用；{}",
                command.diagnostic_message_zh()
            )))
        }
        None => Err(CanvasError::MissingTraceResource(format!(
            "测试程序重构缺少可执行 command replay payload；{}",
            command.diagnostic_message_zh()
        ))),
    }
}

fn replay_trace_commands_into_core(
    core: &mut CanvasCore,
    trace: &SessionTrace,
    base_layers: &[LayerSnapshot],
) -> Result<(), CanvasError> {
    let mut config = trace.canvas;
    config.data_capture.enabled = false;
    core.set_data_capture_config(config.data_capture);
    core.execute(&CanvasCommand::NewCanvas(config))?;
    for command in trace_base_layer_commands(base_layers) {
        core.execute(&command)?;
    }
    let mut stroke_id_map = HashMap::new();
    for (event_index, event) in trace.events.iter().enumerate() {
        let canvas_core::TraceEventKind::Command(command) = &event.kind else {
            continue;
        };
        let command = trace_replay_command_payload(command)?;
        let command = remap_replay_stroke_command(command, &stroke_id_map);
        match &command {
            CanvasCommand::AddLayer(spec)
                if core
                    .layer_snapshot()
                    .iter()
                    .any(|layer| layer.spec.id == spec.id) =>
            {
                continue;
            }
            CanvasCommand::AppendStroke(command)
                if !stroke_id_map.values().any(|id| *id == command.stroke) =>
            {
                continue;
            }
            CanvasCommand::EndStroke(stroke) if !stroke_id_map.values().any(|id| id == stroke) => {
                continue;
            }
            CanvasCommand::BeginStroke(_) => {
                let output = core.execute(&command)?;
                if let CommandOutput::StrokeStarted { id: new_id } = output
                    && let Some(old_id) = trace_stroke_begin_after(trace, event_index)
                {
                    stroke_id_map.insert(old_id, new_id);
                }
                continue;
            }
            _ => {}
        }
        core.execute(&command)?;
    }
    Ok(())
}

fn trace_stroke_begin_after(trace: &SessionTrace, command_index: usize) -> Option<StrokeId> {
    trace
        .events
        .iter()
        .skip(command_index.saturating_add(1))
        .take_while(|event| !matches!(event.kind, canvas_core::TraceEventKind::Command(_)))
        .find_map(|event| match &event.kind {
            canvas_core::TraceEventKind::StrokeBegin(begin) => Some(begin.stroke),
            _ => None,
        })
}

fn remap_replay_stroke_command(
    command: &CanvasCommand,
    stroke_id_map: &HashMap<StrokeId, StrokeId>,
) -> CanvasCommand {
    match command {
        CanvasCommand::AppendStroke(command) => {
            let mut command = command.clone();
            if let Some(stroke) = stroke_id_map.get(&command.stroke) {
                command.stroke = *stroke;
            }
            CanvasCommand::AppendStroke(command)
        }
        CanvasCommand::EndStroke(stroke) => {
            CanvasCommand::EndStroke(stroke_id_map.get(stroke).copied().unwrap_or(*stroke))
        }
        _ => command.clone(),
    }
}

impl CanvasApp {
    fn new(
        core: Arc<Mutex<CanvasCore>>,
        layer_id: LayerId,
        cc: &eframe::CreationContext<'_>,
    ) -> Self {
        install_chinese_fonts(&cc.egui_ctx);

        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("canvas display requires WGPU render state");

        let mut core_guard = core.lock().unwrap();
        core_guard.present().expect("显示画布失败");
        let texture_view = core_guard
            .presentation_texture_view()
            .expect("显示后应存在画布纹理视图");
        drop(core_guard);

        let mut renderer = render_state.renderer.write();
        let texture_id = renderer.register_native_texture_with_sampler_options(
            &render_state.device,
            &texture_view,
            wgpu::SamplerDescriptor {
                label: Some("canvas presentation sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            },
        );

        let trace_base_layers = core.lock().unwrap().layer_snapshot();

        Self {
            core,
            layer_id,
            brush_radius: 15.0,
            brush_color: [0.0, 0.0, 0.0, 1.0],
            brush_mode: BrushMode::Paint,
            canvas_width: 1024.0,
            canvas_height: 1024.0,
            active_stroke: None,
            active_stroke_input_source: None,
            active_stroke_dispatched: false,
            active_tap_candidate: None,
            smooth_strength: 0.5,
            stab_mode: StabilizerMode::Adaptive,
            last_cursor: None,
            pred_vel: egui::Vec2::ZERO,
            prediction: false,
            texture_id: Some(texture_id),
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            debug_overlay: false,
            audit: StrokeAudit::default(),
            fallback_pressure: 1.0,
            last_pressure: 1.0,
            pressure_from_device: false,
            pressure_dynamics: false,
            velocity_dynamics: false,
            texture_grain: false,
            brush_preset: BrushPreset::HardRound,
            core_prediction: false,
            stamp_kind: BrushStampKind::Circle,
            stamp_hardness: 1.0,
            stamp_aspect: 1.0,
            stamp_angle: 0.0,
            stamp_texture: false,
            stamp_texture_id: None,
            tool_mode: ToolMode::Brush,
            selection_mode: SelectionCombineMode::Replace,
            selection_drag: None,
            lasso_points: Vec::new(),
            clipboard: None,
            next_layer_id: layer_id.0 + 1,
            last_error: None,
            new_canvas_width: 1024,
            new_canvas_height: 1024,
            new_canvas_tile_size: 256,
            trace_enabled: true,
            trace_level: CaptureLevel::L1PointStream,
            trace_budget_mb: 8,
            trace_rebuild_result: None,
            last_trace: None,
            trace_base_layers,
        }
    }

    fn color(&self) -> Color {
        Color::rgba(
            self.brush_color[0],
            self.brush_color[1],
            self.brush_color[2],
            self.brush_color[3],
        )
    }

    fn apply_brush_preset(&mut self, preset: BrushPreset) {
        self.brush_preset = preset;
        self.brush_mode = BrushMode::Paint;
        self.stamp_kind = BrushStampKind::Circle;
        self.stamp_aspect = 1.0;
        self.stamp_angle = 0.0;
        self.stamp_texture = false;
        self.velocity_dynamics = false;
        self.texture_grain = false;
        self.brush_radius = 14.0;
        self.brush_color = [0.0, 0.0, 0.0, 1.0];
        self.fallback_pressure = 1.0;
        self.last_pressure = self.fallback_pressure;
        self.pressure_dynamics = matches!(
            preset,
            BrushPreset::SoftRoundPressureSize
                | BrushPreset::HardRoundPressureSize
                | BrushPreset::SoftRoundPressureOpacity
                | BrushPreset::HardRoundPressureOpacity
                | BrushPreset::SoftRoundPressureOpacityFlow
                | BrushPreset::HardRoundPressureOpacityFlow
        );
        self.stamp_hardness = if preset.is_soft_round() { 0.0 } else { 1.0 };
        self.pressure_from_device = false;
        self.sync_core_brush_settings();
    }

    // composite + present, refresh display texture.
    fn refresh(core: &mut CanvasCore) -> Result<(), canvas_core::CanvasError> {
        core.composite()?;
        core.present()
    }

    fn layers(&self) -> Vec<LayerSnapshot> {
        self.core.lock().unwrap().layer_snapshot()
    }

    fn sync_active_layer(&mut self) {
        let layers = self.layers();
        if layers.iter().any(|layer| layer.spec.id == self.layer_id) {
            return;
        }
        if let Some(layer) = layers.last() {
            self.layer_id = layer.spec.id;
        }
    }

    fn execute_canvas_command(&mut self, command: CanvasCommand) {
        let result = {
            let mut core = self.core.lock().unwrap();
            let mut result = core.execute(&command);
            if result.is_ok()
                && let Err(error) = Self::refresh(&mut core)
            {
                result = Err(error);
            }
            result
        };

        match result {
            Ok(_) => self.last_error = None,
            Err(error) => self.last_error = Some(error.to_string()),
        }
        self.sync_active_layer();
    }

    fn copy_selection_to_clipboard(&mut self, cut: bool) {
        let result = {
            let mut core = self.core.lock().unwrap();
            if cut {
                pollster::block_on(core.cut_selected_pixels(self.layer_id))
            } else {
                pollster::block_on(core.copy_selected_pixels(self.layer_id))
            }
        };

        match result {
            Ok(Some(pixels)) => {
                self.clipboard = Some(pixels);
                self.last_error = None;
                if cut {
                    let mut core = self.core.lock().unwrap();
                    if let Err(error) = Self::refresh(&mut core) {
                        self.last_error = Some(error.to_string());
                    }
                }
            }
            Ok(None) => self.last_error = Some("没有可复制的活动选区".to_owned()),
            Err(error) => self.last_error = Some(error.to_string()),
        }
    }

    fn paste_clipboard_to_active_layer(&mut self) {
        let Some(pixels) = self.clipboard.clone() else {
            self.last_error = Some("剪贴板为空".to_owned());
            return;
        };

        let origin_x = pixels.bounds.x;
        let origin_y = pixels.bounds.y;
        let result = {
            let mut core = self.core.lock().unwrap();
            let mut result = core.paste_pixels_to_layer(self.layer_id, &pixels, origin_x, origin_y);
            if result.is_ok()
                && let Err(error) = Self::refresh(&mut core)
            {
                result = Err(error);
            }
            result
        };
        match result {
            Ok(true) => self.last_error = None,
            Ok(false) => self.last_error = Some("粘贴区域超出画布".to_owned()),
            Err(error) => self.last_error = Some(error.to_string()),
        }
    }

    fn paste_clipboard_to_new_layer(&mut self) {
        let Some(pixels) = self.clipboard.clone() else {
            self.last_error = Some("剪贴板为空".to_owned());
            return;
        };

        let layer_id = LayerId(self.next_layer_id);
        let origin_x = pixels.bounds.x;
        let origin_y = pixels.bounds.y;
        let result = {
            let mut core = self.core.lock().unwrap();
            let mut result = core.paste_pixels_to_new_layer(
                LayerSpec::new(layer_id),
                &pixels,
                origin_x,
                origin_y,
            );
            if result.is_ok()
                && let Err(error) = Self::refresh(&mut core)
            {
                result = Err(error);
            }
            result
        };
        match result {
            Ok(true) => {
                self.layer_id = layer_id;
                self.next_layer_id = self.next_layer_id.saturating_add(1);
                self.last_error = None;
            }
            Ok(false) => self.last_error = Some("粘贴区域在画布外".to_owned()),
            Err(error) => self.last_error = Some(error.to_string()),
        }
        self.sync_active_layer();
    }

    fn selection_point_from_canvas(point: StrokePoint) -> SelectionPoint {
        SelectionPoint::new(point.x, point.y)
    }

    fn apply_rect_selection(&mut self, start: SelectionPoint, end: SelectionPoint) {
        let Some(rect) = selection_rect_from_points(start, end) else {
            return;
        };
        if self.selection_mode == SelectionCombineMode::Replace {
            self.execute_canvas_command(CanvasCommand::SetRectSelection(rect));
        } else {
            self.execute_canvas_command(CanvasCommand::CombineRectSelection {
                rect,
                mode: self.selection_mode,
            });
        }
    }

    fn apply_lasso_selection(&mut self) {
        if self.lasso_points.len() < 3 {
            return;
        }
        let polygon = SelectionPolygon::new(self.lasso_points.clone());
        if self.selection_mode == SelectionCombineMode::Replace {
            self.execute_canvas_command(CanvasCommand::SetPolygonSelection(polygon));
        } else {
            self.execute_canvas_command(CanvasCommand::CombinePolygonSelection {
                polygon,
                mode: self.selection_mode,
            });
        }
    }

    fn current_brush_dynamics(&self) -> BrushDynamics {
        let preset = self.brush_preset;
        BrushDynamics {
            size: if preset.pressure_controls_size() {
                PressureCurve::linear(0.05, 1.0)
            } else {
                PressureCurve::constant(1.0)
            },
            opacity: if preset.pressure_controls_opacity() {
                PressureCurve::linear(0.05, 1.0)
            } else {
                BrushDynamics::DEFAULT.opacity
            },
            flow: if preset.pressure_controls_flow() {
                PressureCurve::linear(0.05, 1.0)
            } else {
                BrushDynamics::DEFAULT.flow
            },
            velocity_size: if self.velocity_dynamics {
                PressureCurve::linear(1.25, 0.65)
            } else {
                BrushDynamics::DEFAULT.velocity_size
            },
            velocity_opacity: if self.velocity_dynamics {
                PressureCurve::linear(1.0, 0.55)
            } else {
                BrushDynamics::DEFAULT.velocity_opacity
            },
            velocity_flow: if self.velocity_dynamics {
                PressureCurve::linear(1.0, 0.7)
            } else {
                BrushDynamics::DEFAULT.velocity_flow
            },
            texture_grain: if self.texture_grain {
                PressureCurve::linear(0.15, 0.75)
            } else {
                BrushDynamics::DEFAULT.texture_grain
            },
            texture_scale: 18.0,
            stamp: BrushStamp {
                kind: self.stamp_kind,
                hardness: self.stamp_hardness,
                aspect: self.stamp_aspect,
                angle: self.stamp_angle,
                frequency: BrushStamp::CIRCLE.frequency,
                ..BrushStamp::CIRCLE
            },
            ..BrushDynamics::DEFAULT
        }
    }

    fn sync_core_brush_settings(&mut self) {
        let dynamics = self.current_brush_dynamics();
        let prediction = StrokePredictionConfig {
            enabled: self.core_prediction,
            time_horizon_seconds: 1.0 / 120.0,
            max_distance: self.brush_radius * 0.8,
            min_samples: 2,
        };
        let input = InputDeviceCapabilities {
            pressure_fallback: self.fallback_pressure,
            ..InputDeviceCapabilities::PEN
        };
        let result = {
            let mut core = self.core.lock().unwrap();
            core.set_brush_dynamics(dynamics);
            core.set_prediction_config(prediction);
            core.set_input_device_capabilities(input);
            if self.stamp_texture {
                if self.stamp_texture_id.is_none() {
                    let alpha = Self::stamp_texture_alpha(32, 32);
                    match core.create_brush_stamp_texture(32, 32, &alpha) {
                        Ok(id) => self.stamp_texture_id = Some(id),
                        Err(error) => return self.last_error = Some(error.to_string()),
                    }
                }
                core.set_active_brush_stamp_texture(self.stamp_texture_id)
            } else {
                core.set_active_brush_stamp_texture(None)
            }
        };
        match result {
            Ok(()) => self.last_error = None,
            Err(error) => self.last_error = Some(error.to_string()),
        }
    }

    fn stamp_texture_alpha(width: u32, height: u32) -> Vec<f32> {
        let cx = width as f32 * 0.5;
        let cy = height as f32 * 0.5;
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let dx = (x as f32 + 0.5 - cx).abs() / cx.max(1.0);
                    let dy = (y as f32 + 0.5 - cy).abs() / cy.max(1.0);
                    if dx < 0.22 || dy < 0.22 || (dx - dy).abs() < 0.12 {
                        1.0
                    } else {
                        0.0
                    }
                })
            })
            .collect()
    }

    fn input_pressure(force: Option<f32>) -> Option<f32> {
        force
            .filter(|value| value.is_finite())
            .map(|value| value.clamp(0.0, 1.0))
    }

    fn resolved_input_pressure(
        force: Option<f32>,
        phase: StylusPhase,
        last_pressure: f32,
        pressure_from_device: bool,
    ) -> Option<f32> {
        let pressure = Self::input_pressure(force);
        if pressure.is_some() {
            return pressure;
        }
        if pressure_from_device && matches!(phase, StylusPhase::Move | StylusPhase::Up) {
            return Some(last_pressure.clamp(0.0, 1.0));
        }
        None
    }

    fn input_calibration(&self) -> PressureCalibration {
        PressureCalibration {
            fallback_pressure: self.fallback_pressure,
            ..PressureCalibration::default()
        }
    }

    fn stylus_sample(
        &self,
        point: StrokePoint,
        phase: StylusPhase,
        force: Option<f32>,
        time_seconds: f64,
        batch_size: usize,
    ) -> StylusSample {
        StylusSample {
            device_id: DeviceId(0),
            tool_id: None,
            phase,
            time_seconds,
            x: point.x,
            y: point.y,
            pressure: Self::resolved_input_pressure(
                force,
                phase,
                self.last_pressure,
                self.pressure_from_device,
            ),
            pressure_raw: force.filter(|value| value.is_finite()),
            tilt_x: None,
            tilt_y: None,
            twist: None,
            buttons: StylusButtons {
                primary: true,
                ..StylusButtons::default()
            },
            eraser: self.brush_mode == BrushMode::Erase,
            backend: InputBackend::Synthetic,
            batch_size: batch_size.max(1),
        }
    }

    fn adapt_stylus_sample(&mut self, sample: StylusSample) -> StrokePoint {
        let pressure_from_device = sample.pressure.is_some();
        let adapted = StrokeSampleAdapter::new(self.input_calibration()).adapt(sample);
        self.last_pressure = adapted.pressure.max(0.01);
        self.pressure_from_device = pressure_from_device;
        adapted.into()
    }

    fn input_stroke_point(
        &mut self,
        pos: egui::Pos2,
        phase: StylusPhase,
        force: Option<f32>,
        viewport: CanvasViewport,
        time_seconds: f64,
        batch_size: usize,
    ) -> StrokePoint {
        let point = viewport.to_stroke_point(pos, self.fallback_pressure);
        let sample = self.stylus_sample(point, phase, force, time_seconds, batch_size);
        self.adapt_stylus_sample(sample)
    }

    fn current_input_point(
        &mut self,
        ui: &egui::Ui,
        pos: egui::Pos2,
        phase: StylusPhase,
        viewport: CanvasViewport,
    ) -> StrokePoint {
        let (force, time_seconds) = ui.input(|i| {
            let force = i.events.iter().rev().find_map(|event| match event {
                egui::Event::Touch {
                    phase: egui::TouchPhase::Start | egui::TouchPhase::Move | egui::TouchPhase::End,
                    force,
                    ..
                } => *force,
                _ => None,
            });
            (force, i.time)
        });
        self.input_stroke_point(pos, phase, force, viewport, time_seconds, 1)
    }

    fn record_adapted_pressure_range(&mut self, points: &[StrokePoint]) {
        self.audit.last_adapted_pressure_min = None;
        self.audit.last_adapted_pressure_max = None;
        for pressure in points
            .iter()
            .map(|point| point.pressure)
            .filter(|pressure| pressure.is_finite())
        {
            self.audit.last_adapted_pressure_min = Some(
                self.audit
                    .last_adapted_pressure_min
                    .map_or(pressure, |current| current.min(pressure)),
            );
            self.audit.last_adapted_pressure_max = Some(
                self.audit
                    .last_adapted_pressure_max
                    .map_or(pressure, |current| current.max(pressure)),
            );
        }
    }

    fn touch_events(ui: &egui::Ui) -> (Vec<(egui::Pos2, egui::TouchPhase, Option<f32>)>, f64) {
        ui.input(|i| {
            let events = i
                .events
                .iter()
                .filter_map(|event| match event {
                    egui::Event::Touch {
                        phase, pos, force, ..
                    } => Some((*pos, *phase, *force)),
                    _ => None,
                })
                .collect();
            (events, i.time)
        })
    }

    fn touch_begin_event(
        touch_events: &[(egui::Pos2, egui::TouchPhase, Option<f32>)],
    ) -> Option<(egui::Pos2, egui::TouchPhase, Option<f32>)> {
        touch_events
            .iter()
            .find(|(_, phase, _)| matches!(phase, egui::TouchPhase::Start))
            .or_else(|| {
                touch_events
                    .iter()
                    .find(|(_, phase, _)| touch_phase_can_begin_stroke(*phase))
            })
            .copied()
    }

    fn begin_input_point(
        &mut self,
        ui: &egui::Ui,
        fallback_pos: egui::Pos2,
        viewport: CanvasViewport,
    ) -> InputPoint {
        let (touch_events, time_seconds) = Self::touch_events(ui);
        if let Some((pos, _, force)) = Self::touch_begin_event(&touch_events) {
            let point =
                self.input_stroke_point(pos, StylusPhase::Down, force, viewport, time_seconds, 1);
            let pressure_from_device = Self::input_pressure(force).is_some();
            return InputPoint {
                point,
                source: StrokeInputSource::Touch,
                pressure_from_device,
            };
        }
        let point = self.current_input_point(ui, fallback_pos, StylusPhase::Down, viewport);
        let pressure_from_device = false;
        InputPoint {
            point,
            source: StrokeInputSource::PointerFallback,
            pressure_from_device,
        }
    }

    fn input_stroke_points(&mut self, ui: &egui::Ui, viewport: CanvasViewport) -> InputBatch {
        let (touch_events, time_seconds) = Self::touch_events(ui);

        self.audit.last_touch_events = touch_events.len();
        self.audit.last_touch_starts = touch_events
            .iter()
            .filter(|(_, phase, _)| matches!(phase, egui::TouchPhase::Start))
            .count();
        self.audit.last_touch_moves = touch_events
            .iter()
            .filter(|(_, phase, _)| matches!(phase, egui::TouchPhase::Move))
            .count();
        self.audit.last_touch_ends = touch_events
            .iter()
            .filter(|(_, phase, _)| matches!(phase, egui::TouchPhase::End))
            .count();
        self.audit.last_touch_cancels = touch_events
            .iter()
            .filter(|(_, phase, _)| matches!(phase, egui::TouchPhase::Cancel))
            .count();
        self.audit.ignored_touch_starts = self
            .audit
            .ignored_touch_starts
            .saturating_add(u32::try_from(self.audit.last_touch_starts).unwrap_or(u32::MAX));
        self.audit.last_raw_force_min = None;
        self.audit.last_raw_force_max = None;
        self.audit.last_raw_force_missing = 0;
        for (_, _, force) in &touch_events {
            if let Some(force) = force.filter(|value| value.is_finite()) {
                self.audit.last_raw_force_min = Some(
                    self.audit
                        .last_raw_force_min
                        .map_or(force, |current| current.min(force)),
                );
                self.audit.last_raw_force_max = Some(
                    self.audit
                        .last_raw_force_max
                        .map_or(force, |current| current.max(force)),
                );
            } else {
                self.audit.last_raw_force_missing =
                    self.audit.last_raw_force_missing.saturating_add(1);
            }
        }

        if !touch_events.is_empty() {
            let touch_events = touch_events
                .into_iter()
                .filter(|(_, phase, _)| touch_phase_appends_to_active_stroke(*phase))
                .collect::<Vec<_>>();
            let batch_size = touch_events.len();
            let mut points = Vec::with_capacity(batch_size);
            let mut first_pressure_from_device = false;
            for (pos, phase, force) in touch_events {
                let pressure_from_device = Self::input_pressure(force).is_some();
                let phase = match phase {
                    egui::TouchPhase::Start => StylusPhase::Down,
                    egui::TouchPhase::Move => StylusPhase::Move,
                    egui::TouchPhase::End => StylusPhase::Up,
                    egui::TouchPhase::Cancel => StylusPhase::Cancel,
                };
                let point =
                    self.input_stroke_point(pos, phase, force, viewport, time_seconds, batch_size);
                if points.is_empty() {
                    first_pressure_from_device = pressure_from_device;
                }
                points.push(point);
            }
            self.record_adapted_pressure_range(&points);
            return InputBatch {
                points,
                source: StrokeInputSource::Touch,
                first_pressure_from_device,
            };
        }

        let (pointer_events, time_seconds): (Vec<egui::Pos2>, f64) = ui.input(|i| {
            let events = i
                .events
                .iter()
                .filter_map(|event| match event {
                    egui::Event::PointerMoved(pos) => Some(*pos),
                    _ => None,
                })
                .collect();
            (events, i.time)
        });

        let batch_size = pointer_events.len();
        self.audit.last_touch_events = 0;
        self.audit.last_touch_starts = 0;
        self.audit.last_touch_moves = 0;
        self.audit.last_touch_ends = 0;
        self.audit.last_touch_cancels = 0;
        self.audit.last_raw_force_min = None;
        self.audit.last_raw_force_max = None;
        self.audit.last_raw_force_missing = 0;
        let mut points = Vec::with_capacity(batch_size);
        let mut first_pressure_from_device = false;
        for pos in pointer_events {
            let point = self.input_stroke_point(
                pos,
                StylusPhase::Move,
                None,
                viewport,
                time_seconds,
                batch_size,
            );
            if points.is_empty() {
                first_pressure_from_device = false;
            }
            points.push(point);
        }
        self.record_adapted_pressure_range(&points);
        InputBatch {
            points,
            source: StrokeInputSource::PointerFallback,
            first_pressure_from_device,
        }
    }

    fn should_reanchor_pointer_begin_to_touch(
        &self,
        source: StrokeInputSource,
        first: StrokePoint,
    ) -> bool {
        should_reanchor_pointer_begin_to_touch(
            source,
            self.active_stroke_input_source,
            self.active_stroke_dispatched,
            self.audit.begin_point,
            first,
            self.brush_radius,
        )
    }

    fn should_reanchor_touch_begin_pressure(
        &self,
        source: StrokeInputSource,
        first_pressure_from_device: bool,
    ) -> bool {
        should_reanchor_touch_begin_pressure(
            source,
            self.active_stroke_input_source,
            self.active_stroke_dispatched,
            self.audit.begin_pressure_from_device,
            first_pressure_from_device,
        )
    }

    fn reanchor_active_stroke_to_touch(
        &mut self,
        first: StrokePoint,
        first_pressure_from_device: bool,
    ) -> Result<(), String> {
        if let Some(id) = self.active_stroke.take() {
            let mut core = self.core.lock().unwrap();
            core.execute(&CanvasCommand::EndStroke(id))
                .map_err(|error| error.to_string())?;
        }

        self.sync_core_brush_settings();
        let command = CanvasCommand::BeginStroke(BeginStrokeCommand {
            layer: self.layer_id,
            mode: self.brush_mode,
            radius: self.brush_radius,
            color: self.color(),
            first_point: first,
        });
        let output = {
            let mut core = self.core.lock().unwrap();
            core.execute(&command).map_err(|error| error.to_string())?
        };
        match output {
            CommandOutput::StrokeStarted { id } => {
                self.active_stroke = Some(id);
                self.active_stroke_input_source = Some(StrokeInputSource::Touch);
                self.active_stroke_dispatched = false;
                self.audit.begin_point = Some(first);
                self.audit.begin_pressure_from_device = first_pressure_from_device;
                self.audit.begin_input_source = Some(StrokeInputSource::Touch);
                self.audit.reanchored_touch_begins =
                    self.audit.reanchored_touch_begins.saturating_add(1);
                self.begin_tap_candidate(first);
                Ok(())
            }
            _ => Err("画笔命令未能重新开始笔画".to_owned()),
        }
    }

    fn begin_tap_candidate(&mut self, anchor: StrokePoint) {
        self.active_tap_candidate = Some(TapCandidate {
            anchor,
            max_pressure: anchor.pressure.clamp(0.0, 1.0),
            max_distance: 0.0,
        });
    }

    fn reset_active_stroke_tracking(&mut self) {
        self.active_stroke_input_source = None;
        self.active_stroke_dispatched = false;
        self.active_tap_candidate = None;
    }

    fn update_tap_candidate(&mut self, points: &[StrokePoint]) {
        let Some(candidate) = self.active_tap_candidate.as_mut() else {
            return;
        };
        for point in points.iter().copied() {
            candidate.max_pressure = candidate.max_pressure.max(point.pressure.clamp(0.0, 1.0));
            candidate.max_distance = candidate
                .max_distance
                .max(stroke_point_distance(candidate.anchor, point));
        }
    }

    fn batch_is_tap_jitter(&self, source: StrokeInputSource, points: &[StrokePoint]) -> bool {
        if self.active_stroke_dispatched || source != StrokeInputSource::Touch {
            return false;
        }
        let Some(candidate) = self.active_tap_candidate else {
            return false;
        };
        points_within_tap_jitter(candidate, points, self.brush_radius)
    }

    fn register_presentation_texture(
        &mut self,
        render_state: &eframe::egui_wgpu::RenderState,
    ) -> Result<(), String> {
        let texture_view = {
            let core = self.core.lock().unwrap();
            core.presentation_texture_view()
                .ok_or_else(|| "缺少呈现纹理视图".to_owned())?
        };

        let mut renderer = render_state.renderer.write();
        if let Some(texture_id) = self.texture_id.take() {
            renderer.free_texture(&texture_id);
        }
        self.texture_id = Some(renderer.register_native_texture_with_sampler_options(
            &render_state.device,
            &texture_view,
            wgpu::SamplerDescriptor {
                label: Some("canvas presentation sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            },
        ));
        Ok(())
    }

    fn create_new_canvas(&mut self, frame: &mut eframe::Frame) {
        let Some(render_state) = frame.wgpu_render_state() else {
            self.last_error = Some("无法访问 WGPU 渲染状态".to_owned());
            return;
        };

        let mut config = self.core.lock().unwrap().config();
        config.width = self.new_canvas_width.max(1);
        config.height = self.new_canvas_height.max(1);
        config.tile_size = self.new_canvas_tile_size.max(1);
        config.stabilizer.enabled = true;
        config.stabilizer.mode = self.stab_mode;
        config.stabilizer.smooth_strength = self.smooth_strength;

        let result = {
            let mut core = self.core.lock().unwrap();
            core.execute(&CanvasCommand::NewCanvas(config))
                .and_then(|_| core.execute(&CanvasCommand::AddLayer(LayerSpec::new(LayerId(1)))))
                .and_then(|_| core.composite().map(|_| CommandOutput::None))
                .and_then(|_| core.present().map(|()| CommandOutput::None))
        };

        match result {
            Ok(_) => {
                self.layer_id = LayerId(1);
                self.next_layer_id = 2;
                self.canvas_width = config.width as f32;
                self.canvas_height = config.height as f32;
                self.active_stroke = None;
                self.reset_active_stroke_tracking();
                self.last_cursor = None;
                self.pred_vel = egui::Vec2::ZERO;
                self.zoom = 1.0;
                self.pan = egui::Vec2::ZERO;
                self.audit.reset();
                self.last_pressure = self.fallback_pressure;
                self.pressure_from_device = false;
                self.stamp_texture_id = None;
                self.trace_base_layers = self.core.lock().unwrap().layer_snapshot();
                if let Err(error) = self.register_presentation_texture(render_state) {
                    self.last_error = Some(error);
                } else {
                    self.last_error = None;
                    self.sync_core_brush_settings();
                }
            }
            Err(error) => {
                self.last_error = Some(error.to_string());
            }
        }
    }

    fn apply_trace_config(&mut self) {
        let capture = capture_config(self.trace_enabled, self.trace_level, self.trace_budget_mb);
        let mut core = self.core.lock().unwrap();
        if let Some(trace) = core.session_trace().cloned() {
            self.last_trace = Some(trace);
        }
        if self.trace_enabled {
            self.trace_base_layers = core.layer_snapshot();
        }
        core.set_data_capture_config(capture);
        if self.trace_enabled {
            self.trace_rebuild_result = None;
        }
        self.last_error = None;
    }

    fn trace_report(&self) -> Option<TraceReport> {
        self.core
            .lock()
            .unwrap()
            .trace_report()
            .or_else(|| self.last_trace.as_ref().map(|trace| trace.report.clone()))
    }

    fn rebuild_from_trace(&mut self, frame: &mut eframe::Frame) {
        let Some(render_state) = frame.wgpu_render_state() else {
            self.trace_rebuild_result = Some("无法访问 WGPU 渲染状态，不能显示重构结果".to_owned());
            return;
        };
        let trace = self
            .core
            .lock()
            .unwrap()
            .session_trace()
            .cloned()
            .or_else(|| self.last_trace.clone());
        let Some(trace) = trace else {
            self.trace_rebuild_result = Some(
                trace_rebuild_unavailable_message(
                    self.trace_enabled,
                    false,
                    self.last_trace.is_some(),
                )
                .to_owned(),
            );
            return;
        };

        let base_layers = self.trace_base_layers.clone();
        let result = pollster::block_on(async {
            let original = {
                let mut core = self.core.lock().unwrap();
                core.composite()?;
                core.debug_readback().await?
            };

            let rebuilt = {
                let mut core = self.core.lock().unwrap();
                replay_trace_commands_into_core(&mut core, &trace, &base_layers)?;
                core.composite()?;
                core.present()?;
                core.debug_readback().await?
            };
            let max_diff = max_pixel_abs_diff(&original.rgba_f32, &rebuilt.rgba_f32)
                .ok_or(CanvasError::InvalidCanvasSize)?;
            Ok::<f32, CanvasError>(max_diff)
        });

        self.trace_rebuild_result = Some(match result {
            Ok(max_diff) => {
                self.layer_id = self
                    .core
                    .lock()
                    .unwrap()
                    .layer_snapshot()
                    .last()
                    .map_or(LayerId(1), |layer| layer.spec.id);
                self.next_layer_id = self
                    .core
                    .lock()
                    .unwrap()
                    .layer_snapshot()
                    .iter()
                    .map(|layer| layer.spec.id.0)
                    .max()
                    .unwrap_or(0)
                    .saturating_add(1);
                self.canvas_width = trace.canvas.width as f32;
                self.canvas_height = trace.canvas.height as f32;
                self.trace_enabled = false;
                self.trace_base_layers = self.core.lock().unwrap().layer_snapshot();
                if let Err(error) = self.register_presentation_texture(render_state) {
                    trace_texture_registration_error_message(error)
                } else {
                    trace_rebuild_summary(&trace, max_diff)
                }
            }
            Err(error) => trace_rebuild_error_message(&error),
        });
    }
}

const CHINESE_FONT_PATHS: &[&str] = &[
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/Library/Fonts/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyh.ttf",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
];

const CHINESE_FONT_MISSING_WARNING: &str =
    "未找到可用中文字体，测试程序中文界面可能显示为方框或乱码";

fn install_chinese_fonts(ctx: &egui::Context) {
    let Some((font_path, font_bytes)) = CHINESE_FONT_PATHS
        .iter()
        .find_map(|path| fs::read(path).ok().map(|bytes| (*path, bytes)))
    else {
        eprintln!("{CHINESE_FONT_MISSING_WARNING}");
        return;
    };

    let font_name = "system_chinese".to_owned();
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        font_name.clone(),
        Arc::new(egui::FontData::from_owned(font_bytes)),
    );

    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, font_name.clone());
    }

    ctx.set_fonts(fonts);
    eprintln!("已加载中文字体: {font_path}");
}

const BLEND_MODES: [BlendMode; 15] = [
    BlendMode::Normal,
    BlendMode::Multiply,
    BlendMode::Screen,
    BlendMode::Overlay,
    BlendMode::SoftLight,
    BlendMode::ColorDodge,
    BlendMode::ColorBurn,
    BlendMode::LinearBurn,
    BlendMode::Add,
    BlendMode::Subtract,
    BlendMode::Difference,
    BlendMode::Darken,
    BlendMode::Lighten,
    BlendMode::HardLight,
    BlendMode::Exclusion,
];

fn blend_mode_label(mode: BlendMode) -> &'static str {
    match mode {
        BlendMode::Normal => "正常",
        BlendMode::Multiply => "正片叠底",
        BlendMode::Screen => "滤色",
        BlendMode::Overlay => "叠加",
        BlendMode::SoftLight => "柔光",
        BlendMode::ColorDodge => "颜色减淡",
        BlendMode::LinearBurn => "线性加深",
        BlendMode::ColorBurn => "颜色加深",
        BlendMode::Add => "添加",
        BlendMode::Difference => "差值",
        BlendMode::Subtract => "减去",
        BlendMode::Darken => "变暗",
        BlendMode::Lighten => "变亮",
        BlendMode::HardLight => "强光",
        BlendMode::Exclusion => "排除",
    }
}

fn group_mode_label(mode: LayerGroupMode) -> &'static str {
    match mode {
        LayerGroupMode::PassThrough => "穿透",
        LayerGroupMode::Isolated => "隔离",
    }
}

fn layer_label(id: LayerId) -> String {
    format!("图层 {}", id.0)
}

impl CanvasApp {
    fn show_layer_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("图层");

        let layers = self.layers();
        let selected_index = layers
            .iter()
            .position(|layer| layer.spec.id == self.layer_id);

        ui.horizontal(|ui| {
            if ui.button("新建").clicked() {
                let id = LayerId(self.next_layer_id);
                self.next_layer_id = self.next_layer_id.saturating_add(1);
                self.layer_id = id;
                self.execute_canvas_command(CanvasCommand::AddLayer(LayerSpec::new(id)));
            }

            if ui
                .add_enabled(layers.len() > 1, egui::Button::new("删除"))
                .clicked()
            {
                self.execute_canvas_command(CanvasCommand::RemoveLayer(self.layer_id));
            }

            if ui.button("清空").clicked() {
                self.execute_canvas_command(CanvasCommand::ClearLayer(self.layer_id));
            }
        });

        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    selected_index.is_some_and(|index| index + 1 < layers.len()),
                    egui::Button::new("上移"),
                )
                .clicked()
                && let Some(index) = selected_index
            {
                self.execute_canvas_command(CanvasCommand::MoveLayer {
                    id: self.layer_id,
                    to_index: index + 1,
                });
            }

            if ui
                .add_enabled(
                    selected_index.is_some_and(|index| index > 0),
                    egui::Button::new("下移"),
                )
                .clicked()
                && let Some(index) = selected_index
            {
                self.execute_canvas_command(CanvasCommand::MoveLayer {
                    id: self.layer_id,
                    to_index: index - 1,
                });
            }
        });

        egui::ScrollArea::vertical()
            .max_height(120.0)
            .show(ui, |ui| {
                for layer in layers.iter().rev() {
                    let marker = if layer.spec.visible {
                        "可见"
                    } else {
                        "隐藏"
                    };
                    let response = ui.selectable_label(
                        layer.spec.id == self.layer_id,
                        format!(
                            "{marker} {}  {:.0}%  {} 个 tile",
                            layer_label(layer.spec.id),
                            layer.spec.opacity * 100.0,
                            layer.tile_count
                        ),
                    );
                    if response.clicked() {
                        self.layer_id = layer.spec.id;
                    }
                }
            });

        if let Some(layer) = layers
            .iter()
            .find(|layer| layer.spec.id == self.layer_id)
            .cloned()
        {
            let mut spec = layer.spec.clone();
            let old_spec = spec.clone();

            ui.add(egui::Slider::new(&mut spec.opacity, 0.0..=1.0).text("不透明度"));
            ui.checkbox(&mut spec.alpha_locked, "锁定透明度");
            ui.checkbox(&mut spec.clip_to_below, "裁剪到下方图层");

            egui::ComboBox::from_label("混合模式")
                .selected_text(blend_mode_label(spec.blend_mode))
                .show_ui(ui, |ui| {
                    for mode in BLEND_MODES {
                        ui.selectable_value(&mut spec.blend_mode, mode, blend_mode_label(mode));
                    }
                });
            egui::ComboBox::from_label("蒙版图层")
                .selected_text(spec.mask_layer.map_or_else(|| "无".to_owned(), layer_label))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut spec.mask_layer, None, "无");
                    for candidate in layers
                        .iter()
                        .map(|layer| layer.spec.id)
                        .filter(|id| *id != spec.id)
                    {
                        ui.selectable_value(
                            &mut spec.mask_layer,
                            Some(candidate),
                            layer_label(candidate),
                        );
                    }
                });

            if spec != old_spec {
                self.execute_canvas_command(CanvasCommand::SetLayer(spec));
            }

            let mut parent = layer.parent;
            let old_parent = parent;
            egui::ComboBox::from_label("父图层")
                .selected_text(parent.map_or_else(|| "无".to_owned(), layer_label))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut parent, None, "无");
                    for candidate in layers
                        .iter()
                        .map(|layer| layer.spec.id)
                        .filter(|id| *id != self.layer_id)
                    {
                        ui.selectable_value(&mut parent, Some(candidate), layer_label(candidate));
                    }
                });
            if parent != old_parent {
                self.execute_canvas_command(CanvasCommand::SetLayerParent {
                    id: self.layer_id,
                    parent,
                });
            }

            let mut group_mode = layer.group_mode;
            let old_group_mode = group_mode;
            egui::ComboBox::from_label("组模式")
                .selected_text(group_mode_label(group_mode))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut group_mode,
                        LayerGroupMode::Isolated,
                        group_mode_label(LayerGroupMode::Isolated),
                    );
                    ui.selectable_value(
                        &mut group_mode,
                        LayerGroupMode::PassThrough,
                        group_mode_label(LayerGroupMode::PassThrough),
                    );
                });
            if group_mode != old_group_mode {
                self.execute_canvas_command(CanvasCommand::SetLayerGroupMode {
                    id: self.layer_id,
                    mode: group_mode,
                });
            }
        }

        if let Some(error) = &self.last_error {
            ui.colored_label(egui::Color32::from_rgb(180, 40, 40), error);
        }
    }
}

impl eframe::App for CanvasApp {
    #[allow(clippy::too_many_lines)]
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        egui::Panel::left("controls").show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("工具");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.tool_mode, ToolMode::Brush, "画笔");
                    ui.selectable_value(&mut self.tool_mode, ToolMode::RectSelection, "矩形");
                    ui.selectable_value(&mut self.tool_mode, ToolMode::LassoSelection, "套索");
                });
                egui::ComboBox::from_label("选区组合")
                    .selected_text(selection_mode_label(self.selection_mode))
                    .show_ui(ui, |ui| {
                        for mode in [
                            SelectionCombineMode::Replace,
                            SelectionCombineMode::Add,
                            SelectionCombineMode::Subtract,
                            SelectionCombineMode::Intersect,
                        ] {
                            ui.selectable_value(
                                &mut self.selection_mode,
                                mode,
                                selection_mode_label(mode),
                            );
                        }
                    });
                ui.separator();

                ui.heading("画笔");
                ui.add(egui::Slider::new(&mut self.brush_radius, 1.0..=50.0).text("大小"));
                let mut preset = self.brush_preset;
                egui::ComboBox::from_label("PS 画笔")
                    .selected_text(brush_preset_label(preset))
                    .show_ui(ui, |ui| {
                        for option in BrushPreset::ALL {
                            ui.selectable_value(&mut preset, option, brush_preset_label(option));
                        }
                    });
                if preset != self.brush_preset {
                    self.apply_brush_preset(preset);
                }
                if ui
                    .add(egui::Slider::new(&mut self.smooth_strength, 0.0..=1.0).text("平滑"))
                    .changed()
                {
                    let mut core = self.core.lock().unwrap();
                    core.set_stabilizer_strength(self.smooth_strength);
                }
                ui.horizontal(|ui| {
                    ui.label("稳定器");
                    let mut changed = false;
                    changed |= ui
                        .selectable_value(&mut self.stab_mode, StabilizerMode::Adaptive, "自适应")
                        .changed();
                    changed |= ui
                        .selectable_value(&mut self.stab_mode, StabilizerMode::PulledString, "拉绳")
                        .changed();
                    if changed {
                        let mut core = self.core.lock().unwrap();
                        core.set_stabilizer_mode(self.stab_mode);
                    }
                });

                ui.label("颜色");
                ui.color_edit_button_rgba_unmultiplied(&mut self.brush_color);

                ui.add(egui::Slider::new(&mut self.fallback_pressure, 0.01..=1.0).text("模拟压力"));
                ui.label(format!(
                    "压力: {:.2} ({})",
                    self.last_pressure,
                    if self.pressure_from_device {
                        "设备"
                    } else {
                        "模拟"
                    }
                ));

                ui.checkbox(&mut self.prediction, "笔尖预览");
                ui.checkbox(&mut self.core_prediction, "核心预测尾段");

                ui.separator();
                ui.heading("画笔动态");
                ui.label(format!(
                    "压力映射: {}",
                    if self.pressure_dynamics {
                        "PS 预设"
                    } else {
                        "关闭"
                    }
                ));
                ui.checkbox(&mut self.velocity_dynamics, "速度动态");
                ui.checkbox(&mut self.texture_grain, "压力纹理颗粒");
                egui::ComboBox::from_label("笔尖形状")
                    .selected_text(stamp_kind_label(self.stamp_kind))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.stamp_kind,
                            BrushStampKind::Circle,
                            stamp_kind_label(BrushStampKind::Circle),
                        );
                        ui.selectable_value(
                            &mut self.stamp_kind,
                            BrushStampKind::Square,
                            stamp_kind_label(BrushStampKind::Square),
                        );
                        ui.selectable_value(
                            &mut self.stamp_kind,
                            BrushStampKind::Diamond,
                            stamp_kind_label(BrushStampKind::Diamond),
                        );
                        ui.selectable_value(
                            &mut self.stamp_kind,
                            BrushStampKind::Stripe,
                            stamp_kind_label(BrushStampKind::Stripe),
                        );
                    });
                ui.add(egui::Slider::new(&mut self.stamp_hardness, 0.0..=1.0).text("笔尖硬度"));
                ui.add(egui::Slider::new(&mut self.stamp_aspect, 0.125..=4.0).text("宽高比"));
                ui.add(egui::Slider::new(&mut self.stamp_angle, 0.0..=1.0).text("笔尖角度"));
                ui.checkbox(&mut self.stamp_texture, "外部笔尖纹理遮罩");
                self.sync_core_brush_settings();
                ui.label("画笔模式");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.brush_mode, BrushMode::Paint, "绘制");
                    ui.selectable_value(&mut self.brush_mode, BrushMode::Erase, "擦除");
                });

                ui.separator();
                ui.heading("选区");
                let selection_report = self.core.lock().unwrap().selection_report();
                if let Some(bounds) = selection_report.bounds {
                    ui.label(format!(
                        "范围: {},{} {}x{}  分段={}",
                        bounds.x,
                        bounds.y,
                        bounds.width,
                        bounds.height,
                        selection_report.span_count
                    ));
                } else {
                    ui.label("无活动选区");
                }
                ui.horizontal(|ui| {
                    if ui.button("清除选区").clicked() {
                        self.execute_canvas_command(CanvasCommand::ClearSelection);
                    }
                    if ui.button("反选").clicked() {
                        self.execute_canvas_command(CanvasCommand::InvertSelection);
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("清空选中像素").clicked() {
                        self.execute_canvas_command(CanvasCommand::ClearSelectedPixels(
                            self.layer_id,
                        ));
                    }
                    if ui.button("复制").clicked() {
                        self.copy_selection_to_clipboard(false);
                    }
                    if ui.button("剪切").clicked() {
                        self.copy_selection_to_clipboard(true);
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("粘贴").clicked() {
                        self.paste_clipboard_to_active_layer();
                    }
                    if ui.button("粘贴为新图层").clicked() {
                        self.paste_clipboard_to_new_layer();
                    }
                });
                if let Some(pixels) = &self.clipboard {
                    ui.label(format!(
                        "剪贴板: {}x{} 来自图层 {}",
                        pixels.bounds.width, pixels.bounds.height, pixels.layer.0
                    ));
                } else {
                    ui.label("剪贴板: 空");
                }

                ui.separator();

                if ui.button("清空当前图层").clicked() {
                    self.execute_canvas_command(CanvasCommand::ClearLayer(self.layer_id));
                }
                if ui.button("撤销").clicked() {
                    self.execute_canvas_command(CanvasCommand::Undo);
                }
                if ui.button("重做").clicked() {
                    self.execute_canvas_command(CanvasCommand::Redo);
                }

                ui.separator();
                ui.label("使用鼠标或数位板输入。");

                ui.separator();
                ui.heading("新建画布");
                ui.horizontal(|ui| {
                    ui.label("宽");
                    ui.add(egui::DragValue::new(&mut self.new_canvas_width).range(1..=8192));
                    ui.label("高");
                    ui.add(egui::DragValue::new(&mut self.new_canvas_height).range(1..=8192));
                });
                ui.horizontal(|ui| {
                    ui.label("分块");
                    ui.add(egui::DragValue::new(&mut self.new_canvas_tile_size).range(1..=1024));
                    if ui.button("创建").clicked() {
                        self.create_new_canvas(frame);
                    }
                });
                ui.label(format!(
                    "当前画布: {} x {}",
                    self.canvas_width as u32, self.canvas_height as u32
                ));

                ui.separator();
                ui.heading("缩放");
                ui.add(egui::Slider::new(&mut self.zoom, 1.0..=32.0).text("缩放"));
                ui.horizontal(|ui| {
                    if ui.button("适应").clicked() {
                        self.zoom = 1.0;
                        self.pan = egui::Vec2::ZERO;
                    }
                    if ui.button("4x").clicked() {
                        self.zoom = 4.0;
                    }
                    if ui.button("16x").clicked() {
                        self.zoom = 16.0;
                    }
                });
                ui.label("鼠标滚轮缩放");
                ui.label("右键拖拽平移");

                ui.separator();
                self.show_layer_panel(ui);

                ui.separator();
                ui.heading("输入审计");
                ui.checkbox(&mut self.debug_overlay, "显示原始输入叠加层");
                let a = &self.audit;
                let rate = if a.frames > 0 {
                    f64::from(a.raw_events) / f64::from(a.frames)
                } else {
                    0.0
                };
                ui.label(format!("帧数: {}", a.frames));
                ui.label(format!("原始事件: {}", a.raw_events));
                ui.label(format!("每帧事件: {rate:.2}"));
                ui.label(format!("最大间距: {:.1}px", a.max_event_gap));
                ui.label(format!("平均间距: {:.1}px", a.avg_event_gap()));
                ui.label(format!(
                    "帧间隔: {:.1}ms（最大 {:.1}）",
                    a.last_frame_dt, a.max_frame_dt
                ));
                ui.label(format!(
                    "ignored Touch Start in append: {}",
                    a.ignored_touch_starts
                ));
                ui.label(format!(
                    "last touch phases: total={} start={} move={} end={} cancel={}",
                    a.last_touch_events,
                    a.last_touch_starts,
                    a.last_touch_moves,
                    a.last_touch_ends,
                    a.last_touch_cancels
                ));
                ui.label(format!(
                    "raw force: min={} max={} missing={}",
                    optional_f32_debug(a.last_raw_force_min),
                    optional_f32_debug(a.last_raw_force_max),
                    a.last_raw_force_missing
                ));
                ui.label(format!(
                    "adapted pressure: min={} max={}",
                    optional_f32_debug(a.last_adapted_pressure_min),
                    optional_f32_debug(a.last_adapted_pressure_max)
                ));
                ui.label(format!(
                    "begin: {} ({})",
                    stroke_point_debug(a.begin_point),
                    if a.begin_pressure_from_device {
                        "device"
                    } else {
                        "fallback"
                    }
                ));
                ui.label(format!(
                    "begin source: {}, touch reanchors: {}",
                    input_source_label(a.begin_input_source),
                    a.reanchored_touch_begins
                ));
                ui.label(format!(
                    "append batches: {}, last len: {}",
                    a.append_batches, a.last_append_len
                ));
                ui.label(format!(
                    "append[0]: {}",
                    stroke_point_debug(a.last_append_first)
                ));
                ui.label(format!(
                    "append[1]: {}",
                    stroke_point_debug(a.last_append_second)
                ));
                ui.label(format!(
                    "core upload/dispatch: {}/{}",
                    a.last_core_uploaded_points, a.last_core_dispatches
                ));
                ui.label("红色 = 原始输入点");

                ui.separator();
                ui.heading("绘画记录");
                let mut trace_changed = ui.checkbox(&mut self.trace_enabled, "启用记录").changed();
                egui::ComboBox::from_label("记录等级")
                    .selected_text(capture_level_label(self.trace_level))
                    .show_ui(ui, |ui| {
                        for level in CAPTURE_LEVELS {
                            trace_changed |= ui
                                .selectable_value(
                                    &mut self.trace_level,
                                    level,
                                    capture_level_label(level),
                                )
                                .changed();
                        }
                    });
                trace_changed |= ui
                    .add(egui::Slider::new(&mut self.trace_budget_mb, 1..=128).text("会话预算 MB"))
                    .changed();
                if trace_changed {
                    self.apply_trace_config();
                }
                ui.horizontal(|ui| {
                    if ui.button("重新开始记录").clicked() {
                        self.trace_enabled = true;
                        self.apply_trace_config();
                    }
                    if ui.button("停止记录").clicked() {
                        self.trace_enabled = false;
                        self.apply_trace_config();
                    }
                });
                if ui.button("通过记录重构并校验").clicked() {
                    self.rebuild_from_trace(frame);
                }
                let has_current_trace = self.core.lock().unwrap().session_trace().is_some();
                ui.label(trace_status_label(
                    self.trace_enabled,
                    has_current_trace,
                    self.last_trace.is_some(),
                ));
                if let Some(report) = self.trace_report() {
                    ui.label(trace_report_summary(&report));
                }
                if let Some(result) = &self.trace_rebuild_result {
                    ui.label(result);
                }
            });
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let available = ui.available_size();
            let aspect = self.canvas_width / self.canvas_height.max(1.0);
            let mut display_width = available.x;
            let mut display_height = display_width / aspect;
            if display_height > available.y {
                display_height = available.y;
                display_width = display_height * aspect;
            }
            let (rect, response) = ui.allocate_exact_size(
                egui::vec2(display_width, display_height),
                egui::Sense::drag(),
            );

            // display_scale: screen pixels per canvas pixel
            let mut display_scale = display_width / self.canvas_width * self.zoom;
            let mut max_pan_x = (self.canvas_width * (1.0 - 1.0 / self.zoom)).max(0.0);
            let mut max_pan_y = (self.canvas_height * (1.0 - 1.0 / self.zoom)).max(0.0);
            let mut viewport_width = self.canvas_width / self.zoom;
            let mut viewport_height = self.canvas_height / self.zoom;
            self.pan.x = self.pan.x.clamp(0.0, max_pan_x);
            self.pan.y = self.pan.y.clamp(0.0, max_pan_y);

            // Mouse wheel zoom around the canvas point under the cursor.
            let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll_delta != 0.0
                && rect.contains(
                    ui.input(|i| i.pointer.hover_pos())
                        .unwrap_or(egui::Pos2::ZERO),
                )
            {
                let hover = ui.input(|i| i.pointer.hover_pos()).unwrap_or(rect.min);
                if hover.x >= rect.min.x
                    && hover.x <= rect.max.x
                    && hover.y >= rect.min.y
                    && hover.y <= rect.max.y
                {
                    let cuc = egui::vec2(
                        self.pan.x + (hover.x - rect.min.x) / display_scale,
                        self.pan.y + (hover.y - rect.min.y) / display_scale,
                    );
                    let zoom_factor = 2.0_f32.powf((scroll_delta / 600.0).clamp(-1.0, 1.0));
                    self.zoom = (self.zoom * zoom_factor).clamp(1.0, 32.0);
                    display_scale = display_width / self.canvas_width * self.zoom;
                    max_pan_x = (self.canvas_width * (1.0 - 1.0 / self.zoom)).max(0.0);
                    max_pan_y = (self.canvas_height * (1.0 - 1.0 / self.zoom)).max(0.0);
                    viewport_width = self.canvas_width / self.zoom;
                    viewport_height = self.canvas_height / self.zoom;
                    self.pan.x =
                        (cuc.x - (hover.x - rect.min.x) / display_scale).clamp(0.0, max_pan_x);
                    self.pan.y =
                        (cuc.y - (hover.y - rect.min.y) / display_scale).clamp(0.0, max_pan_y);
                }
            }

            // Right-drag to pan
            if response.dragged_by(egui::PointerButton::Secondary) {
                let delta = response.drag_delta();
                let pan_delta = egui::vec2(delta.x / display_scale, delta.y / display_scale);
                self.pan.x = (self.pan.x - pan_delta.x).clamp(0.0, max_pan_x);
                self.pan.y = (self.pan.y - pan_delta.y).clamp(0.0, max_pan_y);
            }

            let pan_x = self.pan.x;
            let pan_y = self.pan.y;
            let viewport = CanvasViewport {
                origin: rect.min,
                display_scale,
                pan: self.pan,
            };
            let to_canvas = |sp: egui::Pos2, pressure: f32| StrokePoint {
                x: pan_x + (sp.x - rect.min.x) / display_scale,
                y: pan_y + (sp.y - rect.min.y) / display_scale,
                pressure,
            };

            // Stroke start.
            let primary_drag_started = response.drag_started_by(egui::PointerButton::Primary);

            if primary_drag_started {
                self.audit.reset();
                self.last_cursor = response.interact_pointer_pos();
                self.pred_vel = egui::Vec2::ZERO;
                if let Some(pos) = response.interact_pointer_pos() {
                    match self.tool_mode {
                        ToolMode::Brush => {
                            let input = self.begin_input_point(ui, pos, viewport);
                            let p = input.point;
                            self.audit.begin_point = Some(p);
                            self.audit.begin_pressure_from_device = input.pressure_from_device;
                            self.audit.begin_input_source = Some(input.source);
                            self.sync_core_brush_settings();
                            let command = CanvasCommand::BeginStroke(BeginStrokeCommand {
                                layer: self.layer_id,
                                mode: self.brush_mode,
                                radius: self.brush_radius,
                                color: self.color(),
                                first_point: p,
                            });
                            let output = {
                                let mut core = self.core.lock().unwrap();
                                core.execute(&command)
                            };
                            match output {
                                Ok(CommandOutput::StrokeStarted { id }) => {
                                    self.active_stroke = Some(id);
                                    self.active_stroke_input_source = Some(input.source);
                                    self.active_stroke_dispatched = false;
                                    self.begin_tap_candidate(p);
                                    self.last_error = None;
                                }
                                Ok(_) => {
                                    self.active_stroke = None;
                                    self.reset_active_stroke_tracking();
                                    self.last_error = Some("画笔命令未能开始笔画".to_owned());
                                }
                                Err(error) => {
                                    self.active_stroke = None;
                                    self.reset_active_stroke_tracking();
                                    self.last_error = Some(error.to_string());
                                }
                            }
                        }
                        ToolMode::RectSelection => {
                            let point = Self::selection_point_from_canvas(to_canvas(pos, 1.0));
                            self.selection_drag = Some(SelectionDrag::Rect {
                                start: point,
                                current: point,
                            });
                        }
                        ToolMode::LassoSelection => {
                            self.lasso_points.clear();
                            let point = Self::selection_point_from_canvas(to_canvas(pos, 1.0));
                            self.lasso_points.push(point);
                            self.selection_drag = Some(SelectionDrag::Lasso);
                        }
                    }
                }
            }

            // Stroke update: collect high-rate input events and feed the streaming API.
            if response.dragged_by(egui::PointerButton::Primary) && !primary_drag_started {
                match self.tool_mode {
                    ToolMode::Brush => {
                        self.audit.frames += 1;
                        let frame_dt = ui.input(|i| i.stable_dt);
                        let frame_dt_ms = frame_dt * 1000.0;
                        self.audit.last_frame_dt = frame_dt_ms;
                        if frame_dt_ms > self.audit.max_frame_dt {
                            self.audit.max_frame_dt = frame_dt_ms;
                        }

                        // Preview prediction uses a smoothed screen-space velocity estimate.
                        if let Some(cur) = response.interact_pointer_pos() {
                            if let Some(prev) = self.last_cursor {
                                let dt = frame_dt.max(1e-4);
                                let inst_v = (cur - prev) / dt;
                                let a = 0.4;
                                self.pred_vel = self.pred_vel * (1.0 - a) + inst_v * a;
                            }
                            self.last_cursor = Some(cur);
                        }

                        let input_batch = self.input_stroke_points(ui, viewport);
                        let mut batch_source = input_batch.source;
                        let mut batch_first_pressure_from_device =
                            input_batch.first_pressure_from_device;
                        self.audit.raw_events = self.audit.raw_events.saturating_add(
                            u32::try_from(input_batch.points.len()).unwrap_or(u32::MAX),
                        );

                        let mut batch: Vec<StrokePoint> =
                            Vec::with_capacity(input_batch.points.len());
                        for cp in input_batch.points {
                            if let Some(prev) = self.audit.raw_pts.last() {
                                let dx = cp.x - prev.x;
                                let dy = cp.y - prev.y;
                                let gap = (dx * dx + dy * dy).sqrt();
                                self.audit.sum_event_gap += f64::from(gap);
                                self.audit.event_gap_count += 1;
                                if gap > self.audit.max_event_gap {
                                    self.audit.max_event_gap = gap;
                                }
                            }
                            self.audit.raw_pts.push(cp);
                            batch.push(cp);
                        }
                        if batch.is_empty()
                            && let Some(pos) = response.interact_pointer_pos()
                        {
                            batch.push(self.current_input_point(
                                ui,
                                pos,
                                StylusPhase::Move,
                                viewport,
                            ));
                            batch_source = StrokeInputSource::PointerFallback;
                            batch_first_pressure_from_device = false;
                        }

                        if let Some(first) = batch.first().copied() {
                            let should_reanchor = self
                                .should_reanchor_pointer_begin_to_touch(batch_source, first)
                                || self.should_reanchor_touch_begin_pressure(
                                    batch_source,
                                    batch_first_pressure_from_device,
                                );
                            if should_reanchor {
                                match self.reanchor_active_stroke_to_touch(
                                    first,
                                    batch_first_pressure_from_device,
                                ) {
                                    Ok(()) => {
                                        batch.remove(0);
                                    }
                                    Err(error) => {
                                        self.last_error = Some(error);
                                        batch.clear();
                                    }
                                }
                            }
                        }

                        self.update_tap_candidate(&batch);
                        if self.batch_is_tap_jitter(batch_source, &batch) {
                            batch.clear();
                        }

                        if let Some(id) = self.active_stroke
                            && !batch.is_empty()
                        {
                            self.audit.append_batches = self.audit.append_batches.saturating_add(1);
                            self.audit.last_append_len = batch.len();
                            self.audit.last_append_first = batch.first().copied();
                            self.audit.last_append_second = batch.get(1).copied();
                            let mut core = self.core.lock().unwrap();
                            let command = CanvasCommand::AppendStroke(AppendStrokeCommand {
                                stroke: id,
                                points: batch,
                            });
                            match core
                                .execute(&command)
                                .and_then(|_| Self::refresh(&mut core))
                            {
                                Ok(()) => {
                                    let dispatch = core.dispatch_report();
                                    self.audit.last_core_uploaded_points =
                                        dispatch.last_brush_uploaded_points;
                                    self.audit.last_core_dispatches =
                                        dispatch.last_brush_dispatches;
                                    if dispatch.last_brush_dispatches > 0 {
                                        self.active_stroke_dispatched = true;
                                        self.active_tap_candidate = None;
                                    }
                                    self.last_error = None;
                                }
                                Err(error) => self.last_error = Some(error.to_string()),
                            }
                        }
                    }
                    ToolMode::RectSelection => {
                        if let (Some(pos), Some(SelectionDrag::Rect { start, .. })) =
                            (response.interact_pointer_pos(), self.selection_drag)
                        {
                            let current = Self::selection_point_from_canvas(to_canvas(pos, 1.0));
                            self.selection_drag = Some(SelectionDrag::Rect { start, current });
                        }
                    }
                    ToolMode::LassoSelection => {
                        for input_point in self.input_stroke_points(ui, viewport).points {
                            let point = Self::selection_point_from_canvas(input_point);
                            if self.lasso_points.last().is_none_or(|last| {
                                let dx = last.x - point.x;
                                let dy = last.y - point.y;
                                (dx * dx + dy * dy).sqrt() >= 1.5
                            }) {
                                self.lasso_points.push(point);
                            }
                        }
                    }
                }
            }

            // Stroke/selection end.
            if response.drag_stopped() {
                match self.tool_mode {
                    ToolMode::Brush => {
                        if let Some(id) = self.active_stroke.take() {
                            let was_dispatched = self.active_stroke_dispatched;
                            let tap_candidate =
                                self.active_tap_candidate.take().filter(|candidate| {
                                    candidate.max_distance
                                        <= tap_jitter_threshold(self.brush_radius)
                                });
                            let tap_command = (!was_dispatched)
                                .then(|| {
                                    tap_candidate.map(|candidate| {
                                        let mut point = candidate.anchor;
                                        point.pressure = candidate.max_pressure.clamp(0.0, 1.0);
                                        CanvasCommand::Brush(BrushCommand {
                                            layer: self.layer_id,
                                            mode: self.brush_mode,
                                            points: vec![point],
                                            radius: self.brush_radius,
                                            color: self.color(),
                                        })
                                    })
                                })
                                .flatten();
                            self.reset_active_stroke_tracking();
                            let mut core = self.core.lock().unwrap();
                            let result =
                                core.execute(&CanvasCommand::EndStroke(id))
                                    .and_then(|output| {
                                        let committed = matches!(
                                            output,
                                            CommandOutput::StrokeEnded { committed: true }
                                        );
                                        if !committed && let Some(command) = &tap_command {
                                            core.execute(command)?;
                                        }
                                        Self::refresh(&mut core)
                                    });
                            match result {
                                Ok(()) => self.last_error = None,
                                Err(error) => self.last_error = Some(error.to_string()),
                            }
                        }
                    }
                    ToolMode::RectSelection => {
                        if let Some(SelectionDrag::Rect { start, current }) = self.selection_drag {
                            self.apply_rect_selection(start, current);
                        }
                    }
                    ToolMode::LassoSelection => {
                        self.apply_lasso_selection();
                    }
                }
                self.selection_drag = None;
                self.last_cursor = None;
                self.pred_vel = egui::Vec2::ZERO;
            }

            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(245, 245, 245));
            let checker = 16.0;
            let cols = (rect.width() / checker).ceil() as i32;
            let rows = (rect.height() / checker).ceil() as i32;
            for y in 0..rows {
                for x in 0..cols {
                    if (x + y) % 2 == 0 {
                        let min = rect.min + egui::vec2(x as f32 * checker, y as f32 * checker);
                        let max = egui::pos2(
                            (min.x + checker).min(rect.max.x),
                            (min.y + checker).min(rect.max.y),
                        );
                        painter.rect_filled(
                            egui::Rect::from_min_max(min, max),
                            0.0,
                            egui::Color32::from_rgb(224, 224, 224),
                        );
                    }
                }
            }

            // Draw the canvas texture with zoom/pan UV mapping
            if let Some(texture_id) = self.texture_id {
                let uv_min = egui::pos2(
                    self.pan.x / self.canvas_width,
                    self.pan.y / self.canvas_height,
                );
                let uv_max = egui::pos2(
                    (self.pan.x + viewport_width) / self.canvas_width,
                    (self.pan.y + viewport_height) / self.canvas_height,
                );
                painter.image(
                    texture_id,
                    rect,
                    egui::Rect::from_min_max(uv_min, uv_max),
                    egui::Color32::WHITE,
                );
            } else {
                painter.rect_filled(rect, 0.0, egui::Color32::WHITE);
            }
            painter.rect_stroke(
                rect,
                0.0,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 80, 80)),
                egui::StrokeKind::Inside,
            );

            let selection_to_screen = |p: SelectionPoint| -> egui::Pos2 {
                egui::pos2(
                    rect.min.x + (p.x - self.pan.x) * display_scale,
                    rect.min.y + (p.y - self.pan.y) * display_scale,
                )
            };
            let draw_selection_rect = |painter: &egui::Painter, bounds: SelectionRect| {
                let min =
                    selection_to_screen(SelectionPoint::new(bounds.x as f32, bounds.y as f32));
                let max = selection_to_screen(SelectionPoint::new(
                    (bounds.x + bounds.width) as f32,
                    (bounds.y + bounds.height) as f32,
                ));
                painter.rect_stroke(
                    egui::Rect::from_min_max(min, max),
                    0.0,
                    egui::Stroke::new(1.5, egui::Color32::from_rgb(0, 130, 220)),
                    egui::StrokeKind::Inside,
                );
            };
            if let Some(bounds) = self.core.lock().unwrap().selection_report().bounds {
                draw_selection_rect(&painter, bounds);
            }
            if let Some(SelectionDrag::Rect { start, current }) = self.selection_drag
                && let Some(bounds) = selection_rect_from_points(start, current)
            {
                draw_selection_rect(&painter, bounds);
            }
            if self.selection_drag == Some(SelectionDrag::Lasso) && self.lasso_points.len() > 1 {
                let points: Vec<egui::Pos2> = self
                    .lasso_points
                    .iter()
                    .copied()
                    .map(selection_to_screen)
                    .collect();
                painter.add(egui::Shape::line(
                    points,
                    egui::Stroke::new(1.5, egui::Color32::from_rgb(0, 130, 220)),
                ));
            }

            // Core prediction preview.
            if self.core_prediction
                && let Some(id) = self.active_stroke
            {
                let predicted = self.core.lock().unwrap().active_stroke_predicted_tip(id);
                if let Some(tip) = predicted {
                    let preview_center = egui::pos2(
                        rect.min.x + (tip.x - self.pan.x) * display_scale,
                        rect.min.y + (tip.y - self.pan.y) * display_scale,
                    );
                    let r = (self.brush_radius * display_scale).max(1.0);
                    let painter = ui.painter_at(rect);
                    painter.circle_stroke(
                        preview_center,
                        r,
                        egui::Stroke::new(1.5, egui::Color32::from_rgb(20, 180, 210)),
                    );
                }
            }

            if self.prediction
                && let Some(id) = self.active_stroke
            {
                let tip_canvas = self.core.lock().unwrap().active_stroke_tip(id);
                if let (Some(tip), Some(cursor)) = (tip_canvas, self.last_cursor) {
                    let tip_screen = egui::pos2(
                        rect.min.x + (tip.x - self.pan.x) * display_scale,
                        rect.min.y + (tip.y - self.pan.y) * display_scale,
                    );
                    // Match the core prediction horizon at about 1.5 frames at 60fps.
                    let horizon = 0.024_f32;
                    let speed = self.pred_vel.length();
                    let max_ext = (self.brush_radius * 4.0).max(24.0);
                    let mut ext = self.pred_vel * horizon;
                    if ext.length() > max_ext {
                        ext = ext.normalized() * max_ext;
                    }
                    let target = if speed > 5.0 { cursor + ext } else { cursor };
                    let target_to_tip = target - tip_screen;
                    let max_lead = (self.brush_radius * display_scale * 1.5).max(8.0);
                    let preview_center = if target_to_tip.length() > max_lead {
                        tip_screen + target_to_tip.normalized() * max_lead
                    } else {
                        target
                    };

                    let r = (self.brush_radius * display_scale).max(1.0);
                    let fill = egui::Color32::from(egui::Rgba::from_rgba_unmultiplied(
                        self.brush_color[0],
                        self.brush_color[1],
                        self.brush_color[2],
                        55.0 / 255.0,
                    ));
                    let stroke = egui::Color32::from(egui::Rgba::from_rgba_unmultiplied(
                        self.brush_color[0],
                        self.brush_color[1],
                        self.brush_color[2],
                        110.0 / 255.0,
                    ));
                    let painter = ui.painter_at(rect);
                    painter.circle_filled(preview_center, r, fill);
                    painter.circle_stroke(preview_center, r, egui::Stroke::new(1.0, stroke));
                }
            }

            // Zoom indicator
            ui.painter_at(rect).text(
                rect.min + egui::vec2(8.0, 8.0),
                egui::Align2::LEFT_TOP,
                format!("{:.1}x", self.zoom),
                egui::FontId::proportional(14.0),
                egui::Color32::BLACK,
            );

            // Audit overlay: raw input points (red)
            if self.debug_overlay {
                let painter = ui.painter_at(rect);
                let to_screen = |p: &StrokePoint| -> egui::Pos2 {
                    egui::pos2(
                        rect.min.x + (p.x - self.pan.x) * display_scale,
                        rect.min.y + (p.y - self.pan.y) * display_scale,
                    )
                };
                for p in &self.audit.raw_pts {
                    painter.circle_filled(to_screen(p), 2.5, egui::Color32::from_rgb(230, 40, 40));
                }
            }
        });

        ui.ctx().request_repaint();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_level_labels_are_chinese_and_explicit_about_degradation() {
        assert_eq!(
            capture_level_label(CaptureLevel::L0Structure),
            "L0 结构事件"
        );
        assert_eq!(
            capture_level_label(CaptureLevel::L1PointStream),
            "L1 笔触点流"
        );
        assert!(capture_level_label(CaptureLevel::L2PixelDelta).contains("当前降级"));
        assert!(capture_level_label(CaptureLevel::L3SparseCheckpoint).contains("当前降级"));
    }

    #[test]
    fn capture_config_is_conservative_for_test_app() {
        let disabled = capture_config(false, CaptureLevel::L1PointStream, 0);
        assert!(!disabled.enabled);
        assert_eq!(disabled.level, CaptureLevel::L1PointStream);
        assert_eq!(disabled.max_bytes_per_session, 1024 * 1024);
        assert!(!disabled.allow_gpu_readback);

        let enabled = capture_config(true, CaptureLevel::L2PixelDelta, 16);
        assert!(enabled.enabled);
        assert_eq!(enabled.level, CaptureLevel::L2PixelDelta);
        assert_eq!(enabled.max_bytes_per_session, 16 * 1024 * 1024);
        assert!(!enabled.allow_gpu_readback);
    }

    #[test]
    fn initial_capture_config_enables_recording_for_test_app() {
        let capture = initial_capture_config();
        assert!(capture.enabled);
        assert_eq!(capture.level, CaptureLevel::L1PointStream);
        assert!(!capture.allow_gpu_readback);
    }

    #[test]
    fn chinese_font_candidates_cover_macos_and_fallback_platforms() {
        assert!(
            CHINESE_FONT_PATHS
                .iter()
                .any(|path| path.starts_with("/System/Library/Fonts/"))
        );
        assert!(
            CHINESE_FONT_PATHS
                .iter()
                .any(|path| path.contains("Hiragino"))
        );
        assert!(
            CHINESE_FONT_PATHS
                .iter()
                .any(|path| path.contains("NotoSansCJK"))
        );
        assert!(CHINESE_FONT_PATHS.iter().any(|path| path.contains("wqy")));
        assert!(
            CHINESE_FONT_PATHS
                .iter()
                .any(|path| path.contains(r"C:\Windows\Fonts"))
        );
        assert!(CHINESE_FONT_MISSING_WARNING.contains("中文字体"));
        assert!(CHINESE_FONT_MISSING_WARNING.contains("乱码"));
    }

    #[test]
    fn trace_status_label_distinguishes_current_and_cached_records() {
        assert_eq!(trace_status_label(true, true, false), "正在记录当前绘画");
        assert_eq!(
            trace_status_label(false, false, true),
            "记录已停止，可重构上次记录"
        );
        assert_eq!(trace_status_label(false, false, false), "还没有可用记录");
    }

    #[test]
    fn trace_rebuild_unavailable_message_distinguishes_empty_states() {
        assert_eq!(
            trace_rebuild_unavailable_message(false, false, false),
            "记录未启用，无法重构"
        );
        assert_eq!(
            trace_rebuild_unavailable_message(true, false, false),
            "记录为空，无法重构"
        );
        assert_eq!(
            trace_rebuild_unavailable_message(false, false, true),
            "已有可重构记录"
        );
        assert_eq!(
            trace_rebuild_unavailable_message(true, true, false),
            "已有可重构记录"
        );
    }

    #[test]
    fn trace_texture_registration_error_message_reports_display_failure() {
        let message = trace_texture_registration_error_message("纹理无效");
        assert_eq!(message, "重构已完成，但显示纹理注册失败：纹理无效");
    }

    #[test]
    fn trace_report_summary_contains_recording_counters() {
        let report = TraceReport {
            event_count: 12,
            command_count: 4,
            stroke_count: 2,
            stroke_append_count: 3,
            point_count: 1000,
            layer_event_count: 1,
            selection_event_count: 3,
            dirty_summary_count: 2,
            resource_reference_count: 1,
            resource_reference_bytes: 256,
            pixel_delta_payload_bytes: 0,
            checkpoint_payload_bytes: 0,
            degradation_count: 1,
            delta_encoded_estimated_bytes: 2048,
            estimated_bytes: 4096,
        };
        let summary = trace_report_summary(&report);
        assert!(summary.contains("事件=12"));
        assert!(summary.contains("命令=4"));
        assert!(summary.contains("追加=3"));
        assert!(summary.contains("点=1000"));
        assert!(summary.contains("资源引用=1"));
        assert!(summary.contains("降级=1"));
        assert!(summary.contains("估算=4 KB"));
    }

    #[test]
    fn trace_rebuild_error_message_localizes_expected_failures() {
        assert_eq!(
            trace_rebuild_error_message(&CanvasError::MissingLayer(LayerId(7))),
            "重构失败：图层 LayerId(7) 不存在"
        );
        assert_eq!(
            trace_rebuild_error_message(&CanvasError::MissingStroke(StrokeId(29))),
            "重构失败：笔触 StrokeId(29) 不存在"
        );
        assert_eq!(
            trace_rebuild_error_message(&CanvasError::MissingTraceResource(
                "paste:fixture".to_owned()
            )),
            "重构失败：CanvasCore 错误：trace 重放缺少资源引用 paste:fixture"
        );
        assert_eq!(
            trace_rebuild_error_message(&CanvasError::InvalidTraceResource(
                "paste:fixture".to_owned()
            )),
            "重构失败：CanvasCore 错误：trace 重放资源引用与 resolver 返回命令不匹配 paste:fixture"
        );
        assert_eq!(
            trace_rebuild_error_message(&CanvasError::InvalidCanvasSize),
            "重构失败：CanvasCore 错误：画布尺寸必须非零"
        );
    }

    #[test]
    fn max_pixel_abs_diff_reports_mismatch_and_rejects_different_lengths() {
        let left = vec![[0.0, 0.5, 1.0, 1.0], [0.25, 0.0, 0.0, 1.0]];
        let right = vec![[0.0, 0.25, 1.0, 1.0], [0.75, 0.0, 0.0, 1.0]];
        let diff = max_pixel_abs_diff(&left, &right).expect("same length");
        assert!((diff - 0.5).abs() <= f32::EPSILON);
        assert!(max_pixel_abs_diff(&left, &right[..1]).is_none());
    }

    #[test]
    fn trace_rebuild_summary_reports_replay_result() {
        let mut trace = SessionTrace::new(
            CanvasConfig::default(),
            capture_config(true, CaptureLevel::L1PointStream, 8),
        );
        trace.report.command_count = 12;
        trace.report.point_count = 345;
        trace.report.resource_reference_count = 1;
        trace.report.degradation_count = 2;
        let summary = trace_rebuild_summary(&trace, 0.00125);
        assert!(summary.contains("已重构到画布"));
        assert!(summary.contains("命令=12"));
        assert!(summary.contains("点=345"));
        assert!(summary.contains("资源引用=1"));
        assert!(summary.contains("降级=2"));
        assert!(summary.contains("重构前最大像素误差=0.001250"));
    }

    #[test]
    fn trace_base_layer_commands_restore_existing_scene_layers() {
        let mut child = LayerSnapshot {
            spec: LayerSpec::new(LayerId(2)),
            parent: Some(LayerId(1)),
            group_mode: LayerGroupMode::Isolated,
            tile_count: 3,
        };
        child.spec.opacity = 0.5;
        let commands = trace_base_layer_commands(&[
            LayerSnapshot {
                spec: LayerSpec::new(LayerId(1)),
                parent: None,
                group_mode: LayerGroupMode::PassThrough,
                tile_count: 0,
            },
            child,
        ]);
        assert!(matches!(
            &commands[0],
            CanvasCommand::AddLayer(spec) if spec.id == LayerId(1)
        ));
        assert!(matches!(
            &commands[1],
            CanvasCommand::AddLayer(spec) if spec.id == LayerId(2) && (spec.opacity - 0.5).abs() <= f32::EPSILON
        ));
        assert!(matches!(
            &commands[2],
            CanvasCommand::SetLayerParent {
                id: LayerId(2),
                parent: Some(LayerId(1)),
            }
        ));
        assert!(matches!(
            &commands[3],
            CanvasCommand::SetLayerGroupMode {
                id: LayerId(2),
                mode: LayerGroupMode::Isolated,
            }
        ));
    }

    #[test]
    fn trace_replay_command_payload_reports_resource_resolver_state() {
        let brush = CanvasCommand::Brush(canvas_core::BrushCommand {
            layer: LayerId(1),
            mode: BrushMode::Paint,
            radius: 2.0,
            color: Color::rgba(0.0, 0.0, 0.0, 1.0),
            points: vec![StrokePoint {
                x: 1.0,
                y: 2.0,
                pressure: 1.0,
            }],
        });
        let command = canvas_core::TraceCommandEvent {
            position: 0,
            kind: canvas_core::TraceCommandKind::Brush,
            replay: Some(canvas_core::TraceReplayCommand::Command(brush)),
        };
        assert!(matches!(
            trace_replay_command_payload(&command),
            Ok(CanvasCommand::Brush(_))
        ));

        let resource = canvas_core::TraceResourceReference {
            kind: canvas_core::TraceResourceKind::PastePixels,
            id: "paste:fixture".to_owned(),
            byte_len: 64,
            checksum: Some(42),
            width: Some(2),
            height: Some(2),
            pixel_format: Some(canvas_core::TraceResourcePixelFormat::Rgba32FloatLinear),
            lifecycle: canvas_core::TraceResourceLifecycle::ExternalResolverOwned,
        };
        let missing_resource = canvas_core::TraceCommandEvent {
            position: 1,
            kind: canvas_core::TraceCommandKind::PastePixels,
            replay: Some(canvas_core::TraceReplayCommand::MissingResource {
                resource,
                original_kind: canvas_core::TraceCommandKind::PastePixels,
            }),
        };
        let error = trace_replay_command_payload(&missing_resource)
            .expect_err("resource placeholder should require a resolver");
        assert!(matches!(error, CanvasError::MissingTraceResource(_)));
        let message = trace_rebuild_error_message(&error);
        for expected in [
            "重构失败：测试程序重构暂不解析外部资源引用",
            "trace command：position=1",
            "类型=粘贴像素",
            "replay payload=资源占位",
            "trace 资源引用",
            "paste:fixture",
            "尺寸=2x2",
            "checksum=42",
        ] {
            assert!(
                message.contains(expected),
                "资源占位重构错误 `{message}` 应包含 `{expected}`"
            );
        }

        let missing_payload = canvas_core::TraceCommandEvent {
            position: 2,
            kind: canvas_core::TraceCommandKind::Brush,
            replay: None,
        };
        let error = trace_replay_command_payload(&missing_payload)
            .expect_err("missing replay payload should be surfaced");
        let message = trace_rebuild_error_message(&error);
        for expected in [
            "重构失败：测试程序重构缺少可执行 command replay payload",
            "trace command：position=2",
            "类型=笔刷",
            "无 replay payload",
        ] {
            assert!(
                message.contains(expected),
                "缺失 replay payload 错误 `{message}` 应包含 `{expected}`"
            );
        }
    }

    #[test]
    fn replay_stroke_commands_remap_streaming_stroke_ids() {
        let mut stroke_id_map = HashMap::new();
        stroke_id_map.insert(StrokeId(29), StrokeId(1));

        let append = remap_replay_stroke_command(
            &CanvasCommand::AppendStroke(AppendStrokeCommand {
                stroke: StrokeId(29),
                points: vec![StrokePoint {
                    x: 1.0,
                    y: 2.0,
                    pressure: 1.0,
                }],
            }),
            &stroke_id_map,
        );
        assert!(matches!(
            append,
            CanvasCommand::AppendStroke(AppendStrokeCommand {
                stroke: StrokeId(1),
                ..
            })
        ));

        let end =
            remap_replay_stroke_command(&CanvasCommand::EndStroke(StrokeId(29)), &stroke_id_map);
        assert!(matches!(end, CanvasCommand::EndStroke(StrokeId(1))));
    }

    #[test]
    fn trace_stroke_begin_after_finds_original_streaming_id() {
        let mut trace = SessionTrace::new(
            CanvasConfig::default(),
            capture_config(true, CaptureLevel::L1PointStream, 8),
        );
        trace.events.push(canvas_core::TraceEvent {
            sequence: 0,
            estimated_bytes: 1,
            kind: canvas_core::TraceEventKind::Command(canvas_core::TraceCommandEvent {
                position: 0,
                kind: canvas_core::TraceCommandKind::BeginStroke,
                replay: None,
            }),
        });
        trace.events.push(canvas_core::TraceEvent {
            sequence: 1,
            estimated_bytes: 1,
            kind: canvas_core::TraceEventKind::StrokeBegin(canvas_core::TraceStrokeBegin {
                stroke: StrokeId(29),
                layer: LayerId(1),
                mode: BrushMode::Paint,
                radius: 2.0,
                color: Color::rgba(0.0, 0.0, 0.0, 1.0),
                first_point: StrokePoint {
                    x: 1.0,
                    y: 1.0,
                    pressure: 1.0,
                },
                selection_generation: 0,
            }),
        });
        assert_eq!(trace_stroke_begin_after(&trace, 0), Some(StrokeId(29)));
    }

    #[test]
    fn capture_level_list_matches_supported_ui_choices() {
        assert_eq!(CAPTURE_LEVELS.len(), 4);
        assert_eq!(CAPTURE_LEVELS[0], CaptureLevel::L0Structure);
        assert_eq!(CAPTURE_LEVELS[1], CaptureLevel::L1PointStream);
        assert_eq!(CAPTURE_LEVELS[2], CaptureLevel::L2PixelDelta);
        assert_eq!(CAPTURE_LEVELS[3], CaptureLevel::L3SparseCheckpoint);
    }

    #[test]
    fn input_layer_adapts_synthetic_sample_before_core_stroke_point() {
        let viewport = CanvasViewport {
            origin: egui::pos2(10.0, 20.0),
            display_scale: 2.0,
            pan: egui::vec2(100.0, 200.0),
        };
        let point = viewport.to_stroke_point(egui::pos2(14.0, 26.0), 0.75);
        let sample = StylusSample {
            device_id: DeviceId(7),
            tool_id: None,
            phase: StylusPhase::Move,
            time_seconds: 1.25,
            x: point.x,
            y: point.y,
            pressure: Some(1.5),
            pressure_raw: Some(1.5),
            tilt_x: None,
            tilt_y: None,
            twist: None,
            buttons: StylusButtons {
                primary: true,
                ..StylusButtons::default()
            },
            eraser: false,
            backend: InputBackend::Synthetic,
            batch_size: 3,
        };

        let adapted: StrokePoint = StrokeSampleAdapter::default().adapt(sample).into();
        assert!((adapted.x - 102.0).abs() <= f32::EPSILON);
        assert!((adapted.y - 203.0).abs() <= f32::EPSILON);
        assert!((adapted.pressure - 1.0).abs() <= f32::EPSILON);
    }

    #[test]
    fn input_layer_does_not_append_touch_start_as_move_sample() {
        assert!(!touch_phase_appends_to_active_stroke(
            egui::TouchPhase::Start
        ));
        assert!(touch_phase_appends_to_active_stroke(egui::TouchPhase::Move));
        assert!(!touch_phase_appends_to_active_stroke(egui::TouchPhase::End));
        assert!(!touch_phase_appends_to_active_stroke(
            egui::TouchPhase::Cancel
        ));
    }

    #[test]
    fn input_layer_can_begin_from_touch_move_when_start_is_missing() {
        let events = vec![(egui::pos2(24.0, 32.0), egui::TouchPhase::Move, Some(0.42))];
        let begin = CanvasApp::touch_begin_event(&events).expect("move should begin touch stroke");

        assert_eq!(begin.0, egui::pos2(24.0, 32.0));
        assert_eq!(begin.1, egui::TouchPhase::Move);
        assert_eq!(begin.2, Some(0.42));
    }

    #[test]
    fn input_layer_reanchors_large_pointer_fallback_to_first_touch_sample() {
        let begin = StrokePoint {
            x: 500.0,
            y: 818.0,
            pressure: 1.0,
        };
        let first_touch = StrokePoint {
            x: 355.0,
            y: 824.0,
            pressure: 0.216,
        };
        assert!(should_reanchor_pointer_begin_to_touch(
            StrokeInputSource::Touch,
            Some(StrokeInputSource::PointerFallback),
            false,
            Some(begin),
            first_touch,
            15.0,
        ));
        assert!(!should_reanchor_pointer_begin_to_touch(
            StrokeInputSource::Touch,
            Some(StrokeInputSource::PointerFallback),
            true,
            Some(begin),
            first_touch,
            15.0,
        ));
        assert!(!should_reanchor_pointer_begin_to_touch(
            StrokeInputSource::Touch,
            Some(StrokeInputSource::PointerFallback),
            false,
            Some(begin),
            StrokePoint {
                x: 506.0,
                y: 822.0,
                pressure: 0.216,
            },
            15.0,
        ));
    }

    #[test]
    fn input_layer_reanchors_touch_begin_when_first_pressure_arrives() {
        assert!(should_reanchor_touch_begin_pressure(
            StrokeInputSource::Touch,
            Some(StrokeInputSource::Touch),
            false,
            false,
            true,
        ));
        assert!(!should_reanchor_touch_begin_pressure(
            StrokeInputSource::Touch,
            Some(StrokeInputSource::Touch),
            true,
            false,
            true,
        ));
        assert!(!should_reanchor_touch_begin_pressure(
            StrokeInputSource::Touch,
            Some(StrokeInputSource::Touch),
            false,
            true,
            true,
        ));
        assert!(!should_reanchor_touch_begin_pressure(
            StrokeInputSource::Touch,
            Some(StrokeInputSource::Touch),
            false,
            false,
            false,
        ));
    }

    #[test]
    fn input_layer_treats_subpixel_touch_moves_as_tap_jitter() {
        let candidate = TapCandidate {
            anchor: StrokePoint {
                x: 549.76,
                y: 652.01,
                pressure: 0.017,
            },
            max_pressure: 0.323,
            max_distance: 2.86,
        };
        let jitter = [
            StrokePoint {
                x: 550.66,
                y: 649.52,
                pressure: 0.323,
            },
            StrokePoint {
                x: 550.73,
                y: 649.32,
                pressure: 0.323,
            },
        ];
        assert!(points_within_tap_jitter(candidate, &jitter, 15.0));

        let moved = [StrokePoint {
            x: 555.76,
            y: 652.01,
            pressure: 0.323,
        }];
        assert!(!points_within_tap_jitter(candidate, &moved, 15.0));
    }

    #[test]
    fn input_layer_carries_device_pressure_when_move_force_is_missing() {
        assert_eq!(
            CanvasApp::resolved_input_pressure(None, StylusPhase::Move, 0.103, true),
            Some(0.103)
        );
        assert_eq!(
            CanvasApp::resolved_input_pressure(None, StylusPhase::Move, 0.103, false),
            None
        );
        assert_eq!(
            CanvasApp::resolved_input_pressure(None, StylusPhase::Down, 0.103, true),
            None
        );
        assert_eq!(
            CanvasApp::resolved_input_pressure(Some(0.0), StylusPhase::Move, 0.8, true),
            Some(0.0)
        );
    }
}
