#[cfg(target_os = "windows")]
use canva_input::platform::windows::WindowsPointerCapture;
use canva_input::{
    AdapterSample, BackendEvent, DeviceId, InputBackend, PressureCalibration, StrokeInputBatch,
    StrokeInputPoint, StrokeInputSession, StrokeInputSource, StrokeSampleAdapter, StylusButtons,
    StylusPhase, StylusSample, finite_pressure, resolve_stroke_pressure,
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
use std::time::{Duration, Instant};

const DEFAULT_STABILIZER_ENABLED: bool = false;
const DEFAULT_STABILIZER_SMOOTH_STRENGTH: f32 = 0.0;
const RAW_LEAD_POINT_LIMIT: usize = 96;
const RAW_LEAD_SEGMENT_LIMIT: usize = 48;

fn main() -> eframe::Result<()> {
    let config = CanvasConfig {
        width: 1024,
        height: 1024,
        tile_size: 256,
        history_budget_bytes: 512 * 1024 * 1024,
        streaming_sample_min_distance: 0.0,
        stabilizer: low_latency_stabilizer_config(),
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

fn low_latency_stabilizer_config() -> StabilizerConfig {
    StabilizerConfig {
        enabled: DEFAULT_STABILIZER_ENABLED,
        mode: StabilizerMode::Adaptive,
        smooth_strength: DEFAULT_STABILIZER_SMOOTH_STRENGTH,
        spacing_ratio: 0.05,
        ..StabilizerConfig::default()
    }
}

// Input sampling diagnostics.
#[derive(Clone, Copy, Debug, Default)]
struct InputLatencyReport {
    sample_id: u64,
    source_timestamp_age_ms: Option<f32>,
    receive_to_session_ms: Option<f32>,
    receive_to_core_done_ms: Option<f32>,
    core_append_ms: Option<f32>,
    receive_to_paint_ms: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
struct InputLatencyProbe {
    sample_id: u64,
    app_receive: Instant,
    source_timestamp_age_ms: Option<f32>,
    session_emit: Option<Instant>,
    core_append_start: Option<Instant>,
    core_append_done: Option<Instant>,
    paint: Option<Instant>,
}

impl InputLatencyProbe {
    fn new(sample_id: u64, app_receive: Instant, source_timestamp_age_ms: Option<f32>) -> Self {
        Self {
            sample_id,
            app_receive,
            source_timestamp_age_ms,
            session_emit: None,
            core_append_start: None,
            core_append_done: None,
            paint: None,
        }
    }

    fn report(self) -> InputLatencyReport {
        InputLatencyReport {
            sample_id: self.sample_id,
            source_timestamp_age_ms: self.source_timestamp_age_ms,
            receive_to_session_ms: self
                .session_emit
                .map(|time| duration_ms(time.duration_since(self.app_receive))),
            receive_to_core_done_ms: self
                .core_append_done
                .map(|time| duration_ms(time.duration_since(self.app_receive))),
            core_append_ms: self
                .core_append_start
                .zip(self.core_append_done)
                .map(|(start, done)| duration_ms(done.duration_since(start))),
            receive_to_paint_ms: self
                .paint
                .map(|time| duration_ms(time.duration_since(self.app_receive))),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct NativeInputPoint {
    point: StrokeInputPoint,
    phase: StylusPhase,
    window_pos: egui::Pos2,
}

fn native_input_batch_max_gap(points: &[NativeInputPoint]) -> f32 {
    points
        .windows(2)
        .map(|window| {
            let a = window[0].point.point;
            let b = window[1].point.point;
            let dx = a.x - b.x;
            let dy = a.y - b.y;
            (dx * dx + dy * dy).sqrt()
        })
        .fold(0.0, f32::max)
}

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
    last_input_latency_ms: Option<f32>,
    max_input_latency_ms: f32,
    last_batch_newest_source_age_ms: Option<f32>,
    last_batch_oldest_source_age_ms: Option<f32>,
    max_batch_oldest_source_age_ms: f32,
    latest_latency: Option<InputLatencyReport>,
    max_receive_to_core_done_ms: f32,
    max_receive_to_paint_ms: f32,
    native_pointer_events: u32,
    last_native_pointer_batch: usize,
    last_native_batch_max_gap: f32,
    max_native_batch_gap: f32,
    native_pointer_dropped: u64,
    native_pointer_last_error: Option<u32>,
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

#[derive(Clone, Copy)]
struct InputSampleContext {
    phase: StylusPhase,
    force: Option<f32>,
    time_seconds: f64,
    batch_size: usize,
    source: StrokeInputSource,
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
    input_session: StrokeInputSession,
    stabilizer_enabled: bool,
    smooth_strength: f32,
    stab_mode: StabilizerMode,
    /// Last cursor position used for preview prediction and input audit deltas.
    last_cursor: Option<egui::Pos2>,
    pred_vel: egui::Vec2,
    latest_input_preview: Option<StrokePoint>,
    raw_lead_points: Vec<StrokePoint>,
    latency_probe: Option<InputLatencyProbe>,
    next_latency_sample_id: u64,
    #[cfg(target_os = "windows")]
    native_pointer_capture: Option<WindowsPointerCapture>,
    native_pointer_status: String,
    prediction: bool,
    raw_lead_preview: bool,
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

fn native_phase_can_begin_stroke(phase: StylusPhase) -> bool {
    matches!(phase, StylusPhase::Down)
}

fn native_phase_appends_to_active_stroke(phase: StylusPhase) -> bool {
    matches!(phase, StylusPhase::Move)
}

fn native_phase_stops_active_stroke(phase: StylusPhase) -> bool {
    matches!(phase, StylusPhase::Up | StylusPhase::Cancel)
}

fn input_source_label(source: Option<StrokeInputSource>) -> &'static str {
    match source {
        Some(StrokeInputSource::Touch) => "touch",
        Some(StrokeInputSource::WindowsPointer) => "windows pointer",
        Some(StrokeInputSource::PointerFallback) => "pointer fallback",
        None => "-",
    }
}

#[cfg(target_os = "windows")]
#[allow(unsafe_code)]
fn install_native_pointer_capture(
    cc: &eframe::CreationContext<'_>,
) -> (Option<WindowsPointerCapture>, String) {
    use raw_window_handle::{HasWindowHandle as _, RawWindowHandle};

    let handle = match cc.window_handle().map(|handle| handle.as_raw()) {
        Ok(RawWindowHandle::Win32(handle)) => handle,
        Ok(_) => return (None, "unsupported window handle".to_owned()),
        Err(error) => return (None, format!("window handle unavailable: {error}")),
    };

    let hwnd = handle.hwnd.get();
    // Safety: eframe creates the HWND on the GUI thread before app construction.
    // The returned guard is stored in CanvasApp and dropped before the window is destroyed.
    match unsafe { WindowsPointerCapture::install_for_hwnd(hwnd) } {
        Ok(capture) => (Some(capture), "installed".to_owned()),
        Err(error) => (None, error.to_string()),
    }
}

#[cfg(not(target_os = "windows"))]
fn install_native_pointer_capture(_cc: &eframe::CreationContext<'_>) -> String {
    "unsupported platform".to_owned()
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

fn stroke_point_from_input(point: StrokeInputPoint) -> StrokePoint {
    point.point.into()
}

fn stroke_points_from_input(points: Vec<StrokeInputPoint>) -> Vec<StrokePoint> {
    points.into_iter().map(stroke_point_from_input).collect()
}

fn optional_f32_debug(value: Option<f32>) -> String {
    value.map_or_else(|| "-".to_owned(), |value| format!("{value:.3}"))
}

fn optional_ms_debug(value: Option<f32>) -> String {
    value.map_or_else(|| "-".to_owned(), |value| format!("{value:.1}ms"))
}

fn input_latency_ms(sample_time_seconds: f64, frame_time_seconds: f64) -> Option<f32> {
    (sample_time_seconds.is_finite() && frame_time_seconds.is_finite())
        .then(|| ((frame_time_seconds - sample_time_seconds).max(0.0) * 1000.0) as f32)
}

fn input_batch_timestamp_age_range<I>(
    sample_times: I,
    frame_time_seconds: f64,
) -> (Option<f32>, Option<f32>)
where
    I: IntoIterator<Item = f64>,
{
    let mut newest: Option<f32> = None;
    let mut oldest: Option<f32> = None;
    for age in sample_times
        .into_iter()
        .filter_map(|sample_time| input_latency_ms(sample_time, frame_time_seconds))
    {
        newest = Some(newest.map_or(age, |current| current.min(age)));
        oldest = Some(oldest.map_or(age, |current| current.max(age)));
    }
    (newest, oldest)
}

fn raw_lead_tail_start(points: &[StrokePoint], core_tip: Option<StrokePoint>) -> usize {
    if points.len() <= 2 {
        return 0;
    }
    let floor = points.len().saturating_sub(RAW_LEAD_SEGMENT_LIMIT + 1);
    let Some(core_tip) = core_tip else {
        return floor;
    };
    points
        .iter()
        .enumerate()
        .skip(floor)
        .map(|(index, point)| {
            let dx = point.x - core_tip.x;
            let dy = point.y - core_tip.y;
            (index, dx * dx + dy * dy)
        })
        .min_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map_or(floor, |(index, _)| index.saturating_sub(1).max(floor))
}

fn duration_ms(duration: Duration) -> f32 {
    duration.as_secs_f32() * 1000.0
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
        #[cfg(target_os = "windows")]
        let (native_pointer_capture, native_pointer_status) = install_native_pointer_capture(cc);
        #[cfg(not(target_os = "windows"))]
        let native_pointer_status = install_native_pointer_capture(cc);

        Self {
            core,
            layer_id,
            brush_radius: 15.0,
            brush_color: [0.0, 0.0, 0.0, 1.0],
            brush_mode: BrushMode::Paint,
            canvas_width: 1024.0,
            canvas_height: 1024.0,
            active_stroke: None,
            input_session: StrokeInputSession::new(15.0),
            stabilizer_enabled: DEFAULT_STABILIZER_ENABLED,
            smooth_strength: DEFAULT_STABILIZER_SMOOTH_STRENGTH,
            stab_mode: StabilizerMode::Adaptive,
            last_cursor: None,
            pred_vel: egui::Vec2::ZERO,
            latest_input_preview: None,
            raw_lead_points: Vec::new(),
            latency_probe: None,
            next_latency_sample_id: 1,
            #[cfg(target_os = "windows")]
            native_pointer_capture,
            native_pointer_status,
            prediction: false,
            raw_lead_preview: true,
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
        self.input_session.set_brush_radius(self.brush_radius);
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
            core.set_stabilizer_enabled(self.stabilizer_enabled);
            core.set_stabilizer_mode(self.stab_mode);
            core.set_stabilizer_strength(self.smooth_strength);
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
            pressure: resolve_stroke_pressure(
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

    fn adapt_stylus_sample(&mut self, sample: StylusSample) -> AdapterSample {
        let pressure_available = sample.pressure.is_some();
        let adapted = StrokeSampleAdapter::new(self.input_calibration()).adapt(sample);
        self.last_pressure = adapted.pressure.max(0.01);
        self.pressure_from_device = pressure_available;
        adapted
    }

    fn input_stroke_point(
        &mut self,
        pos: egui::Pos2,
        viewport: CanvasViewport,
        context: InputSampleContext,
    ) -> StrokeInputPoint {
        let point = viewport.to_stroke_point(pos, self.fallback_pressure);
        let pressure_from_device = finite_pressure(context.force).is_some();
        let sample = self.stylus_sample(
            point,
            context.phase,
            context.force,
            context.time_seconds,
            context.batch_size,
        );
        let point = self.adapt_stylus_sample(sample);
        StrokeInputPoint::new(point, context.source, pressure_from_device)
    }

    fn current_input_point(
        &mut self,
        ui: &egui::Ui,
        pos: egui::Pos2,
        phase: StylusPhase,
        viewport: CanvasViewport,
    ) -> StrokeInputPoint {
        let time_seconds = ui.input(|i| i.time);
        self.input_stroke_point(
            pos,
            viewport,
            InputSampleContext {
                phase,
                force: None,
                time_seconds,
                batch_size: 1,
                source: StrokeInputSource::PointerFallback,
            },
        )
    }

    fn native_backend_event_to_input_point(
        &mut self,
        event: BackendEvent,
        viewport: CanvasViewport,
    ) -> NativeInputPoint {
        let phase = event.sample.phase;
        let window_pos = egui::pos2(event.sample.x, event.sample.y);
        let mut sample = event.sample;
        let canvas_point = viewport.to_stroke_point(window_pos, self.fallback_pressure);
        sample.x = canvas_point.x;
        sample.y = canvas_point.y;
        let pressure_from_device = finite_pressure(sample.pressure).is_some();
        let adapted = self.adapt_stylus_sample(sample);
        NativeInputPoint {
            point: StrokeInputPoint::new(
                adapted,
                StrokeInputSource::WindowsPointer,
                pressure_from_device,
            ),
            phase,
            window_pos,
        }
    }

    #[cfg(target_os = "windows")]
    fn drain_native_pointer_input(
        &mut self,
        ui: &egui::Ui,
        viewport: CanvasViewport,
        frame_time_seconds: f64,
    ) -> Vec<NativeInputPoint> {
        let events = self
            .native_pointer_capture
            .as_ref()
            .map(|capture| capture.drain_events(ui.ctx().pixels_per_point(), frame_time_seconds))
            .unwrap_or_default();
        if let Some(capture) = &self.native_pointer_capture {
            let stats = capture.stats();
            self.audit.native_pointer_dropped = stats.dropped_events;
            self.audit.native_pointer_last_error = stats.last_error;
        }
        self.audit.last_native_pointer_batch = events.len();
        self.audit.native_pointer_events = self
            .audit
            .native_pointer_events
            .saturating_add(u32::try_from(events.len()).unwrap_or(u32::MAX));
        let points = events
            .into_iter()
            .filter(|event| event.sample.phase != StylusPhase::Hover)
            .map(|event| self.native_backend_event_to_input_point(event, viewport))
            .collect::<Vec<_>>();
        self.audit.last_native_batch_max_gap = native_input_batch_max_gap(&points);
        if self.audit.last_native_batch_max_gap > self.audit.max_native_batch_gap {
            self.audit.max_native_batch_gap = self.audit.last_native_batch_max_gap;
        }
        points
    }

    #[cfg(not(target_os = "windows"))]
    fn drain_native_pointer_input(
        &mut self,
        _ui: &egui::Ui,
        _viewport: CanvasViewport,
        _frame_time_seconds: f64,
    ) -> Vec<NativeInputPoint> {
        Vec::new()
    }

    fn record_adapted_pressure_range(&mut self, points: &[StrokeInputPoint]) {
        self.audit.last_adapted_pressure_min = None;
        self.audit.last_adapted_pressure_max = None;
        for pressure in points
            .iter()
            .map(|point| point.point.pressure)
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

    fn record_input_preview_sample(&mut self, point: StrokeInputPoint, frame_time_seconds: f64) {
        let preview = stroke_point_from_input(point);
        self.latest_input_preview = Some(preview);
        self.push_raw_lead_point(preview);
        let source_timestamp_age_ms =
            input_latency_ms(point.point.time_seconds, frame_time_seconds);
        if let Some(source_timestamp_age_ms) = source_timestamp_age_ms {
            self.audit.last_input_latency_ms = Some(source_timestamp_age_ms);
            if source_timestamp_age_ms > self.audit.max_input_latency_ms {
                self.audit.max_input_latency_ms = source_timestamp_age_ms;
            }
        }
        let sample_id = self.next_latency_sample_id;
        self.next_latency_sample_id = self.next_latency_sample_id.saturating_add(1);
        self.latency_probe = Some(InputLatencyProbe::new(
            sample_id,
            Instant::now(),
            source_timestamp_age_ms,
        ));
        self.update_latency_report();
    }

    fn push_raw_lead_point(&mut self, point: StrokePoint) {
        if self.raw_lead_points.last().is_some_and(|last| {
            let dx = point.x - last.x;
            let dy = point.y - last.y;
            dx * dx + dy * dy <= f32::EPSILON
        }) {
            return;
        }
        self.raw_lead_points.push(point);
        let overflow = self
            .raw_lead_points
            .len()
            .saturating_sub(RAW_LEAD_POINT_LIMIT);
        if overflow > 0 {
            self.raw_lead_points.drain(0..overflow);
        }
    }

    fn record_input_batch_timestamp_ages(
        &mut self,
        points: &[StrokeInputPoint],
        frame_time_seconds: f64,
    ) {
        let (newest, oldest) = input_batch_timestamp_age_range(
            points.iter().map(|point| point.point.time_seconds),
            frame_time_seconds,
        );
        self.audit.last_batch_newest_source_age_ms = newest;
        self.audit.last_batch_oldest_source_age_ms = oldest;
        if let Some(oldest) = oldest
            && oldest > self.audit.max_batch_oldest_source_age_ms
        {
            self.audit.max_batch_oldest_source_age_ms = oldest;
        }
    }

    fn mark_latest_latency_session_emit(&mut self) {
        if let Some(mut probe) = self.latency_probe {
            probe.session_emit = Some(Instant::now());
            self.latency_probe = Some(probe);
            self.update_latency_report();
        }
    }

    fn mark_latest_latency_core_append(&mut self, start: Instant, done: Instant) {
        if let Some(mut probe) = self.latency_probe {
            probe.core_append_start = Some(start);
            probe.core_append_done = Some(done);
            self.latency_probe = Some(probe);
            self.update_latency_report();
        }
    }

    fn mark_latest_latency_paint(&mut self) {
        if let Some(mut probe) = self.latency_probe {
            probe.paint = Some(Instant::now());
            self.latency_probe = Some(probe);
            self.update_latency_report();
        }
    }

    fn update_latency_report(&mut self) {
        let Some(probe) = self.latency_probe else {
            return;
        };
        let report = probe.report();
        if let Some(value) = report.receive_to_core_done_ms
            && value > self.audit.max_receive_to_core_done_ms
        {
            self.audit.max_receive_to_core_done_ms = value;
        }
        if let Some(value) = report.receive_to_paint_ms
            && value > self.audit.max_receive_to_paint_ms
        {
            self.audit.max_receive_to_paint_ms = value;
        }
        self.audit.latest_latency = Some(report);
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
    ) -> StrokeInputPoint {
        let (touch_events, time_seconds) = Self::touch_events(ui);
        if let Some((pos, _, force)) = Self::touch_begin_event(&touch_events) {
            return self.input_stroke_point(
                pos,
                viewport,
                InputSampleContext {
                    phase: StylusPhase::Down,
                    force,
                    time_seconds,
                    batch_size: 1,
                    source: StrokeInputSource::Touch,
                },
            );
        }
        self.current_input_point(ui, fallback_pos, StylusPhase::Down, viewport)
    }

    fn input_stroke_points(&mut self, ui: &egui::Ui, viewport: CanvasViewport) -> StrokeInputBatch {
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
            for (pos, phase, force) in touch_events {
                let phase = match phase {
                    egui::TouchPhase::Start => StylusPhase::Down,
                    egui::TouchPhase::Move => StylusPhase::Move,
                    egui::TouchPhase::End => StylusPhase::Up,
                    egui::TouchPhase::Cancel => StylusPhase::Cancel,
                };
                let point = self.input_stroke_point(
                    pos,
                    viewport,
                    InputSampleContext {
                        phase,
                        force,
                        time_seconds,
                        batch_size,
                        source: StrokeInputSource::Touch,
                    },
                );
                points.push(point);
            }
            self.record_adapted_pressure_range(&points);
            return StrokeInputBatch::new(StrokeInputSource::Touch, points);
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
        for pos in pointer_events {
            let point = self.input_stroke_point(
                pos,
                viewport,
                InputSampleContext {
                    phase: StylusPhase::Move,
                    force: None,
                    time_seconds,
                    batch_size,
                    source: StrokeInputSource::PointerFallback,
                },
            );
            points.push(point);
        }
        self.record_adapted_pressure_range(&points);
        StrokeInputBatch::new(StrokeInputSource::PointerFallback, points)
    }

    fn reanchor_active_stroke_to_touch(&mut self, first: StrokeInputPoint) -> Result<(), String> {
        let first_point = stroke_point_from_input(first);
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
            first_point,
        });
        let output = {
            let mut core = self.core.lock().unwrap();
            core.execute(&command).map_err(|error| error.to_string())?
        };
        match output {
            CommandOutput::StrokeStarted { id } => {
                self.active_stroke = Some(id);
                self.audit.begin_point = Some(first_point);
                self.audit.begin_pressure_from_device = first.pressure_from_device;
                self.audit.begin_input_source = Some(StrokeInputSource::Touch);
                self.audit.reanchored_touch_begins =
                    self.audit.reanchored_touch_begins.saturating_add(1);
                Ok(())
            }
            _ => Err("画笔命令未能重新开始笔画".to_owned()),
        }
    }

    fn reset_active_stroke_tracking(&mut self) {
        self.input_session.reset();
        self.latest_input_preview = None;
        self.raw_lead_points.clear();
        self.latency_probe = None;
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
        config.stabilizer.enabled = self.stabilizer_enabled;
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
                ui.checkbox(&mut self.stabilizer_enabled, "启用稳定器");
                ui.add(egui::Slider::new(&mut self.smooth_strength, 0.0..=1.0).text("平滑"));
                ui.horizontal(|ui| {
                    ui.label("稳定器");
                    ui
                        .selectable_value(&mut self.stab_mode, StabilizerMode::Adaptive, "自适应")
                        .changed();
                    ui
                        .selectable_value(&mut self.stab_mode, StabilizerMode::PulledString, "拉绳")
                        .changed();
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
                ui.checkbox(&mut self.raw_lead_preview, "低延迟原始笔尖");
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
                ui.label(format!("native pointer: {}", self.native_pointer_status));
                ui.label(format!(
                    "native batch: {} total={} dropped={} gap={:.1}px max_gap={:.1}px last_error={}",
                    a.last_native_pointer_batch,
                    a.native_pointer_events,
                    a.native_pointer_dropped,
                    a.last_native_batch_max_gap,
                    a.max_native_batch_gap,
                    a.native_pointer_last_error
                        .map_or_else(|| "-".to_owned(), |error| error.to_string())
                ));
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
                    "source timestamp age: {} (max {:.1}ms)",
                    optional_ms_debug(a.last_input_latency_ms),
                    a.max_input_latency_ms
                ));
                ui.label(format!(
                    "batch source age: newest {} oldest {} (max oldest {:.1}ms)",
                    optional_ms_debug(a.last_batch_newest_source_age_ms),
                    optional_ms_debug(a.last_batch_oldest_source_age_ms),
                    a.max_batch_oldest_source_age_ms
                ));
                if let Some(latency) = a.latest_latency {
                    ui.label(format!("latency sample: #{}", latency.sample_id));
                    ui.label(format!(
                        "receive -> session: {}",
                        optional_ms_debug(latency.receive_to_session_ms)
                    ));
                    ui.label(format!(
                        "receive -> core done: {} (max {:.1}ms)",
                        optional_ms_debug(latency.receive_to_core_done_ms),
                        a.max_receive_to_core_done_ms
                    ));
                    ui.label(format!(
                        "core append: {}",
                        optional_ms_debug(latency.core_append_ms)
                    ));
                    ui.label(format!(
                        "receive -> paint: {} (max {:.1}ms)",
                        optional_ms_debug(latency.receive_to_paint_ms),
                        a.max_receive_to_paint_ms
                    ));
                    ui.label(format!(
                        "source -> receive: {}",
                        optional_ms_debug(latency.source_timestamp_age_ms)
                    ));
                }
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
            let frame_time_seconds = ui.input(|i| i.time);
            let native_input_points =
                self.drain_native_pointer_input(ui, viewport, frame_time_seconds);
            let native_begin_index = native_input_points.iter().position(|input| {
                native_phase_can_begin_stroke(input.phase) && rect.contains(input.window_pos)
            });
            let native_begin_input = native_begin_index.map(|index| native_input_points[index]);
            let native_stopped = native_input_points
                .iter()
                .any(|input| native_phase_stops_active_stroke(input.phase));
            let native_update_points = native_input_points
                .iter()
                .enumerate()
                .filter(|(index, input)| {
                    Some(*index) != native_begin_index
                        && native_phase_appends_to_active_stroke(input.phase)
                })
                .map(|(_, input)| input.point)
                .collect::<Vec<_>>();

            // Stroke start.
            let primary_drag_started = response.drag_started_by(egui::PointerButton::Primary)
                || native_begin_input.is_some();

            if primary_drag_started {
                self.audit.reset();
                self.audit.last_native_pointer_batch = native_input_points.len();
                self.audit.native_pointer_events =
                    u32::try_from(native_input_points.len()).unwrap_or(u32::MAX);
                #[cfg(target_os = "windows")]
                if let Some(capture) = &self.native_pointer_capture {
                    let stats = capture.stats();
                    self.audit.native_pointer_dropped = stats.dropped_events;
                    self.audit.native_pointer_last_error = stats.last_error;
                }
                self.last_cursor = response.interact_pointer_pos();
                self.pred_vel = egui::Vec2::ZERO;
                if let Some(pos) = native_begin_input
                    .map(|input| input.window_pos)
                    .or_else(|| response.interact_pointer_pos())
                {
                    match self.tool_mode {
                        ToolMode::Brush => {
                            let input = native_begin_input.map_or_else(
                                || self.begin_input_point(ui, pos, viewport),
                                |input| input.point,
                            );
                            self.raw_lead_points.clear();
                            self.record_input_preview_sample(input, frame_time_seconds);
                            let p = stroke_point_from_input(input);
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
                                    self.input_session.begin(input);
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
            let stream_native_updates =
                !native_update_points.is_empty() && self.active_stroke.is_some();
            let stream_egui_updates =
                response.dragged_by(egui::PointerButton::Primary) && !primary_drag_started;
            if stream_native_updates || stream_egui_updates {
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

                        let mut input_batch = if native_update_points.is_empty() {
                            self.input_stroke_points(ui, viewport)
                        } else {
                            self.record_adapted_pressure_range(&native_update_points);
                            StrokeInputBatch::new(
                                StrokeInputSource::WindowsPointer,
                                native_update_points.clone(),
                            )
                        };
                        self.record_input_batch_timestamp_ages(
                            &input_batch.points,
                            frame_time_seconds,
                        );
                        self.audit.raw_events = self.audit.raw_events.saturating_add(
                            u32::try_from(input_batch.points.len()).unwrap_or(u32::MAX),
                        );

                        for cp in &input_batch.points {
                            self.record_input_preview_sample(*cp, frame_time_seconds);
                            let point = stroke_point_from_input(*cp);
                            if let Some(prev) = self.audit.raw_pts.last() {
                                let dx = point.x - prev.x;
                                let dy = point.y - prev.y;
                                let gap = (dx * dx + dy * dy).sqrt();
                                self.audit.sum_event_gap += f64::from(gap);
                                self.audit.event_gap_count += 1;
                                if gap > self.audit.max_event_gap {
                                    self.audit.max_event_gap = gap;
                                }
                            }
                            self.audit.raw_pts.push(point);
                        }
                        if input_batch.points.is_empty()
                            && let Some(pos) = response.interact_pointer_pos()
                        {
                            input_batch.source = StrokeInputSource::PointerFallback;
                            let fallback =
                                self.current_input_point(ui, pos, StylusPhase::Move, viewport);
                            self.record_input_preview_sample(fallback, frame_time_seconds);
                            input_batch.points.push(fallback);
                            self.record_input_batch_timestamp_ages(
                                &input_batch.points,
                                frame_time_seconds,
                            );
                        }

                        let append = self.input_session.append_batch(input_batch);
                        self.mark_latest_latency_session_emit();
                        let mut batch = append.points;
                        if let Some(first) = append.reanchor
                            && let Err(error) = self.reanchor_active_stroke_to_touch(first)
                        {
                            self.last_error = Some(error);
                            batch.clear();
                        }

                        if let Some(id) = self.active_stroke
                            && !batch.is_empty()
                        {
                            let batch = stroke_points_from_input(batch);
                            self.audit.append_batches = self.audit.append_batches.saturating_add(1);
                            self.audit.last_append_len = batch.len();
                            self.audit.last_append_first = batch.first().copied();
                            self.audit.last_append_second = batch.get(1).copied();
                            let core_append_start = Instant::now();
                            let mut core = self.core.lock().unwrap();
                            let command = CanvasCommand::AppendStroke(AppendStrokeCommand {
                                stroke: id,
                                points: batch,
                            });
                            let result = core
                                .execute(&command)
                                .and_then(|_| Self::refresh(&mut core));
                            let dispatch = result.as_ref().ok().map(|()| core.dispatch_report());
                            drop(core);
                            let core_append_done = Instant::now();
                            self.mark_latest_latency_core_append(
                                core_append_start,
                                core_append_done,
                            );
                            match result {
                                Ok(()) => {
                                    let dispatch = dispatch.expect("dispatch report after success");
                                    self.audit.last_core_uploaded_points =
                                        dispatch.last_brush_uploaded_points;
                                    self.audit.last_core_dispatches =
                                        dispatch.last_brush_dispatches;
                                    if dispatch.last_brush_dispatches > 0 {
                                        self.input_session.mark_core_dispatched();
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
                            let point = Self::selection_point_from_canvas(stroke_point_from_input(
                                input_point,
                            ));
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
            if response.drag_stopped() || native_stopped {
                match self.tool_mode {
                    ToolMode::Brush => {
                        if let Some(id) = self.active_stroke.take() {
                            let tap_command = self.input_session.end().tap.map(|tap| {
                                CanvasCommand::Brush(BrushCommand {
                                    layer: self.layer_id,
                                    mode: self.brush_mode,
                                    points: vec![stroke_point_from_input(tap)],
                                    radius: self.brush_radius,
                                    color: self.color(),
                                })
                            });
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
                self.latest_input_preview = None;
                self.raw_lead_points.clear();
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

            // Raw input lead preview: draw the live input tail ahead of the
            // stabilized/core brush result so perceived ink stays under the pen.
            if self.raw_lead_preview
                && let Some(id) = self.active_stroke
                && !self.raw_lead_points.is_empty()
            {
                self.mark_latest_latency_paint();
                let core_tip = self.core.lock().unwrap().active_stroke_tip(id);
                let raw_tip = self.raw_lead_points.last().copied();
                let draw_tail = core_tip.zip(raw_tip).is_none_or(|(core_tip, raw_tip)| {
                    let dx = core_tip.x - raw_tip.x;
                    let dy = core_tip.y - raw_tip.y;
                    dx * dx + dy * dy > 0.25
                });
                if draw_tail {
                    let tail_start = raw_lead_tail_start(&self.raw_lead_points, core_tip);
                    let tail = &self.raw_lead_points[tail_start..];
                    let to_screen = |point: StrokePoint| -> egui::Pos2 {
                        egui::pos2(
                            rect.min.x + (point.x - self.pan.x) * display_scale,
                            rect.min.y + (point.y - self.pan.y) * display_scale,
                        )
                    };
                    let painter = ui.painter_at(rect);
                    let preview_color = egui::Color32::from_rgba_unmultiplied(30, 160, 240, 170);
                    let tail_points = tail
                        .iter()
                        .copied()
                        .map(to_screen)
                        .filter(|point| point.x.is_finite() && point.y.is_finite())
                        .collect::<Vec<_>>();
                    if tail_points.len() >= 2 {
                        painter.add(egui::Shape::line(
                            tail_points.clone(),
                            egui::Stroke::new(2.0, preview_color),
                        ));
                    }
                    if let Some(raw_tip) = tail_points.last().copied() {
                        let r = (self.brush_radius * display_scale).max(1.0);
                        painter.circle_stroke(raw_tip, r, egui::Stroke::new(1.5, preview_color));
                        painter.circle_filled(raw_tip, 3.0, preview_color);
                    }
                }
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

    fn input_test_point(
        x: f32,
        y: f32,
        pressure: f32,
        source: StrokeInputSource,
        pressure_from_device: bool,
    ) -> StrokeInputPoint {
        StrokeInputPoint::new(
            AdapterSample {
                x,
                y,
                pressure,
                tilt_x: 0.0,
                tilt_y: 0.0,
                twist: 0.0,
                time_seconds: 0.0,
            },
            source,
            pressure_from_device,
        )
    }

    fn test_core() -> CanvasCore {
        let config = CanvasConfig {
            width: 128,
            height: 128,
            tile_size: 64,
            streaming_sample_min_distance: 0.0,
            stabilizer: low_latency_stabilizer_config(),
            ..CanvasConfig::default()
        };
        let mut core = pollster::block_on(CanvasCore::new(config)).expect("create test core");
        core.execute(&CanvasCommand::AddLayer(LayerSpec::new(LayerId(1))))
            .expect("add test layer");
        core
    }

    fn begin_test_stroke(
        core: &mut CanvasCore,
        session: &mut StrokeInputSession,
        input: StrokeInputPoint,
        radius: f32,
    ) -> StrokeId {
        let output = core
            .execute(&CanvasCommand::BeginStroke(BeginStrokeCommand {
                layer: LayerId(1),
                mode: BrushMode::Paint,
                radius,
                color: Color::rgba(0.0, 0.0, 0.0, 1.0),
                first_point: stroke_point_from_input(input),
            }))
            .expect("begin test stroke");
        let CommandOutput::StrokeStarted { id } = output else {
            panic!("begin test stroke returned unexpected output");
        };
        session.begin(input);
        id
    }

    fn commit_test_end(
        core: &mut CanvasCore,
        session: &mut StrokeInputSession,
        stroke: StrokeId,
        radius: f32,
    ) -> bool {
        let tap = session.end().tap;
        let output = core
            .execute(&CanvasCommand::EndStroke(stroke))
            .expect("end test stroke");
        let CommandOutput::StrokeEnded { committed } = output else {
            panic!("end test stroke returned unexpected output");
        };
        if !committed && let Some(tap) = tap {
            core.execute(&CanvasCommand::Brush(BrushCommand {
                layer: LayerId(1),
                mode: BrushMode::Paint,
                points: vec![stroke_point_from_input(tap)],
                radius,
                color: Color::rgba(0.0, 0.0, 0.0, 1.0),
            }))
            .expect("commit fallback tap");
        }
        committed
    }

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
    fn test_app_uses_low_latency_stabilizer_defaults() {
        let stabilizer = low_latency_stabilizer_config();

        assert!(!stabilizer.enabled);
        assert_eq!(stabilizer.mode, StabilizerMode::Adaptive);
        assert_eq!(
            stabilizer.smooth_strength,
            DEFAULT_STABILIZER_SMOOTH_STRENGTH
        );
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
    fn input_layer_native_pointer_phase_policy_matches_streaming_model() {
        assert!(native_phase_can_begin_stroke(StylusPhase::Down));
        assert!(!native_phase_can_begin_stroke(StylusPhase::Move));
        assert!(!native_phase_can_begin_stroke(StylusPhase::Hover));
        assert!(!native_phase_can_begin_stroke(StylusPhase::Up));
        assert!(native_phase_appends_to_active_stroke(StylusPhase::Move));
        assert!(!native_phase_appends_to_active_stroke(StylusPhase::Down));
        assert!(!native_phase_appends_to_active_stroke(StylusPhase::Hover));
        assert!(native_phase_stops_active_stroke(StylusPhase::Up));
        assert!(native_phase_stops_active_stroke(StylusPhase::Cancel));
        assert!(!native_phase_stops_active_stroke(StylusPhase::Hover));
    }

    #[test]
    fn native_input_batch_gap_uses_canvas_points() {
        let points = vec![
            NativeInputPoint {
                point: input_test_point(10.0, 20.0, 0.5, StrokeInputSource::WindowsPointer, true),
                phase: StylusPhase::Move,
                window_pos: egui::pos2(100.0, 100.0),
            },
            NativeInputPoint {
                point: input_test_point(13.0, 24.0, 0.5, StrokeInputSource::WindowsPointer, true),
                phase: StylusPhase::Move,
                window_pos: egui::pos2(300.0, 300.0),
            },
        ];

        assert!((native_input_batch_max_gap(&points) - 5.0).abs() <= f32::EPSILON);
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
        let mut session = StrokeInputSession::new(15.0);
        session.begin(input_test_point(
            500.0,
            818.0,
            1.0,
            StrokeInputSource::PointerFallback,
            false,
        ));
        let first_touch = input_test_point(355.0, 824.0, 0.216, StrokeInputSource::Touch, true);

        let append = session.append_batch(StrokeInputBatch::new(
            StrokeInputSource::Touch,
            vec![first_touch],
        ));

        assert_eq!(append.reanchor, Some(first_touch));
        assert!(append.points.is_empty());
    }

    #[test]
    fn input_layer_reanchors_touch_begin_when_first_pressure_arrives() {
        let mut session = StrokeInputSession::new(15.0);
        session.begin(input_test_point(
            650.4,
            458.39,
            0.029,
            StrokeInputSource::Touch,
            false,
        ));
        let first_touch = input_test_point(650.0, 458.14, 0.353, StrokeInputSource::Touch, true);

        let append = session.append_batch(StrokeInputBatch::new(
            StrokeInputSource::Touch,
            vec![first_touch],
        ));

        assert_eq!(append.reanchor, Some(first_touch));
        assert!(append.points.is_empty());
    }

    #[test]
    fn input_layer_treats_subpixel_touch_moves_as_tap_jitter() {
        let mut session = StrokeInputSession::new(15.0);
        session.begin(input_test_point(
            549.76,
            652.01,
            0.017,
            StrokeInputSource::Touch,
            true,
        ));

        let append = session.append_batch(StrokeInputBatch::new(
            StrokeInputSource::Touch,
            vec![
                input_test_point(550.66, 649.52, 0.323, StrokeInputSource::Touch, true),
                input_test_point(550.73, 649.32, 0.323, StrokeInputSource::Touch, true),
            ],
        ));

        assert!(append.points.is_empty());
        assert_eq!(session.end().tap.map(|tap| tap.point.pressure), Some(0.323));
    }

    #[test]
    fn built_app_input_tap_jitter_commits_single_dab_through_core() {
        let radius = 15.0;
        let mut core = test_core();
        let mut session = StrokeInputSession::new(radius);
        let stroke = begin_test_stroke(
            &mut core,
            &mut session,
            input_test_point(54.0, 64.0, 0.017, StrokeInputSource::Touch, true),
            radius,
        );

        let append = session.append_batch(StrokeInputBatch::new(
            StrokeInputSource::Touch,
            vec![
                input_test_point(54.9, 61.5, 0.323, StrokeInputSource::Touch, true),
                input_test_point(55.0, 61.3, 0.323, StrokeInputSource::Touch, true),
            ],
        ));
        assert!(append.points.is_empty());

        let committed = commit_test_end(&mut core, &mut session, stroke, radius);

        assert!(!committed);
        let dispatch = core.dispatch_report();
        assert!(dispatch.last_brush_dispatches > 0);
        assert_eq!(dispatch.last_brush_uploaded_points, 1);
    }

    #[test]
    fn built_app_input_long_stroke_streams_append_through_core() {
        let radius = 8.0;
        let mut core = test_core();
        let mut session = StrokeInputSession::new(radius);
        let stroke = begin_test_stroke(
            &mut core,
            &mut session,
            input_test_point(16.0, 16.0, 0.4, StrokeInputSource::Touch, true),
            radius,
        );

        let append = session.append_batch(StrokeInputBatch::new(
            StrokeInputSource::Touch,
            vec![
                input_test_point(28.0, 28.0, 0.5, StrokeInputSource::Touch, true),
                input_test_point(44.0, 44.0, 0.6, StrokeInputSource::Touch, true),
                input_test_point(60.0, 60.0, 0.7, StrokeInputSource::Touch, true),
            ],
        ));
        assert!(append.reanchor.is_none());
        assert!(!append.points.is_empty());

        core.execute(&CanvasCommand::AppendStroke(AppendStrokeCommand {
            stroke,
            points: stroke_points_from_input(append.points),
        }))
        .expect("append normal human stroke");
        let dispatch = core.dispatch_report();
        assert!(dispatch.last_brush_dispatches > 0);
        session.mark_core_dispatched();

        let committed = commit_test_end(&mut core, &mut session, stroke, radius);
        assert!(committed);
    }

    #[test]
    fn input_layer_carries_device_pressure_when_move_force_is_missing() {
        assert_eq!(
            resolve_stroke_pressure(None, StylusPhase::Move, 0.103, true),
            Some(0.103)
        );
        assert_eq!(
            resolve_stroke_pressure(None, StylusPhase::Move, 0.103, false),
            None
        );
        assert_eq!(
            resolve_stroke_pressure(None, StylusPhase::Down, 0.103, true),
            None
        );
        assert_eq!(
            resolve_stroke_pressure(Some(0.0), StylusPhase::Move, 0.8, true),
            Some(0.0)
        );
    }

    #[test]
    fn input_latency_audit_uses_sample_and_frame_timestamps() {
        assert_eq!(input_latency_ms(1.000, 1.016), Some(16.0));
        assert_eq!(input_latency_ms(1.020, 1.016), Some(0.0));
        assert_eq!(input_latency_ms(f64::NAN, 1.016), None);
        assert_eq!(input_latency_ms(1.000, f64::INFINITY), None);
    }

    #[test]
    fn input_batch_timestamp_age_reports_newest_and_oldest_samples() {
        let (newest, oldest) = input_batch_timestamp_age_range([1.000, 1.004, 1.012], 1.016);

        assert!((newest.expect("newest age") - 4.0).abs() <= f32::EPSILON);
        assert!((oldest.expect("oldest age") - 16.0).abs() <= f32::EPSILON);

        let (newest, oldest) = input_batch_timestamp_age_range([f64::NAN, f64::INFINITY], 1.016);
        assert_eq!(newest, None);
        assert_eq!(oldest, None);
    }

    #[test]
    fn raw_lead_tail_start_limits_preview_to_recent_points() {
        let points = (0..80)
            .map(|index| StrokePoint {
                x: index as f32,
                y: 0.0,
                pressure: 1.0,
            })
            .collect::<Vec<_>>();

        assert_eq!(
            raw_lead_tail_start(&points, None),
            points.len() - RAW_LEAD_SEGMENT_LIMIT - 1
        );
    }

    #[test]
    fn raw_lead_tail_start_tracks_core_tip_nearest_recent_point() {
        let points = (0..80)
            .map(|index| StrokePoint {
                x: index as f32,
                y: 0.0,
                pressure: 1.0,
            })
            .collect::<Vec<_>>();
        let core_tip = StrokePoint {
            x: 70.2,
            y: 0.0,
            pressure: 1.0,
        };

        assert_eq!(raw_lead_tail_start(&points, Some(core_tip)), 69);
    }

    #[test]
    fn input_latency_probe_reports_internal_segments() {
        fn assert_ms(actual: Option<f32>, expected: f32) {
            assert!((actual.expect("latency segment") - expected).abs() <= f32::EPSILON);
        }

        let start = Instant::now();
        let probe = InputLatencyProbe {
            sample_id: 7,
            app_receive: start,
            source_timestamp_age_ms: Some(3.0),
            session_emit: Some(start + Duration::from_millis(1)),
            core_append_start: Some(start + Duration::from_millis(2)),
            core_append_done: Some(start + Duration::from_millis(5)),
            paint: Some(start + Duration::from_millis(8)),
        };

        let report = probe.report();

        assert_eq!(report.sample_id, 7);
        assert_ms(report.source_timestamp_age_ms, 3.0);
        assert_ms(report.receive_to_session_ms, 1.0);
        assert_ms(report.core_append_ms, 3.0);
        assert_ms(report.receive_to_core_done_ms, 5.0);
        assert_ms(report.receive_to_paint_ms, 8.0);
    }
}
