use canvas_core::{
    AppendStrokeCommand, BeginStrokeCommand, BlendMode, BrushDynamics, BrushMode, BrushStamp,
    BrushStampKind, BrushStampTextureId, CanvasCommand, CanvasConfig, CanvasCore, CanvasError,
    CaptureLevel, Color, CommandOutput, DataCaptureConfig, InputDeviceCapabilities, LayerGroupMode,
    LayerId, LayerSnapshot, LayerSpec, PressureCurve, SelectedPixels, SelectionCombineMode,
    SelectionPoint, SelectionPolygon, SelectionRect, SessionTrace, StabilizerConfig,
    StabilizerMode, StrokeId, StrokePoint, StrokePredictionConfig, TraceReport,
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
    max_event_gap: f32,
    sum_event_gap: f64,
    event_gap_count: u32,
    max_frame_dt: f32,
    last_frame_dt: f32,
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

struct CanvasApp {
    core: Arc<Mutex<CanvasCore>>,
    layer_id: LayerId,
    brush_radius: f32,
    brush_color: [f32; 4],
    brush_mode: BrushMode,
    canvas_width: f32,
    canvas_height: f32,
    active_stroke: Option<StrokeId>,
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
        let Some(canvas_core::TraceReplayCommand::Command(command)) = &command.replay else {
            return Err(CanvasError::MissingTraceResource(
                "测试程序重构暂不解析外部资源引用".to_owned(),
            ));
        };
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

    fn pressure_value(&mut self, force: Option<f32>) -> f32 {
        let pressure = force
            .filter(|value| value.is_finite())
            .map_or(self.fallback_pressure, |value| value.clamp(0.0, 1.0))
            .max(0.01);
        self.last_pressure = pressure;
        self.pressure_from_device = force.is_some();
        pressure
    }

    fn current_input_pressure(&mut self, ui: &egui::Ui) -> f32 {
        let force = ui.input(|i| {
            i.events.iter().rev().find_map(|event| match event {
                egui::Event::Touch {
                    phase: egui::TouchPhase::Start | egui::TouchPhase::Move | egui::TouchPhase::End,
                    force,
                    ..
                } => *force,
                _ => None,
            })
        });
        self.pressure_value(force)
    }

    fn input_samples(&mut self, ui: &egui::Ui) -> Vec<(egui::Pos2, f32)> {
        let touch_events: Vec<(egui::Pos2, Option<f32>)> = ui.input(|i| {
            i.events
                .iter()
                .filter_map(|event| match event {
                    egui::Event::Touch {
                        phase: egui::TouchPhase::Start | egui::TouchPhase::Move,
                        pos,
                        force,
                        ..
                    } => Some((*pos, *force)),
                    _ => None,
                })
                .collect()
        });

        if !touch_events.is_empty() {
            return touch_events
                .into_iter()
                .map(|(pos, force)| (pos, self.pressure_value(force)))
                .collect();
        }

        let pointer_events: Vec<egui::Pos2> = ui.input(|i| {
            i.events
                .iter()
                .filter_map(|event| match event {
                    egui::Event::PointerMoved(pos) => Some(*pos),
                    _ => None,
                })
                .collect()
        });

        pointer_events
            .into_iter()
            .map(|pos| (pos, self.pressure_value(None)))
            .collect()
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
            self.trace_rebuild_result = Some("还没有可重构的记录".to_owned());
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
                    format!("重构已完成，但显示纹理注册失败：{error}")
                } else {
                    trace_rebuild_summary(&trace, max_diff)
                }
            }
            Err(error) => format!("重构失败：{error}"),
        });
    }
}

fn install_chinese_fonts(ctx: &egui::Context) {
    const FONT_PATHS: &[&str] = &[
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

    let Some((font_path, font_bytes)) = FONT_PATHS
        .iter()
        .find_map(|path| fs::read(path).ok().map(|bytes| (*path, bytes)))
    else {
        eprintln!("未找到可用中文字体，测试程序中文界面可能显示为方框或乱码");
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
            let to_canvas = |sp: egui::Pos2, pressure: f32| StrokePoint {
                x: pan_x + (sp.x - rect.min.x) / display_scale,
                y: pan_y + (sp.y - rect.min.y) / display_scale,
                pressure,
            };

            // Stroke start.
            if response.drag_started_by(egui::PointerButton::Primary) {
                self.audit.reset();
                self.last_cursor = response.interact_pointer_pos();
                self.pred_vel = egui::Vec2::ZERO;
                if let Some(pos) = response.interact_pointer_pos() {
                    match self.tool_mode {
                        ToolMode::Brush => {
                            let pressure = self.current_input_pressure(ui);
                            let p = to_canvas(pos, pressure);
                            self.sync_core_brush_settings();
                            let mut core = self.core.lock().unwrap();
                            let command = CanvasCommand::BeginStroke(BeginStrokeCommand {
                                layer: self.layer_id,
                                mode: self.brush_mode,
                                radius: self.brush_radius,
                                color: self.color(),
                                first_point: p,
                            });
                            match core.execute(&command) {
                                Ok(CommandOutput::StrokeStarted { id }) => {
                                    self.active_stroke = Some(id);
                                    self.last_error = None;
                                }
                                Ok(_) => {
                                    self.last_error = Some("画笔命令未能开始笔画".to_owned());
                                }
                                Err(error) => self.last_error = Some(error.to_string()),
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
            if response.dragged_by(egui::PointerButton::Primary) {
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

                        let screen_positions = self.input_samples(ui);
                        self.audit.raw_events = self.audit.raw_events.saturating_add(
                            u32::try_from(screen_positions.len()).unwrap_or(u32::MAX),
                        );

                        let mut batch: Vec<StrokePoint> =
                            Vec::with_capacity(screen_positions.len());
                        for (sp, pressure) in &screen_positions {
                            let cp = to_canvas(*sp, *pressure);
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
                            batch.push(to_canvas(pos, self.current_input_pressure(ui)));
                        }

                        if let Some(id) = self.active_stroke
                            && !batch.is_empty()
                        {
                            let mut core = self.core.lock().unwrap();
                            let command = CanvasCommand::AppendStroke(AppendStrokeCommand {
                                stroke: id,
                                points: batch,
                            });
                            match core
                                .execute(&command)
                                .and_then(|_| Self::refresh(&mut core))
                            {
                                Ok(()) => self.last_error = None,
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
                        for (pos, _) in self.input_samples(ui) {
                            let point = Self::selection_point_from_canvas(to_canvas(pos, 1.0));
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
                            let mut core = self.core.lock().unwrap();
                            match core
                                .execute(&CanvasCommand::EndStroke(id))
                                .and_then(|_| Self::refresh(&mut core))
                            {
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
    fn trace_status_label_distinguishes_current_and_cached_records() {
        assert_eq!(trace_status_label(true, true, false), "正在记录当前绘画");
        assert_eq!(
            trace_status_label(false, false, true),
            "记录已停止，可重构上次记录"
        );
        assert_eq!(trace_status_label(false, false, false), "还没有可用记录");
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
            degradation_count: 1,
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
}
