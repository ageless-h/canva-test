use canvas_core::{
    AppendStrokeCommand, BeginStrokeCommand, BlendMode, BrushDynamics, BrushMode, BrushStamp,
    BrushStampKind, BrushStampTextureId, CanvasCommand, CanvasConfig, CanvasCore, Color,
    CommandOutput, InputDeviceCapabilities, LayerGroupMode, LayerId, LayerSnapshot, LayerSpec,
    PressureCurve, SelectedPixels, SelectionCombineMode, SelectionPoint, SelectionPolygon,
    SelectionRect, StabilizerConfig, StabilizerMode, StrokeId, StrokePoint, StrokePredictionConfig,
};
use eframe::egui;
use eframe::egui_wgpu::wgpu;
use std::fs;
use std::sync::{Arc, Mutex};

fn main() -> eframe::Result<()> {
    // Headless verification shortcut for the one-shot defect test (same code the
    // in-app button runs). Usage: `cargo run -- --repro`.
    if std::env::args().any(|arg| arg == "--repro") {
        match run_undo_replay_defect_repro() {
            Ok((reproduced, log)) => {
                println!("{log}\n\n[reproduced = {reproduced}]");
            }
            Err(error) => eprintln!("repro error: {error}"),
        }
        return Ok(());
    }

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
        ..CanvasConfig::default()
    };

    let mut core = pollster::block_on(CanvasCore::new(config)).expect("鍒涘缓 CanvasCore 澶辫触");

    let layer_id = LayerId(1);
    core.execute(&CanvasCommand::AddLayer(LayerSpec::new(layer_id)))
        .expect("娣诲姞鍥惧眰澶辫触");
    core.execute(&CanvasCommand::Composite)
        .expect("鍚堟垚鐢诲竷澶辫触");

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
            .with_title("鐢诲竷寮曟搸娴嬭瘯"),
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
        "鐢诲竷寮曟搸娴嬭瘯",
        options,
        Box::new(|cc| Ok(Box::new(CanvasApp::new(core, layer_id, cc)))),
    )
}

// ==========================================================================
// Repro: undo_by_gpu_replay must not destroy pixels on other layers.
// Scenario: layer 1 paints region A, layer 2 paints region B.
// Reordering layers invalidates the replay log without touching resident pixels.
// A later brush on layer 2 leaves the retained log with only that brush.
// undo_by_gpu_replay(1) should match regular undo and keep layer 1 pixels.
// The reveal step paints green below region A; preserved red means layer 1 survived.
// ============================================================================
const REPRO_A_X: usize = 42;
const REPRO_A_Y: usize = 40;
const REPRO_W: u32 = 128;
const REPRO_H: u32 = 128;
const REPRO_TILE: u32 = 32;

fn repro_config() -> CanvasConfig {
    CanvasConfig {
        width: REPRO_W,
        height: REPRO_H,
        tile_size: REPRO_TILE,
        ..CanvasConfig::default()
    }
}

fn repro_paint(layer: u64, x: f32, y: f32, color: Color) -> canvas_core::BrushCommand {
    canvas_core::BrushCommand {
        layer: LayerId(layer),
        mode: BrushMode::Paint,
        points: vec![
            StrokePoint {
                x,
                y,
                pressure: 1.0,
            },
            StrokePoint {
                x: x + 4.0,
                y,
                pressure: 1.0,
            },
        ],
        radius: 8.0,
        color,
    }
}

fn repro_a_index() -> usize {
    REPRO_A_Y * REPRO_W as usize + REPRO_A_X
}

fn repro_layer1_tiles(core: &CanvasCore) -> usize {
    core.layer_snapshot()
        .iter()
        .find(|s| s.spec.id.0 == 1)
        .map_or(0, |s| s.tile_count)
}

// live paint helper
fn live_paint(layer: u64, x: f32, y: f32, radius: f32, color: Color) -> canvas_core::BrushCommand {
    canvas_core::BrushCommand {
        layer: LayerId(layer),
        mode: BrushMode::Paint,
        points: vec![StrokePoint {
            x,
            y,
            pressure: 1.0,
        }],
        radius,
        color,
    }
}

async fn repro_build(core: &mut CanvasCore) -> Result<f32, canvas_core::CanvasError> {
    core.add_layer(LayerSpec::new(LayerId(1))); // 闁告瑦顨呴濠囧炊閹呮勾 -> 闁告牕鎼悡姗?    core.add_layer(LayerSpec::new(LayerId(2))); // 婵炶尪顕ф慨鈺呭炊閹呮勾 -> 闁告牕鎼悡姗?    core.apply_brush(&repro_paint(1, 40.0, 40.0, Color::rgba(1.0, 0.0, 0.0, 1.0)))?;
    core.apply_brush(&repro_paint(2, 96.0, 96.0, Color::rgba(0.0, 1.0, 0.0, 1.0)))?;
    let _handle = core.composite()?;
    let before = core.debug_readback().await?.rgba_f32[repro_a_index()][3];
    // Invalidate replay by moving layer 2 without touching pixels.
    core.move_layer(LayerId(2), 0)?;
    // Paint another stroke on layer 2; layer 1 is no longer in replay log.
    core.apply_brush(&repro_paint(2, 96.0, 70.0, Color::rgba(0.0, 1.0, 0.0, 1.0)))?;
    let _handle = core.composite()?;
    Ok(before)
}

async fn repro_reveal(core: &mut CanvasCore) -> Result<f32, canvas_core::CanvasError> {
    core.apply_brush(&repro_paint(2, 40.0, 40.0, Color::rgba(0.0, 1.0, 0.0, 1.0)))?;
    let _handle = core.composite()?;
    Ok(core.debug_readback().await?.rgba_f32[repro_a_index()][0])
}

// Run one undo replay defect repro.
fn run_undo_replay_defect_repro() -> Result<(bool, String), String> {
    pollster::block_on(async {
        // 闁革妇鍎ゅ▍?闁挎稒顒痭do_by_gpu_replay
        let mut g = CanvasCore::new(repro_config()).await?;
        let g_before = repro_build(&mut g).await?;
        let g_tiles_before = repro_layer1_tiles(&g);
        let report = g.undo_by_gpu_replay(1, Some(std::time::Duration::from_millis(50)))?;
        let _handle = g.composite()?;
        let g_tiles_after = repro_layer1_tiles(&g);
        let g_red = repro_reveal(&mut g).await?;

        // 闁革妇鍎ゅ▍?闁挎稑鐗嗛顕€鎮¤缁辨岸鏁嶅顓熺彯闂?undo()
        let mut s = CanvasCore::new(repro_config()).await?;
        let s_before = repro_build(&mut s).await?;
        let s_tiles_before = repro_layer1_tiles(&s);
        s.undo()?;
        let _handle = s.composite()?;
        let s_tiles_after = repro_layer1_tiles(&s);
        let s_red = repro_reveal(&mut s).await?;

        let baseline_ok = g_before > 0.9 && s_before > 0.9;
        let reproduced = baseline_ok && g_tiles_after == 0 && g_red < 0.1;
        let control_ok = s_tiles_after > 0 && s_red > 0.9;

        let mut log = String::new();
        log.push_str("鍦烘櫙1 undo_by_gpu_replay:\n");
        log.push_str(&format!(
            "  鎾ら攢鍓? 鍥惧眰1 tile={g_tiles_before}, 鍖哄煙A alpha={g_before:.3}\n"
        ));
        log.push_str(&format!(
            "  鎾ら攢: replayed={}, batch={}, key_tile={}\n",
            report.replayed_commands, report.batch_replay_used, report.key_tile_used
        ));
        log.push_str(&format!(
            "  鎾ら攢鍚? 鍥惧眰1 tile={g_tiles_after}, 鎻ず鍖哄煙A red={g_red:.3}\n"
        ));
        log.push_str("鍦烘櫙2 undo() 瀵圭収:\n");
        log.push_str(&format!(
            "  鎾ら攢鍓?tile={s_tiles_before} -> 鎾ら攢鍚?tile={s_tiles_after}, 鎻ず鍖哄煙A red={s_red:.3}\n"
        ));
        log.push_str("------------------------------\n");
        if reproduced && control_ok {
            log.push_str(&format!(
                "result: reproduced. replay undo cleared layer1 tiles from {g_tiles_before}; regular undo preserved layer1.\n"
            ));
        } else if baseline_ok && control_ok {
            log.push_str("result: not reproduced; engine may already be fixed.\n");
        } else {
            log.push_str("result: inconclusive; baseline/control state was unexpected.\n");
        }
        Ok::<(bool, String), canvas_core::CanvasError>((reproduced, log))
    })
    .map_err(|error: canvas_core::CanvasError| format!("{error:?}"))
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
    /// 缂佹鏌ㄩ惃閿嬶紣閸曨噮娼旈柨娑欑煯缁楀倹绋夐埀顒傛暜瑜嶉崢婊堝冀閸パ呮綄妤犵偞娲栧妤呭冀?+ 妤犵偠娅曠划锕傛焻閻斿嘲顔?閻忕偛绻愮粻绌歺/缂?
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
    /// 濞戞挴鍋撴繛鍡忓墲閳ь儸鍛箒闂傚嫬鍢查ˇ鏌ユ偝閻楀牏銈撮悹鍥ㄦ礈缁劑寮稿鎰獥Ok((鐎瑰憡褰冮ˇ鏌ユ偝? 闁哄啨鍎辩换?) / Err(閺夆晜鍔橀、鎴︽煥濞嗘帩鍤?
    defect_test: Option<Result<(bool, String), String>>,
}

fn stamp_kind_label(kind: BrushStampKind) -> &'static str {
    match kind {
        BrushStampKind::Circle => "鍦嗗舰",
        BrushStampKind::Square => "鏂瑰舰",
        BrushStampKind::Diamond => "鑿卞舰",
        BrushStampKind::Stripe => "鏉＄汗",
    }
}

fn brush_preset_label(preset: BrushPreset) -> &'static str {
    match preset {
        BrushPreset::SoftRound => "Soft Round",
        BrushPreset::HardRound => "Hard Round",
        BrushPreset::SoftRoundPressureSize => "Soft Round Pressure Size",
        BrushPreset::HardRoundPressureSize => "Hard Round Pressure Size",
        BrushPreset::SoftRoundPressureOpacity => "Soft Round Pressure Opacity",
        BrushPreset::HardRoundPressureOpacity => "Hard Round Pressure Opacity",
        BrushPreset::SoftRoundPressureOpacityFlow => "Soft Round Pressure Opacity + Flow",
        BrushPreset::HardRoundPressureOpacityFlow => "Hard Round Pressure Opacity + Flow",
    }
}

fn selection_mode_label(mode: SelectionCombineMode) -> &'static str {
    match mode {
        SelectionCombineMode::Replace => "鏇挎崲",
        SelectionCombineMode::Add => "娣诲姞",
        SelectionCombineMode::Subtract => "鍑忓幓",
        SelectionCombineMode::Intersect => "鐩镐氦",
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
        core_guard.present().expect("鏄剧ず鐢诲竷澶辫触");
        let texture_view = core_guard
            .presentation_texture_view()
            .expect("鏄剧ず鍚庡簲瀛樺湪鐢诲竷绾圭悊瑙嗗浘");
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
            defect_test: None,
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
    fn refresh(core: &mut CanvasCore) {
        let _ = core.execute(&CanvasCommand::Composite);
        let _ = core.present();
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
            let result = core.execute(&command);
            if result.is_ok() {
                Self::refresh(&mut core);
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
                    Self::refresh(&mut core);
                }
            }
            Ok(None) => self.last_error = Some("no active selection to copy".to_owned()),
            Err(error) => self.last_error = Some(error.to_string()),
        }
    }

    fn paste_clipboard_to_active_layer(&mut self) {
        let Some(pixels) = self.clipboard.clone() else {
            self.last_error = Some("clipboard is empty".to_owned());
            return;
        };

        let origin_x = pixels.bounds.x;
        let origin_y = pixels.bounds.y;
        let result = {
            let mut core = self.core.lock().unwrap();
            let result = core.paste_pixels_to_layer(self.layer_id, &pixels, origin_x, origin_y);
            if result.is_ok() {
                Self::refresh(&mut core);
            }
            result
        };
        match result {
            Ok(true) => self.last_error = None,
            Ok(false) => self.last_error = Some("paste area is outside canvas".to_owned()),
            Err(error) => self.last_error = Some(error.to_string()),
        }
    }

    fn paste_clipboard_to_new_layer(&mut self) {
        let Some(pixels) = self.clipboard.clone() else {
            self.last_error = Some("clipboard is empty".to_owned());
            return;
        };

        let layer_id = LayerId(self.next_layer_id);
        let origin_x = pixels.bounds.x;
        let origin_y = pixels.bounds.y;
        let result = {
            let mut core = self.core.lock().unwrap();
            let result = core.paste_pixels_to_new_layer(
                LayerSpec::new(layer_id),
                &pixels,
                origin_x,
                origin_y,
            );
            if result.is_ok() {
                Self::refresh(&mut core);
            }
            result
        };
        match result {
            Ok(true) => {
                self.layer_id = layer_id;
                self.next_layer_id = self.next_layer_id.saturating_add(1);
                self.last_error = None;
            }
            Ok(false) => self.last_error = Some("绮樿创鍖哄煙鍦ㄧ敾甯冨".to_owned()),
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
                .ok_or_else(|| "presentation texture view is missing".to_owned())?
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
            self.last_error = Some("cannot access WGPU render state".to_owned());
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
                .and_then(|_| core.execute(&CanvasCommand::Composite))
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
    // Run one-shot live canvas defect repro.
    fn run_defect_on_live_canvas(&mut self, frame: &mut eframe::Frame) {
        let Some(render_state) = frame.wgpu_render_state() else {
            self.defect_test = Some(Err("cannot access WGPU render state".to_owned()));
            return;
        };

        let core_arc = self.core.clone();
        let cfg = core_arc.lock().unwrap().config();
        let w = cfg.width as f32;
        let h = cfg.height as f32;
        let cx = w * 0.5;
        let cy = h * 0.5;
        let full_r = w.max(h);
        let a_index = (cy as usize) * cfg.width as usize + cx as usize;
        let red = Color::rgba(1.0, 0.0, 0.0, 1.0);
        let green = Color::rgba(0.0, 1.0, 0.0, 1.0);

        let result: Result<(bool, String), String> = pollster::block_on(async {
            let mut core = core_arc.lock().unwrap();
            let to_err = |e: canvas_core::CanvasError| e.to_string();

            core.execute(&CanvasCommand::NewCanvas(cfg))
                .map_err(to_err)?;
            core.execute(&CanvasCommand::AddLayer(LayerSpec::new(LayerId(1))))
                .map_err(to_err)?;
            core.execute(&CanvasCommand::AddLayer(LayerSpec::new(LayerId(2))))
                .map_err(to_err)?;

            // Layer 1 fills red; layer 2 gets a green corner stroke.
            core.apply_brush(&live_paint(1, cx, cy, full_r, red))
                .map_err(to_err)?;
            core.apply_brush(&live_paint(2, w * 0.85, h * 0.15, h * 0.08, green))
                .map_err(to_err)?;
            let _handle = core.composite().map_err(to_err)?;
            let before_red = core.debug_readback().await.map_err(to_err)?.rgba_f32[a_index][0];

            // 濠㈡儼椴搁弲銉╂晬濮樿泛娅㈤柟鐑樺笒濞存浠?-> 婵炴挸鎳愰埞鏍川閹存帗濮㈤梺鎻掔У閺備線寮妷銉х闁挎稑鐗嗛崕姘辨閻樿京鐭濋柛锔荤厜缁?            core.move_layer(LayerId(2), 0).map_err(to_err)?;
            // Draw another layer 2 stroke that will be undone.
            core.apply_brush(&live_paint(2, w * 0.85, h * 0.85, h * 0.08, green))
                .map_err(to_err)?;
            let _handle = core.composite().map_err(to_err)?;
            let tiles_before = core
                .layer_snapshot()
                .iter()
                .find(|s| s.spec.id.0 == 1)
                .map_or(0, |s| s.tile_count);

            // 闁?undo_by_gpu_replay 闁逛勘鍊濋弨銏ゆ焽閿濆嫮顏辩紒妤佹⒐濡倝宕楀畷鍥ㄧ暠闁搞儲鍎抽惇? 缂佹妫佽
            let report = core
                .undo_by_gpu_replay(1, Some(std::time::Duration::from_millis(50)))
                .map_err(to_err)?;
            let _handle = core.composite().map_err(to_err)?;
            let tiles_after = core
                .layer_snapshot()
                .iter()
                .find(|s| s.spec.id.0 == 1)
                .map_or(0, |s| s.tile_count);

            core.apply_brush(&live_paint(2, cx, cy, full_r, green))
                .map_err(to_err)?;
            let _handle = core.composite().map_err(to_err)?;
            let after_red = core.debug_readback().await.map_err(to_err)?.rgba_f32[a_index][0];

            let reproduced = before_red > 0.9 && tiles_after == 0 && after_red < 0.1;
            let mut log = String::new();
            log.push_str(&format!(
                "before replay undo: layer1 tiles={tiles_before}, red={before_red:.3}\n"
            ));
            log.push_str(&format!(
                "undo replay: replayed={}, batch={}\n",
                report.replayed_commands, report.batch_replay_used
            ));
            log.push_str(&format!("after replay undo: layer1 tiles={tiles_after}\n"));
            log.push_str(&format!("after green reveal: red={after_red:.3}\n"));
            log.push_str("------------------------------\n");
            if reproduced {
                log.push_str("result: reproduced. live canvas turned green after replay undo.\n");
            } else {
                log.push_str("result: not reproduced or inconclusive.\n");
            }
            Ok::<(bool, String), String>((reproduced, log))
        });

        // Refresh display texture so the canvas reflects the final state.
        {
            let mut core = core_arc.lock().unwrap();
            Self::refresh(&mut core);
        }
        // NewCanvas reallocates the presentation texture, so the previously
        // registered egui texture id is stale; re-register it or the canvas
        // would not visibly update after the demo.
        if let Err(error) = self.register_presentation_texture(render_state) {
            self.last_error = Some(error);
        }
        // Reset interaction state.
        self.layer_id = LayerId(1);
        self.next_layer_id = 3;
        self.active_stroke = None;
        self.last_cursor = None;
        self.pred_vel = egui::Vec2::ZERO;
        self.zoom = 1.0;
        self.pan = egui::Vec2::ZERO;
        self.canvas_width = w;
        self.canvas_height = h;
        self.audit.reset();
        self.defect_test = Some(result);
    }
}

fn install_chinese_fonts(ctx: &egui::Context) {
    const FONT_PATHS: &[&str] = &[
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ];

    let Some(font_bytes) = FONT_PATHS.iter().find_map(|path| fs::read(path).ok()) else {
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
        BlendMode::Normal => "姝ｅ父",
        BlendMode::Multiply => "姝ｇ墖鍙犲簳",
        BlendMode::Screen => "婊よ壊",
        BlendMode::Overlay => "鍙犲姞",
        BlendMode::SoftLight => "鏌斿厜",
        BlendMode::ColorDodge => "棰滆壊鍑忔贰",
        BlendMode::LinearBurn => "Linear Burn",
        BlendMode::ColorBurn => "棰滆壊鍔犳繁",
        BlendMode::Add => "娣诲姞",
        BlendMode::Difference => "Difference",
        BlendMode::Subtract => "鍑忓幓",
        BlendMode::Darken => "鍙樻殫",
        BlendMode::Lighten => "鍙樹寒",
        BlendMode::HardLight => "寮哄厜",
        BlendMode::Exclusion => "鎺掗櫎",
    }
}

fn group_mode_label(mode: LayerGroupMode) -> &'static str {
    match mode {
        LayerGroupMode::PassThrough => "Pass Through",
        LayerGroupMode::Isolated => "闅旂",
    }
}

fn layer_label(id: LayerId) -> String {
    format!("鍥惧眰 {}", id.0)
}

impl CanvasApp {
    fn show_layer_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("鍥惧眰");

        let layers = self.layers();
        let selected_index = layers
            .iter()
            .position(|layer| layer.spec.id == self.layer_id);

        ui.horizontal(|ui| {
            if ui.button("鏂板缓").clicked() {
                let id = LayerId(self.next_layer_id);
                self.next_layer_id = self.next_layer_id.saturating_add(1);
                self.layer_id = id;
                self.execute_canvas_command(CanvasCommand::AddLayer(LayerSpec::new(id)));
            }

            if ui
                .add_enabled(layers.len() > 1, egui::Button::new("鍒犻櫎"))
                .clicked()
            {
                self.execute_canvas_command(CanvasCommand::RemoveLayer(self.layer_id));
            }

            if ui.button("Clear").clicked() {
                self.execute_canvas_command(CanvasCommand::ClearLayer(self.layer_id));
            }
        });

        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    selected_index.is_some_and(|index| index + 1 < layers.len()),
                    egui::Button::new("涓婄Щ"),
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
                    egui::Button::new("涓嬬Щ"),
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
                        "visible"
                    } else {
                        "hidden"
                    };
                    let response = ui.selectable_label(
                        layer.spec.id == self.layer_id,
                        format!(
                            "{marker} {}  {:.0}%  {} tiles",
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

            ui.add(egui::Slider::new(&mut spec.opacity, 0.0..=1.0).text("Opacity"));
            ui.checkbox(&mut spec.alpha_locked, "Lock alpha");
            ui.checkbox(&mut spec.clip_to_below, "Clip to below");

            egui::ComboBox::from_label("Blend mode")
                .selected_text(blend_mode_label(spec.blend_mode))
                .show_ui(ui, |ui| {
                    for mode in BLEND_MODES {
                        ui.selectable_value(&mut spec.blend_mode, mode, blend_mode_label(mode));
                    }
                });
            egui::ComboBox::from_label("Mask layer")
                .selected_text(spec.mask_layer.map_or_else(|| "None".to_owned(), layer_label))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut spec.mask_layer, None, "None");
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
            egui::ComboBox::from_label("Parent")
                .selected_text(parent.map_or_else(|| "None".to_owned(), layer_label))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut parent, None, "None");
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
            egui::ComboBox::from_label("Group mode")
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
                ui.heading("宸ュ叿");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.tool_mode, ToolMode::Brush, "鐢荤瑪");
                    ui.selectable_value(&mut self.tool_mode, ToolMode::RectSelection, "Rect");
                    ui.selectable_value(&mut self.tool_mode, ToolMode::LassoSelection, "濂楃储");
                });
                egui::ComboBox::from_label("閫夊尯缁勫悎")
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

                ui.heading("鐢荤瑪");
                ui.add(egui::Slider::new(&mut self.brush_radius, 1.0..=50.0).text("澶у皬"));
                let mut preset = self.brush_preset;
                egui::ComboBox::from_label("PS Brush")
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
                    .add(egui::Slider::new(&mut self.smooth_strength, 0.0..=1.0).text("Smooth"))
                    .changed()
                {
                    let mut core = self.core.lock().unwrap();
                    core.set_stabilizer_strength(self.smooth_strength);
                }
                ui.horizontal(|ui| {
                    ui.label("Stabilizer");
                    let mut changed = false;
                    changed |= ui
                        .selectable_value(&mut self.stab_mode, StabilizerMode::Adaptive, "Adaptive")
                        .changed();
                    changed |= ui
                        .selectable_value(
                            &mut self.stab_mode,
                            StabilizerMode::PulledString,
                            "Pulled String",
                        )
                        .changed();
                    if changed {
                        let mut core = self.core.lock().unwrap();
                        core.set_stabilizer_mode(self.stab_mode);
                    }
                });

                ui.label("棰滆壊");
                ui.color_edit_button_rgba_unmultiplied(&mut self.brush_color);

                ui.add(egui::Slider::new(&mut self.fallback_pressure, 0.01..=1.0).text("妯℃嫙鍘嬪姏"));
                ui.label(format!(
                    "鍘嬪姏: {:.2} ({})",
                    self.last_pressure,
                    if self.pressure_from_device {
                        "璁惧"
                    } else {
                        "妯℃嫙"
                    }
                ));

                ui.checkbox(&mut self.prediction, "绗斿皷棰勮");
                ui.checkbox(&mut self.core_prediction, "鏍稿績棰勬祴灏炬");

                ui.separator();
                ui.heading("Brush dynamics");
                ui.label(format!(
                    "Pressure mapping: {}",
                    if self.pressure_dynamics {
                        "PS preset"
                    } else {
                        "off"
                    }
                ));
                ui.checkbox(&mut self.velocity_dynamics, "Velocity dynamics");
                ui.checkbox(&mut self.texture_grain, "鍘嬪姏绾圭悊棰楃矑");
                egui::ComboBox::from_label("Brush shape")
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
                ui.add(egui::Slider::new(&mut self.stamp_hardness, 0.0..=1.0).text("绗斿皷纭害"));
                ui.add(egui::Slider::new(&mut self.stamp_aspect, 0.125..=4.0).text("Aspect"));
                ui.add(egui::Slider::new(&mut self.stamp_angle, 0.0..=1.0).text("绗斿皷瑙掑害"));
                ui.checkbox(&mut self.stamp_texture, "澶栭儴绗斿皷绾圭悊閬僵");
                self.sync_core_brush_settings();
                ui.label("Brush mode");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.brush_mode, BrushMode::Paint, "Paint");
                    ui.selectable_value(&mut self.brush_mode, BrushMode::Erase, "Erase");
                });

                ui.separator();
                ui.heading("閫夊尯");
                let selection_report = self.core.lock().unwrap().selection_report();
                if let Some(bounds) = selection_report.bounds {
                    ui.label(format!(
                        "鑼冨洿: {},{} {}x{}  鍒嗘={}",
                        bounds.x,
                        bounds.y,
                        bounds.width,
                        bounds.height,
                        selection_report.span_count
                    ));
                } else {
                    ui.label("鏃犳椿鍔ㄩ€夊尯");
                }
                ui.horizontal(|ui| {
                    if ui.button("娓呴櫎閫夊尯").clicked() {
                        self.execute_canvas_command(CanvasCommand::ClearSelection);
                    }
                    if ui.button("Invert").clicked() {
                        self.execute_canvas_command(CanvasCommand::InvertSelection);
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("娓呯┖閫変腑鍍忕礌").clicked() {
                        self.execute_canvas_command(CanvasCommand::ClearSelectedPixels(
                            self.layer_id,
                        ));
                    }
                    if ui.button("澶嶅埗").clicked() {
                        self.copy_selection_to_clipboard(false);
                    }
                    if ui.button("鍓垏").clicked() {
                        self.copy_selection_to_clipboard(true);
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("绮樿创").clicked() {
                        self.paste_clipboard_to_active_layer();
                    }
                    if ui.button("绮樿创涓烘柊鍥惧眰").clicked() {
                        self.paste_clipboard_to_new_layer();
                    }
                });
                if let Some(pixels) = &self.clipboard {
                    ui.label(format!(
                        "鍓创鏉? {}x{} 鏉ヨ嚜鍥惧眰 {}",
                        pixels.bounds.width, pixels.bounds.height, pixels.layer.0
                    ));
                } else {
                    ui.label("Clipboard: empty");
                }

                ui.separator();

                if ui.button("娓呯┖褰撳墠鍥惧眰").clicked() {
                    self.execute_canvas_command(CanvasCommand::ClearLayer(self.layer_id));
                }
                if ui.button("鎾ら攢").clicked() {
                    self.execute_canvas_command(CanvasCommand::Undo);
                }
                if ui.button("閲嶅仛").clicked() {
                    self.execute_canvas_command(CanvasCommand::Redo);
                }

                ui.separator();
                ui.heading("閲嶆斁缂洪櫡澶嶇幇");
                ui.label("Run undo_by_gpu_replay repro on the current canvas.");
                if ui.button("鍦ㄧ敾甯冧笂杩愯澶嶇幇").clicked() {
                    self.run_defect_on_live_canvas(frame);
                }
                match &self.defect_test {
                    Some(Ok((reproduced, report))) => {
                        if *reproduced {
                            ui.colored_label(
                                egui::Color32::from_rgb(220, 60, 60),
                                "Reproduced: undo_by_gpu_replay damaged other layers.",
                            );
                        } else {
                            ui.colored_label(egui::Color32::from_rgb(60, 180, 90), "Not reproduced.");
                        }
                        ui.monospace(report);
                    }
                    Some(Err(error)) => {
                        ui.colored_label(
                            egui::Color32::from_rgb(220, 60, 60),
                            format!("閿欒: {error}"),
                        );
                    }
                    None => {}
                }

                ui.separator();
                ui.label("Use mouse or tablet input.");

                ui.separator();
                ui.heading("鏂板缓鐢诲竷");
                ui.horizontal(|ui| {
                    ui.label("W");
                    ui.add(egui::DragValue::new(&mut self.new_canvas_width).range(1..=8192));
                    ui.label("H");
                    ui.add(egui::DragValue::new(&mut self.new_canvas_height).range(1..=8192));
                });
                ui.horizontal(|ui| {
                    ui.label("Tile");
                    ui.add(egui::DragValue::new(&mut self.new_canvas_tile_size).range(1..=1024));
                    if ui.button("鍒涘缓").clicked() {
                        self.create_new_canvas(frame);
                    }
                });
                ui.label(format!(
                    "褰撳墠鐢诲竷: {} x {}",
                    self.canvas_width as u32, self.canvas_height as u32
                ));

                ui.separator();
                ui.heading("缂╂斁");
                ui.add(egui::Slider::new(&mut self.zoom, 1.0..=32.0).text("缂╂斁"));
                ui.horizontal(|ui| {
                    if ui.button("閫傚簲").clicked() {
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
                ui.label("Mouse wheel zoom");
                ui.label("Right drag to pan");

                ui.separator();
                self.show_layer_panel(ui);

                ui.separator();
                ui.heading("Input audit");
                ui.checkbox(&mut self.debug_overlay, "Show raw input overlay");
                let a = &self.audit;
                let rate = if a.frames > 0 {
                    f64::from(a.raw_events) / f64::from(a.frames)
                } else {
                    0.0
                };
                ui.label(format!("甯ф暟: {}", a.frames));
                ui.label(format!("鍘熷浜嬩欢: {}", a.raw_events));
                ui.label(format!("姣忓抚浜嬩欢: {rate:.2}"));
                ui.label(format!("Max gap: {:.1}px", a.max_event_gap));
                ui.label(format!("骞冲潎闂磋窛: {:.1}px", a.avg_event_gap()));
                ui.label(format!(
                    "Frame dt: {:.1}ms (max {:.1})",
                    a.last_frame_dt, a.max_frame_dt
                ));
                ui.label("Red = raw input points");
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
                            if let Ok(CommandOutput::StrokeStarted { id }) = core.execute(&command)
                            {
                                self.active_stroke = Some(id);
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
                            let _ = core.execute(&command);
                            Self::refresh(&mut core);
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
                            let _ = core.execute(&CanvasCommand::EndStroke(id));
                            Self::refresh(&mut core);
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
                    // 闁哄牆顦遍弲顐﹀箒閹烘せ鍋撻悢宄邦唺濠㈣埖鐗楃敮褰掓晬濮橀鏆曟繛鏉戭儐濡炲倿宕?~1.5 閻㈩垎宥囩闁圭顦甸埀顒傚枎鐎硅櫕寰勮閻剟姊介幇顒傜暯
                    let horizon = 0.024_f32; // 缂佸甯槐婵堢棯?1.5 閻?@60fps
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

