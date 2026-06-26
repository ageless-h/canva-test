use canvas_core::{
    AppendStrokeCommand, BeginStrokeCommand, BlendMode, BrushMode, CanvasCommand, CanvasConfig,
    CanvasCore, Color, CommandOutput, LayerGroupMode, LayerId, LayerSnapshot, LayerSpec,
    StabilizerConfig, StabilizerMode, StrokeId, StrokePoint,
};
use eframe::egui;
use eframe::egui_wgpu::wgpu;
use std::fs;
use std::sync::{Arc, Mutex};

fn main() -> eframe::Result<()> {
    let config = CanvasConfig {
        width: 1024,
        height: 1024,
        tile_size: 256,
        history_budget_bytes: 512 * 1024 * 1024,
        streaming_sample_min_distance: 0.0,
        // 启用引擎内置稳定器（One Euro 自适应平滑 + 向心 Catmull-Rom + 弧长重采样）
        stabilizer: StabilizerConfig {
            enabled: true,
            mode: StabilizerMode::Adaptive,
            smooth_strength: 0.5,
            spacing_ratio: 0.05,
        },
    };

    let mut core = pollster::block_on(CanvasCore::new(config)).expect("创建 CanvasCore 失败");

    let layer_id = LayerId(1);
    core.execute(&CanvasCommand::AddLayer(LayerSpec::new(layer_id)))
        .expect("添加图层失败");
    core.execute(&CanvasCommand::Composite)
        .expect("合成画布失败");

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

/// A0 审计层：记录输入采样诊断数据（平滑算法已下沉引擎，这里只看原始输入）。
#[derive(Default)]
struct StrokeAudit {
    frames: u32,
    raw_events: u32,
    max_event_gap: f32,
    sum_event_gap: f64,
    event_gap_count: u32,
    max_frame_dt: f32,
    last_frame_dt: f32,
    /// 原始事件点（画布坐标，drag 期间累积，下一笔清空），用于叠加显示
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

struct CanvasApp {
    core: Arc<Mutex<CanvasCore>>,
    layer_id: LayerId,
    brush_radius: f32,
    brush_color: [f32; 4],
    brush_mode: BrushMode,
    canvas_width: f32,
    canvas_height: f32,
    /// 当前活动笔触（引擎 streaming API），None 表示未在绘制
    active_stroke: Option<StrokeId>,
    /// 平滑强度 0~1（实时调引擎稳定器）
    smooth_strength: f32,
    /// 稳定器模式
    stab_mode: StabilizerMode,
    /// 笔尖预览：上一帧光标屏幕坐标 + 平滑速度(屏幕px/秒)
    last_cursor: Option<egui::Pos2>,
    pred_vel: egui::Vec2,
    /// 笔尖预览开关
    prediction: bool,
    texture_id: Option<egui::TextureId>,
    zoom: f32,
    pan: egui::Vec2,
    debug_overlay: bool,
    audit: StrokeAudit,
    fallback_pressure: f32,
    last_pressure: f32,
    pressure_from_device: bool,
    next_layer_id: u64,
    last_error: Option<String>,
    new_canvas_width: u32,
    new_canvas_height: u32,
    new_canvas_tile_size: u32,
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
            .expect("画布显示需要 wgpu 渲染状态");

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
            debug_overlay: true,
            audit: StrokeAudit::default(),
            fallback_pressure: 0.5,
            last_pressure: 0.5,
            pressure_from_device: false,
            next_layer_id: layer_id.0 + 1,
            last_error: None,
            new_canvas_width: 1024,
            new_canvas_height: 1024,
            new_canvas_tile_size: 256,
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

    /// composite + present，刷新显示纹理。
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
                .ok_or_else(|| "画布显示纹理尚未生成".to_owned())?
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
                if let Err(error) = self.register_presentation_texture(render_state) {
                    self.last_error = Some(error);
                } else {
                    self.last_error = None;
                }
            }
            Err(error) => {
                self.last_error = Some(error.to_string());
            }
        }
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
        BlendMode::Normal => "正常",
        BlendMode::Multiply => "正片叠底",
        BlendMode::Screen => "滤色",
        BlendMode::Overlay => "叠加",
        BlendMode::SoftLight => "柔光",
        BlendMode::ColorDodge => "颜色减淡",
        BlendMode::ColorBurn => "颜色加深",
        BlendMode::LinearBurn => "线性加深",
        BlendMode::Add => "添加",
        BlendMode::Subtract => "减去",
        BlendMode::Difference => "差值",
        BlendMode::Darken => "变暗",
        BlendMode::Lighten => "变亮",
        BlendMode::HardLight => "强光",
        BlendMode::Exclusion => "排除",
    }
}

fn group_mode_label(mode: LayerGroupMode) -> &'static str {
    match mode {
        LayerGroupMode::Isolated => "隔离",
        LayerGroupMode::PassThrough => "穿透",
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
                    let marker = if layer.spec.visible { "●" } else { "○" };
                    let response = ui.selectable_label(
                        layer.spec.id == self.layer_id,
                        format!(
                            "{marker} {}  {:.0}%  {} 块",
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

            ui.label(format!("当前: {}", layer_label(spec.id)));
            ui.checkbox(&mut spec.visible, "可见");
            ui.add(egui::Slider::new(&mut spec.opacity, 0.0..=1.0).text("不透明度"));
            ui.checkbox(&mut spec.alpha_locked, "锁定透明度");
            ui.checkbox(&mut spec.clip_to_below, "剪贴到下方");

            egui::ComboBox::from_label("混合模式")
                .selected_text(blend_mode_label(spec.blend_mode))
                .show_ui(ui, |ui| {
                    for mode in BLEND_MODES {
                        ui.selectable_value(&mut spec.blend_mode, mode, blend_mode_label(mode));
                    }
                });

            egui::ComboBox::from_label("蒙版")
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
            egui::ComboBox::from_label("父级")
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
                ui.heading("画笔设置");
                ui.add(egui::Slider::new(&mut self.brush_radius, 1.0..=50.0).text("大小"));
                if ui
                    .add(egui::Slider::new(&mut self.smooth_strength, 0.0..=1.0).text("平滑"))
                    .changed()
                {
                    let mut core = self.core.lock().unwrap();
                    core.set_stabilizer_strength(self.smooth_strength);
                }
                ui.horizontal(|ui| {
                    ui.label("稳定模式:");
                    let mut changed = false;
                    changed |= ui
                        .selectable_value(&mut self.stab_mode, StabilizerMode::Adaptive, "自适应")
                        .changed();
                    changed |= ui
                        .selectable_value(
                            &mut self.stab_mode,
                            StabilizerMode::PulledString,
                            "牵引线",
                        )
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
                    "当前压力: {:.2} ({})",
                    self.last_pressure,
                    if self.pressure_from_device {
                        "设备"
                    } else {
                        "模拟"
                    }
                ));

                ui.checkbox(&mut self.prediction, "笔尖预览");

                ui.label("绘制模式");
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.brush_mode, BrushMode::Paint, "绘制");
                    ui.selectable_value(&mut self.brush_mode, BrushMode::Erase, "橡皮擦");
                });

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
                ui.label("使用鼠标或数位板绘制");

                ui.separator();
                ui.heading("新建画布");
                ui.horizontal(|ui| {
                    ui.label("宽");
                    ui.add(egui::DragValue::new(&mut self.new_canvas_width).range(1..=8192));
                    ui.label("高");
                    ui.add(egui::DragValue::new(&mut self.new_canvas_height).range(1..=8192));
                });
                ui.horizontal(|ui| {
                    ui.label("块");
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
                ui.label("滚轮缩放");
                ui.label("右键拖拽平移");

                ui.separator();
                self.show_layer_panel(ui);

                ui.separator();
                ui.heading("输入审计");
                ui.checkbox(&mut self.debug_overlay, "显示叠加层（原始输入）");
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

            // 笔触开始
            if response.drag_started_by(egui::PointerButton::Primary) {
                self.audit.reset();
                self.last_cursor = response.interact_pointer_pos();
                self.pred_vel = egui::Vec2::ZERO;
                if let Some(pos) = response.interact_pointer_pos() {
                    let pressure = self.current_input_pressure(ui);
                    let p = to_canvas(pos, pressure);
                    let mut core = self.core.lock().unwrap();
                    let command = CanvasCommand::BeginStroke(BeginStrokeCommand {
                        layer: self.layer_id,
                        mode: self.brush_mode,
                        radius: self.brush_radius,
                        color: self.color(),
                        first_point: p,
                    });
                    if let Ok(CommandOutput::StrokeStarted { id }) = core.execute(&command) {
                        self.active_stroke = Some(id);
                    }
                }
            }

            // 笔触进行中：收集本帧所有高频 PointerMoved 事件，喂引擎 streaming API
            if response.dragged_by(egui::PointerButton::Primary) {
                self.audit.frames += 1;
                let frame_dt = ui.input(|i| i.stable_dt);
                let frame_dt_ms = frame_dt * 1000.0;
                self.audit.last_frame_dt = frame_dt_ms;
                if frame_dt_ms > self.audit.max_frame_dt {
                    self.audit.max_frame_dt = frame_dt_ms;
                }

                // 笔尖预览：估计屏幕空间速度（指数平滑），用于轻量外推
                if let Some(cur) = response.interact_pointer_pos() {
                    if let Some(prev) = self.last_cursor {
                        let dt = frame_dt.max(1e-4);
                        let inst_v = (cur - prev) / dt;
                        // 指数平滑速度，抑制抖动
                        let a = 0.4;
                        self.pred_vel = self.pred_vel * (1.0 - a) + inst_v * a;
                    }
                    self.last_cursor = Some(cur);
                }

                let screen_positions = self.input_samples(ui);
                self.audit.raw_events = self
                    .audit
                    .raw_events
                    .saturating_add(u32::try_from(screen_positions.len()).unwrap_or(u32::MAX));

                let mut batch: Vec<StrokePoint> = Vec::with_capacity(screen_positions.len());
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

            // 笔触结束
            if response.drag_stopped() {
                if let Some(id) = self.active_stroke.take() {
                    let mut core = self.core.lock().unwrap();
                    let _ = core.execute(&CanvasCommand::EndStroke(id));
                    Self::refresh(&mut core);
                }
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

            // 笔尖预览：只画落笔位置的小圆，不再画粗线，避免高速移动时形成横向残影。
            if self.prediction
                && let Some(id) = self.active_stroke
            {
                let tip_canvas = self.core.lock().unwrap().active_stroke_tip(id);
                if let (Some(tip), Some(cursor)) = (tip_canvas, self.last_cursor) {
                    let tip_screen = egui::pos2(
                        rect.min.x + (tip.x - self.pan.x) * display_scale,
                        rect.min.y + (tip.y - self.pan.y) * display_scale,
                    );
                    // 有界恒速度外推：预测时域 ~1.5 帧，按速度大小限幅
                    let horizon = 0.024_f32; // 秒，约 1.5 帧 @60fps
                    let speed = self.pred_vel.length();
                    // 高速时限制外推距离，防过冲；速度极低时不外推
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
