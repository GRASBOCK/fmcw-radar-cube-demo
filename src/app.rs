use crate::radar::{Object, Radar, scene_to_data};
use ndarray::{Array2, Array3, s};
use rustfft::num_complex::Complex64;

/// Range-Angle heatmap viewer (multi receiver; uses Radar::radar_cube which already includes angle FFT).
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct App {
    // Radar setup:
    carrier_frequency: f64,
    c: f64,
    sampling_frequency: f64,
    bandwidth: f64,
    chirp_duration: f64,
    chirp_count: usize,

    receivers: usize,
    receiver_spacing: f64,

    // Scene (two objects):
    obj1_range: f64,
    obj1_velocity: f64,
    obj1_angle_deg: f64,

    obj2_range: f64,
    obj2_velocity: f64,
    obj2_angle_deg: f64,
}

impl Default for App {
    fn default() -> Self {
        Self {
            carrier_frequency: 77e9,
            c: 299_792_458.0,
            sampling_frequency: 2e6,
            bandwidth: 1.6e9,
            chirp_duration: 40e-6,
            chirp_count: 64,

            receivers: 10,
            receiver_spacing: (299_792_458.0 / 77e9) / 2.0,

            obj1_range: 20.0,
            obj1_velocity: 10.0,
            obj1_angle_deg: -15.0,

            obj2_range: 35.0,
            obj2_velocity: -8.0,
            obj2_angle_deg: 25.0,
        }
    }
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Default::default()
        }
    }

    fn radar(&self) -> Radar {
        Radar {
            carrier_frequency: self.carrier_frequency,
            c: self.c,
            sampling_frequency: self.sampling_frequency,
            bandwidth: self.bandwidth,
            chirp_duration: self.chirp_duration,
            chirp_count: self.chirp_count,
            receivers: self.receivers, // <- 10 receivers
            receiver_spacing: self.receiver_spacing,
        }
    }

    fn scene(&self) -> Vec<Object> {
        vec![
            Object {
                angle: self.obj1_angle_deg.to_radians(),
                velocity: self.obj1_velocity,
                range: self.obj1_range,
            },
            Object {
                angle: self.obj2_angle_deg.to_radians(),
                velocity: self.obj2_velocity,
                range: self.obj2_range,
            },
        ]
    }

    /// Returns Range-Angle map as (angle_bins=nz, range_bins=nx).
    fn range_angle_map(&self) -> (Array2<f64>, Radar, usize, usize, usize) {
        let radar = self.radar();
        let objects = self.scene();

        // Raw data: (receivers=nz, chirps=ny, samples=nx)
        let data: Array3<Complex64> = scene_to_data(&radar, &objects);

        let nx = radar.sample_count_chirp();
        let ny = radar.chirp_count;
        let nz = radar.receivers;

        // radar_cube already does Range FFT, Doppler FFT, Angle FFT and returns magnitude:
        // shape (nz, ny, nx)
        let cube_mag = radar.radar_cube(&data);

        let doppler_bin = 20.min(ny.saturating_sub(1));

        // Range-Angle slice: (nz, nx) = take doppler bin
        let mut ra2d = cube_mag.slice(s![.., doppler_bin, ..]).to_owned();

        // fftshift along "angle bins" axis so broadside-ish energy is centered.
        // new[a, r] = old[(a + nz/2) % nz, r]
        let shift = nz / 2;
        let mut shifted = Array2::<f64>::zeros((nz, nx));
        for a in 0..nz {
            let src_a = (a + shift) % nz;
            for r in 0..nx {
                shifted[(a, r)] = ra2d[(src_a, r)];
            }
        }
        ra2d = shifted;

        (ra2d, radar, nz, ny, nx)
    }
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                let is_web = cfg!(target_arch = "wasm32");
                if !is_web {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.add_space(16.0);
                }
                egui::widgets::global_theme_preference_buttons(ui);
            });
        });

        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.heading("Radar parameters");

            ui.add(
                egui::Slider::new(&mut self.sampling_frequency, 0.2e6..=10e6)
                    .text("Sampling frequency (Hz)"),
            );
            ui.add(egui::Slider::new(&mut self.bandwidth, 0.1e9..=8e9).text("Bandwidth (Hz)"));
            ui.add(
                egui::Slider::new(&mut self.chirp_duration, 1e-6..=100e-6)
                    .text("Chirp duration (s)"),
            );
            ui.add(egui::Slider::new(&mut self.chirp_count, 4..=256).text("Chirp count"));

            ui.add(egui::Slider::new(&mut self.receivers, 2..=64).text("Receivers (Nz)"));
            ui.add(
                egui::Slider::new(&mut self.receiver_spacing, 0.0..=0.02)
                    .text("Receiver spacing (m)"),
            );
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Range–Angle heatmap (slice cube on Doppler)");



            ui.heading("Objects");
            ui.label("Object 1");
            ui.add(egui::Slider::new(&mut self.obj1_range, 1.0..=200.0).text("Range (m)"));
            ui.add(egui::Slider::new(&mut self.obj1_velocity, -80.0..=80.0).text("Velocity (m/s)"));
            ui.add(egui::Slider::new(&mut self.obj1_angle_deg, -80.0..=80.0).text("Angle (deg)"));

            ui.label("Object 2");
            ui.add(egui::Slider::new(&mut self.obj2_range, 1.0..=200.0).text("Range (m)"));
            ui.add(egui::Slider::new(&mut self.obj2_velocity, -80.0..=80.0).text("Velocity (m/s)"));
            ui.add(egui::Slider::new(&mut self.obj2_angle_deg, -80.0..=80.0).text("Angle (deg)"));

            ui.separator();

            // Build map first so we know ny for the slider max
            let (ra2d, radar, nz, ny, nx) = self.range_angle_map();

            fn clamp01(x: f64) -> f32 {
                if x <= 0.0 {
                    0.0
                } else if x >= 1.0 {
                    1.0
                } else {
                    x as f32
                }
            }

            fn colormap_turbo_like(t: f32) -> egui::Color32 {
                let t = t.clamp(0.0, 1.0);
                let r = (0.10 + 1.25 * t - 0.35 * t * t).clamp(0.0, 1.0);
                let g = (0.05 + 1.60 * t - 1.10 * t * t + 0.35 * t * t * t).clamp(0.0, 1.0);
                let b = (0.30 + 1.10 * t - 1.50 * t * t + 0.90 * t * t * t).clamp(0.0, 1.0);
                egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
            }

            let (min_v, max_v) = {
                let max_v = ra2d
                    .iter()
                    .fold(f64::NEG_INFINITY, |acc, &v| if v > acc { v } else { acc })
                    .max(0.0);
                (0.0, max_v)
            };
            let denom = (max_v - min_v).max(1e-12);

            // Image dims: width = range bins (x), height = angle bins (y)
            let mut img = egui::ColorImage::new([nx, nz], vec![egui::Color32::BLACK; nx * nz]);
            for a in 0..nz {
                for r in 0..nx {
                    let v = ra2d[(a, r)];
                    let t = (v - min_v) / denom;
                    img.pixels[a * nx + r] = colormap_turbo_like(clamp01(t));
                }
            }

            let texture = ui.ctx().load_texture(
                "range_angle_texture",
                img,
                egui::TextureOptions::NEAREST,
            );

            let max_range_m = radar.max_range();

            // We don't have a physical bin->angle calibration here; label bins roughly [-90,+90].
            let angle_min = -90.0;
            let angle_max = 90.0;

            ui.label(format!(
                "Range–Angle map. cube shape (angle, doppler, range)=({}, {}, {}). Max range ≈ {:.1} m",
                nz, ny, nx, max_range_m
            ));

            let available = ui.available_size();
            let aspect = nz as f32 / nx as f32;
            let desired_w = available.x.max(1.0);
            let desired_h = (desired_w * aspect).min(available.y.max(1.0));

            ui.horizontal(|ui| {
                ui.label("Range (m):");
                ui.label("0");
                ui.add_space(8.0);
                ui.label(format!("{:.2}", max_range_m));
            });

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label("Angle (deg)");
                    let ticks = [0.0, 0.25, 0.5, 0.75, 1.0];
                    for t in ticks {
                        let ang = angle_max - t * (angle_max - angle_min);
                        ui.label(format!("{ang:.0}"));
                        ui.add_space((desired_h / (ticks.len().saturating_sub(1) as f32)).max(0.0));
                    }
                });

                ui.add(egui::Image::from_texture(&texture).fit_to_exact_size(egui::vec2(
                    desired_w,
                    desired_h,
                )));
            });

            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                powered_by_egui_and_eframe(ui);
                egui::warn_if_debug_build(ui);
            });
        });
    }
}

fn powered_by_egui_and_eframe(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.label("Powered by ");
        ui.hyperlink_to("egui", "https://github.com/emilk/egui");
        ui.label(" and ");
        ui.hyperlink_to(
            "eframe",
            "https://github.com/emilk/egui/tree/master/crates/eframe",
        );
        ui.label(".");
    });
}
