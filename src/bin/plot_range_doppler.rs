use fmcw_radar_cube_demo::radar::{Object, Radar, scene_to_data};
use ndarray::{Array2, Array3, s};
use ndrustfft::{Complex, FftHandler, ndfft};
use rustfft::num_complex::Complex64;

/// 2D Range-Doppler heatmap viewer (single receiver).
///
/// Pipeline:
/// 1) Synthesize raw data for a scene: (receivers=1, chirps=Ny, samples=Nx)
/// 2) Range FFT along samples axis (axis=2)
/// 3) Doppler FFT along chirps axis (axis=1)
/// 4) Take magnitude, extract receiver 0 => (Ny, Nx)
/// 5) Plot as a heatmap using egui_plot.
pub struct App {
    // Radar setup:
    carrier_frequency: f64,
    c: f64,
    sampling_frequency: f64,
    bandwidth: f64,
    chirp_duration: f64,
    chirp_count: usize,

    // Scene (two objects):
    obj1_range: f64,
    obj1_velocity: f64,
    obj2_range: f64,
    obj2_velocity: f64,

    // Display:
    log_scale: bool,
    floor_db: f64,
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

            obj1_range: 20.0,
            obj1_velocity: 10.0,
            obj2_range: 20.0,
            obj2_velocity: -8.0,

            log_scale: true,
            floor_db: -60.0,
        }
    }
}

impl App {
    pub fn new(_: &eframe::CreationContext<'_>) -> Self {
        Default::default()
    }

    fn radar(&self) -> Radar {
        Radar {
            carrier_frequency: self.carrier_frequency,
            c: self.c,
            sampling_frequency: self.sampling_frequency,
            bandwidth: self.bandwidth,
            chirp_duration: self.chirp_duration,
            chirp_count: self.chirp_count,
            receivers: 1,          // single receiver as requested
            receiver_spacing: 0.0, // irrelevant with receivers=1
        }
    }

    fn scene(&self) -> Vec<Object> {
        vec![
            Object {
                angle: 0.0,
                velocity: self.obj1_velocity,
                range: self.obj1_range,
            },
            Object {
                angle: 0.0,
                velocity: self.obj2_velocity,
                range: self.obj2_range,
            },
        ]
    }

    fn range_doppler_map(&self) -> (Array2<f64>, usize, usize) {
        let radar = self.radar();
        let objects = self.scene();

        // Raw data: shape (receivers=1, chirps=Ny, samples=Nx)
        let data: Array3<Complex64> = scene_to_data(&radar, &objects);

        let nx = radar.sample_count_chirp();
        let ny = radar.chirp_count;
        let nz = radar.receivers;

        debug_assert_eq!(nz, 1);

        // Range FFT along samples axis (axis 2): (nz, ny, nx)
        let mut range_fft = Array3::<Complex<f64>>::zeros((nz, ny, nx));
        let range_handler = FftHandler::<f64>::new(nx);
        ndfft(&data.view(), &mut range_fft.view_mut(), &range_handler, 2);

        // Doppler FFT along chirp axis (axis 1): (nz, ny, nx)
        let mut rd_fft = Array3::<Complex<f64>>::zeros((nz, ny, nx));
        let doppler_handler = FftHandler::<f64>::new(ny);
        ndfft(
            &range_fft.view(),
            &mut rd_fft.view_mut(),
            &doppler_handler,
            1,
        );

        // Magnitude, take receiver 0 => (ny, nx)
        let rd_mag = rd_fft.map(|v| v.norm());
        let rd2d: Array2<f64> = rd_mag.slice(s![0, .., ..]).to_owned();

        (rd2d, ny, nx)
    }

    fn signal_map(&self) -> (Array2<f64>, usize, usize) {
        let radar = self.radar();
        let objects = self.scene();

        let nx = radar.sample_count_chirp();
        let ny = radar.chirp_count;

        // Raw data: shape (receivers=1, chirps=Ny, samples=Nx)
        let data: Array3<Complex64> = scene_to_data(&radar, &objects);

        let rd2d: Array2<f64> = data.slice(s![0, .., ..]).map(|c| c.re);

        (rd2d, ny, nx)
    }
}

impl eframe::App for App {
    fn save(&mut self, _: &mut dyn eframe::Storage) {}

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Range-Doppler (2D FFT) heatmap (single receiver)");

            ui.collapsing("Radar parameters", |ui| {
                ui.add(
                    egui::Slider::new(&mut self.sampling_frequency, 0.2e6..=10e6)
                        .text("Sampling frequency (Hz)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.bandwidth, 0.1e9..=8e9).text("Bandwidth (Hz)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.chirp_duration, 1e-6..=100e-6)
                        .text("Chirp duration (s)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.chirp_count, 4..=256).text("Chirp count"),
                );
            });

            ui.collapsing("Objects", |ui| {
                ui.label("Object 1");
                ui.add(egui::Slider::new(&mut self.obj1_range, 1.0..=200.0).text("Range (m)"));
                ui.add(
                    egui::Slider::new(&mut self.obj1_velocity, -40.0..=40.0).text("Velocity (m/s)"),
                );

                ui.separator();

                ui.label("Object 2");
                ui.add(egui::Slider::new(&mut self.obj2_range, 1.0..=200.0).text("Range (m)"));
                ui.add(
                    egui::Slider::new(&mut self.obj2_velocity, -80.0..=80.0).text("Velocity (m/s)"),
                );
            });

            ui.separator();

            ui.horizontal(|ui| {
                ui.checkbox(&mut self.log_scale, "Log scale (dB)");
                ui.add(egui::Slider::new(&mut self.floor_db, -120.0..=0.0).text("Floor (dB)"));
            });

            let radar = self.radar();
            let (sig2d, ny, nx) = self.signal_map();

            // Render a heatmap ourselves using egui::ColorImage + egui::Image,
            // because egui_plot 0.34 doesn't provide a heatmap primitive.
            let (min_v, max_v) = if self.log_scale {
                (self.floor_db, 0.0)
            } else {
                let max_v = sig2d
                    .iter()
                    .fold(f64::NEG_INFINITY, |acc, &v| if v > acc { v } else { acc })
                    .max(0.0);
                (0.0, max_v)
            };

            fn clamp01(x: f64) -> f32 {
                if x <= 0.0 {
                    0.0
                } else if x >= 1.0 {
                    1.0
                } else {
                    x as f32
                }
            }

            // A simple "turbo-like" gradient (not exact turbo; good enough for visualization).
            fn colormap_turbo_like(t: f32) -> egui::Color32 {
                // Piecewise polynomial-ish approximation (hand-tuned).
                // t in [0,1]
                let t = t.clamp(0.0, 1.0);
                let r = (0.10 + 1.25 * t - 0.35 * t * t).clamp(0.0, 1.0);
                let g = (0.05 + 1.60 * t - 1.10 * t * t + 0.35 * t * t * t).clamp(0.0, 1.0);
                let b = (0.30 + 1.10 * t - 1.50 * t * t + 0.90 * t * t * t).clamp(0.0, 1.0);
                egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
            }

            let denom = (max_v - min_v).max(1e-12);

            // Image dims: width = range bins (x), height = doppler bins (y).
            // egui 0.33 `ColorImage::new` expects the initial pixel buffer as a Vec<Color32>.
            let mut img = egui::ColorImage::new([nx, ny], vec![egui::Color32::BLACK; nx * ny]);
            for y in 0..ny {
                for x in 0..nx {
                    let v = sig2d[(y, x)];
                    let t = (v - min_v) / denom;
                    let t = clamp01(t);
                    img.pixels[y * nx + x] = colormap_turbo_like(t);
                }
            }

            let texture = ui.ctx().load_texture(
                "range_doppler_texture",
                img,
                egui::TextureOptions::NEAREST,
            );

            let available = ui.available_size();
            let aspect = ny as f32 / nx as f32;
            let desired_w = available.x.max(1.0);
            let desired_h = (desired_w * aspect).min(available.y.max(1.0));

            ui.add(egui::Image::from_texture(&texture).fit_to_exact_size(egui::vec2(
                desired_w,
                desired_h,
            )));

            // ==============

            let (rd2d, ny, nx) = self.range_doppler_map();

            let max_range_m = radar.max_range();
            let max_velocity_mps = radar.max_velocity();
            let range_bin_width_m = if nx > 0 {
                max_range_m / nx as f64
            } else {
                0.0
            };

            // Convert to dB if requested.
            let rd_display: Array2<f64> = if self.log_scale {
                rd2d.map(|v| {
                    // epsilon to avoid log(0)
                    let eps = 1e-12;
                    let db = 20.0 * (v + eps).log10();
                    db.max(self.floor_db)
                })
            } else {
                rd2d
            };

            // fftshift along Doppler axis (rows) so 0-velocity is centered.
            // new[y, x] = old[(y + ny/2) % ny, x]
            let doppler_shift = ny / 2;
            let rd_display = {
                let mut shifted = Array2::<f64>::zeros((ny, nx));
                for y in 0..ny {
                    let src_y = (y + doppler_shift) % ny;
                    for x in 0..nx {
                        shifted[(y, x)] = rd_display[(src_y, x)];
                    }
                }
                shifted
            };

            // Render a heatmap ourselves using egui::ColorImage + egui::Image,
            // because egui_plot 0.34 doesn't provide a heatmap primitive.
            let (min_v, max_v) = if self.log_scale {
                (self.floor_db, 0.0)
            } else {
                let max_v = rd_display
                    .iter()
                    .fold(f64::NEG_INFINITY, |acc, &v| if v > acc { v } else { acc })
                    .max(0.0);
                (0.0, max_v)
            };

            let denom = (max_v - min_v).max(1e-12);

            // Image dims: width = range bins (x), height = doppler bins (y).
            // egui 0.33 `ColorImage::new` expects the initial pixel buffer as a Vec<Color32>.
            let mut img = egui::ColorImage::new([nx, ny], vec![egui::Color32::BLACK; nx * ny]);
            for y in 0..ny {
                for x in 0..nx {
                    let v = rd_display[(y, x)];
                    let t = (v - min_v) / denom;
                    let t = clamp01(t);
                    img.pixels[y * nx + x] = colormap_turbo_like(t);
                }
            }

            let texture = ui.ctx().load_texture(
                "range_doppler_texture",
                img,
                egui::TextureOptions::NEAREST,
            );

            // Proper axis labels + simple tick marks (range in meters, doppler in bins).
            // Note: This is an image, so we draw labels/ticks around it manually.
            let range_ticks = [0.0, 0.25, 0.5, 0.75, 1.0];
            let doppler_ticks = [0.0, 0.25, 0.5, 0.75, 1.0];

            ui.label(format!(
                "Range–Doppler map (single receiver). Color = {}. Max velocity ≈ {:.2} m/s",
                if self.log_scale { "magnitude (dB)" } else { "magnitude" },
                max_velocity_mps
            ));

            let available = ui.available_size();
            let aspect = ny as f32 / nx as f32;
            let desired_w = available.x.max(1.0);
            let desired_h = (desired_w * aspect).min(available.y.max(1.0));

            // Top: X axis label
            ui.horizontal(|ui| {
                ui.label("Range (m):");
                ui.label(format!("0"));
                ui.add_space(8.0);
                ui.label(format!("{:.2}", max_range_m));
                ui.add_space(8.0);
                ui.label(format!("(≈ {:.4} m/bin)", range_bin_width_m));
            });

            // Middle: Y axis ticks + image
            ui.horizontal(|ui| {
                // Y-axis labels (velocity, after fftshift)
                ui.vertical(|ui| {
                    ui.label("Velocity (m/s)");
                    // Labels from top to bottom (image y increases downward).
                    // After fftshift, we interpret the axis as spanning [-max_velocity, +max_velocity).
                    for t in doppler_ticks {
                        let v = (1.0 - 2.0 * t) * max_velocity_mps;
                        ui.label(format!("{v:.1}"));
                        ui.add_space((desired_h / (doppler_ticks.len().saturating_sub(1) as f32)).max(0.0));
                    }
                });

                ui.add(egui::Image::from_texture(&texture).fit_to_exact_size(egui::vec2(
                    desired_w,
                    desired_h,
                )));
            });

            // Bottom: X-axis tick labels
            ui.horizontal(|ui| {
                ui.label("Range ticks:");
                for t in range_ticks {
                    let r = t * max_range_m;
                    ui.add_space(8.0);
                    ui.label(format!("{:.1}", r));
                }
            });

            ui.separator();

            ui.small(format!(
                "Shape: doppler_bins (chirps) = {}, range_bins (samples) = {}. Receiver: 1 (no angle FFT).",
                ny, nx
            ));
        });
    }
}

fn main() -> eframe::Result {
    env_logger::init();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([820.0, 640.0])
            .with_min_inner_size([520.0, 420.0])
            .with_icon(
                eframe::icon_data::from_png_bytes(&include_bytes!("../../assets/icon-256.png")[..])
                    .expect("Failed to load icon"),
            ),
        ..Default::default()
    };

    eframe::run_native(
        "Range-Doppler Heatmap",
        native_options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
