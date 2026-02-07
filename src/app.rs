use crate::radar::{Object, Radar, detect, scene_to_data};
use egui::vec2;
use egui_plot::{Arrows, Legend, Plot, PlotImage, PlotPoint, PlotPoints, Points};
use ndarray::{Array3, s};
use rustfft::num_complex::Complex64;

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

    // Scene:
    objects: Vec<Object>,
}

impl Default for App {
    fn default() -> Self {
        let c = 299_792_458.0;
        let cf = 77e9;
        Self {
            carrier_frequency: cf,
            c,
            sampling_frequency: 2.4e6,
            bandwidth: 320e6,
            chirp_duration: 90e-6,
            chirp_count: 64,

            receivers: 36,
            receiver_spacing: (c / cf) / 2.0,

            objects: vec![
                Object {
                    angle: -10.0,
                    range: 69.0,
                    velocity: 8.0,
                },
                Object {
                    angle: 38.0,
                    range: 27.0,
                    velocity: -2.5,
                },
            ],
        }
    }
}

impl App {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
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
            receivers: self.receivers, // <- 10 receivers
            receiver_spacing: self.receiver_spacing,
        }
    }

    fn cube_mag(&self) -> (Array3<f64>, Radar, usize, usize, usize) {
        let radar = self.radar();
        let objects: Vec<Object> = self
            .objects
            .iter()
            .map(|obj| Object {
                angle: obj.angle.to_radians(),
                velocity: obj.velocity,
                range: obj.range,
            })
            .collect();

        // Raw data: (receivers=nz, chirps=ny, samples=nx)
        let data: Array3<Complex64> = scene_to_data(&radar, &objects);

        let nx = radar.sample_count_chirp();
        let ny = radar.chirp_count;
        let nz = radar.receivers;

        // radar_cube already does Range FFT, Doppler FFT, Angle FFT and returns magnitude:
        // shape (nz, ny, nx)
        let cube_mag = radar.radar_cube(&data);
        (cube_mag, radar, nz, ny, nx)
    }
}

impl eframe::App for App {
    fn save(&mut self, _storage: &mut dyn eframe::Storage) {}

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.heading("Radar parameters");
            ui.label(format!(
                "carrier frequency: {:.1} GHz",
                self.radar().carrier_frequency / 1E9
            ));
            ui.label(format!(
                "receiver spacing: {:.2} mm",
                self.radar().receiver_spacing * 1000.0
            ));

            ui.add(
                egui::Slider::new(&mut self.sampling_frequency, 0.2e6..=3.5e6)
                    .text("Sampling frequency (Hz)"),
            );
            ui.add(egui::Slider::new(&mut self.bandwidth, 0.1e9..=1e9).text("Bandwidth (Hz)"));
            ui.add(
                egui::Slider::new(&mut self.chirp_duration, 1e-6..=200e-6)
                    .text("Chirp duration (s)"),
            );
            ui.add(egui::Slider::new(&mut self.chirp_count, 4..=256).text("Chirp count"));

            ui.add(egui::Slider::new(&mut self.receivers, 2..=64).text("Receivers (Nz)"));
            ui.separator();
            ui.heading("Objects");
            for (i, obj) in self.objects.iter_mut().enumerate() {
                ui.label(format!("Object {}", i + 1));
                ui.add(egui::Slider::new(&mut obj.range, 1.0..=100.0).text("Range (m)"));
                ui.add(egui::Slider::new(&mut obj.velocity, -20.0..=20.0).text("Velocity (m/s)"));
                ui.add(egui::Slider::new(&mut obj.angle, -80.0..=80.0).text("Angle (deg)"));
            }
        });

        // Build map first so we know ny for the slider max
        let (fft_cube, radar, nz, ny, nx) = self.cube_mag();

        let doppler_bin = 20.min(ny.saturating_sub(1));

        // Range-Angle slice: (nz, nx) = take doppler bin
        let ra2d = fft_cube.slice(s![.., doppler_bin, ..]).to_owned();

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

        let max_range = radar.max_range();
        let max_angle = radar.max_angle();
        let max_velocity = radar.max_velocity();
        let max_angle_deg = max_angle.to_degrees();

        // Image dims: width = range bins (x), height = angle bins (y)
        let nz_ = nz * 3;
        let mut img = egui::ColorImage::new([nx, nz_], vec![egui::Color32::BLACK; nx * nz_]);
        for z_ in 0..nz_ {
            for x in 0..nx {
                let angle = -max_angle_deg * 2.0 * (z_ as f64 / nz_ as f64) + max_angle_deg;
                let z = radar.z_from_angle(angle.to_radians()) as usize % nz;
                let v = ra2d[(z, x)];
                let t = (v - min_v) / denom;
                img.pixels[z_ * nx + x] = colormap_turbo_like(clamp01(t));
            }
        }

        let detection_coords = detect(&fft_cube);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Range–Angle heatmap (slice cube on Doppler)");

            let texture =
                ui.ctx()
                    .load_texture("range_angle_texture", img, egui::TextureOptions::NEAREST);
            let image = PlotImage::new(
                "range_angle_texture",
                &texture,
                PlotPoint::new(max_range/2.0, 0.0),
                vec2(max_range as f32, (max_angle_deg*2.0) as f32),
            );

            let true_velocity_arrows = {
                let arrow_origins = PlotPoints::from_iter(self.objects.iter().map(|obj| [obj.range, obj.angle]));
                let arrow_tips = PlotPoints::from_iter(self.objects.iter().map(|obj| [obj.range-obj.velocity, obj.angle]));

                Arrows::new("arrows", arrow_origins, arrow_tips)
            };
            let detections = detection_coords.iter().map(|d|{
                let d = radar.coord_to_adr(d);
                println!("detection: {:.2} m, {:.2} m/s, {:.2}°", d.2, d.1, d.0);
                [d.0, d.1, d.2]
            }).collect::<Vec<[f64; 3]>>();

            let detection_points = PlotPoints::from_iter(detections.iter().map(|d|{
                [d[2], d[0]]
            }));

            let detection_velocity_arrows = {
                let arrow_origins = PlotPoints::from_iter(detections.iter().map(|d| [d[2], d[0]]));
                let arrow_tips = PlotPoints::from_iter(detections.iter().map(|d| [d[2]-d[1], d[0]]));

                Arrows::new("arrows", arrow_origins, arrow_tips)
            };

            let plot = Plot::new("items_demo")
                        .legend(
                            Legend::default()
                                .position(egui_plot::Corner::RightBottom)
                                .title("Items"),
                        )
                        .show_x(false)
                        .show_y(false)
                        .default_x_bounds(-20.0, 120.0)
                        .default_y_bounds(-90.0, 90.0)
                        .view_aspect(2.0);
            plot.show(ui, |plot_ui| {
                plot_ui.image(image.name("Image"));
                plot_ui.arrows(true_velocity_arrows.name("Actual Velocity"));
                plot_ui.arrows(detection_velocity_arrows.name("Detected Velocity"));
                plot_ui.points(Points::new("Detections", detection_points).radius(3.0));
            });

            ui.label(format!(
                "Range–Angle map. cube shape (angle, doppler, range)=({nz}, {ny}, {nx}). Max range ≈ {max_range:.1} m"
            ));

            ui.horizontal(|ui| {
                ui.label(format!("Max Range: {max_range:.2} m"));
                ui.add_space(8.0);
                ui.label(format!("Max Angle: {max_angle_deg:.2}°"));
                ui.add_space(8.0);
                ui.label(format!("Max Velocity: {max_velocity:.2} m/s"));
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
