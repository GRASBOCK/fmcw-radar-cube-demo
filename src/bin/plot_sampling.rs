use fmcw_radar_cube_demo::radar;
use ndarray::{Array3, s};
use std::f64::consts::PI;

pub struct App {
    // Example stuff:
    frequency: f64,
    phase: f64,
    count: usize,
    sample_time: f64,
}

impl Default for App {
    fn default() -> Self {
        Self {
            // Example stuff:
            frequency: 1.0,
            phase: 0.0,
            count: 100,
            sample_time: 0.1,
        }
    }
}

impl App {
    /// Called once before the first frame.
    pub fn new(_: &eframe::CreationContext<'_>) -> Self {
        Default::default()
    }
}

impl eframe::App for App {
    /// Called by the framework to save state before shutdown.
    fn save(&mut self, _: &mut dyn eframe::Storage) {}

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        egui::CentralPanel::default().show(ctx, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's
            ui.heading("eframe template");

            ui.add(egui::Slider::new(&mut self.frequency, 0.1..=2.0).text("Frequency"));
            ui.add(egui::Slider::new(&mut self.phase, 0.0..=2.0 * PI).text("Phase"));
            ui.add(egui::Slider::new(&mut self.count, 2..=1000).text("Count"));
            ui.add(egui::Slider::new(&mut self.sample_time, 0.01..=1.0).text("Sample Time"));

            ui.separator();

            egui_plot::Plot::new("my_plot").show(ui, |plot_ui| {
                let sampled_data =
                    radar::sample_signal(self.frequency, self.phase, self.count, self.sample_time);
                let sample_times: Vec<f64> = (0..self.count)
                    .map(|i| i as f64 * self.sample_time)
                    .collect();
                let lines: [(Vec<[f64; 2]>, &str, egui::Color32); 2] = [
                    (
                        sample_times
                            .iter()
                            .zip(sampled_data.map(|x| x.re))
                            .map(|(x, y)| [*x, y])
                            .collect(),
                        "Re",
                        egui::Color32::RED,
                    ),
                    (
                        sample_times
                            .iter()
                            .zip(sampled_data.map(|x| x.im))
                            .map(|(x, y)| [*x, y])
                            .collect(),
                        "Im",
                        egui::Color32::BLUE,
                    ),
                ];
                for (line, label, color) in lines {
                    let plot_line = egui_plot::Line::new(
                        label,
                        egui_plot::PlotPoints::from_iter(line.into_iter()),
                    )
                    .color(color)
                    .width(2.0)
                    .name(label);
                    plot_ui.line(plot_line);
                }
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

fn main() -> eframe::Result {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 300.0])
            .with_min_inner_size([300.0, 220.0])
            .with_icon(
                // NOTE: Adding an icon is optional
                eframe::icon_data::from_png_bytes(&include_bytes!("../../assets/icon-256.png")[..])
                    .expect("Failed to load icon"),
            ),
        ..Default::default()
    };
    eframe::run_native(
        "Plot samples",
        native_options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
