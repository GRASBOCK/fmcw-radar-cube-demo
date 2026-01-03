use fmcw_radar_cube_demo::radar;
use ndarray::Array1;
use ndrustfft::{Complex, FftHandler, ndfft};
use std::f64::consts::PI;

pub struct App {
    frequency: f64,
    phase: f64,
    count: usize,
    sample_time: f64,
}

impl Default for App {
    fn default() -> Self {
        Self {
            frequency: 1.0,
            phase: 0.0,
            count: 256,
            sample_time: 0.1,
        }
    }
}

impl App {
    pub fn new(_: &eframe::CreationContext<'_>) -> Self {
        Default::default()
    }
}

impl eframe::App for App {
    fn save(&mut self, _: &mut dyn eframe::Storage) {}

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Radar 1D FFT");

            ui.add(egui::Slider::new(&mut self.frequency, 0.1..=80E9).text("Frequency"));
            ui.add(egui::Slider::new(&mut self.phase, 0.0..=2.0 * PI).text("Phase"));
            ui.add(egui::Slider::new(&mut self.count, 2..=80).text("Count"));
            ui.add(egui::Slider::new(&mut self.sample_time, 0.000040..=1.0).text("Sample Time"));

            ui.separator();

            // Sample the signal
            let sampled_data =
                radar::sample_signal(self.frequency, self.phase, self.count, self.sample_time);

            // Convert sampled_data to ndarray Array1<Complex<f64>>
            let signal: Array1<Complex<f64>> =
                Array1::from_iter(sampled_data.iter().map(|x| Complex { re: x.re, im: x.im }));

            // FFT using ndrustfft (ndfft for 1D)
            let handler = FftHandler::<f64>::new(self.count);
            let mut spectrum = Array1::<Complex<f64>>::zeros(self.count);
            ndfft(&signal.view(), &mut spectrum.view_mut(), &handler, 0);

            // Compute magnitude
            let fft_magnitude: Vec<f64> = spectrum
                .iter()
                .map(|c| c.re.hypot(c.im))
                .collect();

            // Frequency bins
            let sample_rate = 1.0 / self.sample_time;
            let freq_bins: Vec<f64> = (0..self.count)
                .map(|i| i as f64 * sample_rate / self.count as f64)
                .collect();

            // Prepare plot points
            let fft_points: Vec<[f64; 2]> = freq_bins
                .iter()
                .zip(fft_magnitude.iter())
                .map(|(f, m)| [*f, *m])
                .collect();

            egui_plot::Plot::new("fft_plot").show(ui, |plot_ui| {
                let plot_line = egui_plot::Line::new(
                    "FFT Magnitude",
                    egui_plot::PlotPoints::from_iter(fft_points),
                )
                .color(egui::Color32::GREEN)
                .width(2.0)
                .name("FFT Magnitude");
                plot_ui.line(plot_line);
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
    env_logger::init();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([500.0, 350.0])
            .with_min_inner_size([300.0, 220.0])
            .with_icon(
                eframe::icon_data::from_png_bytes(&include_bytes!("../../assets/icon-256.png")[..])
                    .expect("Failed to load icon"),
            ),
        ..Default::default()
    };
    eframe::run_native(
        "Radar FFT",
        native_options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
