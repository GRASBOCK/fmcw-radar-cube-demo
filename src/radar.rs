use ndarray::Array1;
use ndarray::{Array3, s};
use ndrustfft::{Complex, FftHandler, ndfft};
use rustfft::num_complex::Complex64;

struct Object {
    angle: f64,
    velocity: f64,
    range: f64,
}

fn beat_frequency(_carrier_frequency: f64, _obj: &Object) -> f64 {
    100.0E3
}

struct Radar {
    pub carrier_frequency: f64,
    pub sampling_frequency: f64,
    pub chirp_duration: f64,
    pub chirp_count: usize,
    pub receivers: usize,
}

impl Radar {
    pub fn frame_time(&self) -> f64 {
        self.chirp_count as f64 * self.chirp_duration
    }

    pub fn sample_count_chirp(&self) -> usize {
        (self.sampling_frequency * self.chirp_duration) as usize
    }

    pub fn radar_cube(radar: &Self, data: &Array3<Complex64>) -> Array3<f64> {
        // extract doppler FFT across time frame
        let nx = radar.sample_count_chirp();
        let ny = radar.chirp_count;
        let nz = radar.receivers;
        assert_eq!(
            *data.shape(),
            [nz, ny, nx],
            "Shape of input data is not as expected"
        );

        let mut range_fft_data = Array3::<Complex<f64>>::zeros((nz, ny, nx / 2 + 1));
        let mut fft_handler = FftHandler::<f64>::new(nx);
        ndfft(
            &data.view(),
            &mut range_fft_data.view_mut(),
            &fft_handler,
            2,
        );

        let mut range_doppler_fft_data =
            Array3::<Complex<f64>>::zeros((nz, ny / 2 + 1, nx / 2 + 1));
        let mut fft_handler = FftHandler::<f64>::new(ny);
        ndfft(
            &range_fft_data.view(),
            &mut range_doppler_fft_data.view_mut(),
            &fft_handler,
            1,
        );

        let mut angle_range_doppler_fft_data =
            Array3::<Complex<f64>>::zeros((nz / 2 + 1, ny / 2 + 1, nx / 2 + 1));
        let mut fft_handler = FftHandler::<f64>::new(nz);
        ndfft(
            &range_doppler_fft_data.view(),
            &mut angle_range_doppler_fft_data.view_mut(),
            &fft_handler,
            0,
        );

        angle_range_doppler_fft_data.map(|v| v.norm())
    }
}

fn sample_signal(frequency: f64, phase: f64, count: usize, sample_time: f64) -> Array1<Complex64> {
    let mut data = Array1::<Complex64>::zeros(count);
    for (i, v) in data.iter_mut().enumerate() {
        let t = i as f64 * sample_time;
        *v += Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI * frequency * t + phase).sin();
    }
    data
}

fn obj_to_data(radar: &Radar, obj: &Object) -> Array3<Complex64> {
    let fb = beat_frequency(radar.carrier_frequency, obj);
    let sample_count = radar.sample_count_chirp();
    let mut data = Array3::<Complex64>::zeros((radar.receivers, radar.chirp_count, sample_count));
    for r in 0..radar.receivers {
        for c in 0..radar.chirp_count {
            let phase = 0.0;
            let v = sample_signal(fb, phase, sample_count, 1.0 / radar.carrier_frequency);
            data.slice_mut(s![r, c, ..]).assign(&v);
        }
    }
    data
}

fn scene_to_data(radar: &Radar, objects: &Vec<Object>) -> Array3<Complex64> {
    let sample_count = radar.sample_count_chirp();
    let mut data = Array3::<Complex64>::zeros((radar.receivers, radar.chirp_count, sample_count));
    for obj in objects {
        data = data + obj_to_data(radar, obj);
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_in_one() {
        let radar = Radar {
            carrier_frequency: 77e9,
            sampling_frequency: 2e6,
            chirp_duration: 1e-3,
            chirp_count: 8,
            receivers: 4,
        };
        let obj1 = Object {
            angle: 0.0,
            velocity: 0.0,
            range: 10.0,
        };
        let obj2 = Object {
            angle: 0.0,
            velocity: 0.0,
            range: 10.0,
        };
        let data = scene_to_data(&radar, &vec![obj1, obj2]);
        let _ = Radar::radar_cube(&radar, &data);
    }
}
