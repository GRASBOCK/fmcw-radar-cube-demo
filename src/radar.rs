use ndarray::Array1;
use ndarray::{Array3, s};
use ndrustfft::{Complex, FftHandler, ndfft};
use rustfft::num_complex::Complex64;

#[derive(Debug, Clone)]
pub struct Object {
    pub angle: f64,
    pub velocity: f64,
    pub range: f64,
}

fn beat_frequency(radar: &Radar, obj: &Object) -> f64 {
    obj.range * 2.0 * radar.chirp_rate() / radar.c
    // + TODO: doppler
}

pub struct Radar {
    pub carrier_frequency: f64,
    pub c: f64,
    pub sampling_frequency: f64,
    pub bandwidth: f64,
    pub chirp_duration: f64,
    pub chirp_count: usize,
    pub receivers: usize,
    pub receiver_spacing: f64,
}

impl Radar {
    pub fn frame_time(&self) -> f64 {
        self.chirp_count as f64 * self.chirp_duration
    }
    pub fn chirp_rate(&self) -> f64 {
        self.bandwidth / self.chirp_duration
    }

    pub fn max_range(&self) -> f64 {
        self.c * self.sampling_frequency / 2.0 / self.chirp_rate()
    }

    pub fn range_resolution(&self) -> f64 {
        self.c / 2.0 / self.bandwidth
    }

    pub fn max_velocity(&self) -> f64 {
        self.wavelength() / (4.0 * self.chirp_duration)
    }

    pub fn max_angle(&self) -> f64 {
        (self.wavelength() / (2.0 * self.receiver_spacing)).asin()
    }

    pub fn wavelength(&self) -> f64 {
        self.c / self.carrier_frequency
    }

    pub fn sample_count_chirp(&self) -> usize {
        (self.sampling_frequency * self.chirp_duration) as usize
    }

    pub fn radar_cube(&self, data: &Array3<Complex64>) -> Array3<f64> {
        let nx = self.sample_count_chirp();
        let ny = self.chirp_count;
        let nz = self.receivers;
        println!("{nz} {ny} {nx}");
        assert_eq!(
            *data.shape(),
            [nz, ny, nx],
            "Shape of input data is not as expected"
        );

        let mut range_fft_data = Array3::<Complex<f64>>::zeros((nz, ny, nx));
        let fft_handler = FftHandler::<f64>::new(nx);
        ndfft(
            &data.view(),
            &mut range_fft_data.view_mut(),
            &fft_handler,
            2,
        );

        let mut range_doppler_fft_data = Array3::<Complex64>::zeros((nz, ny, nx));
        let fft_handler = FftHandler::<f64>::new(ny);
        ndfft(
            &range_fft_data.view(),
            &mut range_doppler_fft_data.view_mut(),
            &fft_handler,
            1,
        );

        let mut angle_range_doppler_fft_data = Array3::<Complex64>::zeros((nz, ny, nx));
        let fft_handler = FftHandler::<f64>::new(nz);
        ndfft(
            &range_doppler_fft_data.view(),
            &mut angle_range_doppler_fft_data.view_mut(),
            &fft_handler,
            0,
        );

        angle_range_doppler_fft_data.map(|v| v.norm())
    }
}

pub fn sample_signal(
    frequency: f64,
    phase: f64,
    count: usize,
    sample_time: f64,
) -> Array1<Complex64> {
    let mut data = Array1::<Complex64>::zeros(count);
    for (i, v) in data.iter_mut().enumerate() {
        let t = i as f64 * sample_time;
        *v = Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI * frequency * t + phase);
    }
    data
}

fn obj_to_data(radar: &Radar, obj: &Object) -> Array3<Complex64> {
    let fb = beat_frequency(radar, obj);
    println!("{obj:?} beat: {fb}");
    let sample_count = radar.sample_count_chirp();
    let mut data = Array3::<Complex64>::zeros((radar.receivers, radar.chirp_count, sample_count));
    let wavelength = radar.c / radar.carrier_frequency;
    let phase_shift_due_to_antenna_array =
        2.0 * std::f64::consts::PI * radar.receiver_spacing * obj.angle.sin() / wavelength;
    let time_shift = 2.0 * obj.velocity * radar.chirp_duration / radar.c;
    let phase_shift_due_to_velocity =
        -2.0 * std::f64::consts::PI * radar.carrier_frequency * time_shift;
    for r in 0..radar.receivers {
        for c in 0..radar.chirp_count {
            let phase = phase_shift_due_to_velocity * c as f64
                + r as f64 * phase_shift_due_to_antenna_array;
            let v = sample_signal(fb, phase, sample_count, 1.0 / radar.sampling_frequency);
            data.slice_mut(s![r, c, ..]).assign(&v);
        }
    }
    data
}

pub fn scene_to_data(radar: &Radar, objects: &Vec<Object>) -> Array3<Complex64> {
    let sample_count = radar.sample_count_chirp();
    let mut data = Array3::<Complex64>::zeros((radar.receivers, radar.chirp_count, sample_count));
    for obj in objects {
        data = data + obj_to_data(radar, obj);
    }
    data
}

fn detections(array: &Array3<f64>) -> Vec<(f64, f64, f64)> {
    let threshold = 1e4;
    let mut indices = Vec::new();
    for ((i, j, k), &val) in array.indexed_iter() {
        if val > threshold {
            indices.push((i, j, k));
        }
    }
    println!("{indices:?}");
    vec![(0.0, 0.0, 0.0); 2]
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;

    #[test]
    fn test_sample_signal_basic() {
        let frequency = 1.0;
        let phase = 0.0;
        let count = 10;
        let sample_time = 0.25;
        let signal = sample_signal(frequency, phase, count, sample_time);
        assert_eq!(signal.len(), count);
        // Check that all values are finite
        let expected = [
            (1.0, 0.0),
            (1.0, PI / 2.0),
            (1.0, PI),
            (1.0, PI / 2.0 * 3.0),
            (1.0, 0.0),
            (1.0, PI / 2.0),
            (1.0, PI),
            (1.0, PI / 2.0 * 3.0),
            (1.0, 0.0),
            (1.0, PI / 2.0),
        ];
        let margin = 1e-10;
        for (i, &val) in expected.iter().enumerate() {
            let exp = Complex::from_polar(val.0, val.1);
            assert!(
                (signal[i].re - exp.re).abs() < margin && (signal[i].im - exp.im).abs() < margin,
                "Mismatch at index {}: got {:?}, expected {:?}",
                i,
                signal[i],
                exp
            );
        }
    }

    #[test]
    fn test_all_in_one() {
        let radar = Radar {
            carrier_frequency: 77e9,
            c: 300000.0,
            sampling_frequency: 2e6,
            chirp_duration: 1e-3,
            bandwidth: 4e9,
            chirp_count: 8,
            receivers: 4,
            receiver_spacing: 0.1,
        };
        let obj1 = Object {
            angle: 0.0,
            velocity: 0.0,
            range: 10.0,
        };
        let obj2 = Object {
            angle: -4.0,
            velocity: 20.0,
            range: 40.0,
        };
        let data = scene_to_data(&radar, &vec![obj1.clone(), obj2.clone()]);
        let output = Radar::radar_cube(&radar, &data);
        let detections = detections(&output);
        assert_eq!(detections.len(), 2);
        let expected = vec![
            (obj1.angle, obj1.velocity, obj1.range),
            (obj2.angle, obj2.velocity, obj2.range),
        ];
        for (det, exp) in detections.iter().zip(expected.iter()) {
            let (det_angle, det_velocity, det_range) = det;
            let (exp_angle, exp_velocity, exp_range) = exp;
            // Allow some tolerance for floating point comparison
            assert!(
                (det_angle - exp_angle).abs() < 1e-2,
                "Angle mismatch: detected {}, expected {}",
                det_angle,
                exp_angle
            );
            assert!(
                (det_velocity - exp_velocity).abs() < 1e-2,
                "Velocity mismatch: detected {}, expected {}",
                det_velocity,
                exp_velocity
            );
            assert!(
                (det_range - exp_range).abs() < 1e-2,
                "Range mismatch: detected {}, expected {}",
                det_range,
                exp_range
            );
        }
    }
}
