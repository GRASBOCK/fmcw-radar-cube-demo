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

    pub fn velocity_resolution(&self) -> f64 {
        self.wavelength() / (2.0 * self.frame_time())
    }

    pub fn angular_resolution(&self, angle_rad: f64) -> f64 {
        self.wavelength() / (self.receivers as f64 * self.receiver_spacing * angle_rad.cos())
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

    fn nx(&self) -> usize {
        self.sample_count_chirp()
    }

    fn ny(&self) -> usize {
        self.chirp_count
    }
    fn nz(&self) -> usize {
        self.receivers
    }

    pub fn coord_to_rda(&self, zyx: &(usize, usize, usize)) -> (f64, f64, f64) {
        let angle_shifted = (zyx.0 + self.nz() / 2) % self.nz();
        let max_angle = self.max_angle();
        let angle = (angle_shifted as f64 / self.nz() as f64 * (max_angle * 2.0) - max_angle)
            .asin()
            .to_degrees()
            / 2.0;
        let velocity_shifted = (zyx.1 + 1 + self.ny() / 2) % self.ny();
        let max_vel = self.max_velocity();
        let velocity = -(velocity_shifted as f64 / self.ny() as f64 * (max_vel * 2.0) - max_vel);
        let range = zyx.2 as f64 * self.max_range() / self.nx() as f64;
        (range, velocity, angle)
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

pub fn detect(array: &Array3<f64>) -> Vec<(usize, usize, usize)> {
    let threshold = 3e4;
    let kernel_size = 9;
    let k2 = kernel_size / 2;
    let mut indices = Vec::new();

    // Local maximum filter (3D sliding window) + thresholding.
    // Keep only samples that are the strict maximum within a kernel_size^3 neighborhood.
    let (nz, ny, nx) = array.dim();

    for ((i, j, k), &val) in array.indexed_iter() {
        if val <= threshold {
            continue;
        }

        let i0 = i.saturating_sub(k2);
        let i1 = (i + k2).min(nz.saturating_sub(1));
        let j0 = j.saturating_sub(k2);
        let j1 = (j + k2).min(ny.saturating_sub(1));
        let k0 = k.saturating_sub(k2);
        let k1 = (k + k2).min(nx.saturating_sub(1));

        let mut is_local_max = true;
        'neigh: for ii in i0..=i1 {
            for jj in j0..=j1 {
                for kk in k0..=k1 {
                    if (ii, jj, kk) != (i, j, k) && array[(ii, jj, kk)] >= val {
                        is_local_max = false;
                        break 'neigh;
                    }
                }
            }
        }

        if is_local_max {
            indices.push((i, j, k));
        }
    }

    indices
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
        let c = 299_792_458.0;
        let cf = 77e9;
        let radar = Radar {
            carrier_frequency: cf,
            c: c,
            sampling_frequency: 2.4e6,
            bandwidth: 320e6,
            chirp_duration: 90e-6,
            chirp_count: 64,
            receivers: 36,
            receiver_spacing: (c / cf) / 2.0,
        };
        let obj1 = Object {
            angle: 38.0,
            range: 27.0,
            velocity: -2.5,
        };
        let obj2 = Object {
            angle: -10.0,
            range: 69.0,
            velocity: 8.0,
        };
        let data = scene_to_data(&radar, &vec![obj1.clone(), obj2.clone()]);
        let output = Radar::radar_cube(&radar, &data);
        let detection_coords = detect(&output);
        assert_eq!(detection_coords.len(), 2);
        let detections: Vec<(f64, f64, f64)> = detection_coords
            .iter()
            .map(|coord| radar.coord_to_rda(&coord))
            .collect();

        let expected = vec![
            (obj1.range, obj1.velocity, obj1.angle),
            (obj2.range, obj2.velocity, obj2.angle),
        ];

        dbg!(&expected, &detections);

        // Sort detections by range (ascending) to match expected ordering.
        let mut detections = detections;
        detections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for (det, exp) in detections.iter().zip(expected.iter()) {
            let (det_range, det_velocity, det_angle) = det;
            let (exp_range, exp_velocity, exp_angle) = exp;
            let range_tolerance = radar.range_resolution() * 3.0;
            assert!(
                (det_range - exp_range).abs() < range_tolerance,
                "Range mismatch: detected {:.3} m, expected {:.3} m; tolerance {:.3} m",
                det_range,
                exp_range,
                range_tolerance
            );
            let angle_tolerance = radar.angular_resolution(*exp_angle / 180.0 * PI) / PI * 180.0;
            assert!(
                (det_angle - exp_angle).abs() < angle_tolerance,
                "Angle mismatch: detected {:.3}°, expected {:.3}°; tolerance {:.3}°",
                det_angle,
                exp_angle,
                angle_tolerance
            );
            let velocity_tolerance = radar.velocity_resolution() * 3.0;
            assert!(
                (det_velocity - exp_velocity).abs() < velocity_tolerance,
                "Velocity mismatch: detected {:.3} m/s, expected {:.3} m/s; tolerance {:.3} m/s",
                det_velocity,
                exp_velocity,
                velocity_tolerance
            );
        }
    }
}
