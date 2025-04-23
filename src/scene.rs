use palette::{FromColor, LinSrgb, Oklch};

use super::*;

#[derive(Clone, Copy, Debug)]
pub enum Brush {
    Rect(f32, f32),
    Circle(f32),
}

#[derive(Clone, Copy, Debug)]
pub struct Draw {
    pub brush: Brush,
    pub center: Vec2<f32>,
    pub color: Color,
}

pub struct Scene {
    pub draws: Vec<Draw>,
}

impl Scene {
    pub fn new<const N: usize>(draws: [Draw; N]) -> Self {
        Self {
            draws: draws.to_vec(),
        }
    }
    pub fn simple() -> Self {
        Self::new([
            Draw {
                brush: Brush::Circle(1.0),
                center: Vec2::new(256.0, 256.0),
                color: Color::new(Vec3::splat(5.0), Vec3::splat(100.0)),
            },
            Draw {
                brush: Brush::Rect(20.0, 5.0),
                center: Vec2::new(256.0, 384.0),
                color: Color::new(Vec3::splat(0.0), Vec3::splat(100.0)),
            },
        ])
    }
    pub fn pinhole(t: u32) -> Self {
        let l = (t as f32 / 200.0 * TAU).sin() * 200.0;
        let mut draws = vec![
            Draw {
                brush: Brush::Rect(1.0, 512.0 + l),
                center: Vec2::new(512.0, -5.0),
                color: Color::solid(Vec3::splat(0.0)),
            },
            Draw {
                brush: Brush::Rect(1.0, 512.0 - l),
                center: Vec2::new(512.0, 1024.0 + 5.0),
                color: Color::solid(Vec3::splat(0.0)),
            },
        ];
        for i in -3..=3 {
            let color = Oklch::new(0.5, 0.15, (1.618033988 * 360.0 * i as f64) % 360.0);
            let color = LinSrgb::from_color(color);
            draws.push(Draw {
                brush: Brush::Rect(5.0, 20.0),
                center: Vec2::new(1024.0 - 1000.0, -i as f32 * 40.0 + 512.0),
                color: Color::solid(Vec3::new(
                    50.0 * color.red.max(0.0) as f32,
                    50.0 * color.green.max(0.0) as f32,
                    50.0 * color.blue.max(0.0) as f32,
                )),
            })
        }
        Self { draws }
    }
}
