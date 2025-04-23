use std::f32::consts::TAU;

use keter::{
    lang::types::vector::{Vec2, Vec3},
    prelude::*,
};
use keter_testbed::{App, KeyCode, MouseButton};
use utils::pcg3df;
use voxel::VoxelTracer;

mod utils;
mod voxel;

pub type Emission = Vec3<f32>;
pub type Opacity = Vec3<f32>;
pub type Radiance = Vec3<f32>;
pub type Transmittance = Vec3<f32>;

#[derive(Clone, Copy, Debug, PartialEq, Value)]
#[repr(C)]
pub struct Fluence {
    pub radiance: Radiance,
    pub transmittance: Transmittance,
}
impl Fluence {
    pub fn empty() -> Self {
        Fluence {
            radiance: Vec3::splat(0.0),
            transmittance: Vec3::splat(1.0),
        }
    }
    pub fn expr(radiance: Expr<Radiance>, transmittance: Expr<Transmittance>) -> Expr<Self> {
        Fluence::from_comps_expr(FluenceComps {
            radiance,
            transmittance,
        })
    }
}
impl FluenceExpr {
    #[tracked]
    pub fn over(self, far: Expr<Fluence>) -> Expr<Fluence> {
        Fluence::expr(
            self.transmittance * far.radiance + self.radiance,
            self.transmittance * far.transmittance,
        )
    }
    #[tracked]
    pub fn over_radiance(self, far: Expr<Radiance>) -> Expr<Radiance> {
        self.transmittance * far + self.radiance
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Value)]
#[repr(C)]
pub struct Color {
    pub emission: Emission,
    pub opacity: Opacity,
}
impl Color {
    pub fn empty() -> Self {
        Color {
            emission: Vec3::splat(0.0),
            opacity: Vec3::splat(0.0),
        }
    }
    pub fn new(emission: Emission, opacity: Opacity) -> Self {
        Color { emission, opacity }
    }
    pub fn expr(emission: Expr<Emission>, opacity: Expr<Opacity>) -> Expr<Self> {
        Color::from_comps_expr(ColorComps { emission, opacity })
    }
}
impl ColorExpr {
    #[tracked]
    fn to_fluence(self, segment_length: Expr<f32>) -> Expr<Fluence> {
        let transmittance = (-self.opacity * segment_length).exp();
        Fluence::expr(self.emission * (1.0 - transmittance), transmittance)
    }
}

#[tracked]
fn bilinear(pos: Expr<Vec2<f32>>) -> [(Expr<Vec2<u32>>, Expr<f32>); 4] {
    let f = pos.fract();
    let nf = 1.0 - f;
    let pos = pos.floor().cast_u32();
    [
        (pos, nf.x * nf.y),
        (pos + Vec2::x(), f.x * nf.y),
        (pos + Vec2::y(), nf.x * f.y),
        (pos + Vec2::splat(1), f.x * f.y),
    ]
}

struct CascadeStorage {
    data: Buffer<f32>,
    base_size: Vec2<u32>,
    base_spacing: f32,
    base_angles: u32,
    angular_scale: u32,
    num_cascades: u32,
}
impl CascadeStorage {
    fn cascade_size(&self) -> u32 {
        self.base_size.x * self.base_size.y * self.base_angles
    }
    #[tracked]
    fn index(&self, cascade: Expr<u32>, pos: Expr<Vec2<u32>>, angle: Expr<u32>) -> Expr<u32> {
        let angles = self.base_angles << (cascade * self.angular_scale);
        cascade * self.cascade_size()
            + pos.x * (self.base_size.y >> cascade) * angles
            + pos.y * angles
            + angle
    }
    fn get(&self, cascade: Expr<u32>, pos: Expr<Vec2<u32>>, angle: Expr<u32>) -> Expr<f32> {
        let index = self.index(cascade, pos, angle);
        self.data.read(index)
    }
    #[tracked]
    fn probe_position(&self, cascade: Expr<u32>, world_pos: Expr<Vec2<f32>>) -> Expr<Vec2<f32>> {
        let spacing = self.base_spacing * (1 << cascade).cast_f32();
        let pos = world_pos / spacing - 0.5; // TODO: Make this +?
        pos.clamp(0.0, ((self.base_size >> cascade) - 1).cast_f32())
    }
    #[tracked]
    fn get_bilinear(
        &self,
        cascade: Expr<u32>,
        pos: Expr<Vec2<f32>>,
        angle: Expr<u32>,
    ) -> Expr<f32> {
        let pos = self.probe_position(cascade, pos);
        bilinear(pos)
            .into_iter()
            .map(|(pos, w)| self.get(cascade, pos, angle) * w)
            .reduce(|a, b| a + b)
            .unwrap()
    }
    fn add(&self, cascade: Expr<u32>, pos: Expr<Vec2<u32>>, angle: Expr<u32>, value: Expr<f32>) {
        let index = self.index(cascade, pos, angle);
        self.data.atomic_fetch_add(index, value);
    }
    #[tracked]
    fn add_bilinear(
        &self,
        cascade: Expr<u32>,
        pos: Expr<Vec2<f32>>,
        angle: Expr<u32>,
        value: Expr<f32>,
    ) {
        let pos = self.probe_position(cascade, pos);
        bilinear(pos).into_iter().for_each(|(pos, w)| {
            let value = value * w;
            self.add(cascade, pos, angle, value)
        })
    }
}

const DISPLAY_SIZE: u32 = 512;

fn main() {
    let app = App::new("Vlam", [DISPLAY_SIZE; 2])
        .scale(2048 / DISPLAY_SIZE)
        .agx()
        .init();

    let world = VoxelTracer::new(Vec2::splat(DISPLAY_SIZE));

    let [storage, next_storage] = [(); 2].map(|()| CascadeStorage {
        data: DEVICE.create_buffer((DISPLAY_SIZE * DISPLAY_SIZE * 4 * 5) as usize),
        base_size: Vec2::splat(DISPLAY_SIZE),
        base_spacing: 1.0,
        base_angles: 4,
        angular_scale: 2,
        num_cascades: 5,
    });

    let display =
        DEVICE.create_tex2d::<Vec3<f32>>(PixelStorage::Float4, DISPLAY_SIZE, DISPLAY_SIZE, 1);

    let compute_diff = DEVICE.create_kernel::<fn()>(&track!(|| {
        world.compute_diff();
    }));
    let draw_kernel = DEVICE.create_kernel::<fn(u32)>(&track!(|iterations| {
        let pixel = dispatch_id().xy();
        app.display()
            .write(pixel, display.read(pixel) / iterations.cast_f32());
    }));
    let trace_kernel = DEVICE.create_kernel::<fn(u32)>(&track!(|t| {
        let pixel = dispatch_id().xy();
        let pos = pixel.cast_f32() + 0.5;
        let angle = pcg3df(pixel.extend(t)).x * TAU;
        let radiance = world.trace(pos, angle.direction(), 9999.0.expr()).radiance;
        display.write(pixel, display.read(pixel) + radiance);
    }));

    let rect_brush =
        DEVICE.create_kernel::<fn(Vec2<f32>, Vec2<f32>, Color)>(&track!(|center, size, color| {
            let pos = dispatch_id().xy();
            if ((pos.cast_f32() + 0.5 - center).abs() < size).all() {
                world.write(pos, color);
            }
        }));
    let circle_brush =
        DEVICE.create_kernel::<fn(Vec2<f32>, f32, Color)>(&track!(|center, radius, color| {
            let pos = dispatch_id().xy();
            if (pos.cast_f32() + 0.5 - center).length() < radius {
                world.write(pos, color);
            }
        }));

    let mut iterations = 0;

    app.run(|rt| {
        let brushes = [
            (
                MouseButton::Middle,
                Color::new(Vec3::splat(1.0), Vec3::splat(0.5)),
            ),
            (
                MouseButton::Left,
                Color::new(Vec3::splat(0.0), Vec3::splat(100.0)),
            ),
            (MouseButton::Right, Color::empty()),
        ];
        for brush in brushes {
            if rt.button_down(brush.0) {
                circle_brush.dispatch(
                    [world.size.x, world.size.y, 1],
                    &rt.cursor_position,
                    &4.0,
                    &brush.1,
                );
            }
        }

        if rt.key_pressed(KeyCode::Space) {
            compute_diff.dispatch_blocking([DISPLAY_SIZE / 8, DISPLAY_SIZE / 8, 1]);
            trace_kernel.dispatch(rt.dispatch_size(), &iterations);
            iterations += 1;
        }

        draw_kernel.dispatch(rt.dispatch_size(), &iterations);
    });
}
