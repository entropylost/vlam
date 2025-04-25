#![feature(more_float_constants)]

use std::f32::consts::{PHI, TAU};

use analytic::{AnalyticTracer, Object};
use keter::{
    lang::types::vector::{Vec2, Vec3, Vec4},
    prelude::*,
};
use keter_testbed::{App, KeyCode, MouseButton};
use scene::{Brush, Scene};
use utils::{luma, pcg3df};
use voxel::VoxelTracer;

mod analytic;
mod scene;
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
    pub fn solid(emission: Emission) -> Self {
        Color {
            emission,
            opacity: Vec3::splat(999999.0),
        }
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
        let pos = keter::min(pos, (self.base_size >> cascade) - 1);
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
        // TODO: Also this results in OOB which has to be fixed later.
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

fn cascade_colors() -> [Vec3<f32>; 6] {
    fn vec(r: f32, g: f32, b: f32) -> Vec3<f32> {
        Vec3::new(r, g, b).map(|x| x.powf(2.2))
    }
    [
        vec(0.64178, 0.22938, 0.33132),
        vec(0.60514, 0.27555, 0.1076),
        vec(0.30633, 0.39017, 0.11297),
        vec(0.06965, 0.44936, 0.3549),
        vec(0.11775, 0.38443, 0.67507),
        vec(0.39088, 0.2793, 0.65571),
    ]
}

const DISPLAY_SIZE: u32 = 1024;
const MAX_ITERS: u32 = 1000;

fn main() {
    let app = App::new("Vlam", [DISPLAY_SIZE; 2])
        .scale(2048 / DISPLAY_SIZE)
        .agx()
        .init();

    let world = AnalyticTracer::new(&[
        Object {
            center: Vec2::new(3.0 * DISPLAY_SIZE as f32 / 4.0, DISPLAY_SIZE as f32 / 2.0),
            radius: 5.0,
            refraction_index: 1.0,
            color: Color::new(Vec3::splat(20.0), Vec3::splat(2.0)),
        },
        Object {
            center: Vec2::new(DISPLAY_SIZE as f32 / 2.0, DISPLAY_SIZE as f32 / 2.0),
            radius: 100.0,
            refraction_index: 1.5,
            color: Color::new(Vec3::splat(0.0), Vec3::splat(0.0)),
        },
        Object {
            center: Vec2::new(DISPLAY_SIZE as f32 / 4.0, DISPLAY_SIZE as f32 / 2.0),
            radius: 50.0,
            refraction_index: 1.5,
            color: Color::new(Vec3::splat(0.0), Vec3::splat(0.0)),
        },
    ]);
    // VoxelTracer::new(Vec2::splat(DISPLAY_SIZE));

    let [storage, next_storage] = [(); 2].map(|()| CascadeStorage {
        data: DEVICE.create_buffer_from_fn((DISPLAY_SIZE * DISPLAY_SIZE * 4 * 6) as usize, |_| 1.0),
        base_size: Vec2::splat(DISPLAY_SIZE),
        base_spacing: 1.0,
        base_angles: 4,
        angular_scale: 2,
        num_cascades: 6,
    });

    let display =
        DEVICE.create_tex2d::<Vec3<f32>>(PixelStorage::Float4, DISPLAY_SIZE, DISPLAY_SIZE, 1);

    // let compute_diff = DEVICE.create_kernel::<fn()>(&track!(|| {
    //     world.compute_diff();
    // }));
    let copy_storage = DEVICE.create_kernel::<fn()>(&track!(|| {
        let index = dispatch_id().x;
        storage
            .data
            .write(index, next_storage.data.read(index) * 0.5 + 0.01);
    }));
    let draw_kernel = DEVICE.create_kernel::<fn(u32)>(&track!(|iterations| {
        let pixel = dispatch_id().xy();
        app.display()
            .write(pixel, display.read(pixel) / iterations.cast_f32());
    }));
    let trace_simple_kernel = DEVICE.create_kernel::<fn(u32)>(&track!(|t| {
        let pixel = dispatch_id().xy();
        let pos = pixel.cast_f32() + 0.5;

        let angle = (pcg3df(pixel.extend(35)).x + (t.cast_f32() * PHI) % 1.0) * TAU;
        let dir = angle.direction();
        let radiance = world.trace(pos, dir, 9999.0.expr()).fluence.radiance;
        display.write(pixel, display.read(pixel) + radiance);
    }));
    let trace_kernel = DEVICE.create_kernel::<fn(u32)>(&track!(|t| {
        let pixel = dispatch_id().xy();
        let pos = pixel.cast_f32() + 0.5;
        let index = 0_u32.var();
        let bias = 1.0_f32.var();
        for i in (0..storage.num_cascades) {
            let i = i.expr();
            *index *= 4; // TODO: Fixed with angular_scale = 2, base_angles = 4.
            let weights = Vec4::expr(
                storage.get_bilinear(i, pos, index + 0),
                storage.get_bilinear(i, pos, index + 1),
                storage.get_bilinear(i, pos, index + 2),
                storage.get_bilinear(i, pos, index + 3),
            );
            let weights = weights / weights.reduce_sum();
            let rand = pcg3df(pixel.extend(t * (storage.num_cascades + 1) + i)).x;
            let j = if rand < weights.x {
                0.expr()
            } else if rand < weights.x + weights.y {
                1.expr()
            } else if rand < weights.x + weights.y + weights.z {
                2.expr()
            } else {
                3.expr()
            };
            *index += j;
            *bias *= 4.0 * Expr::<[f32; 4]>::from(weights).read(j);
        }
        let max_index = 1 << (storage.angular_scale * storage.num_cascades);
        let angle = (index.cast_f32()
            + pcg3df(pixel.extend(t * (storage.num_cascades + 1) + storage.num_cascades)).x)
            / (max_index as f32).expr()
            * TAU;
        let dir = angle.direction();

        let orig_pos = pos;
        let mut fluences = vec![];
        let mut pos = pos;
        let mut dir = dir;

        for i in (0..storage.num_cascades) {
            let traced = world.trace(
                pos,
                dir,
                (2.0 * if i == 0 {
                    1.0
                } else {
                    (3 << ((i - 1) * storage.angular_scale)) as f32
                })
                .expr(),
            );
            pos = traced.final_pos;
            dir = traced.final_dir;
            fluences.push(traced.fluence);
        }

        let mut radiance = Radiance::splat(0.0).expr();

        for i in (0..storage.num_cascades).rev() {
            radiance = fluences[i as usize].over_radiance(radiance);
            let index = index >> (storage.angular_scale * (storage.num_cascades - 1 - i));
            next_storage.add_bilinear(i.expr(), orig_pos, index, luma(radiance) / bias);
        }

        display.write(pixel, display.read(pixel) + radiance / bias);
    }));

    let draw_rc_overlay =
        DEVICE.create_kernel::<fn(Vec2<f32>, f32)>(&track!(|cursor, exposure| {
            let pixel = dispatch_id().xy();
            let pos = pixel.cast_f32() + 0.5;
            let delta = pos - cursor;
            let dist = delta.length();
            let angle = delta.angle();
            let cascade = keter::max((dist / 2.0).log2() / 2.0, 0.0).ceil().cast_u32();
            if cascade >= storage.num_cascades {
                return;
            }
            let index =
                (angle / TAU).rem_euclid(1.0) * (4 << (storage.angular_scale * cascade)).cast_f32();
            let index = index.cast_u32();
            let weight = storage.get_bilinear(cascade, cursor, index);
            let color = weight / exposure * cascade_colors().expr().read(cascade);
            app.display().write(pixel, color);
        }));
    let draw_probe_overlay = DEVICE.create_kernel::<fn(u32)>(&track!(|cascade| {
        let pos = dispatch_id().xy().cast_f32() + 0.5;
        let pos = pos * storage.base_spacing * (1 << cascade).cast_f32();
        app.set_pixel(
            pos.floor().cast_i32(),
            cascade_colors().expr().read(cascade),
        );
    }));

    /*
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

    let scene = Scene::sunflower4();

    for draw in scene.draws {
        match draw.brush {
            Brush::Rect(width, height) => {
                rect_brush.dispatch(
                    [world.size.x, world.size.y, 1],
                    &draw.center,
                    &Vec2::new(width, height),
                    &draw.color,
                );
            }
            Brush::Circle(radius) => {
                circle_brush.dispatch(
                    [world.size.x, world.size.y, 1],
                    &draw.center,
                    &radius,
                    &draw.color,
                );
            }
        }
    }
    */

    let mut iterations = 0;

    let mut cpos = Vec2::splat(-f32::INFINITY);
    let mut display_cascades = false;

    app.run(|rt| {
        /*
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
                iterations = 0;
                rt.display().view(0).copy_to_texture(&display.view(0));
                next_storage
                    .data
                    .copy_from(&vec![1.0; next_storage.data.len()]);
            }
        }
        */

        if iterations < MAX_ITERS {
            // compute_diff.dispatch_blocking([DISPLAY_SIZE / 8, DISPLAY_SIZE / 8, 1]);
            trace_kernel.dispatch(rt.dispatch_size(), &iterations);
            iterations += 1;
            copy_storage.dispatch([storage.data.len() as u32, 1, 1]);
            if iterations == MAX_ITERS {
                println!("Done");
            }
        }

        draw_kernel.dispatch(rt.dispatch_size(), &iterations);

        if rt.key_pressed(KeyCode::KeyQ) {
            display_cascades ^= true;
            cpos = rt.cursor_position;
        }

        if display_cascades {
            draw_rc_overlay.dispatch(rt.dispatch_size(), &cpos, &10.0);
        }
        if rt.key_down(KeyCode::KeyW) {
            for i in 4..storage.num_cascades {
                draw_probe_overlay.dispatch([DISPLAY_SIZE, DISPLAY_SIZE, 1], &i);
            }
        }
    });
}
