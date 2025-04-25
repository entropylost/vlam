use core::f32;

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Value)]
#[repr(C)]
pub struct Object {
    pub center: Vec2<f32>,
    // Assume circular; change sometime?
    pub radius: f32,
    pub refraction_index: f32,
    pub color: Color,
}

pub struct AnalyticTracer {
    pub objects: Buffer<Object>,
}

#[derive(Debug, Clone, Copy, PartialEq, Value)]
#[repr(C)]
struct RayHit {
    distance: f32, // Inf if no hit.
    normal: Vec2<f32>,
    object: u32,
    leaving: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Value)]
#[repr(C)]
pub struct TracedRay {
    pub fluence: Fluence,
    pub final_pos: Vec2<f32>,
    pub final_dir: Vec2<f32>,
}

#[tracked]
fn intersect_circle(
    ray_start: Expr<Vec2<f32>>,
    ray_dir: Expr<Vec2<f32>>,
    radius: Expr<f32>,
) -> (Expr<f32>, Expr<f32>, Expr<f32>, Expr<bool>) {
    let dist_to_parallel = -ray_start.dot(ray_dir);
    let min_point = ray_start + dist_to_parallel * ray_dir;
    let dist_to_center = min_point.length();
    let penetration = radius - dist_to_center;
    if penetration < 0.0.expr() {
        (0.0.expr(), 0.0.expr(), penetration, false.expr())
    } else {
        let dist_to_intersection = (radius.sqr() - dist_to_center.sqr()).sqrt();
        let min_t = dist_to_parallel - dist_to_intersection;
        let max_t = dist_to_parallel + dist_to_intersection;
        (min_t, max_t, penetration, true.expr())
    }
}

impl AnalyticTracer {
    pub fn new(objects: &[Object]) -> Self {
        Self {
            objects: DEVICE.create_buffer_from_slice(objects),
        }
    }
    #[tracked]
    fn trace_once(
        &self,
        pos: Expr<Vec2<f32>>,
        start: Expr<f32>,
        dir: Expr<Vec2<f32>>,
    ) -> Expr<RayHit> {
        let closest_hit = RayHit {
            distance: f32::INFINITY,
            normal: Vec2::splat(0.0),
            object: 0,
            leaving: false,
        }
        .var();
        for i in 0_u32.expr()..self.objects.len_expr_u32() {
            let object = self.objects.read(i);
            let (min_t, max_t, _penetration, hit) =
                intersect_circle(pos - object.center, dir, object.radius);
            // TODO: Figure out what to do if circles intersect.
            if hit {
                if start < min_t && min_t < closest_hit.distance {
                    *closest_hit.distance = min_t;
                    *closest_hit.normal = (pos + min_t * dir - object.center).normalize();
                    *closest_hit.object = i;
                    *closest_hit.leaving = false;
                } else if start < max_t && max_t < closest_hit.distance {
                    *closest_hit.distance = max_t;
                    *closest_hit.normal = (pos + max_t * dir - object.center).normalize();
                    *closest_hit.object = i;
                    *closest_hit.leaving = true;
                }
            }
        }
        **closest_hit
    }
    // Needs RNG; otherwise refraction only.
    #[tracked]
    pub fn trace(
        &self,
        pos: Expr<Vec2<f32>>,
        dir: Expr<Vec2<f32>>,
        len: Expr<f32>,
    ) -> Expr<TracedRay> {
        let pos = pos.var();
        let dir = dir.var();
        let len = len.var();
        let color = Color::empty().var();
        let refr_index = 1.0_f32.var();
        let fluence = Fluence::empty().var();
        for i in 0_u32.expr()..self.objects.len_expr_u32() {
            let object = self.objects.read(i);
            if (pos - object.center).length() < object.radius {
                *color = object.color;
                *refr_index = object.refraction_index;
            }
        }
        loop {
            let hit = self.trace_once(**pos, 0.001_f32.expr(), **dir);
            if hit.distance > len {
                *pos += len * dir;
                *fluence = fluence.over(color.to_fluence(**len));
                break;
            }
            *pos += hit.distance * dir;
            *len -= hit.distance;
            *fluence = fluence.over(color.to_fluence(hit.distance));
            let normal = if hit.leaving { hit.normal } else { -hit.normal };
            // let a = normal.dot(dir) >= 0.0;
            // lc_assert!(a);
            let obj = self.objects.read(hit.object);
            let next_refr_index = if hit.leaving {
                1.0_f32.expr()
            } else {
                obj.refraction_index
            };
            let angle = (1.0 - normal.dot(dir).sqr()).sqrt() * refr_index / next_refr_index; // sin theta
            if angle.abs() >= 1.0 {
                *fluence =
                    fluence.over(Fluence::expr(Vec3::splat_expr(0.0), Vec3::splat_expr(0.0)));
                break;
            }
            let tangent = Vec2::expr(normal.y, -normal.x);
            let sign = dir.dot(tangent).signum();
            *dir = angle * sign * tangent + (1.0 - angle.sqr()).sqrt() * normal;
            *refr_index = next_refr_index;
            *color = obj.color;
        }
        TracedRay::from_comps_expr(TracedRayComps {
            fluence: **fluence,
            final_pos: **pos,
            final_dir: **dir,
        })
    }
}
