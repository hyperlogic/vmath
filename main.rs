extern mod vmath;

use mod vmath::vec2::{Vec2};
use mod vmath::vec3::{Vec3};
use mod vmath::vec4::{Vec4};
use mod vmath::mat2::{Mat2};

fn main() {
    let pi = 3.141592653589793;

    let theta = pi / 3.0;
    let m = Mat2(Vec2(f32::cos(theta), f32::sin(theta)),
                 Vec2(f32::sin(theta), -f32::cos(theta)));
    let v = Vec2(1.0, 0.0);
    let v_prime = m.xform(&-v);
    println(fmt!("v_prime = %s", v_prime.to_str()));

    let a = Vec3(1.0, 0.0, 0.0);
    let b = Vec3(0.0, 1.0, 0.0);
    let c = a + b - a;
    assert!(a % b == Vec3(0.0, 0.0, 1.0));
    assert!(a ^ b == 0.0);
    println(fmt!("c = %s", c.to_str()));
}