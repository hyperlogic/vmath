use std::num::cos;
use std::num::sin;

//use vmath::Vec2;
use vmath::Vec3;
use vmath::Vec4;
use vmath::Mat4;
use vmath::Complex;
//use vmath::Quat;

mod vmath;

fn main() {
    let theta = vmath::deg_to_rad(60.0);
    let m = Mat4(Vec4(cos(theta), sin(theta), 0.0, 0.0),
                 Vec4(sin(theta), -cos(theta), 0.0, 0.0),
                 Vec4(0.0, 0.0, 1.0, 0.0),
                 Vec4(0.0, 0.0, 0.0, 1.0));
    let v = Vec4(1.0, 0.0, 0.0, 0.0);
    let v_prime = m.xform4x4(&-v);
    println(format!("v_prime = {}", v_prime.to_str()));

    let aa = Complex(cos(theta), sin(theta));
    let bb = Complex::exp_i(theta);
    assert!(Complex::fuzzy_eq(aa, bb));

    /*
    for i in range(0, 10) {
        println(format!("rand_int(0, 10) = %?", vmath::rand_int(0, 10)));
    }
    */

    let a = Vec3(1.0, 0.0, 0.0);
    let b = Vec3(0.0, 1.0, 0.0);
    let c = vmath::lerp_vec(&a, &b, 0.5);
    assert!(a % b == Vec3(0.0, 0.0, 1.0));
    assert!(a ^ b == 0.0);
    assert!(vmath::fuzzy_eq_vec(&a, &a));
    println(format!("c = {}", c.to_str()));
    println(format!("c.len = {}", c.len()));
    println(format!("1/2 = {}", vmath::lerp_f32(0.0, 1.0, 0.5)));
}