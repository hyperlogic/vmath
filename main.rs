mod vmath;

fn main() {
    let theta = vmath::deg_to_rad(60.0);
    let m = vmath::Mat4(vmath::Vec4(f32::cos(theta), f32::sin(theta), 0.0, 0.0),
                        vmath::Vec4(f32::sin(theta), -f32::cos(theta), 0.0, 0.0),
                        vmath::Vec4(0.0, 0.0, 1.0, 0.0),
                        vmath::Vec4(0.0, 0.0, 0.0, 1.0));
    let v = vmath::Vec4(1.0, 0.0, 0.0, 0.0);
    let v_prime = m.xform4x4(&-v);
    println(fmt!("v_prime = %s", v_prime.to_str()));

    let aa = vmath::Complex(f32::cos(theta), f32::sin(theta));
    let bb = vmath::Complex::exp_i(theta);
    assert!(vmath::Complex::fuzzy_eq(aa, bb));

    for 10.times {
        println(fmt!("rand_int(0, 10) = %?", vmath::rand_int(0, 10)));
    }

    let a = vmath::Vec3(1.0, 0.0, 0.0);
    let b = vmath::Vec3(0.0, 1.0, 0.0);
    let c = vmath::lerp_vec(&a, &b, 0.5);
    assert!(a % b == vmath::Vec3(0.0, 0.0, 1.0));
    assert!(a ^ b == 0.0);
    assert!(vmath::fuzzy_eq_vec(&a, &a));
    println(fmt!("c = %s", c.to_str()));
    println(fmt!("c.len = %?", c.len()));
    println(fmt!("1/2 = %?", vmath::lerp_f32(0.0, 1.0, 0.5)));
}