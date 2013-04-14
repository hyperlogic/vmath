mod vmath;

fn main() {
    let theta = vmath::deg_to_rad(60.0);
    let m = vmath::Mat2(vmath::Vec2(f32::cos(theta), f32::sin(theta)),
                        vmath::Vec2(f32::sin(theta), -f32::cos(theta)));
    let v = vmath::Vec2(1.0, 0.0);
    let v_prime = m.xform(&-v);
    println(fmt!("v_prime = %s", v_prime.to_str()));

    let a = vmath::Vec3(1.0, 0.0, 0.0);
    let b = vmath::Vec3(0.0, 1.0, 0.0);
    let c = a + b - a;
    assert!(a % b == vmath::Vec3(0.0, 0.0, 1.0));
    assert!(a ^ b == 0.0);
    println(fmt!("c = %s", c.to_str()));
    println(fmt!("c.len = %?", c.len()));
}