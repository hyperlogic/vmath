use vec2::Vec2;

pub struct Mat2(Vec2, Vec2);

pub impl Mat2 {
    fn from_cols(col0: &Vec2, col1: &Vec2) -> Mat2 {
        let Vec2(m0, m2) = *col0;
        let Vec2(m1, m3) = *col1;
        Mat2(Vec2(m0, m1), Vec2(m2, m3))
    }
    fn xform(&self, v: &Vec2) -> Vec2 {
        let Mat2(Vec2(m0, m1), Vec2(m2, m3)) = *self;
        let Vec2(vx, vy) = *v;
        Vec2(m0 * vx + m1 * vy,
             m2 * vx + m3 * vy)
    }
    fn transpose(&self) -> Mat2 {
        let Mat2(row0, row1) = *self;
        Mat2::from_cols(&row0, &row1)
    }
    fn det(&self) -> f32 {
        let Mat2(Vec2(m0, m1), Vec2(m2, m3)) = *self;
        m0 * m3 - m1 * m2
    }
}

impl ToStr for Mat2 {
    fn to_str(&self) -> ~str {
        let Mat2(Vec2(m0, m1), Vec2(m2, m3)) = *self;
        fmt!("Mat4(%?, %?, %?, %?)", m0, m1, m2, m3)
    }
}
