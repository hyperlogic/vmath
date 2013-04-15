pub struct Vec2(f32, f32);
pub struct Vec3(f32, f32, f32);
pub struct Vec4(f32, f32, f32, f32);

pub struct Complex(f32, f32);

pub struct Mat4(Vec4, Vec4, Vec4, Vec4);

pub static pi:f32 = f32::consts::pi;

////////////////////////////////////////////////////////////////////////////////

pub fn deg_to_rad(deg: f32) -> f32 {
    deg * (pi / 180.0)
}

pub fn rad_to_deg(rad: f32) -> f32 {
    rad * (180.0 / pi)
}

pub fn clamp(val: f32, min: f32, max: f32) -> f32 {
    if val > max {
        max
    } else if val < min {
        min
    } else {
        val
    }
}

pub fn limit_pi(theta: f32) -> f32 {
    mod_2pi(theta + pi) - pi
}

pub fn mod_2pi(theta: f32) -> f32 {
    theta - 2.0 * pi * f32::floor(theta / (2.0 * pi))
}

pub fn fuzzy_eq(lhs: f32, rhs: f32) -> bool {
    fuzzy_eq_epsilon(lhs, rhs, 0.0001)
}

pub fn fuzzy_eq_epsilon(lhs: f32, rhs: f32, epsilon: f32) -> bool {
    f32::abs(rhs - lhs) <= epsilon
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/*
pub fn rand_int(min: int, max: int) -> int {
    fail!(~"not implemented");
}

pub fn rand_f32(min: f32, max: f32) -> f32 {
    fail!(~"not implemented");
}
*/

////////////////////////////////////////////////////////////////////////////////
// Vec2

impl Vec2 {
    /*
    pub fn rand_unit() -> Vec2 {
        let theta = rand_f32(0.0, 2.0 * pi);
        Vec2(f32::cos(theta), f32::sin(theta))
    }
    */
    fn lerp(a: Vec2, b: Vec2, t: f32) -> Vec2 {
        a + (b - a).fmul(t)
    }
    fn fuzzy_eq(lhs: &Vec2, rhs: &Vec2) -> bool {
        Vec2::fuzzy_eq_epsilon(lhs, rhs, 0.0001)
    }
    fn fuzzy_eq_epsilon(lhs: &Vec2, rhs: &Vec2, epsilon: f32) -> bool {
        let Vec2(lx, ly) = *lhs;
        let Vec2(rx, ry) = *rhs;
        (f32::abs(rx - lx) <= epsilon &&
         f32::abs(ry - ly) <= epsilon)
    }
    fn len(&self) -> f32 {
        f32::sqrt(*self ^ *self)
    }
    fn min_len(&self, len: f32) -> Vec2 {
        let l = self.len();
        if l > len {
            self.fdiv(l).fmul(len)
        } else {
            self.dup()
        }
    }
    fn unit(&self) -> Vec2 {
        self.fdiv(self.len())
    }
    fn dup(&self) -> Vec2 {
        let Vec2(x, y) = *self;
        Vec2(x, y)
    }
    fn fmul(&self, rhs: f32) -> Vec2 {
        let Vec2(lx, ly) = *self;
        Vec2(lx * rhs, ly * rhs)
    }
    fn fdiv(&self, rhs: f32) -> Vec2 {
        let Vec2(lx, ly) = *self;
        Vec2(lx / rhs, ly / rhs)
    }
}

// ^ dot product
impl ops::BitXor<Vec2, f32> for Vec2 {
    fn bitxor(&self, rhs: &Vec2) -> f32 {
        let Vec2(lx, ly) = *self;
        let Vec2(rx, ry) = *rhs;
        lx * rx + ly * ry
    }
}

// +
impl ops::Add<Vec2, Vec2> for Vec2 {
    fn add(&self, rhs: &Vec2) -> Vec2 {
        let Vec2(lx, ly) = *self;
        let Vec2(rx, ry) = *rhs;
        Vec2(lx + rx, ly + ry)
    }
}

// -
impl ops::Sub<Vec2, Vec2> for Vec2 {
    fn sub(&self, rhs: &Vec2) -> Vec2 {
        let Vec2(lx, ly) = *self;
        let Vec2(rx, ry) = *rhs;
        Vec2(lx - rx, ly - ry)
    }
}

// *
impl ops::Mul<Vec2, Vec2> for Vec2 {
    fn mul(&self, rhs: &Vec2) -> Vec2 {
        let Vec2(lx, ly) = *self;
        let Vec2(rx, ry) = *rhs;
        Vec2(lx * rx, ly * ry)
    }
}

// /
impl ops::Div<Vec2, Vec2> for Vec2 {
    fn div(&self, rhs: &Vec2) -> Vec2 {
        let Vec2(lx, ly) = *self;
        let Vec2(rx, ry) = *rhs;
        Vec2(lx / rx, ly / ry)
    }
}

// unary -
impl ops::Neg<Vec2> for Vec2 {
    fn neg(&self) -> Vec2 {
        let Vec2(x, y) = *self;
        Vec2(-x, -y)
    }
}

// []
impl ops::Index<int, f32> for Vec2 {
    fn index(&self, i: &int) -> f32 {
        let Vec2(x, y) = *self;
        match *i {
            0 => x,
            1 => y,
            _ => fail!(~"Vec2: index out of bounds")
        }
    }
}

// ==
impl cmp::Eq for Vec2 {
    fn eq(&self, rhs: &Vec2) -> bool {
        let Vec2(lx, ly) = *self;
        let Vec2(rx, ry) = *rhs;
        lx == rx && ly == ry
    }
    fn ne(&self, rhs: &Vec2) -> bool { !(*self).eq(rhs) }
}

impl ToStr for Vec2 {
    fn to_str(&self) -> ~str {
        let Vec2(x, y) = *self;
        fmt!("Vec2(%?, %?)", x, y)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Vec3

pub impl Vec3 {
    /*
    pub fn rand_unit() -> Vec2 {
        let theta = rand_f32(0.0, 2.0 * pi);
        let phi = rand_f32(0.0, 2.0 * pi);
        Vec3(f32::cos(theta) * f32::sin(phi), f32::sin(theta) * f32::sin(phi), f32::cos(phi));
    }
    */
    fn lerp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
        a + (b - a).fmul(t)
    }
    fn fuzzy_eq(lhs: &Vec3, rhs: &Vec3) -> bool {
        Vec3::fuzzy_eq_epsilon(lhs, rhs, 0.0001)
    }
    fn fuzzy_eq_epsilon(lhs: &Vec3, rhs: &Vec3, epsilon: f32) -> bool {
        let Vec3(lx, ly, lz) = *lhs;
        let Vec3(rx, ry, rz) = *rhs;
        (f32::abs(rx - lx) <= epsilon &&
         f32::abs(ry - ly) <= epsilon &&
         f32::abs(rz - lz) <= epsilon)
    }
    fn len(&self) -> f32 {
        f32::sqrt(*self ^ *self)
    }
    fn min_len(&self, len: f32) -> Vec3 {
        let l = self.len();
        if l > len {
            self.fdiv(l).fmul(len)
        } else {
            self.dup()
        }
    }
    fn unit(&self) -> Vec3 {
        self.fdiv(self.len())
    }
    fn dup(&self) -> Vec3 {
        let Vec3(x, y, z) = *self;
        Vec3(x, y, z)
    }
    fn fmul(&self, rhs: f32) -> Vec3 {
        let Vec3(lx, ly, lz) = *self;
        Vec3(lx * rhs, ly * rhs, lz * rhs)
    }
    fn fdiv(&self, rhs: f32) -> Vec3 {
        let Vec3(lx, ly, lz) = *self;
        Vec3(lx / rhs, ly / rhs, lz / rhs)
    }
    fn to_vec2(&self) -> Vec2 {
        let Vec3(x, y, _) = *self;
        Vec2(x, y)
    }
}

// +
impl ops::Add<Vec3, Vec3> for Vec3 {
    fn add(&self, rhs: &Vec3) -> Vec3 {
        let Vec3(lx, ly, lz) = *self;
        let Vec3(rx, ry, rz) = *rhs;
        Vec3(lx + rx, ly + ry, lz + rz)
    }
}

// -
impl ops::Sub<Vec3, Vec3> for Vec3 {
    fn sub(&self, rhs: &Vec3) -> Vec3 {
        let Vec3(lx, ly, lz) = *self;
        let Vec3(rx, ry, rz) = *rhs;
        Vec3(lx - rx, ly - ry, lz - rz)
    }
}

// *
impl ops::Mul<Vec3, Vec3> for Vec3 {
    fn mul(&self, rhs: &Vec3) -> Vec3 {
        let Vec3(lx, ly, lz) = *self;
        let Vec3(rx, ry, rz) = *rhs;
        Vec3(lx * rx, ly * ry, lz * rz)
    }
}

// /
impl ops::Div<Vec3, Vec3> for Vec3 {
    fn div(&self, rhs: &Vec3) -> Vec3 {
        let Vec3(lx, ly, lz) = *self;
        let Vec3(rx, ry, rz) = *rhs;
        Vec3(lx / rx, ly / ry, lz / rz)
    }
}

// ^ dot product
impl ops::BitXor<Vec3, f32> for Vec3 {
    fn bitxor(&self, rhs: &Vec3) -> f32 {
        let Vec3(lx, ly, lz) = *self;
        let Vec3(rx, ry, rz) = *rhs;
        lx * rx + ly * ry + lz * rz
    }
}

// % cross product
impl ops::Modulo<Vec3, Vec3> for Vec3 {
    fn modulo(&self, rhs: &Vec3) -> Vec3 {
        let Vec3(lx, ly, lz) = *self;
        let Vec3(rx, ry, rz) = *rhs;
        Vec3(ly * rz - lz * ry,
             lz * rx - lx * rz,
             lx * ry - ly * rx)
    }
}

// unary -
impl ops::Neg<Vec3> for Vec3 {
    fn neg(&self) -> Vec3 {
        let Vec3(x, y, z) = *self;
        Vec3(-x, -y, -z)
    }
}

// []
impl ops::Index<int, f32> for Vec3 {
    fn index(&self, i: &int) -> f32 {
        let Vec3(x, y, z) = *self;
        match *i {
            0 => x,
            1 => y,
            2 => z,
            _ => fail!(~"Vec3: index out of bounds")
        }
    }
}

// ==
impl cmp::Eq for Vec3 {
    fn eq(&self, rhs: &Vec3) -> bool {
        let Vec3(lx, ly, lz) = *self;
        let Vec3(rx, ry, rz) = *rhs;
        lx == rx && ly == ry && lz == rz
    }
    fn ne(&self, rhs: &Vec3) -> bool { !(*self).eq(rhs) }
}

impl ToStr for Vec3 {
    fn to_str(&self) -> ~str {
        let Vec3(x, y, z) = *self;
        fmt!("Vec3(%?, %?, %?)", x, y, z)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Vec4

impl Vec4 {
    fn lerp(a: Vec4, b: Vec4, t: f32) -> Vec4 {
        a + (b - a).fmul(t)
    }
    fn fuzzy_eq(lhs: &Vec4, rhs: &Vec4) -> bool {
        Vec4::fuzzy_eq_epsilon(lhs, rhs, 0.0001)
    }
    fn fuzzy_eq_epsilon(lhs: &Vec4, rhs: &Vec4, epsilon: f32) -> bool {
        let Vec4(lx, ly, lz, lw) = *lhs;
        let Vec4(rx, ry, rz, rw) = *rhs;
        (f32::abs(rx - lx) <= epsilon &&
         f32::abs(ry - ly) <= epsilon &&
         f32::abs(rz - lz) <= epsilon &&
         f32::abs(rw - lw) <= epsilon)
    }
    fn len(&self) -> f32 {
        f32::sqrt(*self ^ *self)
    }
    fn min_len(&self, len: f32) -> Vec4 {
        let l = self.len();
        if l > len {
            self.fdiv(l).fmul(len)
        } else {
            self.dup()
        }
    }
    fn unit(&self) -> Vec4 {
        self.fdiv(self.len())
    }
    fn dup(&self) -> Vec4 {
        let Vec4(x, y, z, w) = *self;
        Vec4(x, y, z, w)
    }
    fn fmul(&self, rhs: f32) -> Vec4 {
        let Vec4(lx, ly, lz, lw) = *self;
        Vec4(lx * rhs, ly * rhs, lz * rhs, lw * rhs)
    }
    fn fdiv(&self, rhs: f32) -> Vec4 {
        let Vec4(lx, ly, lz, lw) = *self;
        Vec4(lx / rhs, ly / rhs, lz / rhs, lw / rhs)
    }
    fn to_vec2(&self) -> Vec2 {
        let Vec4(x, y, _, _) = *self;
        Vec2(x, y)
    }
    fn to_vec3(&self) -> Vec3 {
        let Vec4(x, y, z, _) = *self;
        Vec3(x, y, z)
    }
}

// +
impl ops::Add<Vec4, Vec4> for Vec4 {
    fn add(&self, rhs: &Vec4) -> Vec4 {
        let Vec4(lx, ly, lz, lw) = *self;
        let Vec4(rx, ry, rz, rw) = *rhs;
        Vec4(lx + rx, ly + ry, lz + rz, lw + rw)
    }
}

// -
impl ops::Sub<Vec4, Vec4> for Vec4 {
    fn sub(&self, rhs: &Vec4) -> Vec4 {
        let Vec4(lx, ly, lz, lw) = *self;
        let Vec4(rx, ry, rz, rw) = *rhs;
        Vec4(lx - rx, ly - ry, lz - rz, lw - rw)
    }
}

// *
impl ops::Mul<Vec4, Vec4> for Vec4 {
    fn mul(&self, rhs: &Vec4) -> Vec4 {
        let Vec4(lx, ly, lz, lw) = *self;
        let Vec4(rx, ry, rz, rw) = *rhs;
        Vec4(lx * rx, ly * ry, lz * rz, lw * rw)
    }
}

// /
impl ops::Div<Vec4, Vec4> for Vec4 {
    fn div(&self, rhs: &Vec4) -> Vec4 {
        let Vec4(lx, ly, lz, lw) = *self;
        let Vec4(rx, ry, rz, rw) = *rhs;
        Vec4(lx / rx, ly / ry, lz / rz, lw / rw)
    }
}

// ^ dot product
impl ops::BitXor<Vec4, f32> for Vec4 {
    fn bitxor(&self, rhs: &Vec4) -> f32 {
        let Vec4(lx, ly, lz, lw) = *self;
        let Vec4(rx, ry, rz, rw) = *rhs;
        lx * rx + ly * ry + lz * rz + lw * rw
    }
}

// unary -
impl ops::Neg<Vec4> for Vec4 {
    fn neg(&self) -> Vec4 {
        let Vec4(x, y, z, w) = *self;
        Vec4(-x, -y, -z, -w)
    }
}

// []
impl ops::Index<int, f32> for Vec4 {
    fn index(&self, i: &int) -> f32 {
        let Vec4(x, y, z, w) = *self;
        match *i {
            0 => x,
            1 => y,
            2 => z,
            3 => w,
            _ => fail!(~"Vec4: index out of bounds")
        }
    }
}

// ==
impl cmp::Eq for Vec4 {
    fn eq(&self, rhs: &Vec4) -> bool {
        let Vec4(lx, ly, lz, lw) = *self;
        let Vec4(rx, ry, rz, rw) = *rhs;
        lx == rx && ly == ry && lz == rz && lw == rw
    }
    fn ne(&self, rhs: &Vec4) -> bool { !(*self).eq(rhs) }
}

impl ToStr for Vec4 {
    fn to_str(&self) -> ~str {
        let Vec4(x, y, z, w) = *self;
        fmt!("Vec4(%?, %?, %?, %?)", x, y, z, w)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Complex

pub impl Complex {
    fn from_v2(vec2: &Vec2) -> Complex {
        let Vec2(real, imag) = *vec2;
        Complex(real, imag)
    }
    fn fuzzy_eq(lhs: Complex, rhs: Complex) -> bool {
        Complex::fuzzy_eq_epsilon(lhs, rhs, 0.0001)
    }
    fn fuzzy_eq_epsilon(lhs: Complex, rhs: Complex, epsilon: f32) -> bool {
        let Complex(l_real, l_imag) = lhs;
        let Complex(r_real, r_imag) = rhs;
        (f32::abs(r_real - l_real) <= epsilon &&
         f32::abs(r_imag - l_imag) <= epsilon)
    }
    fn sqrt(z: &Complex) -> Complex {
        let Complex(x, y) = *z;

        if x == 0.0 {
            let t = f32::sqrt(f32::abs(y) / 2.0);
            if y < 0.0 {
                Complex(t, -t)
            } else {
                Complex(t, t)
            }
        } else {
            let t = f32::sqrt(2.0 * (z.len() + f32::abs(x)));
            let u = t / 2.0;
            if x > 0.0 {
                Complex(u, y / t)
            } else {
                if y < 0.0 {
                    Complex(f32::abs(y) / t, -u)
                } else {
                    Complex(f32::abs(y) / t, u)
                }
            }
        }
    }
    fn exp(z: &Complex) -> Complex {
        let Complex(real, imag) = *z;
        let e = f32::exp(real);
        Complex(e * f32::cos(imag), e * f32::sin(imag))
    }
    fn exp_i(theta: f32) -> Complex {
        Complex(f32::cos(theta), f32::sin(theta))
    }
    fn ln(z: &Complex) -> Complex {
        let Complex(real, imag) = *z;
        Complex(f32::ln(z.len()), f32::atan2(imag, real))
    }
    fn len(&self) -> f32 {
        let Complex(real, imag) = *self;
        f32::sqrt(real * real + imag * imag)
    }
    fn dup(&self) -> Complex {
        let Complex(real, imag) = *self;
        Complex(real, imag)
    }
}

// +
impl ops::Add<Complex, Complex> for Complex {
    fn add(&self, rhs: &Complex) -> Complex {
        let Complex(l_real, l_imag) = *self;
        let Complex(r_real, r_imag) = *rhs;
        Complex(l_real + r_real, l_imag + r_imag)
    }
}

// -
impl ops::Sub<Complex, Complex> for Complex {
    fn sub(&self, rhs: &Complex) -> Complex {
        let Complex(l_real, l_imag) = *self;
        let Complex(r_real, r_imag) = *rhs;
        Complex(l_real - r_real, l_imag - r_imag)
    }
}

// *
impl ops::Mul<Complex, Complex> for Complex {
    fn mul(&self, rhs: &Complex) -> Complex {
        let Complex(a, b) = *self;
        let Complex(c, d) = *rhs;
        let ac = a * c;
        let bd = b * d;
        Complex(ac - bd, (a + b) * (c + d) - ac - bd)
    }
}

// /
impl ops::Div<Complex, Complex> for Complex {
    fn div(&self, rhs: &Complex) -> Complex {
        let Complex(a, b) = *self;
        let Complex(c, d) = *rhs;
        let denom = c * c - b * b;
        Complex((a * c + b * d) / denom, (b * c - a * d) / denom)
    }
}

// unary -
impl ops::Neg<Complex> for Complex {
    fn neg(&self) -> Complex {
        let Complex(real, imag) = *self;
        Complex(-real, -imag)
    }
}

// ! (complex conjugate
impl ops::Not<Complex> for Complex {
    fn not(&self) -> Complex {
        let Complex(real, imag) = *self;
        Complex(real, -imag)
    }
}

// []
impl ops::Index<int,f32> for Complex {
    fn index(&self, i: &int) -> f32 {
        let Complex(real, imag) = *self;
        match *i {
            0 => real,
            1 => imag,
            _ => fail!(~"Complex: index out of bounds")
        }
    }
}

// ==
impl cmp::Eq for Complex {
    fn eq(&self, rhs: &Complex) -> bool {
        let Complex(l_real, l_imag) = *self;
        let Complex(r_real, r_imag) = *rhs;
        l_real == r_real && l_imag == r_imag
    }
    fn ne(&self, rhs: &Complex) -> bool { !(*self).eq(rhs) }
}

impl ToStr for Complex {
    fn to_str(&self) -> ~str {
        let Complex(real, imag) = *self;
        fmt!("Complex(%?, %?)", real, imag)
    }
}

////////////////////////////////////////////////////////////////////////////////
// Mat4

pub impl Mat4 {
    fn from_axes(x_axis: &Vec3, y_axis: &Vec3, z_axis: &Vec3, trans: &Vec3) -> Mat4 {
        let Vec3(x0, x1, x2) = *x_axis;
        let Vec3(y0, y1, y2) = *y_axis;
        let Vec3(z0, z1, z2) = *z_axis;
        let Vec3(t0, t1, t2) = *trans;
        Mat4(Vec4(x0, x1, x2, 0.0),
             Vec4(y0, y1, y2, 0.0),
             Vec4(z0, z1, z2, 0.0),
             Vec4(t0, t1, t2, 1.0))
    }
    fn from_cols(col0: &Vec4, col1: &Vec4, col2: &Vec4, col3: &Vec4) -> Mat4 {
        let Vec4(x0, x1, x2, x3) = *col0;
        let Vec4(y0, y1, y2, y3) = *col1;
        let Vec4(z0, z1, z2, z3) = *col2;
        let Vec4(t0, t1, t2, t3) = *col3;
        Mat4(Vec4(x0, x1, x2, x3),
             Vec4(y0, y1, y2, y3),
             Vec4(z0, z1, z2, z3),
             Vec4(t0, t1, t2, t3))
    }
    fn from_rows(row0: &Vec4, row1: &Vec4, row2: &Vec4, row3: &Vec4) -> Mat4 {
        let Vec4(x0, y0, z0, t0) = *row0;
        let Vec4(x1, y1, z1, t1) = *row1;
        let Vec4(x2, y2, z2, t2) = *row2;
        let Vec4(x3, y3, z3, t3) = *row3;
        Mat4(Vec4(x0, x1, x2, x3),
             Vec4(y0, y1, y2, y3),
             Vec4(z0, z1, z2, z3),
             Vec4(t0, t1, t2, t3))
    }
    fn from_scale(scale: &Vec3) -> Mat4 {
        let Vec3(sx, sy, sz) = *scale;
        Mat4(Vec4(sx, 0.0, 0.0, 0.0),
             Vec4(0.0, sy, 0.0, 0.0),
             Vec4(0.0, 0.0, sz, 0.0),
             Vec4(0.0, 0.0, 0.0, 1.0))
    }
    fn identity() -> Mat4 {
        Mat4(Vec4(1.0, 0.0, 0.0, 0.0),
             Vec4(0.0, 1.0, 0.0, 0.0),
             Vec4(0.0, 0.0, 1.0, 0.0),
             Vec4(0.0, 0.0, 0.0, 1.0))
    }
    fn frustum(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let f = 1.0 / f32::tan(fovy / 2.0);
        let col0 = Vec4(f / aspect, 0.0, 0.0, 0.0);
        let col1 = Vec4(0.0, f, 0.0, 0.0);
        let col2 = Vec4(0.0, 0.0, (far + near) / (near - far), -1.0);
        let col3 = Vec4(0.0, 0.0, (2.0 * far * near) / (near - far), 0.0);
        Mat4(col0, col1, col2, col3)
    }
    fn ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
        let tx = -(right + left) / (right - left);
        let ty = -(top + bottom) / (top - bottom);
        let tz = -(far + near) / (far - near);
        let col0 = Vec4(2.0 / (right - left), 0.0, 0.0, 0.0);
        let col1 = Vec4(0.0, 2.0 / (top - bottom), 0.0, 0.0);
        let col2 = Vec4(0.0, 0.0, -2.0 / (far - near), 0.0);
        let col3 = Vec4(tx, ty, tz, 1.0);
        Mat4(col0, col1, col2, col3)
    }
    fn xform4x4(&self, v: &Vec4) -> Vec4 {
        let Mat4(col0, col1, col2, col3) = *self;
        let Vec4(x, y, z, w) = *v;
        col0.fmul(x) + col1.fmul(y) + col2.fmul(z) + col3.fmul(w)
    }
    fn xform3x4(&self, v: &Vec3) -> Vec3 {
        let Mat4(col0, col1, col2, col3) = *self;
        let Vec3(x, y, z) = *v;
        (col0.fmul(x) + col1.fmul(y) + col2.fmul(z) + col3).to_vec3()
    }
    fn xform3x3(&self, v: &Vec3) -> Vec3 {
        let Mat4(col0, col1, col2, _) = *self;
        let Vec3(x, y, z) = *v;
        (col0.fmul(x) + col1.fmul(y) + col2.fmul(z)).to_vec3()
    }
    fn transpose(&self) -> Mat4 {
        let Mat4(col0, col1, col2, col3) = *self;
        Mat4::from_rows(&col0, &col1, &col2, &col3)
    }
    fn ortho_inverse(&self) -> Mat4 {
        let Mat4(_, _, _, pos) = *self;
        let t = self.transpose();
        let Vec3(x, y, z) = -t.xform3x3(&pos.to_vec3());
        let Mat4(col0, col1, col2, _) = t;
        Mat4(col0, col1, col2, Vec4(x, y, z, 1.0))
    }
    /*
    fn full_inverse(&self) -> Mat4 {
        fail!(~"not implemented");
    }
    */
}

impl ToStr for Mat4 {
    fn to_str(&self) -> ~str {
        let Mat4(Vec4(m0, m1, m2, m3),
                 Vec4(m4, m5, m6, m7),
                 Vec4(m8, m9, m10, m11),
                 Vec4(m12, m13, m14, m15)) = *self;
        fmt!("Mat4(col0 = %?, %?, %?, %?, col1 = %?, %?, %?, %?, col2 = %?, %?, %?, %?, col3 = %?, %?, %?, %?)",
             m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15)
    }
}
