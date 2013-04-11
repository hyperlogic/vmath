use vec2::Vec2;
use vec3::Vec3;

pub struct Vec4(f32, f32, f32, f32);

impl Vec4 {
    fn fmul(&self, rhs: f32) -> Vec4 {
        let Vec4(lx, ly, lz, lw) = *self;
        Vec4(lx * rhs, ly * rhs, lz * rhs, lw * rhs)
    }
    fn fdiv(&self, rhs: f32) -> Vec4 {
        let Vec4(lx, ly, lz, lw) = *self;
        Vec4(lx / rhs, ly / rhs, lz / rhs, lw / rhs)
    }
    fn len(&self) -> f32 {
        f32::sqrt(*self ^ *self)
    }
    fn unit(&self) -> Vec4 {
        self.fdiv(self.len())
    }
    fn to_v2(&self) -> Vec2 {
        let Vec4(x, y, _, _) = *self;
        Vec2(x, y)
    }
    fn to_v3(&self) -> Vec3 {
        let Vec4(x, y, z, _) = *self;
        Vec3(x, y, z)
    }
}

// +
impl ops::Add<Vec4,Vec4> for Vec4 {
    fn add(&self, rhs: &Vec4) -> Vec4 {
        let Vec4(lx, ly, lz, lw) = *self;
        let Vec4(rx, ry, rz, rw) = *rhs;
        Vec4(lx + rx, ly + ry, lz + rz, lw + rw)
    }
}

// -
impl ops::Sub<Vec4,Vec4> for Vec4 {
    fn sub(&self, rhs: &Vec4) -> Vec4 {
        let Vec4(lx, ly, lz, lw) = *self;
        let Vec4(rx, ry, rz, rw) = *rhs;
        Vec4(lx - rx, ly - ry, lz - rz, lw - rw)
    }
}

// *
impl ops::Mul<Vec4,Vec4> for Vec4 {
    fn mul(&self, rhs: &Vec4) -> Vec4 {
        let Vec4(lx, ly, lz, lw) = *self;
        let Vec4(rx, ry, rz, rw) = *rhs;
        Vec4(lx * rx, ly * ry, lz * rz, lw * rw)
    }
}

// /
impl ops::Div<Vec4,Vec4> for Vec4 {
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
impl ops::Index<int,f32> for Vec4 {
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
