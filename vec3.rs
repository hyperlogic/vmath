use vec2::*;

pub struct Vec3(f32, f32, f32);

impl Vec3 {
    fn fmul(&self, rhs: f32) -> Vec3 {
        let Vec3(lx, ly, lz) = *self;
        Vec3(lx * rhs, ly * rhs, lz * rhs)
    }
    fn fdiv(&self, rhs: f32) -> Vec3 {
        let Vec3(lx, ly, lz) = *self;
        Vec3(lx / rhs, ly / rhs, lz / rhs)
    }
    fn len(&self) -> f32 {
        f32::sqrt(*self ^ *self)
    }
    fn unit(&self) -> Vec3 {
        self.fdiv(self.len())
    }
    fn to_v2(&self) -> Vec2 {
        let Vec3(x, y, _) = *self;
        Vec2(x, y)
    }
}

// +
impl ops::Add<Vec3,Vec3> for Vec3 {
    fn add(&self, rhs: &Vec3) -> Vec3 {
        let Vec3(lx, ly, lz) = *self;
        let Vec3(rx, ry, rz) = *rhs;
        Vec3(lx + rx, ly + ry, lz + rz)
    }
}

// -
impl ops::Sub<Vec3,Vec3> for Vec3 {
    fn sub(&self, rhs: &Vec3) -> Vec3 {
        let Vec3(lx, ly, lz) = *self;
        let Vec3(rx, ry, rz) = *rhs;
        Vec3(lx - rx, ly - ry, lz - rz)
    }
}

// *
impl ops::Mul<Vec3,Vec3> for Vec3 {
    fn mul(&self, rhs: &Vec3) -> Vec3 {
        let Vec3(lx, ly, lz) = *self;
        let Vec3(rx, ry, rz) = *rhs;
        Vec3(lx * rx, ly * ry, lz * rz)
    }
}

// /
impl ops::Div<Vec3,Vec3> for Vec3 {
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
impl ops::Index<int,f32> for Vec3 {
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
