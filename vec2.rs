pub struct Vec2(f32, f32);

impl Vec2 {
    fn fmul(&self, rhs: f32) -> Vec2 {
        let Vec2(lx, ly) = *self;
        Vec2(lx * rhs, ly * rhs)
    }
    fn fdiv(&self, rhs: f32) -> Vec2 {
        let Vec2(lx, ly) = *self;
        Vec2(lx / rhs, ly / rhs)
    }
    fn len(&self) -> f32 {
        f32::sqrt(*self ^ *self)
    }
    fn unit(&self) -> Vec2 {
        self.fdiv(self.len())
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
impl ops::Add<Vec2,Vec2> for Vec2 {
    fn add(&self, rhs: &Vec2) -> Vec2 {
        let Vec2(lx, ly) = *self;
        let Vec2(rx, ry) = *rhs;
        Vec2(lx + rx, ly + ry)
    }
}

// -
impl ops::Sub<Vec2,Vec2> for Vec2 {
    fn sub(&self, rhs: &Vec2) -> Vec2 {
        let Vec2(lx, ly) = *self;
        let Vec2(rx, ry) = *rhs;
        Vec2(lx - rx, ly - ry)
    }
}

// *
impl ops::Mul<Vec2,Vec2> for Vec2 {
    fn mul(&self, rhs: &Vec2) -> Vec2 {
        let Vec2(lx, ly) = *self;
        let Vec2(rx, ry) = *rhs;
        Vec2(lx * rx, ly * ry)
    }
}

// /
impl ops::Div<Vec2,Vec2> for Vec2 {
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
impl ops::Index<int,f32> for Vec2 {
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
