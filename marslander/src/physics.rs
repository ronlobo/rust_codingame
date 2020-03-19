use std::ops;
use crate::cartesian::*;

pub const GRAVITY_Y: f32 = -3.711;
pub const GRAVITY: Acceleration = acceleration(0.0, GRAVITY_Y);

pub const fn acceleration(dx: f32, dy: f32) -> Acceleration { Acceleration(Vector2((dx, dy))) }

pub type Seconds = f32;
pub type Position = Point2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Acceleration(pub Vector2);

impl ops::Mul<Time> for Acceleration {
    type Output = Speed;

    fn mul(self, mul: Time) -> Self::Output {
        Speed { direction: self.0 * mul.sec() }
    }
}

impl ops::Add<Acceleration> for Acceleration {
    type Output = Self;

    fn add(self, other: Acceleration) -> Self::Output {
        Acceleration(self.0 + other.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Time(pub Seconds);

impl Time {
    pub fn sec(&self) -> Seconds { self.0 }
}

pub trait ToTime { fn sec(self) -> Time; }

impl ToTime for f32 {
    fn sec(self) -> Time { Time(self) }
}

#[derive(Copy, Clone, PartialEq, Default, Debug)]
pub struct Particle {
    pub position: Position,
    pub speed: Speed,
}

impl Particle {
    pub fn accelerate(&self, acceleration: Acceleration, time: Time) -> Self {
        Particle {
            position: self.position + self.speed.direction * time.0 + acceleration.0 * time.0 * time.0 * 0.5,
            speed: self.speed + acceleration * time,
        }
    }
}

impl Default for Time {
    fn default() -> Self { Time(1.0_f32) }
}

impl ops::Mul<Time> for Time {
    type Output = Self;

    fn mul(self, other: Time) -> Self::Output {
        Time(self.0 * other.0)
    }
}

#[derive(Copy, Clone, PartialEq, Default, Debug)]
pub struct Speed {
    pub direction: Vector2,
}

impl ops::Add<Speed> for Speed {
    type Output = Speed;

    fn add(self, rhs: Speed) -> Self::Output {
        Speed { direction: self.direction + rhs.direction }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_can_accelerate() {
        assert_eq!(
            GRAVITY + acceleration(-2.0, 1.0),
            Acceleration(Vector2((-2.0, -2.711)))
        );
    }

    #[test]
    fn it_can_accelerate_a_particle() {
        let time = Time::default();
        let acceleration = GRAVITY;
        let expected = Particle { position: Point2::new(0.0, GRAVITY_Y / 2.0), speed: Speed { direction: Vector2((0.0, GRAVITY_Y)) } };
        let particle = Particle { position: Point2::default(), speed: Speed { direction: Vector2((0.0, 0.0)) } };

        assert_eq!(particle.accelerate(acceleration, time), expected);
    }
}