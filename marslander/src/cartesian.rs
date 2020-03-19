use std::{ops, cmp};
use std::cmp::Ordering;

pub const SIZE_X: usize = 7000;
pub const SIZE_Y: usize = 3000;

pub type Radian = f32;
pub type Segment = (Point2, Point2);

#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, Default)]
pub struct Point2((f32, f32));

impl Point2 {
    pub fn new(x: f32, y: f32) -> Self { Self((x, y)) }
    pub fn x(&self) -> f32 { (self.0).0 }
    pub fn y(&self) -> f32 { (self.0).1 }
}

impl ops::Add<Vector2> for Point2 {
    type Output = Self;

    fn add(self, vector: Vector2) -> Self::Output {
        Self((self.x() + vector.dx(), self.y() + vector.dy()))
    }
}

impl ops::Sub<Point2> for Point2 {
    type Output = Vector2;

    fn sub(self, other: Point2) -> Self::Output {
        Vector2((self.x() - other.x(), self.y() - other.y()))
    }
}

#[derive(Copy, Clone, PartialEq, Default, Debug)]
pub struct Angle {
    pub rad: Radian
}

impl Angle {
    fn new(rad: Radian) -> Self { Self { rad } }
    fn deg(&self) -> Radian { self.rad.to_degrees() }
}

impl ops::Mul<f32> for Angle {
    type Output = Angle;

    fn mul(self, mul: f32) -> Self::Output {
        Angle::new(self.deg() * mul)
    }
}

#[derive(Copy, Clone, PartialEq, Default, Debug)]
pub struct Vector2(pub (f32, f32));

impl Vector2 {
    pub fn dx(&self) -> f32 { (self.0).0 }
    pub fn dy(&self) -> f32 { (self.0).1 }
}

impl ops::Add<Vector2> for Vector2 {
    type Output = Vector2;

    fn add(self, rhs: Vector2) -> Vector2 {
        Vector2((self.dx() + rhs.dx(), self.dy() + rhs.dy()))
    }
}

impl ops::Mul<f32> for Vector2 {
    type Output = Vector2;

    fn mul(self, times: f32) -> Self::Output {
        Vector2((self.dx() * times, self.dy() * times))
    }
}

impl Vector2 {
    pub fn length(&self) -> f32 { (self.dx() * self.dx() + self.dy() * self.dy()).sqrt() }

    pub fn rotate(&self, angle: Angle) -> Self {
        Self((
            (self.dx() * angle.rad.cos() - self.dy() * angle.rad.sin()).round(),
            (self.dx() * angle.rad.sin() + self.dy() * angle.rad.cos()).round(),
        ))
    }
}

pub type LineValue = Vec<Point2>; // cap: SURFACE_N

#[derive(Clone, PartialOrd, PartialEq, Debug)]
pub struct Line(LineValue);

impl Line {
    pub fn new(line_value: LineValue) -> Self {
        assert!(
            line_value.iter().count() > 1,
            "At least two points required for line."
        );

        Line(line_value)
    }

    pub fn points(&self) -> &LineValue {
        &self.0
    }

    pub fn land_zone(&self) -> (Point2, Point2) {
        let points = self.points();
        let segment = points
            .windows(2)
            .find(|win| win.get(0).unwrap().y().eq(&win.get(1).unwrap().y()))
            .unwrap();

        (segment[0], segment[1])
    }

    pub fn is_horizontal_at_x(&self, x: i32) -> bool {
        let segment = self.get_segment_for(x);
        let y1 = &segment.0.y();
        let y2 = &segment.1.y();
        let is_horizontal = y1.eq(y2);

        is_horizontal
    }

    pub fn get_segment_for(&self, x: i32) -> Segment {
        let points = self.points();
        let segment = points
            .windows(2)
            .find(|win| x.max(0).min(SIZE_X as i32 - 1) <= win.get(1).unwrap().x() as i32)
            .unwrap();

        (segment[0], segment[1])
    }

    pub fn get_y_for_x(&self, x: f32) -> i32 {
        let segment = self.get_segment_for(x as i32);

        (segment.0.y() + (x - segment.0.x()) * (segment.1.y() - segment.0.y()) / (segment.1.x() - segment.0.x())) as i32
    }
}

impl cmp::PartialEq<Point2> for Line {
    fn eq(&self, _point: &Point2) -> bool { unimplemented!() }

    fn ne(&self, _point: &Point2) -> bool { unimplemented!() }
}

impl cmp::PartialOrd<Point2> for Line {
    fn partial_cmp(&self, other: &Point2) -> Option<Ordering> {
        let y_for_x = self.get_y_for_x(other.x());

        if y_for_x >= 0 || y_for_x == 0 {
            Some(Ordering::Greater)
        } else {
            None
        }
    }

    fn gt(&self, point: &Point2) -> bool {
        let y_for_x = self.get_y_for_x(point.x()) as f32;
        let is_greater = y_for_x - point.y() >= 0.0_f32;

        if is_greater {
            return is_greater;
        }

        is_greater
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::Time;

    #[test]
    fn it_can_calculates_a_vectors_length() {
        assert_eq!(std::f32::consts::SQRT_2, Vector2((1.0, 1.0)).length());
    }

    #[test]
    fn it_can_add_two_vectors() {
        assert_eq!(Vector2((3.0, 4.0)), Vector2((1.0, 3.0)) + Vector2((2.0, 1.0)));
    }

    #[test]
    fn it_can_mul_a_vector() {
        assert_eq!(Vector2((-1.0, 2.0)) * Time(2.0).sec(), Vector2((-2.0, 4.0)));
    }

    #[test]
    fn it_can_rotate_a_vector() {
        assert_eq!(Vector2((-1.0, 1.0)).rotate(Angle { rad: 90.0_f32.to_radians() }), Vector2((-1.0, -1.0)));
        assert_eq!(Vector2((0.0, -2.0)).rotate(Angle { rad: 180.0_f32.to_radians() }), Vector2((0.0, 2.0)));
        assert_eq!(Vector2((2.0, 2.0)).rotate(Angle { rad: 270.0_f32.to_radians() }), Vector2((2.0, -2.0)));
        assert_eq!(Vector2((-4.0, 4.0)).rotate(Angle { rad: 360.0_f32.to_radians() }), Vector2((-4.0, 4.0)));
    }

    #[test]
    fn it_can_add_a_vector_to_a_point() {
        assert_eq!(
            Point2((-2.0, 4.0)),
            Point2((-1.0, 2.0)) + Vector2((-1.0, 2.0))
        );
    }

    #[test]
    fn it_can_determine_a_segment_for_x() {
        assert_eq!((|| {
            let mut points: LineValue = Vec::with_capacity(29 + 2);

            points.push(Point2((0.0, 0.0)));
            points.push(Point2((1.0, 2.0)));
            points.push(Point2((2.0, 2.0)));

            let line = Line::new(points);

            line.get_segment_for(2)
        })(), (Point2((1.0, 2.0)), Point2((2.0, 2.0))))
    }

    #[test]
    fn it_can_determine_a_collision() {
        assert_eq!((|| {
            let mut points: LineValue = Vec::with_capacity(29 + 2);

            points.push(Point2((0.0, 0.0)));
            points.push(Point2((1.0, 1.0)));
            points.push(Point2((2.0, 2.0)));

            let line = Line::new(points);

            line > Point2((2.0, 2.0))
        })(), true)
    }
}