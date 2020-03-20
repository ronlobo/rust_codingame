pub mod env {
    #[derive(Copy, Clone, Debug, PartialOrd, PartialEq)]
    pub enum EnvOption {
        PROD,
        DEV,
    }

    pub const ENV: EnvOption = EnvOption::PROD;

    pub fn is_prod() -> bool {
        match ENV {
            EnvOption::PROD => true,
            EnvOption::DEV => false
        }
    }
}

pub mod cartesian {
    use std::{ops, cmp};
    use std::cmp::Ordering;

    use crate::lander::SURFACE_N;

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
            Self(((self.x() + vector.dx()).round(), (self.y() + vector.dy()).round()))
        }
    }

    impl ops::Sub<Point2> for Point2 {
        type Output = Vector2;

        fn sub(self, other: Point2) -> Self::Output {
            Vector2(((self.x() - other.x()).round(), (self.y() - other.y()).round()))
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
        fn length(&self) -> f32 { (self.dx() * self.dx() + self.dy() * self.dy()).sqrt() }

        pub fn rotate(&self, angle: Angle) -> Self {
            Self(
                (
                    self.dx() * angle.rad.cos() - self.dy() * angle.rad.sin(),
                    self.dx() * angle.rad.sin() + self.dy() * angle.rad.cos(),
                )
            )
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

    impl Default for Line {
        fn default() -> Self {
            Line(vec![Point2::default(); SURFACE_N])
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
}

pub mod physics {
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
}

pub mod genetic {
    use crate::lander::*;
    use rand::{Rng, thread_rng};
    use rand::prelude::ThreadRng;
    use std::borrow::Borrow;

    const ELITISM: bool = true;
    const GENERATIONS_COUNT: usize = 20;
    const POPULATION_SIZE: usize = 128;
    const GENOME_SIZE: usize = 256;

    const UNIFORM_RATE: f32 = 0.7;
    const MUTATION_RATE: f32 = 0.05;
    const SELECTION_RATIO: f32 = 0.7;

    pub type Genome = [Gene; GENOME_SIZE];

    pub type Populations = Vec<LanderWithGenomeAndResult>; // cap: POPULATION_SIZE

    pub type Generation = Vec<Populations>;  // cap: GENERATIONS_COUNT

    pub type FnGeneratePopulation = dyn Fn() -> Box<dyn Fn(&Genome) -> Lander>;

    pub type FnCalcFitness = dyn Fn() -> Box<dyn Fn(&Lander) -> f32>;

    #[derive(Copy, Clone, PartialEq, Default, Debug)]
    pub struct Gene(pub f32);

    impl Gene {
        pub fn as_int(&self, max: i32) -> i32 { (self.0 * (max + 1) as f32) as i32 }
    }

    pub trait GenomeAndResult {
        type Result;

        fn from_genome(genome: Genome, gen_pop: &FnGeneratePopulation) -> Self;
        fn genome(&self) -> Genome;
        fn result(&self) -> &Self::Result;
    }

    #[derive(Clone)]
    pub struct LanderWithGenomeAndResult {
        genome: Genome,
        result: Lander,
    }

    impl GenomeAndResult for LanderWithGenomeAndResult {
        type Result = Lander;

        fn from_genome(genome: Genome, gen_lander: &FnGeneratePopulation) -> Self {
            LanderWithGenomeAndResult {
                genome,
                result: gen_lander()(&genome),
            }
        }

        fn genome(&self) -> Genome { self.genome }

        fn result(&self) -> &Self::Result { self.result.borrow() }
    }

    fn build_next_generation(
        rng: &mut ThreadRng,
        population: &Populations,
        gen_pop: &FnGeneratePopulation,
    ) -> Populations {
        let elitism_offset = if ELITISM { 1 } else { 0 };
        let best_pop: &LanderWithGenomeAndResult = population.get(0).unwrap();
        let mut new_pop: Populations = Vec::with_capacity(POPULATION_SIZE);

        if elitism_offset == 1 {
            new_pop.push(best_pop.clone());
        }

        for _ in elitism_offset..(population.len()) {
            let genome1 = select(&population, rng).genome();
            let genome2 = select(&population, rng).genome();
            let mut genome = crossover(&genome1, &genome2, rng);

            mutate(rng, &mut genome);

            new_pop.push(LanderWithGenomeAndResult::from_genome(genome, gen_pop));
        }

        new_pop
    }

    pub fn find_best_population(gen_pop: &FnGeneratePopulation, calc_fitness: &FnCalcFitness) -> LanderWithGenomeAndResult
    {
        let mut rng = thread_rng();

        let build_lander = |mut genome: &Genome|
            LanderWithGenomeAndResult::from_genome(build_genome(&mut genome.to_owned(), &mut rng), gen_pop);

        let compare_fitness = |a: &LanderWithGenomeAndResult, b: &LanderWithGenomeAndResult|
            (calc_fitness()(&b.result()) as i32).cmp(&(calc_fitness()(&a.result()) as i32));

        let mut populations: Populations = [[Gene::default(); GENOME_SIZE]; POPULATION_SIZE]
            .iter()
            .map(build_lander)
            .collect::<Populations>();

        populations.sort_by(compare_fitness);

        let populations: Populations = [0; GENERATIONS_COUNT]
            .iter()
            .fold(populations, |cur: Populations, _| {
                let mut next_pop = build_next_generation(&mut rng, &cur, gen_pop);

                next_pop.sort_by(compare_fitness);

                next_pop
            });

        populations.first().unwrap().to_owned()
    }

    pub fn build_genome(buf: &mut Genome, rng: &mut ThreadRng) -> Genome {
        for i in 0..buf.len() {
            buf[i] = Gene(rng.gen::<f32>());
        }

        *buf
    }

    pub fn select(population: &Populations, rng: &mut ThreadRng) -> LanderWithGenomeAndResult {
        for i in 0..population.len() {
            if rng.gen::<f32>() <= SELECTION_RATIO * (population.len() - i) as f32 / population.len() as f32 {
                return population.get(i).unwrap().clone();
            }
        }

        population.first().unwrap().clone()
    }

    pub fn crossover<'a, 'b>(genome1: &'a Genome, genome2: &'a Genome, rng: &'b mut ThreadRng) -> Genome {
        return *if rng.gen::<f32>() <= UNIFORM_RATE {
            genome1
        } else {
            genome2
        };
    }

    pub fn mutate(rng: &mut ThreadRng, genome: &mut Genome) {
        for i in 0..genome.len() {
            if rng.gen::<f32>() <= MUTATION_RATE {
                genome[i] = Gene(rng.gen::<f32>());
            }
        }
    }
}

pub mod lander {
    use crate::cartesian::*;
    use crate::physics::*;

    pub const SURFACE_N: usize = 29; // max 29 points to paint the mars surface

    pub const VERTICAL_SPEED_TOLERANCE: f32 = 40.0;

    pub const HORIZONTAL_SPEED_TOLERANCE: f32 = 20.0;

    pub type Commands = Vec<ControlCmd>; // cap: GENOME_SIZE

    pub type Trajectory = Vec<LanderState>; // cap: GENOME_SIZE

    pub type ControlCmdBuilder = Box<dyn Fn(usize) -> ControlCmd>;

    #[derive(Copy, Clone, PartialEq, Default, Debug)]
    pub struct ControlCmd {
        pub power: i32,
        pub angle: i32,
    }

    #[derive(Copy, Clone, PartialEq, Debug)]
    pub enum CrashResult {
        Outside,
        NoFuel,
        Collision,
    }

    #[derive(Copy, Clone, PartialEq, Debug)]
    pub enum FlyState {
        Flying,
        Crashed(CrashResult),
        Landed,
    }

    impl Default for FlyState {
        fn default() -> Self {
            FlyState::Flying
        }
    }

    #[derive(Clone)]
    pub struct Lander {
        ground: Line,
        init_lander_state: LanderState,
        commands: Commands,
        trajectory: Trajectory,
        fly_state: FlyState,
    }

    impl Lander {
        pub fn new(
            ground: Line,
            init_lander_state: LanderState,
            commands: Commands,
        ) -> Self {
            let (fly_state, trajectory) = Lander::compute_trajectory(
                &ground,
                &init_lander_state,
                &commands,
            );

            Lander {
                ground,
                init_lander_state,
                commands,
                trajectory,
                fly_state,
            }
        }

        pub fn land_zone(&self) -> Segment { self.ground.land_zone() }

        pub fn init_lander_state(&self) -> &LanderState { &self.init_lander_state }

        pub fn commands(&self) -> &Commands { &self.commands }

        pub fn trajectory(&self) -> &Trajectory { &self.trajectory }

        pub fn fly_state(&self) -> &FlyState { &self.fly_state }

        fn compute_trajectory(
            ground: &Line,
            init_lander_state: &LanderState,
            commands: &Commands,
        ) -> (FlyState, Trajectory) {
            let fly_state = FlyState::Flying;
            let mut trajectory: Trajectory = Vec::with_capacity(commands.len() + 1);
            let mut cur_state: LanderState = *init_lander_state;

            trajectory.push(cur_state.clone());

            for i in 1..commands.len() + 1 {
                cur_state = *trajectory.get(i - 1).unwrap();
                let next_state = cur_state.compute_next_state(
                    commands.get(i - 1).unwrap(),
                    1.0_f32.sec(),
                );

                trajectory.push(next_state);

                let fly_state = Lander::evaluate_outside(&next_state);
                if fly_state != FlyState::Flying {
                    return (fly_state, trajectory);
                }

                let fly_state = Lander::evaluate_no_fuel(&next_state);
                if fly_state != FlyState::Flying {
                    return (fly_state, trajectory);
                }

                let fly_state = Lander::evaluate_hit_the_ground(&ground, &next_state);
                if fly_state != FlyState::Flying {
                    return (fly_state, trajectory);
                }
            }

            (fly_state, trajectory)
        }

        fn evaluate_outside(next_state: &LanderState) -> FlyState {
            if next_state.position().x() as usize > SIZE_X - 1
                || next_state.position().x() < 0.0_f32
                || next_state.position().y() as usize > SIZE_Y - 1 {
                FlyState::Crashed(CrashResult::Outside)
            } else {
                FlyState::Flying
            }
        }

        fn evaluate_no_fuel(next_state: &LanderState) -> FlyState {
            if next_state.fuel <= 0 {
                FlyState::Crashed(CrashResult::NoFuel)
            } else {
                FlyState::Flying
            }
        }

        fn evaluate_hit_the_ground(ground: &Line, next_state: &LanderState) -> FlyState {
            return if *ground > next_state.position() {
                return if Lander::is_landed(&ground, &next_state) {
                    FlyState::Landed
                } else {
                    FlyState::Crashed(CrashResult::Collision)
                };
            } else {
                FlyState::Flying
            };
        }

        fn is_landed(ground: &Line, lander_state: &LanderState) -> bool {
            Self::is_centered(lander_state)
                && Self::is_within_horizontal_speed(lander_state)
                && Self::is_within_vertical_speed(lander_state)
                && Self::is_on_land_zone(ground, lander_state)
        }

        fn is_centered(lander_state: &LanderState) -> bool { lander_state.angle == 0 }

        fn is_within_horizontal_speed(lander_state: &LanderState) -> bool {
            lander_state.speed().direction.dy() > -40.0_f32
        }

        fn is_within_vertical_speed(lander_state: &LanderState) -> bool {
            lander_state.speed().direction.dx().abs() <= 20.0_f32
        }

        fn is_on_land_zone(ground: &Line, lander_state: &LanderState) -> bool {
            ground.is_horizontal_at_x(lander_state.position().x() as i32)
        }
    }

    #[derive(Copy, Clone, PartialEq, Default, Debug)]
    pub struct LanderState {
        pub fuel: i32,
        pub power: i32,
        pub angle: i32,
        pub particle: Particle,
    }

    impl LanderState {
        pub fn compute_next_state(&self, cmd: &ControlCmd, time: Time) -> LanderState {
            let angle = cmd.angle;
            let power = cmd.power;
            let vector = (Vector2((0.0, 1.0)) * power as f32).rotate(Angle { rad: (angle as f32).to_radians() });
            let fuel = self.fuel - power;
            let thrust = Acceleration(vector);
            let acceleration = GRAVITY + thrust;
            let particle = self.particle.accelerate(acceleration, time);

            LanderState {
                fuel,
                power,
                angle,
                particle,
            }
        }

        pub fn position(&self) -> Position { self.particle.position }

        pub fn speed(&self) -> Speed {
            self.particle.speed
        }
    }
}

pub mod game {
    use crate::lander::{SURFACE_N, LanderState};
    use crate::cartesian::{Line, LineValue, Point2, Vector2};
    use crate::physics::{Particle, Speed};
    use crate::env::is_prod;

    use std::{io, mem};
    use std::sync::{Arc, Mutex, Once};

    macro_rules! parse_input {
    ($x:expr, $t:ident) => ($x.trim().parse::<$t>().unwrap())
}

    type GameLevel = [Option<&'static str>; SURFACE_N + 2];

    const RUN_LEVEL: usize = 0;
    const GAME_LEVELS: [GameLevel; 2] = [
        [
            Some("6"),
            Some("0 100"),
            Some("1000 500"),
            Some("1500 100"),
            Some("3000 100"),
            Some("5000 1500"),
            Some("6999 1000"),
            Some("2500 2500 0 0 500 0 0"),
            None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None
        ],
        [
            Some("7"),
            Some("0 100"),
            Some("1000 500"),
            Some("1500 1500"),
            Some("3000 1000"),
            Some("4000 150"),
            Some("5500 150"),
            Some("6999 800"),
            Some("2500 2700 0 0 550 0 0"),
            None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None,
        ]
    ];

    #[derive(Clone)]
    pub struct GameStateSingleton {
        pub inner: Arc<Mutex<GameState>>,
    }

    pub fn game_state_singleton() -> &'static GameStateSingleton {
        static mut SINGLETON: *const GameStateSingleton = 0 as *const GameStateSingleton;
        static ONCE: Once = Once::new();

        unsafe {
            ONCE.call_once(|| {
                let singleton = GameStateSingleton {
                    inner: Arc::new(Mutex::new(GameState::build())),
                };

                SINGLETON = mem::transmute(Box::new(singleton));
            });

            &*SINGLETON
        }
    }

    #[derive(Clone, Debug)]
    pub struct GameState {
        pub ground: Line,
        pub init_lander_state: LanderState,
    }

    impl GameState {
        const fn new(
            ground: Line,
            init_lander_state: LanderState,
        ) -> Self {
            Self { ground, init_lander_state }
        }

        //noinspection RsLiveness
        pub fn build() -> Self {
            let mut input_line = String::new();
            let mut line_value: LineValue = Vec::with_capacity(SURFACE_N);

            if !is_prod() {
                input_line = GAME_LEVELS
                    .get(RUN_LEVEL)
                    .unwrap()
                    .get(0)
                    .unwrap()
                    .unwrap()
                    .to_string();
            } else {
                io::stdin().read_line(&mut input_line).unwrap();
            };

            let surface_n = parse_input!(input_line, usize);
            for i in 0..surface_n {
                let mut input_line = String::new();

                if !is_prod() {
                    input_line = GAME_LEVELS
                        .get(RUN_LEVEL)
                        .unwrap()
                        .get(i + 1)
                        .unwrap()
                        .unwrap()
                        .to_string();
                } else {
                    io::stdin().read_line(&mut input_line).unwrap();
                }

                let inputs = input_line.split(" ").collect::<Vec<_>>();
                let land_x = parse_input!(inputs.get(0).unwrap(), f32); // X coordinate of a surface point. (0 to 6999)
                let land_y = parse_input!(inputs.get(1).unwrap(), f32); // Y coordinate of a surface point. By linking all the points together in a sequential fashion, you form the surface of Mars.

                line_value.push(Point2::new(land_x, land_y));
            }

            let mut input_line = String::new();
            if !is_prod() {
                input_line = GAME_LEVELS
                    .get(RUN_LEVEL)
                    .unwrap()
                    .get(surface_n + 1)
                    .unwrap()
                    .unwrap()
                    .to_string();
            } else {
                io::stdin().read_line(&mut input_line).unwrap();
            }

            let inputs = input_line.split(" ").collect::<Vec<_>>();
            let x = parse_input!(inputs.get(0).unwrap(), f32);
            let y = parse_input!(inputs.get(1).unwrap(), f32);
            let dx = parse_input!(inputs.get(2).unwrap(), f32); // the horizontal speed (in m/s), can be negative.
            let dy = parse_input!(inputs.get(3).unwrap(), f32); // the vertical speed (in m/s), can be negative.
            let fuel = parse_input!(inputs.get(4).unwrap(), i32); // the quantity of remaining fuel in liters.
            let angle = parse_input!(inputs.get(5).unwrap(), i32); // the rotation angle in degrees (-90 to 90).
            let power = parse_input!(inputs.get(6).unwrap(), i32); // the thrust power (0 to 4).

            let init_lander_state = LanderState {
                fuel,
                power,
                angle,
                particle: Particle {
                    position: Point2::new(x, y),
                    speed: Speed {
                        direction: Vector2((dx, dy))
                    },
                },
            };

            let ground = Line::new(line_value);

            if !is_prod() {
                eprintln!(
                    "[{}]",
                    ground
                        .points()
                        .iter()
                        .fold(
                            String::new(),
                            |acc, num|
                                acc + format!("x: {}, y: {}", num.x(), num.y()).as_ref() + ", ",
                        )
                );
            }

            Self::new(ground, init_lander_state)
        }
    }
}

use std::thread;
use rand::prelude::ThreadRng;
use {genetic::*, lander::*, game::*};
use std::time::Instant;

const COMMAND_SIZE: usize = 9;
const AVAILABLE_COMMANDS: [(i32, i32); COMMAND_SIZE] = [(-15, -1), (-15, 0), (-15, 1), (0, -1), (0, 0), (0, 1), (15, -1), (15, 0), (15, 1)];

fn main() {
    let timer = Instant::now();
    let best_population: LanderWithGenomeAndResult = find_best_population(
        &|| Box::new(|genome: &Genome| lander_from_genome(&genome)),
        &|| Box::new(|lander: &Lander| calculate_lander_fitness(&lander)),
    ) as LanderWithGenomeAndResult;
    let duration = timer.elapsed();

    let best_population_result = best_population.result();
    let trajectory = best_population_result.trajectory();

    eprintln!("Time elapsed: {} ms", duration.as_millis());
    eprintln!("Lander state: {:?}", best_population_result.fly_state());
    eprintln!("Lander fuel: {:?}", trajectory.last().unwrap().fuel);
    eprintln!("Lander pos: ({:?})", trajectory.last().unwrap().particle.position);
    eprintln!("Lander speed: ({:?})", trajectory.last().unwrap().speed());
    eprintln!("Fitness: {}", calculate_lander_fitness(&best_population.result()));
    eprintln!("Trajectory Len: {}", trajectory.len());

    // eprintln!(
    //     "Landing phase starting\nangle {} speed {} X {} Y {}",
    //     trajectory.get(0).unwrap().angle as i32,
    //     trajectory.get(0).unwrap().power as i32,
    //     trajectory.get(0).unwrap().position().x() as i32,
    //     trajectory.get(0).unwrap().position().y() as i32,
    // );

    // eprintln!(
    //     "X={}m, Y={}m, HSpeed={}m/s VSpeed={}m/s",
    //     trajectory.last().unwrap().position().x() as i32,
    //     trajectory.last().unwrap().position().y() as i32,
    //     trajectory.last().unwrap().speed().direction.dx() as i32,
    //     trajectory.last().unwrap().speed().direction.dy() as i32,
    // );

    // eprintln!(
    //     "Fuel={}l, Angle={}˚, Power={}m/s*s",
    //     trajectory.last().unwrap().fuel as i32,
    //     trajectory.last().unwrap().angle as i32,
    //     trajectory.last().unwrap().power as i32,
    // );

    let trajectory_len = trajectory.len();
    let mut counter = -1;

    eprintln!(
        "{}",
        trajectory
            .iter()
            .fold(
                String::new(),
                |acc, state| {
                    counter += 1;

                    acc + format!(
                        "X={}m, Y={}m, HSpeed={}m/s VSpeed={}m/s",
                        state.position().x(),
                        state.position().y(),
                        state.speed().direction.dx(),
                        state.speed().direction.dy(),
                    ).as_ref() + "\n" + format!(
                        "Fuel={}l, Angle={}˚, Power={}m/s*s",
                        state.fuel,
                        state.angle,
                        state.power,
                    ).as_ref() + "\t\t" + format!("{}/{}", counter, trajectory_len).as_ref() + "\n\n"
                },
            )
    );

    for i in 0..trajectory.len() - 1 {
        let command = best_population_result.commands().get(i).unwrap();
        println!(
            "{} {}",
            command.angle as i32,
            command.power as i32
        );
    }
}

fn lander_from_genome(genome: &Genome) -> Lander {
    let mut commands = Vec::with_capacity(genome.len());
    let mut prev_power = 0_i32;
    let mut prev_angle = 0_i32;

    for i in 0..commands.capacity() {
        let tmp_command = AVAILABLE_COMMANDS.get(((genome.get(i).unwrap().as_int(COMMAND_SIZE as i32 - 1) + 1) % COMMAND_SIZE as i32) as usize).unwrap();
        let tmp_angle: i32 = prev_angle + tmp_command.0;
        let tmp_power: i32 = prev_power + tmp_command.1;

        let power = tmp_power.min(4).max(0);
        let angle = tmp_angle.min(90).max(-90);

        commands.push(ControlCmd { power, angle });

        prev_angle = tmp_angle;
        prev_power = tmp_power;
    }

    let singleton = game_state_singleton();
    let game_state = singleton.inner.lock().unwrap();

    Lander::new(game_state.ground.to_owned(), game_state.init_lander_state.to_owned(), commands)
}

fn calculate_lander_fitness(lander: &Lander) -> f32 {
    let last_trajectory = *lander.trajectory().last().unwrap();

    let last_y_pos = last_trajectory.position().y();
    let last_x_pos = last_trajectory.position().x();
    let last_x_neg = -1.0 * last_x_pos;

    let land_zone = lander.land_zone();
    let land_zone_y = land_zone.0.y();
    let land_x_left = land_zone.0.x();
    let land_x_right = land_zone.1.x();

    let speed_x = last_trajectory.speed().direction.dx();
    let speed_y = last_trajectory.speed().direction.dy();

    let distance_x = if last_x_pos < land_x_left {
        last_x_pos - land_x_left
    } else if last_x_pos > land_x_right {
        last_x_neg + land_x_right
    } else {
        0.0
    };
    let distance_y = (land_zone_y - last_y_pos).min(0.0);

    let distance_fitness = distance_y + distance_x;
    let is_on_land_zone = distance_y == 0.0 && distance_x == 0.0;

    let vertical_speed_fitness = (-1.0 * speed_x.abs()).min(-1.0 * VERTICAL_SPEED_TOLERANCE) + VERTICAL_SPEED_TOLERANCE;
    let horizontal_speed_fitness = (-1.0 * speed_y.abs()).min(-1.0 * HORIZONTAL_SPEED_TOLERANCE) + HORIZONTAL_SPEED_TOLERANCE;

    let angle_fitness = -1.0 * last_trajectory.angle.abs() as f32;

    return match lander.fly_state() {
        FlyState::Landed => {
            lander.trajectory().iter().last().unwrap().fuel as f32
        }
        FlyState::Flying => {
            vertical_speed_fitness + horizontal_speed_fitness + distance_fitness + angle_fitness
        }
        FlyState::Crashed(_) => {
            if is_on_land_zone {
                vertical_speed_fitness + horizontal_speed_fitness + angle_fitness
            } else {
                vertical_speed_fitness + horizontal_speed_fitness + distance_fitness + angle_fitness
            }
        }
    };
}
