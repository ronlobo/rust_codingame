use crate::cartesian::*;
use crate::physics::*;

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
