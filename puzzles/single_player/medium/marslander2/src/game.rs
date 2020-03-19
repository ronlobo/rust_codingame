use std::{io, mem};
use std::sync::{Arc, Mutex, Once};
use marslander::cartesian::{Line, LineValue, Point2, Vector2};
use marslander::lander::LanderState;
use marslander::physics::{Particle, Speed};
use crate::env::is_prod;

macro_rules! parse_input {
    ($x:expr, $t:ident) => ($x.trim().parse::<$t>().unwrap())
}

type GameLevel = [Option<&'static str>; SURFACE_N + 2];


pub const SURFACE_N: usize = 29;
// max 29 points to paint the mars surface
pub const VERTICAL_SPEED_TOLERANCE: f32 = 40.0;
pub const HORIZONTAL_SPEED_TOLERANCE: f32 = 20.0;

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
    // Since this could be used in many threads, we need to protect concurrent access
    pub inner: Arc<Mutex<GameState>>,
}

pub fn game_state_singleton() -> &'static GameStateSingleton {
    // Initialize it to a null value
    static mut SINGLETON: *const GameStateSingleton = 0 as *const GameStateSingleton;
    static ONCE: Once = Once::new();

    unsafe {
        ONCE.call_once(|| {
            // Make it
            let singleton = GameStateSingleton {
                inner: Arc::new(Mutex::new(GameState::build())),
            };

            // Put it in the heap so it can outlive this call
            SINGLETON = mem::transmute(Box::new(singleton));
        });

        // Hand out a reference to our static singleton
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

        // init starting mars lander state
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
