extern crate marslander2;

use marslander2::{game::*};
use std::time::Instant;
use marslander::genetic::{LanderWithGenomeAndResult, find_best_population, Genome, GenomeAndResult};
use marslander::lander::{Lander, FlyState, ControlCmd};

// A list of possible lander commands to reduce the search space
const COMMAND_SIZE: usize = 9;
const AVAILABLE_COMMANDS: [(i32, i32); COMMAND_SIZE] = [
    (-15, -1), (-15, 0), (-15, 1),
    (0, -1), (0, 0), (0, 1),
    (15, -1), (15, 0), (15, 1)
];

fn main() {
    let timer = Instant::now();
    let best_population: LanderWithGenomeAndResult = find_best_population(
        &|genome: &Genome| lander_from_genome(&genome),
        &|lander: &Lander| calculate_lander_fitness(&lander),
    );
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

    eprintln!(
        "angle {} speed {} X {} Y {}",
        trajectory.get(0).unwrap().angle as i32,
        trajectory.get(0).unwrap().power as i32,
        trajectory.get(0).unwrap().position().x() as i32,
        trajectory.get(0).unwrap().position().y() as i32,
    );

    eprintln!(
        "X={}m, Y={}m, HSpeed={}m/s VSpeed={}m/s",
        trajectory.last().unwrap().position().x() as i32,
        trajectory.last().unwrap().position().y() as i32,
        trajectory.last().unwrap().speed().direction.dx() as i32,
        trajectory.last().unwrap().speed().direction.dy() as i32,
    );

    eprintln!(
        "Fuel={}l, Angle={}˚, Power={}m/s*s",
        trajectory.last().unwrap().fuel as i32,
        trajectory.last().unwrap().angle as i32,
        trajectory.last().unwrap().power as i32,
    );

    let trajectory_len = trajectory.len();
    let mut counter = 0;

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
                        state.position().x() as i32,
                        state.position().y() as i32,
                        state.speed().direction.dx() as i32,
                        state.speed().direction.dy() as i32,
                    ).as_ref() + "\n" + format!(
                        "Fuel={}l, Angle={}˚, Power={}m/s*s",
                        state.fuel as i32,
                        state.angle as i32,
                        state.power as i32,
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
