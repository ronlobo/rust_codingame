use crate::lander::*;
use rand::{Rng, thread_rng};
use rand::prelude::ThreadRng;
use std::borrow::Borrow;

const ELITISM: bool = true;
const GENERATIONS_COUNT: usize = 10;
const POPULATION_SIZE: usize = 128;
const GENOME_SIZE: usize = 256;

const UNIFORM_RATE: f32 = 0.7;
const MUTATION_RATE: f32 = 0.05;
const SELECTION_RATIO: f32 = 0.7;

pub type Genome = [Gene; GENOME_SIZE];

pub type Population = Vec<LanderWithGenomeAndResult>; // cap: POPULATION_SIZE

pub type Generation = Vec<Population>;  // cap: GENERATIONS_COUNT

pub type FnGeneratePopulation = dyn Fn(&Genome) -> Lander;

pub type FnCalcFitness = dyn Fn(&Lander) -> f32;

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
            result: gen_lander(&genome),
        }
    }

    fn genome(&self) -> Genome { self.genome }

    fn result(&self) -> &Self::Result { self.result.borrow() }
}

fn build_next_generation(
    rng: &mut ThreadRng,
    population: &Population,
    gen_pop: &FnGeneratePopulation,
) -> Population {
    let elitism_offset = if ELITISM { 1 } else { 0 };
    let best_pop: &LanderWithGenomeAndResult = population.get(0).unwrap();
    let mut new_pop: Population = Vec::with_capacity(POPULATION_SIZE);

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

    let build_lander = |genome: &Genome|
        LanderWithGenomeAndResult::from_genome(build_genome(&mut genome.to_owned(), &mut rng), gen_pop);

    let compare_fitness = |a: &LanderWithGenomeAndResult, b: &LanderWithGenomeAndResult|
        (calc_fitness(&b.result()) as i32).cmp(&(calc_fitness(&a.result()) as i32));

    let mut population: Population = [[Gene::default(); GENOME_SIZE]; POPULATION_SIZE]
        .iter()
        .map(build_lander)
        .collect::<Population>();

    population.sort_by(compare_fitness);

    let population: Population = [0; GENERATIONS_COUNT]
        .iter()
        .fold(population, |cur: Population, _| {
            let mut next_pop = build_next_generation(&mut rng, &cur, gen_pop);

            next_pop.sort_by(compare_fitness);

            next_pop
        });

    population.first().unwrap().to_owned()
}

pub fn build_genome(buf: &mut Genome, rng: &mut ThreadRng) -> Genome {
    for i in 0..buf.len() {
        buf[i] = Gene(rng.gen::<f32>());
    }

    *buf
}

pub fn select(population: &Population, rng: &mut ThreadRng) -> LanderWithGenomeAndResult {
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
