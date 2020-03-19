#[derive(Copy, Clone, Debug, PartialOrd, PartialEq)]
pub enum EnvOption {
    PROD,
    DEV,
}

pub const ENV: EnvOption = EnvOption::DEV;

pub fn is_prod() -> bool {
    match ENV {
        EnvOption::PROD => true,
        EnvOption::DEV => false
    }
}