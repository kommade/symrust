pub trait Antiderivative {
    fn get_antiderivative(&self) -> Self;
}

trait AntiderivativeInternal {
    fn get_antiderivative(&self) -> Self;
    fn power_rule(&self) -> Self;
    fn exp_rule(&self) -> Self;
    fn basic_function(&self) -> Self;
    fn u_substitution(&self) -> Self;
    fn integration_by_parts(&self) -> Self;
    fn partial_fraction(&self) -> Self;
}
