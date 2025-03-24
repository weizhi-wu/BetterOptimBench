from optimizer_toolkit import create_card as card
from toolkit_functions import *

__all_test_functions__ = [
    Rastrigin,
    Ackley,
    Sphere,
    Rosenbrock,
    Beale,
    GoldsteinPrice,
    Booth,
    BukinN6,
    Matyas,
    LeviN13,
    Griewank,
    Himmelblau,
    ThreeHumpCamel,
    Easom,
    CrossInTray,
    Eggholder,
    HolderTable,
    Mccormick,
    SchafferN2,
    SchafferN4,
    StyblinskiTang,
    Shekel
]

__all_strict_2D_functions__ = [
    Ackley,
    Beale,
    GoldsteinPrice,
    Booth,
    BukinN6,
    Matyas,
    LeviN13,
    Himmelblau,
    ThreeHumpCamel,
    Easom,
    CrossInTray,
    Eggholder,
    Mccormick,
    SchafferN2,
    SchafferN4

]

__all_scalable_functions__ = [
    Rastrigin,
    Sphere,
    Rosenbrock,
    Griewank,
    StyblinskiTang
]

Standard2D = [
    card(Rastrigin),
    card(Ackley),
    card(Rosenbrock),
    card(Griewank),
    card(Himmelblau),
    card(ThreeHumpCamel),
]

Standard100 = [
    card(Rastrigin, {'num_params': 100}),
    card(Sphere, {'num_params': 100}),
    card(Rosenbrock, {'num_params': 100}),
    card(Griewank, {'num_params': 100}),
    card(StyblinskiTang, {'num_params': 100}),
]

Standard1K = [
    card(Rastrigin, {'num_params': 1000}),
    card(Sphere, {'num_params': 1000}),
    card(Rosenbrock, {'num_params': 1000}),
    card(Griewank, {'num_params': 1000}),
    card(StyblinskiTang, {'num_params': 1000}),
]

Standard10K = [
    card(Rastrigin, {'num_params': 10000}),
    card(Sphere, {'num_params': 10000}),
    card(Rosenbrock, {'num_params': 10000}),
    card(Griewank, {'num_params': 10000}),
    card(StyblinskiTang, {'num_params': 10000}),
]

Standard100K = [
    card(Rastrigin, {'num_params': 100000}),
    card(Sphere, {'num_params': 100000}),
    card(Rosenbrock, {'num_params': 100000}),
    card(Griewank, {'num_params': 100000}),
    card(StyblinskiTang, {'num_params': 100000}),
]

Standard1M = [
    card(Rastrigin, {'num_params': 1000000}),
    card(Sphere, {'num_params': 1000000}),
    card(Rosenbrock, {'num_params': 1000000}),
    card(Griewank, {'num_params': 1000000}),
    card(StyblinskiTang, {'num_params': 1000000}),
]
