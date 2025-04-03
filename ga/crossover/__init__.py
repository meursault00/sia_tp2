from .one_point import one_point_crossover
from .two_point import two_point_crossover
from .uniform import uniform_crossover

crossover_strategies = {
    "one_point": one_point_crossover,
    "two_point": two_point_crossover,
    "uniform": uniform_crossover
}