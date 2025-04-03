from .basic import basic_mutation
from .multi_gen import multi_gen_mutation

mutation_strategies = {
    "basic": basic_mutation,
    "multi_gen": multi_gen_mutation
}