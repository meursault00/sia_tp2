from .tournament import tournament_selection
from .roulette import roulette_selection
from .ranking import ranking_selection

# Mapeo por string => función
selection_strategies = {
    "tournament": tournament_selection,
    "roulette": roulette_selection,
    "ranking": ranking_selection
}