import math

def log_decay(generation, T0=100):
    """
    Logarithmic decay temperature function.
    
    Parameters:
    - generation: current generation number (starts from 0).
    - T0: initial temperature (default = 100).

    Returns:
    - Temperature value at current generation.
    """
    return T0 / math.log(generation + 2)  # log(2) = 0.69, avoids division by 0