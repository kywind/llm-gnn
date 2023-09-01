from types import Optional, List, Union, Any

class Particle:

    x: float
    y: float
    z: float
    attr: Any

    def __init__():
        pass


class Relation:

    attr: Any
    threshold: float
    max_degree: int

    def __init__():
        pass

class GNN:
    """
    Constructs a GNN model using perticles and relations."""

    particles: List[Particle]
    relations: List[Relation]
    attr: Any

    def __init__():
        pass

    def forward():
        pass