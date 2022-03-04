from dataclasses import dataclass, field
from typing import Any


@dataclass
class Options:
    """_summary_"""

    gdp_formulation: str = "gdp.bigm"
    solver_factory: str = "gurobi"
    solver_language: str = "python"
    model_output_type: str = "keras"
    max_weight_bound: float = 1
    optimizer_options: dict = field(default_factory=lambda: {"timelimit": 200})
