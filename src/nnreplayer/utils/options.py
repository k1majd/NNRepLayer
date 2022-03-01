from dataclasses import dataclass
from typing import Any

@dataclass
class Options:
    gdp_formulation: Any
    solver_factory: Any
    solver_language: Any
    model_output_type: Any
    weightSlack: Any
    optimizer_time_limit: Any
    optimizer_mip_gap: Any