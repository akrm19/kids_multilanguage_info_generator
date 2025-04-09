import operator

from dataclasses import dataclass, field
from typing import List, Annotated, Dict

@dataclass(kw_only=True)
class GeneratorInput:
    topic: str = field(default="")
    target_languages: List[str] = Annotated[dict, operator.add]

@dataclass(kw_only=True)
class GeneratorOutput:
    topic: str = field(default="")
    #summary_outputs: Dict[str, str]# = field(default={})  #Annotated[dict, operator.add]
    existing_summary: str = field(default="")
    output_languages_summaries: Dict[str, str] # = field(default={}) #Annotated[dict, operator.add]

@dataclass(kw_only=True)
class GeneratorState:
    topic: str = field(default="")
    is_valid: bool = field(default=False)
    target_languages: List[str] = Annotated[dict, operator.add]
    existing_summary: str = field(default="")
    output_languages_summaries: Dict[str, str] = Annotated[dict, operator.add]