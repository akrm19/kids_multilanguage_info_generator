from dataclasses import dataclass, field

@dataclass(kw_only=True)
class Configuration:
    llm_model: str = field(default="llama3.2")
    temperature: float = field(default=0.0)
    target_languages: dict = field(default_factory=dict)