from dataclasses import dataclass
from PIL.Image import Image
from sampler.Sampler import Sampler


@dataclass(frozen=True)
class Face:
    image: Image
    age: int
    gender: int  # 0 = male, 1 = female
    race: int  # 0 = white, 1 = black, 2 = asian, 3 = indian, 4 = others
    filename: str
    sampler: Sampler
    age_bin: int = None
    gdpr_knowledge: float = None
    changed_privacy_settings: bool = None
    requests_unlearning: bool = None

    def __post_init__(self):
        object.__setattr__(self, 'age_bin',
                           next(x[0] for x in enumerate([2, 9, 20, 27, 45, 65, 120]) if x[1] >= self.age))
        object.__setattr__(self, 'gdpr_knowledge',
                           self.sampler.get_gdpr_knowledge(self))
        object.__setattr__(self, 'changed_privacy_settings',
                           self.sampler.changed_privacy_settings(self))
        object.__setattr__(self, 'requests_unlearning',
                           self.sampler.sample_unlearning_request(self))
