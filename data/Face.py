from dataclasses import dataclass
from PIL import Image


@dataclass(frozen=True)
class Face:
    image: Image.Image
    age: int
    gender: int  # 0 = male, 1 = female
    race: int  # 0 = white, 1 = black, 2 = asian, 3 = indian, 4 = others
    filename: str
    age_bin: int = None
    requests_unlearning: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'age_bin',
                           next(x[0] for x in enumerate([2, 9, 20, 27, 45, 65, 120]) if x[1] >= self.age))
