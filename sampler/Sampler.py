from statistics import harmonic_mean
from scipy.interpolate import CubicSpline
import random


class Sampler:
    def __init__(self, strategy=None):
        if strategy is None:
            strategy = ['age', 'gender', 'race']
        if not set(strategy).issubset({'age', 'gender', 'race'}):
            raise ValueError('Strategy can only be age, gender and/or race or empty list (=random)')

        # approximated from Special Eurobarometer 487a
        # 0 = 0-2, 1 = 3-9, 2 = 10-20, 3 = 21-27, 4 = 28-45, 5 = 46-65, 6 = 66-120
        x = [0, 10, 20, 25, 33, 40, 48, 53, 62, 80, 90, 100, 120]
        y = [0.3, 0.32, 0.36, 0.4, 0.45, 0.46, 0.43, 0.36, 0.29, 0.23, 0.21, 0.2, 0.18]
        cs = CubicSpline(x, y)
        keys = list(range(0, 121))
        values = [float(cs(x)) for x in keys]
        age_probabilities = dict(zip(keys, values))  # {0: 0.3, 1: 0.302, ..., 120: 0.18}
        # directly taken from Special Eurobarometer 487a
        # 0 = male, 1 = female
        gender_probabilities = {0: 0.38, 1: 0.34}
        # roughly estimated with SAT score distribution over races
        # 0 = white, 1 = black, 2 = asian, 3 = indian, 4 = others
        race_probabilities = {0: 0.4, 1: 0.3, 2: 0.43, 3: 0.43, 4: 0.32}

        self.probabilities = {'age': age_probabilities,
                              'gender': gender_probabilities,
                              'race': race_probabilities}

        self.strategy = strategy

    def get_gdpr_knowledge(self, face):
        probabilities = [self.probabilities[feature][face.__dict__[feature]] for feature in self.strategy]
        return harmonic_mean(probabilities) if probabilities else 0.36

    def changed_privacy_settings(self, face):
        knowledge = self.get_gdpr_knowledge(face)
        # around 23.7% of users have ever changed their privacy settings (0.36 * 0.66 = 0.237)
        return random.random() < (knowledge * 0.66)

    def sample_unlearning_request(self, face):
        knowledge = self.get_gdpr_knowledge(face)
        # 0.1 percent of people who know the GDPR will make a request
        # for 23700 people, this means an average of 8.532 unlearning requests (23700 * 0.36 * 0.001 = 8.532)
        return random.random() < (knowledge * 0.001)
