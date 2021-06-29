import math

class FeatureExtractor:
    def __init__(self):
        self.features_n = 2
        self.state = None

    def get_features(self, state):
        self.state = state
        # this is a numeric vector with the features corresponding to the current state
        # normalized preferably.
        features = [
            state['x'] + state['y'],  # Replace with actual feature calculation logic
            state['x'] * state['y']   # Replace with actual feature calculation logic
        ]
        return features
