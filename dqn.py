from collections import deque
import random
import sys
from featureExtractor import FeatureExtractor

sys.path.append('./scratchNeuralNetwork')
from scratchNeuralNetwork.neuralNetwork import NeuralNetwork

class DQN:
    def __init__(self, config, n_actions, alpha, discount, replay_mem_size, batch_size, ott_interval, **kwargs):
        self.features = FeatureExtractor()
        self.n_actions = n_actions
        self.alpha = alpha
        self.discount = discount
        self.replay_mem = deque([], replay_mem_size)
        self.batch_size = batch_size
        self.ott_interval = ott_interval
        self.counter = 1

        self.nn_params = {
            # Add Input Layer from the state length
            # Add Output layer from n. of actions
            'config': [self.features.features_n] + config + [self.n_actions],
            'learning_rate': self.alpha,
            **kwargs,
        }
        self.target_nn = NeuralNetwork(**self.nn_params)
        self.online_nn = NeuralNetwork(**self.nn_params)

    def predict(self, state):
        return self.online_nn.predict(self.features.get_features(state))

    def update(self, state, action, reward, next_state):
        if len(self.replay_mem) == self.replay_mem.maxlen:
            self.batch_train(random.sample(self.replay_mem, self.batch_size))
            for _ in range(self.batch_size):
                self.replay_mem.popleft()
            if self.counter % self.ott_interval == 0:
                self.online_to_target()
        self.replay_mem.append((state, action, reward, next_state))
        self.counter += 1

    def batch_train(self, batch):
        ft_state, ft_action, ft_reward, ft_next_state = batch[0]
        total_err = [0.0 for _ in range(self.n_actions)]

        for state, action, reward, next_state in batch:
            q_values = self.predict(state)
            target = self.target_nn.predict(self.features.get_features(next_state))
            total_err = [e + (t - q) for e, q, t in zip(total_err, q_values, target)]

        q_values = self.predict(ft_state)
        targets = [q + e for q, e in zip(q_values, total_err)]
        self.online_nn.train(self.features.get_features(ft_state), targets)

    def online_to_target(self):
        # Basically copy with same weights
        self.target_nn = NeuralNetwork(**{
            'model': self.online_nn.get_model(),
            **self.nn_params
        })

    def get_model(self):
        return self.target_nn.get_model()

