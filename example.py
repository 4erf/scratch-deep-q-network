from dqn import DQN
import random

dqn = DQN(**{
    'alpha': 0.1,
    'config': [
        2,
        2
    ],
    'model': None,
    'activation_fn': 'relu',
    'dropout': 0,
    'n_actions': 4,
    'discount': 0.9,
    'replay_mem_size': 10000,
    'batch_size': 32,
    'ott_interval': 100
})

# this has to be replaced by actual actions allowed
dummy_actions = {'north': 0, 'east': 1, 'south': 2, 'west': 3}
inv_actions = {v: k for k, v in dummy_actions.items()}

# this has to be replaced by actual game state
dummy_state = {
    'x': 0,
    'y': 1
}

# Obtain best actions for current state
q_values = dqn.predict(dummy_state)
best_action = max(q_values)
best_actions = [inv_actions[ix] for ix, action in enumerate(q_values) if action == best_action]
action_to_execute = random.choice(best_actions)

print(best_actions)

# This is the next state
dummy_state_new = {
    'x': 0.5,
    'y': 0.75
}

# Train with experience tuple
dqn.update(dummy_state, dummy_actions[action_to_execute], 1, dummy_state_new)

# Obtain best actions for current state
q_values = dqn.predict(dummy_state_new)
best_action = max(q_values)
best_actions = [inv_actions[ix] for ix, action in enumerate(q_values) if action == best_action]
action_to_execute = random.choice(best_actions)

# Can be different than the last
print(best_actions)

print(dqn.get_model())