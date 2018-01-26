from network import DQNetwork
import random
import numpy as np


class DQAgent:
    def __init__(self,
                 actions,
                 batch_size=1024,
                 alpha=0.01,
                 gamma=0.9,
                 dropout_prob=0.1,
                 epsilon=1,
                 epsilon_rate=0.99,
                 network_input_shape=(2, 84, 84),
                 load_path='',
                 logger=None):

        self.actions = actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_rate = epsilon_rate
        self.min_epsilon = 0.3

        self.experiences = []
        self.training_count = 0

        self.DQN = DQNetwork(
            self.actions,
            network_input_shape,
            alpha=alpha,
            gamma=self.gamma,
            dropout_prob=dropout_prob,
            load_path=load_path,
            logger=logger
        )

        if logger is not None:
            logger.log({
                'Learning rate': alpha,
                'Discount factor': self.gamma,
                'Starting epsilon': self.epsilon,
                'Epsilon decrease rate': self.epsilon_rate,
                'Batch size': self.batch_size
            })

    def get_action(self, state, testing=False):
        q_values = self.DQN.predict(state)
        if (random.random() < self.epsilon) and not testing:
            return random.randint(0, self.actions - 1)
        else:
            return np.argmax(q_values)

    def add_experience(self, source, action, reward, dest, final):
        self.experiences.append({'source': source,
                                 'action': action,
                                 'reward': reward,
                                 'dest': dest,
                                 'final': final})

    def sample_batch(self):
        out = [self.experiences.pop(random.randrange(0, len(self.experiences)))
               for _ in range(self.batch_size)]
        return np.asarray(out)

    def must_train(self):
        return len(self.experiences) >= self.batch_size

    def train(self, update_epsilon=True):
        self.training_count += 1
        print 'Training sessio #', self.training_count, ' - epsilon:', self.epsilon
        batch = self.sample_batch()
        self.DQN.train(batch)
        if update_epsilon:
            self.epsilon = self.epsilon * self.epsilon_rate if self.epsilon > self.min_epsilon else self.min_epsilon

    def quit(self):
        self.DQN.save()
