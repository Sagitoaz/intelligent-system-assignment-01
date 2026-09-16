"""Manual Adam optimizer for NumPy parameters."""

import numpy as np


class Adam:
    def __init__(self, parameters_and_grads, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.items = list(parameters_and_grads)
        self.learning_rate = learning_rate
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.m = [np.zeros_like(parameter) for parameter, _ in self.items]
        self.v = [np.zeros_like(parameter) for parameter, _ in self.items]
        self.step_number = 0

    def step(self):
        self.step_number += 1
        for index, (parameter, gradient) in enumerate(self.items):
            self.m[index] = self.beta1 * self.m[index] + (1 - self.beta1) * gradient
            self.v[index] = self.beta2 * self.v[index] + (1 - self.beta2) * gradient * gradient
            m_hat = self.m[index] / (1 - self.beta1 ** self.step_number)
            v_hat = self.v[index] / (1 - self.beta2 ** self.step_number)
            parameter -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
