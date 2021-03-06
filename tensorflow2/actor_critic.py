import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import RandomUniform

class Critic:
    def __init__(self, obs_dim, action_dim, learning_rate=0.001):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.model = self.make_network()
        self.optimizer = keras.optimizers.Adam(learning_rate)
        # self.model.compile(loss="mse", optimizer=self.optimizer)

    def make_network(self):
        obs_input = keras.Input(shape=(self.obs_dim,), dtype="float32", name="obs")
        action_input = keras.Input(shape=(self.action_dim,), dtype="float32", name="action")

        # layer 0 - with obs input
        w_range = 1 / np.sqrt(self.obs_dim)
        lr_0 = keras.layers.Dense(400, activation="relu", name="c_lr_0", kernel_initializer=RandomUniform(-w_range, w_range))(obs_input)
        # add
        lr_0 = keras.layers.BatchNormalization()(lr_0)

        # layer 1 with concatenated input of [lr_0, action]
        lr_1_input = keras.layers.concatenate([lr_0, action_input])
        w_range = 1 / np.sqrt(400.0)
        lr_1 = keras.layers.Dense(300, activation="relu", name="c_lr_1", kernel_initializer=RandomUniform(-w_range, w_range))(lr_1_input)

        # final layers with linear activation
        w_range = 0.003
        q_val = keras.layers.Dense(1, activation="linear", name="q_val", kernel_initializer=RandomUniform(-w_range, w_range))(lr_1)

        model = keras.Model(inputs=[obs_input, action_input], outputs=q_val)
        return model

    def estimate_q(self, obs, action):
        obs = tf.reshape(obs, (-1, self.obs_dim))
        action = tf.reshape(action, (-1, self.action_dim))
        return self.model([obs, action])


class Actor:
    # 输入特征数，动作特征数，奖励
    def __init__(self, obs_dim, action_dim, action_gain, learning_rate=0.0001):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_gain = action_gain
        self.model = self.make_network()
        self.optimizer = keras.optimizers.Adam(learning_rate)

    def make_network(self):
        obs_input = keras.Input(shape=(self.obs_dim,), dtype="float32", name="obs")

        # layer 0 - with obs input
        w_range = 1 / np.sqrt(self.obs_dim)
        lr_0 = keras.layers.Dense(400, activation="relu", name="a_lr_0", kernel_initializer=RandomUniform(-w_range, w_range))(obs_input)

        # add
        lr_0 = keras.layers.BatchNormalization()(lr_0)

        # layer 1
        w_range = 1 / np.sqrt(400.0)
        lr_1 = keras.layers.Dense(300, activation="relu", name="a_lr_1", kernel_initializer=RandomUniform(-w_range, w_range))(lr_0)
        # add
        lr_1 = keras.layers.BatchNormalization()(lr_1)

        # action layer
        # tanh 函数输出在(-1, 1)之间，用action_gain缩放
        w_range = 0.003
        action = self.action_gain * keras.layers.Dense(self.action_dim, activation="tanh", name="action", kernel_initializer=RandomUniform(-w_range, w_range))(lr_1)

        model = keras.Model(inputs=obs_input, outputs=action)
        return model

    def act(self, obs):
        # 将状态转换为批量的形式
        obs = tf.reshape(obs, (-1, self.obs_dim))
        return self.model(obs)


if __name__ == "__main__":
    actor = Actor(4, 1, 2)
    critic = Critic(4, 1)

    obs = np.random.rand(4)

    action = actor.act(obs)[0]
    q_val = critic.estimate_q(obs, action)[0]

    # keras.utils.plot_model(actor, 'actor.png', show_shapes=True)
    # keras.utils.plot_model(critic, 'critic.png', show_shapes=True)


    print("\nRandom actor-critic output for obs={}:".format(obs))
    print("Action: {}, Qval: {}".format(action, q_val))
