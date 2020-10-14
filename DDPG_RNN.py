### Required Package

import os
import sys
# os._exit(0)
import random
from collections import deque
from math import ceil

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import get_data as GD
import data_preprocessing as DP
from model import RNN_cell

data_test = DP.data_test
train_norm, test_norm, sc = DP.train_norm, DP.test_norm, DP.sc
# print(train_norm.shape)
# print(test_norm.shape)
Close_range = sc.data_range_[-1]
Close_max   = sc.data_max_[-1]
Close_min   = sc.data_min_[-1]
Close_denormalize = lambda x: x * Close_range + Close_min
# buy approx. 100 USD each time
buy_per_unit = 100 / ((Close_max+Close_min)/2)
Transaction_fee = 0.01
fee_rate = lambda sign: 1 + sign * Transaction_fee

currency_type = GD.SYMBOL
model_type = sys.argv[3]
mul_size_layer = 2 if model_type=='LSTM' else 1
test_length = DP.test_length
for file_name in ["fig", "output"]:
    path = os.path.join(file_name, test_length, currency_type, model_type)
    if not os.path.exists(path):
        os.makedirs(path)

print("#################################")
print("Data Type:", currency_type)
print("Model Type:", model_type)
print("Test Length:", test_length)
print("#################################")

### Agent
class Actor:
    def __init__(self, name, time_step, input_dim, output_size, size_layer):
        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, (None, time_step, input_dim))
            # attention?
            self.hidden_layer = tf.placeholder(tf.float32, (None, mul_size_layer * size_layer))
            cell = RNN_cell(model_type, size_layer)
            self.rnn, self.last_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.X,
                                                          dtype=tf.float32,
                                                          initial_state=self.hidden_layer)
            self.logits = tf.layers.dense(self.rnn[:,-1], output_size) # output_size = 3: prob. of buy, sell, hold

class Critic:
    def __init__(self, name, time_step, input_dim, output_size, size_layer, learning_rate):
        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, (None, time_step, input_dim))
            self.Y = tf.placeholder(tf.float32, (None, output_size))
            self.hidden_layer = tf.placeholder(tf.float32, (None, mul_size_layer * size_layer))
            self.REWARD = tf.placeholder(tf.float32, (None, 1))
            cell = RNN_cell(model_type, size_layer)
            self.rnn, self.last_state = tf.nn.dynamic_rnn(cell=cell, inputs=self.X,
                                                          dtype=tf.float32,
                                                          initial_state=self.hidden_layer)
            feed_critic = tf.layers.dense(self.rnn[:,-1], output_size, activation = tf.nn.relu) + self.Y
            feed_critic = tf.layers.dense(feed_critic, size_layer//2, activation = tf.nn.relu)
            self.logits = tf.layers.dense(feed_critic, 1)
            self.cost = tf.reduce_mean(tf.square(self.REWARD - self.logits))
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

class Agent:
    Episodes_counter = 0
    # Hyperparameters
    ACTOR_LEARNING_RATE = 0.0001
    CRITIC_LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    LAYER_SIZE = 256
    OUTPUT_SIZE = 3
    EPSILON = 0.5
    DECAY_RATE = 0.005
    MIN_EPSILON = 0.1
    GAMMA = 0.99
    MEMORIES = deque()
    MEMORY_SIZE = 10000
    COPY = 100
    T_COPY = 0
    # Results
    result = data_test[['Symbol','Open','High','Low','Close','Volume %s' % GD.SYMBOL,'Volume USD']]
    for i in ['Signal', 'Investment', 'Total Balance(USD)']:
        result[i] = 0
    result.reset_index(inplace=True)
    Inv_per_episode_train = []
    Inv_per_episode_test = []

    def __init__(self, input_dim, window_size, trend, skip):
        self.input_dim = input_dim
        self.window_size = window_size
        self.trend = trend
        self.skip = skip
        tf.reset_default_graph()
        self.actor = Actor('actor-eval', self.window_size, self.input_dim, self.OUTPUT_SIZE, self.LAYER_SIZE)
        self.actor_target = Actor('actor-target', self.window_size, self.input_dim, self.OUTPUT_SIZE, self.LAYER_SIZE)
        self.critic = Critic('critic-eval', self.window_size, self.input_dim, self.OUTPUT_SIZE, self.LAYER_SIZE, self.CRITIC_LEARNING_RATE)
        self.critic_target = Critic('critic-target', self.window_size, self.input_dim, self.OUTPUT_SIZE, self.LAYER_SIZE, self.CRITIC_LEARNING_RATE)
        self.grad_critic = tf.gradients(self.critic.logits, self.critic.Y)
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.OUTPUT_SIZE])
        weights_actor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        self.grad_actor = tf.gradients(self.actor.logits, weights_actor, -self.actor_critic_grad)
        grads = zip(self.grad_actor, weights_actor)
        self.optimizer = tf.train.AdamOptimizer(self.ACTOR_LEARNING_RATE).apply_gradients(grads)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # TensorBoard
        # $ tensorboard --logdir="TensorBoard/"
        tf.summary.FileWriter("TensorBoard/", graph = self.sess.graph)

    def _assign(self, from_name, to_name):
        from_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_name)
        to_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=to_name)
        for i in range(len(from_w)):
            assign_op = to_w[i].assign(from_w[i])
            self.sess.run(assign_op)

    def _memorize(self, state, action, reward, next_state, rnn_state):
        self.MEMORIES.append((state, action, reward, next_state, rnn_state))
        if len(self.MEMORIES) > self.MEMORY_SIZE:
            self.MEMORIES.popleft()

    def _select_action(self, state, init_value):
        # Episilon greedy
        if np.random.rand() < self.EPSILON:
            action = np.random.randint(self.OUTPUT_SIZE)
        else:
            action, last_state = self.sess.run([self.actor.logits,
                                                self.actor.last_state],
                                                feed_dict={self.actor.X:[state],
                                                           self.actor.hidden_layer:init_value})
            action, init_value = np.argmax(action[0]), last_state

        return action, init_value

    # Replay Buffer
    def _construct_memories_and_train(self, replay):
        states = np.array([exp[0] for exp in replay])
        next_states = np.array([exp[3] for exp in replay])
        init_values = np.array([exp[-1] for exp in replay])
        Q_eval = self.sess.run(self.actor.logits, feed_dict={self.actor.X: states,
                                                       self.actor.hidden_layer: init_values})
        Q_target = self.sess.run(self.actor_target.logits, feed_dict={self.actor_target.X: states,
                                                                      self.actor_target.hidden_layer: init_values})
        grads = self.sess.run(self.grad_critic, feed_dict={self.critic.X: states, self.critic.Y: Q_eval,
                                                           self.critic.hidden_layer: init_values})[0]
        self.sess.run(self.optimizer, feed_dict={self.actor.X: states, self.actor_critic_grad: grads,
                                                 self.actor.hidden_layer: init_values})

        rewards_eval = np.array([exp[2] for exp in replay]).reshape((-1, 1))
        rewards_target = self.sess.run(self.critic_target.logits, 
                                       feed_dict={self.critic_target.X: next_states,self.critic_target.Y: Q_target,
                                                  self.critic_target.hidden_layer: init_values})
        for i in range(len(rewards_eval)):
            rewards_eval[i] += self.GAMMA * rewards_target[i]
        cost, _ = self.sess.run([self.critic.cost, self.critic.optimizer], 
                                feed_dict={self.critic.X: states, self.critic.Y: Q_eval, self.critic.REWARD: rewards_eval,
                                           self.critic.hidden_layer: init_values})
        return cost

    def get_state(self, t):
        window_size = self.window_size + 1
        d = t - window_size + 1
        if d >= 0:
            block = self.trend[d : t + 1]
        else:
            block = np.concatenate((np.tile(self.trend[0], (-d, 1)), self.trend[0 : t + 1]))
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        return np.array(res)

    def train(self, iterations, initial_USD):
        print("Training...")
        for i in range(iterations):
            self.Episodes_counter += 1
            print("%d episode:" % self.Episodes_counter)
            state = self.get_state(0)
            remain_USD = initial_USD
            remain_ETH = 0
            init_value = np.zeros((1, mul_size_layer * self.LAYER_SIZE))
            for t in range(0, len(self.trend) - 1, self.skip):
                if (self.T_COPY + 1) % self.COPY == 0:
                    self._assign('actor-eval', 'actor-target')
                    self._assign('critic-eval', 'critic-target')

                action, init_value = self._select_action(state, init_value)
                next_state = self.get_state(t + 1)
                
                Close_value = Close_denormalize(self.trend[t][-1])
                Tomorrow_close = Close_denormalize(self.trend[t+1][-1])
                # Buy
                if action == 1 and remain_USD >= Close_value * fee_rate(1):
                    signal = +1
                # Sell
                elif action == 2 and remain_USD + remain_ETH * Close_value >= Close_value * fee_rate(1):
                    signal = -1
                # Hold
                else:
                    signal = 0

                remain_USD -= signal * buy_per_unit * (Close_value * fee_rate(signal))
                remain_ETH += signal * buy_per_unit

                Total_Balance = remain_USD + remain_ETH * Close_value
                Value_Change = (Tomorrow_close - Close_value) / Close_value
                # Buy/Sell
                if signal:
                    profit = signal * buy_per_unit * Value_Change
                # Hold
                else:
                    profit = 2**(-np.sign(Value_Change)) * buy_per_unit * Value_Change
                self._memorize(state, action, profit, next_state, init_value[0])
                batch_size = min(len(self.MEMORIES), self.BATCH_SIZE)
                replay = random.sample(self.MEMORIES, batch_size)
                cost = self._construct_memories_and_train(replay)
                self.T_COPY += 1
                self.EPSILON = self.MIN_EPSILON + (1.0 - self.MIN_EPSILON) * np.exp(-self.DECAY_RATE * i)
                state = next_state
            Investment = (Total_Balance - initial_USD) / initial_USD
            self.Inv_per_episode_train.append(Investment)
        print("Cost: %f, Investment: %.2f, Total Balance(USD): %f" % (cost, 100*Investment, Total_Balance))

    def test(self, trend_test, initial_USD):
        print("Testing...")
        remain_USD = initial_USD
        remain_ETH = 0
        self.trend = trend_test
        states_sell = []
        states_buy = []
        init_value = np.zeros((1, mul_size_layer * self.LAYER_SIZE))
        for t in range(0, len(self.trend) - 1, self.skip):
            state = self.get_state(t)
            action, init_value = self._select_action(state, init_value)

            Close_value = Close_denormalize(self.trend[t][-1])
            # Buy
            if action == 1 and remain_USD >= Close_value * fee_rate(1):
                states_buy.append(t)
                signal = +1
            # Sell
            elif action == 2 and remain_USD + remain_ETH * Close_value >= Close_value * fee_rate(1):
                states_sell.append(t)
                signal = -1
            # Hold
            else:
                signal = 0

            remain_USD -= signal * buy_per_unit * (Close_value * fee_rate(signal))
            remain_ETH += signal * buy_per_unit

            Total_Balance = remain_USD + remain_ETH * Close_value
            Investment = (Total_Balance - initial_USD) / initial_USD
            # Buy/Sell
            if signal:
                print("Day %d, %s 1 unit at price %f, Investment %.2f %%, Total Balance %f"
                      % (t, 'buy' if signal>0 else 'sell', Close_value, 100*Investment, Total_Balance))

            self.result.iloc[t,-3] = signal
            self.result.iloc[t,-2] = Investment
            self.result.iloc[t,-1] = Total_Balance

        self.Inv_per_episode_test.append(Investment)
        return states_buy, states_sell, Total_Balance, Investment

train_feature = train_norm.values
test_feature = test_norm.values
initial_USD = 10000
window_size = 7
skip = 1
episodes = 100
eval_per_episode = 1
plot_per_episode = 10
N = ceil(episodes//eval_per_episode)

agent = Agent(input_dim = train_feature.shape[1], 
              window_size = window_size, 
              trend = train_feature, 
              skip = skip)

pd.plotting.register_matplotlib_converters()
for i in range(N):
    ### Training
    agent.train(iterations=eval_per_episode, initial_USD=initial_USD)
    file_name = "DDPG_%s_ep%d" % (model_type, agent.Episodes_counter)

    ### Evaluate
    states_buy, states_sell, Total_Balance, Investment = agent.test(trend_test=test_feature, initial_USD=initial_USD)
    result = agent.result

    ### Plot fig and output result
    if (i+1) % plot_per_episode == 0:
        fig = plt.figure(figsize = (20, 10))
        plt.plot(test_norm.iloc[:,-1], color='r', lw=2.)
        plt.plot(test_norm.iloc[:,-1], '^', markersize=10, color='m', label = "buying signal: %d" % (result['Signal']>0).sum(), markevery = states_buy)
        plt.plot(test_norm.iloc[:,-1], 'v', markersize=10, color='k', label = "selling signal: %d" % (result['Signal']<0).sum(), markevery = states_sell)
        plt.title("Initial money: %d, Total balance: %.3f, Investment: %.3f %%, Market change: %.3f %%"
                   % (initial_USD, Total_Balance, 100*Investment, 100*(test_norm.iloc[-1,-1] - test_norm.iloc[0,-1])/test_norm.iloc[0,-1]))
        plt.legend()
        fig_path = os.path.join("./fig", test_length, currency_type, model_type, file_name+".png")
        # plt.tight_layout()
        plt.savefig(fig_path)
        plt.clf()
        plt.close()

        ### Result Output
        for i in ['Investment','Total Balance(USD)']:
            result.loc[i] = result[i].replace(to_replace=0, method='ffill')
        # print(result)
        result_path = os.path.join("./output", test_length, currency_type, model_type, file_name+".json")
        result.to_json(result_path, orient='records')

# Investment versus episode
fig = plt.figure(figsize = (20, 10))
plt.plot(np.linspace(eval_per_episode, episodes, N), np.array(agent.Inv_per_episode_train) * 100, lw=2., label='train')
plt.plot(np.linspace(eval_per_episode, episodes, N), np.array(agent.Inv_per_episode_test) * 100, lw=2., label='test')
plt.xlabel('Episodes')
plt.ylabel('Investment(%)')
plt.title("Investment versus episode")
plt.legend()
path = os.path.join("./fig", test_length, currency_type + '_' + model_type + '_' + "Investment-versus-episode.png")
# plt.tight_layout()
plt.savefig(path)
plt.clf()
plt.close()

# TRY:
# attention, emphasize later, importance of last 7 days
