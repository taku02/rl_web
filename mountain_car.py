import gym
import numpy as np
from tqdm import tqdm


class Discretizer:
    def __init__(self):
        self.res_pos = 15
        self.res_vel = 15

    def descretize_state(self, state):
        POSITION_MIN = -1.2
        POSITION_MAX = 0.5
        VELOCITY_MIN = -0.07
        VELOCITY_MAX = 0.07

        position, velocity = state
        vec = [int(self.res_pos * (position - POSITION_MIN) / (POSITION_MAX - POSITION_MIN)), \
               int(self.res_vel * (velocity - VELOCITY_MIN) / (VELOCITY_MAX - VELOCITY_MIN))]
        return vec

    def get_resolution(self):
        return self.res_pos, self.res_vel


class Agent:
    def __init__(self):
        self.last_action = None
        self.last_state = None
        self.epsilon = 0.1
        self.gamma = 1.0
        self.alpha = 0.1
        self.num_actions = 3

        self.dc = Discretizer()
        m, n = self.dc.get_resolution()
        self.q_tab = np.zeros((m, n, self.num_actions))

    def argmax(self, arr):
        maxval = max(arr)
        ties = []
        for i, el in enumerate(arr):
            if maxval == el:
                ties.append(i)
        id_sel = np.random.choice(ties)
        return id_sel

    def select_action(self, state):
        x1, x2 = self.dc.descretize_state(state)

        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(self.num_actions)
        else:
            chosen_action = self.argmax(self.q_tab[x1, x2, :])

        return chosen_action

    def agent_start(self, state):
        current_action = self.select_action(state)
        self.last_action = current_action
        self.last_state = state
        return self.last_action

    def agent_step(self, reward, state):
        current_action = self.select_action(state)
        x1, x2 = self.dc.descretize_state(state)
        last_x1, last_x2 = self.dc.descretize_state(self.last_state)

        max_action_value = np.max(self.q_tab[x1, x2, :])
        last_action_value = self.q_tab[last_x1, last_x2, self.last_action]
        delta = reward + self.gamma * max_action_value - last_action_value

        self.q_tab[last_x1, last_x2, self.last_action] += self.alpha * delta
        self.last_action = current_action
        self.last_state = state

        return self.last_action

    def agent_end(self, reward):
        last_x1, last_x2 = self.dc.descretize_state(self.last_state)
        last_action_value = self.q_tab[last_x1, last_x2, self.last_action]
        delta = reward - last_action_value
        self.q_tab[last_x1, last_x2, self.last_action] += self.alpha * delta

        self.last_action = None
        self.last_state = None


class TestEnv():
    def __init__(self):
        self.max_episode_steps = 1000
        gym.envs.register(
            id='MountainCarMyEasyVersion-v0',
            entry_point='gym.envs.classic_control:MountainCarEnv',
            max_episode_steps=self.max_episode_steps
        )

        self.agent = Agent()
        self.steps_to_end = []
        self.env = gym.make('MountainCarMyEasyVersion-v0')
        self.end_reward = 10

    def avarage_by_runs(self):
        num_runs = 10
        all_steps = []
        for i in tqdm(range(num_runs)):
            self.agent = Agent()
            self.repeat_episode()
            all_steps.append(self.steps_to_end)
            print('run complete')
        self.mean_steps_to_end = np.mean(np.array(all_steps), axis=0)

    def repeat_episode(self):
        num_episode = 500  # 500
        self.steps_to_end = []
        for i in tqdm(range(num_episode)):
            counter = self.run_one_episode()
            self.steps_to_end.append(counter)

    def plot_steps_to_end(self):
        import matplotlib.pyplot as plt
        plt.plot(self.steps_to_end)
        plt.show()

    def plot_mean_steps_to_end(self):
        import matplotlib.pyplot as plt
        plt.plot(self.mean_steps_to_end)
        plt.show()

    def simulate_with_policy(self):
        self.env.reset()
        state = self.env.env.state
        action = self.agent.agent_start(state)
        while True:
            out = self.env.step(action)
            state, reward, done, info = out
            self.env.render()
            if done:
                break
            action = self.agent.agent_step(reward, state)

    def run_one_episode(self):
        self.env.reset()
        state = self.env.env.state
        action = self.agent.agent_start(state)
        counter = 1

        while True:
            out = self.env.step(action)
            state, reward, done, info = out
            if done:
                break
            action = self.agent.agent_step(reward, state)
            counter += 1

        if counter < self.max_episode_steps:
            self.agent.agent_end(self.end_reward)

        return counter


if __name__ == "__main__":
    te = TestEnv()
    te.repeat_episode()
    te.plot_steps_to_end()
    te.simulate_with_policy()
