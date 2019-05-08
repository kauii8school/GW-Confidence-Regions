import gym

env = gym.make('CartPole-v0')

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1