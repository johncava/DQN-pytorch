import gym
import torch
import random
from torch.autograd import Variable

# Initialize replay memory D
D = []

# Definition of Neural Network Model
model = torch.nn.Sequential(
    torch.nn.Linear(4, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100,100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 2),
    torch.nn.Sigmoid()
)

# Parameters 
max_episodes = 10
max_iterations = 100
epsilon = 0.7
batch_size = 128
T = 0

# Initialize the environment
env = gym.make('CartPole-v0')
env.reset()

def pick_action(state):
    # Given random probability, pick random state or pick max of Q-function (DQN)
    if random.random() > epsilon:
        return 0 if random.random() < 0.5 else 1
    else:
        return model(Variable(state))

def optimize_model():
    # Sample transitions in memory replay D
    batch = random.sample(D, batch_size)
    # TODO: Calculate Model Loss from batch states  


for episode in xrange(max_episodes):
    env.render()
    # Initial observation
    state, reward, done, _ = env.step(0)
    for iteration in xrange(max_iterations):
        # Retrieve next action
        action = pick_action(state)
        # Emulate the chosen action 
        next_state, reward, done, _ = env.step(action)
        # Add new transition state to replay memory D
        D.append((state, action, reward, next_state))
        ##Optimize model
        optimize_model()
        ###
        if done:
            break