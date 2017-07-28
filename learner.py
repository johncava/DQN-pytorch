import gym
import torch
import random
from torch.autograd import Variable
from torch import FloatTensor
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

# Definition of MSE loss function
loss_function = torch.nn.MSELoss()

# Parameters 
max_episodes = 10
max_iterations = 100
epsilon = 0.7
batch_size = 128
gamma = 0.7
T = 0

# Initialize the environment
env = gym.make('CartPole-v0')
env.reset()

def pick_action(state):
    # Given random probability, pick random state or pick max of Q-function (DQN)
    if random.random() > epsilon:
        return 0 if random.random() < 0.5 else 1
    else:
        # Given the state, make the DQN decide the "best action"
        return model(Variable(FloatTensor(state), volatile=True))

def optimize_model():
    # Sample transitions in memory replay D
    batch = random.sample(D, batch_size)
    # Get transitions from batch
    state_batch = []
    action_batch = []
    reward_batch = []
    next_state_batch = []
    for state, action, reward, next_state in batch:
        state_batch.append(state)
        action_batch.append(action)
        reward_batch.append(reward)
        next_state_batch.append(next_state)
    # Compute Q(s_t, a) from current state
    q_values = model(Variable(FloatTensor(state_batch)))
    print q_values
    # TODO: Get best actions for the state_batch
    #
    # Compute V(s_{t+1}) for next_state batch
    next_state_values = model(Variable(FloatTensor(next_state_batch)))
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + Variable(FloatTensor(reward_batch))
    print expected_state_action_values
    # Compute loss function
    loss = loss_function(q_values, expected_state_action_values)
    # TODO: Optimize the parameters

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