import gym
import torch
import random
import numpy as np
import torch.optim as optim
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

# Definition of the RMSprop optimizer function
optimizer = optim.RMSprop(model.parameters())

# Parameters 
max_episodes = 10
max_iterations = 5000
epsilon = 0.7
batch_size = 1
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
        prediction = model(Variable(FloatTensor([state])))
        prediction = prediction.data.numpy()
        #print prediction
        #print np.argmax(prediction, axis=1)
        return np.argmax(prediction, axis=1)[0]

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
    print Variable(torch.LongTensor([action_batch]))
    q_values = q_values.gather(1, Variable(torch.LongTensor([action_batch])))
    print 'q_values ', q_values
    '''
    # Equivalent to "q_values.gather(1, Variable(torch.LongTensor([action_batch])))"
    q_values = q_values.data.numpy()
    q_values_from_action = []
    print q_values
    print action_batch
    for index in xrange(len(batch)):
        q_values_from_action.append(q_values[index][action_batch[index]])
    print q_values_from_action
    '''
    # Compute V(s_{t+1}) for next_state_batch
    next_state_values = model(Variable(FloatTensor(next_state_batch), volatile=True)).max(1)[0]
    # Compute the expected Q values
    print 'next_state_values: ', next_state_values
    print reward_batch
    next_state_values.volatile = False
    expected_state_action_values = (next_state_values * gamma) + Variable(FloatTensor(reward_batch))
    print 'Expected state action values: ' ,expected_state_action_values
    # Compute loss function
    loss = loss_function(q_values, expected_state_action_values)
    # Zero the gradients for the optimizer
    optimizer.zero_grad()
    # Backpropagate the error through the DQN model
    loss.backward()

    optimizer.step()

for episode in xrange(max_episodes):
    #env.render()
    env.reset()
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