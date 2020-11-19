import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(torch.nn.Module):
    def __init__(self, n=4, in_dim=128):
        super(PolicyNetwork, self).__init__()

        self.fc1 = torch.nn.Linear(in_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)

        self.fc4 = torch.nn.Linear(128, n)

        self.l_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, x):

        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))

        y = self.fc4(x)

        y = F.softmax(y, dim=-1)

        return y

    def sample_action(self, state):

        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)

        if len(state.size()) == 1:
            state = state.unsqueeze(0)

        y = self(state)

        dist = Categorical(y)

        action = dist.sample()

        log_probability = dist.log_prob(action)

        return action.item(), log_probability.item()

    def evaluate_actions(self, states, actions):
        y = self(states)

        dist = Categorical(y)

        entropy = dist.entropy()

        log_probabilities = dist.log_prob(actions)

        return log_probabilities, entropy


class ValueNetwork(torch.nn.Module):
    def __init__(self, in_dim=128):
        super(ValueNetwork, self).__init__()

        self.fc1 = torch.nn.Linear(in_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)

        self.fc4 = torch.nn.Linear(128, 1)

        self.l_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, x):

        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))

        y = self.fc4(x)

        return y.squeeze(1)

    def state_value(self, state):

        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)

        if len(state.size()) == 1:
            state = state.unsqueeze(0)

        y = self(state)

        return y.item()
