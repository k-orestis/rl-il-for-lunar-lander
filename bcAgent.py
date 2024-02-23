import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Categorical
from networks import mlp as mlp



class BCAgent:
    def __init__(self, env, experience, batch_size = 20, hidden_layers = [128,128], lr = 0.0001, device = 'cpu'):
        self.env = env
        self.n_actions = env.action_space.n
        self.state_space = env.observation_space.shape[0]
        self.device = device

        TD = TensorDataset(experience['states'].to(self.device), experience['actions'].to(self.device))
        self.dataloader = DataLoader(TD, batch_size=batch_size, shuffle=True)
        self.net = mlp(input_dim = self.state_space, output_dim = self.n_actions, hidden_layers = hidden_layers)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
    
    def train(self, epochs = 1000):
        self.net.to(self.device)
        self.net.train()
        loss_ = []
        print_loss = []
        for i in range(epochs):
            for _, (states, actions) in enumerate(self.dataloader):
                actions_dist = self.net(states)
                actions_dist = F.softmax(actions_dist, dim = -1)
                action_probs = Categorical(actions_dist).log_prob(actions)
                loss = -(action_probs.mean())
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.5)
                self.opt.step()
                loss_.append(loss.item())
                print_loss.append(loss.item())
            
            if i%20 == 0:
                print('Epoch %i, Loss %0.4f' % (i, np.mean(print_loss)))
                print_loss = []
        return loss_
    
    def forward(self, state):
        self.net.eval()
        state = torch.tensor(state, dtype = torch.float32, requires_grad=False)
        return self.net(state)
    
    def load_models(self, mod, dev):
        self.net.load_state_dict(torch.load('models/' + mod, map_location=dev))
    
    def visualize(self, ep=4):
        rew_ = []
        for _ in range(ep):
            s = self.env.reset()
            done = False
            r = 0
            while not done:
                self.env.render()
                q = self.forward(s)
                a  = torch.argmax(q).item()
                s_, rew, done, _ = self.env.step(a)
                r+=rew
                
                s = s_
            rew_.append(r)
        self.env.close()
        return rew_