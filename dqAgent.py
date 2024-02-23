import numpy as np
import torch
import torch.nn as nn
import random
from networks import mlp as mlp


class DQAgent:
    def __init__(self, env, layers = [128, 32], eps=0.5, batch_size = 32, d_eps = 0.00005, eps_min = 0.1, 
                 target_net_update_tau = 1e-2, max_cap = 100000, gamma = 0.99, lr = 1e-3, tr_stride = 5, max_ep = 1000, 
                 double_qn = False, device = 'cpu'):
        #set hyperparameters
        self.gamma = gamma
        self.env = env
        self.eps = eps
        self.d_eps = d_eps
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.max_cap = max_cap
        self.max_ep = max_ep
        self.device = device

        # initialize buffers
        self.n_actions = env.action_space.n
        self.state_space = env.observation_space.shape[0]
        self.states = np.zeros((self.max_cap, self.state_space))
        self.states_ = np.zeros((self.max_cap, self.state_space))
        self.rews = np.zeros((self.max_cap, 1))
        self.actions = np.zeros((self.max_cap, 1))
        self.done = np.zeros((self.max_cap, 1))
        self.target_net_update_tau = target_net_update_tau
        self.index = 0

        #initialize learning
        self.double_qn = double_qn
        self.q = mlp(input_dim = self.state_space, output_dim = self.n_actions, hidden_layers = layers)
        self.target_q = mlp(input_dim = self.state_space, output_dim = self.n_actions, hidden_layers = layers)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr = lr)
        self.criterion = nn.MSELoss()
        self.tr_stride = tr_stride
    
    def act(self, state): 
        if np.random.random()<self.eps:
            return np.random.randint(self.n_actions)
        state = torch.tensor(state, requires_grad=False)
        with torch.no_grad():
            self.q.eval()
            q = self.q(state)
        return torch.argmax(q).item() #q

    def forward(self, state):
        state = torch.tensor(state, dtype = torch.float32)

        return self.q(state)

    def forward_t(self, state):
        self.target_q.eval()
        state = torch.tensor(state, dtype = torch.float32, requires_grad=False)
        return self.target_q(state)
    
    def soft_update_target_net(self):
        params1 = self.q.named_parameters()
        params2 = self.target_q.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(\
                    self.target_net_update_tau*param1.data\
                + (1-self.target_net_update_tau)*dict_params2[name1].data)
        self.target_q.load_state_dict(dict_params2)

    def save_models(self, name):
        torch.save(self.q.state_dict(), 'models/' + name + '_q')
        torch.save(self.target_q.state_dict(), 'models/' + name + '_t')

    def load_models(self, name):
        self.q.load_state_dict(torch.load('models/' + name + '_q'))
        self.target_q.load_state_dict(torch.load('models/' + name + '_t'))
    
    def train(self):
        losses = []
        rew_ep = 0
        actions = []
        step, stride = (0, 5)

        for ep in range(self.max_ep):
            s = self.env.reset()
            done = False
            while not done:
                step += 1
                # gather experience
                a = self.act(s)
                actions.append(a)

                s_, rew, done, _ = self.env.step(a)
                rew_ep += rew

                self.fill_buffer((s, a, rew, s_, done))

                if step % self.tr_stride == 0:
                    self.q.train()
                    st, a, rew, st_, done_t, = self.get_from_buf()
                    rew = torch.tensor(rew, dtype = torch.float32, requires_grad=False)
                    done_t = torch.tensor(done_t, dtype = torch.int8, requires_grad=False)
                    a = torch.tensor(a, dtype = torch.int16, requires_grad=False)
                    #print(st.shape)


                    #calculate target q values
                    if self.double_qn:
                        actions_ = self.forward(st_).argmax(dim = 1).unsqueeze(dim=1)
                        q_ = self.forward_t(st_)
                        q_m = torch.gather(q_, dim=1, index = actions_)
                        
                    else:
                        q_ = self.forward_t(st_)
                        q_m, _ = torch.max(q_, dim = 1)
                        q_m = q_m.unsqueeze(dim=1)
                    
                    q_d = rew + self.gamma * (1.-done_t) *q_m

                    #print(rew.shape, q_.shape, q_m.shape, q_d.shape)

                    
                    q = self.forward(st)
                    qm = torch.gather(q, dim = 1, index = a)
                    #print(q.shape, qm.shape, a.shape)
                    
                    loss = self.criterion(qm, q_d)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    #if ep%4 == 0:
                    self.update_epsilon()
                    self.soft_update_target_net()
                    losses.append(loss.item())
                
                s = s_
            if ep%50 == 0:
                print(ep, rew_ep/50)
                rew_ep = 0

        self.env.close()
        return losses, actions 
    
    def update_epsilon(self):
        if self.eps > self.eps_min:
            self.eps -= self.d_eps

    def fill_buffer(self, t):
        self.states[self.index] = t[0]
        self.actions[self.index] = t[1]
        #self.qs[self.index] = t[2]
        self.rews[self.index] = t[2]
        self.states_[self.index] = t[3]
        self.done[self.index] = t[4]
        #self.buffer[self.index] = t
        self.index += 1
        if self.index == self.max_cap:
            self.index = 0

    def get_from_buf(self):
        if self.index > self.batch_size:
            batch_idx = random.sample(range(self.index), self.batch_size )
        else:
            batch_idx = np.arange(self.index)
        return self.states[batch_idx], self.actions[batch_idx], self.rews[batch_idx], self.states_[batch_idx], self.done[batch_idx]
    

    def visualize(self, ep=4):
        rew_ = []
        for _ in range(ep):
            s = self.env.reset()
            done = False
            r = 0
            while not done:
                self.env.render()
                q = self.forward_t(s)
                a  = torch.argmax(q).item()
                s_, rew, done, _ = self.env.step(a)
                r+=rew
                
                s = s_
            rew_.append(r)
        self.env.close()
        return rew_
    
    def get_experiences(self, num_exp = 25000):
        experience  = {
            'states': torch.empty(num_exp, self.state_space),
            'states_': torch.empty(num_exp, self.state_space),
            'actions': torch.empty(num_exp),
            'rewards': torch.empty(num_exp),
            'dones': torch.empty(num_exp)
        }
        step = 0
        while step < num_exp:
            s = self.env.reset()
            done = False
            while not done:
                q = self.forward_t(s)
                a  = torch.argmax(q).item()
                s_, rew, done, _ = self.env.step(a)
                s = s_
                experience['states'][step] = torch.tensor(s, dtype=torch.float32)
                experience['states_'][step] = torch.tensor(s_, dtype=torch.float32)
                experience['actions'][step] = torch.tensor(a, dtype=torch.int16)
                experience['rewards'][step] = torch.tensor(rew, dtype=torch.float32)
                experience['dones'][step] = torch.tensor(done, dtype=torch.int8)
                step+=1
                if step == num_exp:
                    break

        self.env.close()
        return experience