import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from networks import mlp as mlp

# CODE AND HYPERPARAMETERS BASED ON
# PPO IMPLEMENTATION IN https://github.com/mitre/ilpyt.git

class PPOAgent:
    def __init__(self, env, lr = 0.0005, gamma = 0.99, clip_ratio = 0.1, entropy_coeff = 0.01, layers = [128,128], device = 'cpu', 
                 num_env = 16, rollout_steps=1):
        self.env = env
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.lr = lr
        self.device = device
        self.n_actions = self.env[0].action_space.n
        self.state_space = self.env[0].observation_space.shape[0]
        self.best_reward = -np.inf
        self.num_env = num_env
        self.rollout_steps = rollout_steps
        self.log_probs = []
        self.reward_tracker = self.best_reward * np.ones(self.num_env)

        self.actor = mlp(input_dim = self.state_space, output_dim = self.n_actions, hidden_layers = layers, act = 'tanh')
        self.critic = mlp(input_dim = self.state_space, output_dim = 1, hidden_layers = layers, act = 'tanh')

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr / 2)
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.state = torch.tensor(np.array([e.reset() for e in env]), device = self.device)
        self.episode_stats = {
            'reward': np.zeros(self.num_env),
            'length': np.zeros(self.num_env),
            'count': np.zeros(self.num_env),
        }
    
    @torch.no_grad()
    def generate_batch(self, rollout_steps = 1):

        obs_shape = (rollout_steps, self.num_env, self.state_space)
        act_shape = (rollout_steps, self.num_env)
        rew_shape = dones_shape = (rollout_steps, self.num_env)

        batch= {
            'states': torch.empty(obs_shape),
            'next_states': torch.empty(obs_shape),
            'actions': torch.empty(act_shape),
            'rewards': torch.empty(rew_shape),
            'dones': torch.empty(dones_shape),
            'infos': [],
        }
        next_state = np.empty(obs_shape)
        reward = np.empty(rew_shape)
        done = np.empty(dones_shape)
        for step in range(rollout_steps):
            # Agent takes action
            action = self.step(self.state)
            #print(action.shape)
            # Update environment
            for i in range(self.num_env):
                next_state[step, i], reward[step, i], done[step, i], _ = self.env[i].step(action[i]) 
            
            # Record transition to batch
            batch['states'][step] = torch.as_tensor(self.state, dtype=torch.float32)
            batch['next_states'][step] = torch.as_tensor(next_state[step], dtype=torch.float32)
            batch['actions'][step] = torch.tensor(
                    action, dtype=torch.float, requires_grad=True
                )
            batch['rewards'][step] = torch.as_tensor(reward[step])
            batch['dones'][step] = torch.as_tensor(done[step])

            # Update episode stats
            self.episode_stats['reward'] += reward[step]
            self.episode_stats['length'] += np.ones(self.num_env)
            self.episode_stats['count'] += done[step]

            # On episode end, update batch infos and reset
            for i in range(self.num_env):
                if done[step, i]:
                    #print('reward/%i ' % i, self.episode_stats['reward'][i],
                    #    'length/%i ' % i, self.episode_stats['length'][i])
                    update_dict = {
                        'reward/%i' % i: self.episode_stats['reward'][i],
                        'length/%i' % i: self.episode_stats['length'][i],
                    }
                    update = [self.episode_stats['count'][i], update_dict]
                    batch['infos'].append(update)
                    self.episode_stats['reward'][i] = 0
                    self.episode_stats['length'][i] = 0
                    next_state[step, i] = self.env[i].reset()

            # Update state
            self.state = torch.from_numpy(next_state[step]).float().to(self.device)
        
        # Batch to GPU
        if self.device == 'cuda':
            for (k, v) in batch.items():
                if k != 'infos':
                    batch[k] = v.cuda()

        return batch

    def update(self, batch):
        """
        Returns
        -------
        Dict[str, float]:
            losses for the update step, key strings and loss values can be 
            automatically recorded to TensorBoard
        """
        # Update critic
        final_states = batch['next_states'][-1]
        value_final = self.critic(final_states).squeeze()
        #print('value final:', value_final.shape)
        targets = self.compute_target(
            value_final, batch['rewards'], 1 - batch['dones'], self.gamma
        ).reshape(-1)
        #print('targets', targets.shape)
        if self.device == 'cuda':
            targets = targets.cuda()
        
        for key, value in batch.items():
            if key == 'infos':
                continue
            batch[key] = value.view((self.rollout_steps*self.num_env, -1))
        
        values = self.critic(batch['states']).squeeze()
        #print('values', values.shape)
        advantages = targets - values
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        loss_critic = F.smooth_l1_loss(values, targets)
        self.opt_critic.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.5)
        self.opt_critic.step()

        # Update actor
        logits = self.actor(batch['states'])
        #print(logits.shape)
        dist = Categorical(F.softmax(logits, dim=-1))
        log_action_probs = dist.log_prob(batch['actions'])

        
        if len(self.log_probs[0].shape) != 0:
            old_log_action_probs = torch.cat(self.log_probs)
        else:
            old_log_action_probs = torch.tensor(self.log_probs)
        if self.device == 'cuda':
            old_log_action_probs = old_log_action_probs.cuda()

        ratio = torch.exp(log_action_probs - old_log_action_probs.detach())
        clipped_advantages = (
            torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            * advantages.detach()
        )

        # Compute losses
        loss_entropy = self.entropy_coeff * dist.entropy().mean()
        loss_action = -(
            torch.min(ratio * advantages.detach(), clipped_advantages)
        ).mean()
        loss_actor = loss_action - loss_entropy

        # Updates
        self.opt_actor.zero_grad()
        loss_actor.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.5)
        self.opt_actor.step()

        # Reset log_probs
        self.log_probs = []

        # Return loss dictionary
        loss_dict = {
            'loss/actor': loss_actor.item(),
            'loss/critic': loss_critic.item(),
            'loss/total': loss_actor.item() + loss_critic.item(),
        }
        return loss_dict
 
    def step(self, state):
        with torch.no_grad():
            logits = self.actor(state)
            dist = Categorical(F.softmax(logits, dim=-1))
            actions = dist.sample()

            log_probs = dist.log_prob(actions)
            self.log_probs.append(log_probs)
            

        if self.device == 'cuda':
            actions = actions.cpu().numpy()
        else:
            actions = actions.numpy()
        return actions
    
    def compute_target(
        self,
        value_final: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """
        Compute target (sum of total discounted rewards) for rollout.

        Parameters
        -----------
        value_final: torch.Tensor
            state values from final time step of rollout, size (num_env,)
        rewards: torch.Tensor
            rewards across rollout, size (rollout_steps, num_env)
        masks: torch.Tensor
            masks for episode end states, 0 if end state, 1 otherwise,
            size (rollout_steps, num_env)
        gamma: float
            discount factor for rollout

        Returns
        -------
        torch.Tensor: targets, size (rollout_steps, num_env)
        """
        G = value_final
        T = rewards.shape[0]
        targets = torch.zeros(rewards.shape)

        for i in range(T - 1, -1, -1):
            G = rewards[i] + gamma * G * masks[i]
            targets[i] = G

        return targets

    
    def train(self, num_episodes = 10000):
        
        self.actor.train()
        self.critic.train()
        step = 0
        ep_count = 0

        while ep_count < num_episodes:
            # Step agent and environment
            batch = self.generate_batch(self.rollout_steps)
            #print(ep_count, batch['states'].shape, batch['actions'].shape, batch['rewards'].shape)
            step += 1

            # Update agent
            loss_dict = self.update(batch)
            if step % 100 == 0:
                print(ep_count, loss_dict) 

            for ep_count, info_dict in batch['infos']:
                for (k, v) in info_dict.items():
                    if 'reward' in k:
                        agent_num = int(k.split('/')[1])
                        self.reward_tracker[agent_num] = v
            # should this be on this level?
            # Save agent
            reward = np.mean(self.reward_tracker)
            if reward > self.best_reward:
                torch.save(self.actor.state_dict(), "models/actor")
                torch.save(self.critic.state_dict(), "models/critic")
                self.best_reward = reward
                print(
                    "Save new best model at episode %i with reward %0.4f."
                    % (ep_count, reward)
                )

    def load_models(self, file, dev = 'cpu'):
        self.actor.load_state_dict(torch.load('models/' + file + '/actor', map_location=dev))
        self.critic.load_state_dict(torch.load('models/' + file + '/critic', map_location=dev))

    def visualize(self, ep=4):
        rew_ = []
        for _ in range(ep):
            s = self.env[0].reset()
            it = 0
            done = False
            r = 0
            self.actor.eval()
            while not done:
                it+=1
                self.env[0].render()
                dist = self.actor(torch.tensor(s, dtype = torch.float32, requires_grad=False))
                action = torch.argmax(dist)
                s_, rew, done, _ = self.env[0].step(action.item())
                r+=rew
                s = s_
            #print(s)
            rew_.append(r)
        self.env[0].close()
        return rew_ 
