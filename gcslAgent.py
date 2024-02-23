import numpy as np
import torch
from torch.distributions import Categorical
import time
import tqdm
from networks import mlp as mlp

# CODE AND HYPERPARAMETERS BASED ON
# GCSL IMPLEMENTATION IN https://github.com/dibyaghosh/gcsl.git


class GCSLBuffer:
    def __init__(self, env, max_trajectory_length, buffer_size):
            self.env = env
            self.actions = np.zeros((buffer_size, max_trajectory_length), dtype = np.float64 )
            self.states = np.zeros((buffer_size, max_trajectory_length, env.observation_space.shape[0]), dtype = np.float64 )
            self.goal_states = np.zeros((buffer_size, max_trajectory_length, env.observation_space.shape[0]), dtype = np.float64 )
            self.mid_goals = np.zeros((buffer_size, max_trajectory_length, 5), dtype = np.float64 )
            self.traj_length = np.zeros((buffer_size), dtype = np.int32)

            self.max_tlength = max_trajectory_length
            self.buffer_size = buffer_size
            self.cur_buffer_size = 0
            self.cur = 0
    
    def add_trajectory(self, states, actions, desired_state, length_of_traj = 50):

        self.actions[self.cur] = actions
        self.states[self.cur] = states
        self.mid_goals[self.cur] = states[..., [0,1,4,6,7]]
        self.goal_states[self.cur] = desired_state
        self.traj_length[self.cur] = length_of_traj

        self.cur += 1
        self.cur_buffer_size = max(self.cur, self.cur_buffer_size)
        if self.cur == self.buffer_size:
            self.cur = 0

    def _sample_indices(self, batch_size):
        traj_idxs = np.random.choice(self.cur_buffer_size, batch_size)

        prop_idxs_1 = np.random.rand(batch_size)
        prop_idxs_2 = np.random.rand(batch_size)
        time_idxs_1 = np.floor(prop_idxs_1 * (self.traj_length[traj_idxs]-1)).astype(int)
        time_idxs_2 = np.floor(prop_idxs_2 * (self.traj_length[traj_idxs])).astype(int)
        time_idxs_2[time_idxs_1 == time_idxs_2] += 1

        time_state_idxs = np.minimum(time_idxs_1, time_idxs_2)
        time_goal_idxs = np.maximum(time_idxs_1, time_idxs_2)
        return traj_idxs, time_state_idxs, time_goal_idxs

    def sample_batch(self, batch_size):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """

        traj_idxs, time_state_idxs, time_goal_idxs = self._sample_indices(batch_size)
        return self._get_batch(traj_idxs, time_state_idxs, time_goal_idxs)

    def _get_batch(self, traj_idxs, time_state_idxs, time_goal_idxs):
        batch_size = len(traj_idxs)
        observations = self.states[traj_idxs, time_state_idxs]
        actions = self.actions[traj_idxs, time_state_idxs]
        goals = self.states[traj_idxs, time_goal_idxs][..., [0,1,4,6,7]]
        
        lengths = time_goal_idxs - time_state_idxs
        horizons = np.tile(np.arange(self.max_tlength), (batch_size, 1))
        horizons = horizons >= lengths[..., None]


        return observations, actions, goals, lengths, horizons
       
        
#max_timesteps = 2e5
class GCSLAgent:
    def __init__(self, env, layers = [400, 300], batch_size = 256, buffer_size = 20000, 
                 max_trajectory_length = 50, lr=5e-4, max_timesteps = 1e6, expl_noise=0.0,
                 device = 'cpu'):
        self.explore_timesteps=10000
        self.start_policy_timesteps=1000
        self.max_trajectory_length = max_trajectory_length
        self.env = env 
        self.batch_size = batch_size
        self.n_actions = env.action_space.n
        self.state_space = env.observation_space.shape[0]
        self.goal_space = 5
        self.policy = mlp(input_dim = self.state_space + self.goal_space, output_dim = self.n_actions, hidden_layers = layers)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.expl_noise = expl_noise

        self.buffer = GCSLBuffer(env, max_trajectory_length, buffer_size)
        self.validation_buffer = GCSLBuffer(env, max_trajectory_length, int(0.2*buffer_size))
        self.max_timesteps = max_timesteps
        self.eval_episodes = 200
        self.device = device
        self.policy.to(self.device)
        self.frame_skip = 2

    def take_policy_step(self):
        buffer = self.buffer

        avg_loss = 0
        self.policy_optimizer.zero_grad()
        

        observations, actions, goals, _, horizons = buffer.sample_batch(self.batch_size)
        loss = self.loss_fn(observations, goals, actions, horizons)
        loss.backward()
        avg_loss += loss.detach().cpu().numpy()
        
        self.policy_optimizer.step()

        
        return avg_loss
    

    def sample_trajectory(self, greedy=False, noise=0, render=False):

        base_goal = np.array([0.0, 0, 0, 0, 0, 0, 1, 1,])
        base_goal[0] += 0.3 * np.random.randn()
        if np.random.rand() > 0.5:
            base_goal[6:] = 0.0
            # base_goal[1] += np.abs(np.random.randn() * 0.3)
            base_goal[2:6] += np.random.randn(4) * 0.2

        goal = base_goal[..., [0,1,4,6,7]]

        states = []
        actions = []

        state = self.env.reset()
        for t in range(self.max_trajectory_length):
            if render:
                self.env.render()

            states.append(state)

            observation = state
            horizon = np.arange(self.max_trajectory_length) >= (self.max_trajectory_length - 1 - t) # Temperature encoding of horizon
            action = self.act_vectorized(observation[None], goal[None], horizon=horizon[None], greedy=greedy, noise=noise)[0]
            
            
            actions.append(action)
            for _ in range(self.frame_skip):
                state, _, _, _ = self.env.step(action)
            #state, _, _, _ = self.env.step(action)
        print(np.histogram(np.array(actions), bins=[-.5,.5,1.5,2.5,3.5])[0])
        #print(states)
        return np.stack(states), np.array(actions), base_goal
    
    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0):
        obs = torch.tensor(obs, dtype=torch.float32, device = self.device)
        goal = torch.tensor(goal, dtype=torch.float32, device = self.device)
        if horizon is not None:
            horizon = torch.tensor(horizon, dtype=torch.float32, device = self.device)

        embed = torch.cat((obs, goal), dim=1).detach()       
        #logits = self.policy.forward(obs, goal, horizon=horizon)
        logits = self.policy.forward(embed)
        noisy_logits = logits  * (1 - noise)
        probs = torch.softmax(noisy_logits, 1)
        #print(probs)
        #print(embed)
        if greedy:
            samples = torch.argmax(probs, dim=-1)
        else:
            samples = Categorical(probs=probs).sample()
        return samples.detach().cpu().numpy()
    
    def loss_fn(self, obs, goals, actions, horizons):
        actions = torch.tensor(actions, dtype=torch.int32, device = self.device)
        obs = torch.tensor(obs, dtype=torch.float32, device = self.device)
        goals = torch.tensor(goals, dtype=torch.float32, device = self.device)
        horizons = torch.tensor(horizons, dtype=torch.float32, device = self.device)

        embed = torch.cat((obs, goals), dim=1).detach()
        #logits = self.policy.forward(obs, goals, actions, horizon=horizons)
        logits = self.policy.forward(embed)
        #print(logits)
        one_hot_mask = torch.eye(self.n_actions, dtype=torch.bool, device = self.device)[actions]
        nll = torch.logsumexp(logits, dim=1) - logits.masked_select(one_hot_mask)
        return torch.mean(nll)
    
    def validation_loss(self):
        buffer = self.validation_buffer

        if buffer is None or buffer.cur_buffer_size == 0:
            return 0

        avg_loss = 0
        
        observations, actions, goals, lengths, horizons = buffer.sample_batch(self.batch_size)
        loss = self.loss_fn(observations, goals, actions, horizons)
        avg_loss += loss.detach().cpu().numpy()

        return avg_loss
    
    def train(self):
        self.policy.eval()
        start_time = time.time()
        last_time = start_time

        # Evaluate untrained policy
        total_timesteps = 0
        timesteps_since_eval = 0
        running_loss = None
        running_validation_loss = None

        self.evaluate_policy(self.eval_episodes, greedy=True, prefix='Eval')
        with tqdm.tqdm(total=5e3, smoothing=0) as ranger:
            while total_timesteps < self.max_timesteps:

                # Interact in environment according to exploration strategy.
                if total_timesteps < self.explore_timesteps:
                    states, actions, goal_state = self.sample_trajectory(noise=1)
                else:
                    states, actions, goal_state = self.sample_trajectory(greedy=True, noise=self.expl_noise)

                # With some probability, put this new trajectory into the validation buffer
                if self.validation_buffer is not None and np.random.rand() < 0.2:
                    self.validation_buffer.add_trajectory(states, actions, goal_state)
                else:
                    self.buffer.add_trajectory(states, actions, goal_state)

                total_timesteps += self.max_trajectory_length
                timesteps_since_eval += self.max_trajectory_length
                
                ranger.update(self.max_trajectory_length)
                
                # Take training steps
                if  total_timesteps > self.start_policy_timesteps:
                    self.policy.train()
                    for _ in range(int(self.max_trajectory_length)):
                        loss = self.take_policy_step()
                        validation_loss = self.validation_loss()
                        if running_loss is None:
                            running_loss = loss
                        else:
                            running_loss = 0.9 * running_loss + 0.1 * loss

                        if running_validation_loss is None:
                            running_validation_loss = validation_loss
                        else:
                            running_validation_loss = 0.9 * running_validation_loss + 0.1 * validation_loss

                    self.policy.eval()
                    print('Loss: %s Validation Loss: %s'%(running_loss, running_validation_loss))

                
                # Evaluate, log, and save to disk
                if timesteps_since_eval >= 5e3:
                    timesteps_since_eval %= 5e3
                    # Evaluation Code
                    self.policy.eval()
                    self.evaluate_policy(self.eval_episodes, greedy=True, prefix='Eval')
                    print('policy loss', running_loss or 0) # Handling None case
                    print('timesteps', total_timesteps)
                    print('epoch time (s)', time.time() - last_time)
                    print('total time (s)', time.time() - start_time)
                    
                    # Logging Code
                    torch.save(
                            self.policy.state_dict(),
                            'models/gcsl'
                        )

                
    def evaluate_policy(self, eval_episodes=200, greedy=True, prefix='Eval'):
        
        all_states = []
        all_goal_states = []
        all_actions = []
        final_dist_vec = np.zeros(eval_episodes)
        success_vec = np.zeros(eval_episodes)

        for index in tqdm.trange(eval_episodes, leave=True):
            states, actions, goal_state = self.sample_trajectory(noise=0, greedy=greedy)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_state)
            final_dist = np.linalg.norm(states[-1][..., [0,1,]] - goal_state[..., [0,1,]], axis=-1)
            
            final_dist_vec[index] = final_dist
            success_vec[index] = (final_dist < 0.05)

        all_states = np.stack(all_states)
        all_goal_states = np.stack(all_goal_states)

        print('%s num episodes'%prefix, eval_episodes)
        print('%s avg final dist'%prefix,  np.mean(final_dist_vec))
        print('%s success ratio'%prefix, np.mean(success_vec))
        
        return all_states, all_goal_states
    

    def load_models(self, name, dev='cpu'):
        self.policy.load_state_dict(torch.load('models/' + name, map_location=torch.device(dev)))

    def visualize(self, ep=4):
        base_goal = np.array([0, 0, 0, 0, 0, 0, 1, 1,])

        goal = base_goal[..., [0,1,4,6,7]]

        for _ in range(ep):
            s = self.env.reset()
            it = 0
            done = False
            self.policy.eval()
            while not done:
                it+=1
                self.env.render()
                action = self.act_vectorized([s], [goal])
                for _ in range(1):
                    s_, rew, done, _ = self.env.step(action[0])
                s = s_
            print('final state: ', s[np.array([0,1,6,7])])
        self.env.close()