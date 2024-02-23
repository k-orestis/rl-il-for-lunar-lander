import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from networks import mlp as mlp
from ppoAgent import PPOAgent

# CODE AND HYPERPARAMETERS BASED ON
# GAIL IMPLEMENTATION IN https://github.com/mitre/ilpyt.git

class GAILAgent:
    def __init__(self, env, expert_demos, gen = None, device = 'cpu', 
                 rollout_steps = 64,  lr = 0.00065):
        
        self.env = env
        self.num_envs = len(env)
        self.device = device
        self.n_actions = self.env[0].action_space.n
        self.state_space = self.env[0].observation_space.shape[0]
        self.expert_demos = expert_demos
        self.rollout_steps = rollout_steps
        #self.batch_size = batch_size

        if gen == None:
            self.gen = PPOAgent(self.env, num_env=len(self.env), rollout_steps=rollout_steps, layers=[128,128,128], device=self.device)
        else:
            self.gen = gen
        self.disc = mlp(input_dim= self.state_space+self.n_actions, output_dim=1, hidden_layers=[128,128, 128], act = 'tanh')
        self.opt_disc = torch.optim.Adam(self.disc.parameters(), lr)
        TD = TensorDataset(expert_demos['states'].to(self.device), expert_demos['actions'].to(self.device))
        self.dataloader = DataLoader(TD, batch_size=self.num_envs * rollout_steps, shuffle=True)
        self.disc.to(self.device)
   
    @torch.no_grad()
    def step(self, state):
        return self.gen.step(state)

    def update(self, batch):
        """
        Update agent policy based on batch of experiences.

        Parameters
        ----------
        batch: Dict[str, torch.Tensor]
            batch of transitions, with keys `states`, `actions`, 
            `expert_states`, and `expert_actions`. Values should be of size 
            (num_steps, num_env, item_shape)

        Returns
        -------
        Dict[str, float]:
            losses for the update step, key strings and loss values can be 
            automatically recorded to TensorBoard
        """
        # Rewards
        rollout_steps = batch['states'].shape[0]
        with torch.no_grad():
            rewards = []
            for i in range(rollout_steps):
                a = F.one_hot(
                batch['actions'][i].to(torch.int64), num_classes=self.n_actions)
                xa = torch.cat([batch['states'][i], a], dim=-1)
                logits = torch.sigmoid(self.disc(xa))
                reward = -torch.log(logits)
                rewards.append(reward.squeeze())
            rewards = torch.stack(rewards)

        # Update discriminator
        actions = F.one_hot(
            batch['actions'].view(rollout_steps*self.num_envs).to(torch.int64), num_classes=self.n_actions)
        expert_actions = F.one_hot(
            batch['expert_actions'].to(torch.int64), num_classes=self.n_actions)
        #print(actions.shape, expert_actions.shape)
        
        learner_logits = self.disc(
            torch.cat((batch['states'].view(rollout_steps*self.num_envs, -1), actions), dim = -1), 
        ).squeeze()
        
        expert_logits = self.disc(
            torch.cat((batch['expert_states'], expert_actions), dim = -1)
        ).squeeze()
        
        loss_disc = F.binary_cross_entropy_with_logits(
            learner_logits, torch.ones_like(learner_logits)
        ) + F.binary_cross_entropy_with_logits(
            expert_logits, torch.zeros_like(expert_logits)
        )
        
        self.opt_disc.zero_grad()
        
        loss_disc.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.disc.parameters(), 1.5)
        self.opt_disc.step()

        # Update generator
        batch['rewards'] = rewards
        loss_gen_dict = self.gen.update(batch)

        # Return loss dictionary
        loss_dict = {
            'loss/disc': loss_disc.item(),
            'loss/gen': loss_gen_dict['loss/total'],
            'loss/total': loss_disc.item() + loss_gen_dict['loss/total'],
        }
        return loss_dict

    def train(self, num_episodes = 10000):
        expert_gen = iter(self.dataloader)

        # Start training
        self.reward_tracker = -np.inf * np.ones(self.num_envs)
        self.gen.actor.train()
        self.gen.critic.train()
        self.disc.train()
        step = 0
        ep_count = 0

        #self.agent.save(self.save_path, 0, keep=num_save)
        #logging.info("Save initial model at episode %i:%i." % (ep_count, step))

        while ep_count < num_episodes:
            # Step agent and environment
            batch = self.gen.generate_batch(self.rollout_steps)
            step += self.rollout_steps

            # Get expert rollouts
            try:
                expert_states, expert_actions = next(expert_gen)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                expert_gen = iter(self.dataloader)
                expert_states, expert_actions = next(expert_gen)
            batch['expert_states'] = expert_states
            batch['expert_actions'] = expert_actions

            # Update agent
            loss_dict = self.update(batch)
            if step % 200 == 0:
                print(ep_count, loss_dict) 

            for ep_count, info_dict in batch['infos']:
                for (k, v) in info_dict.items():
                    if 'reward' in k:
                        agent_num = int(k.split('/')[1])
                        self.reward_tracker[agent_num] = v

                # Save agent
                if ep_count % 200 == 0:
                    torch.save(self.gen.actor.state_dict(), "models/gen_actor")
                    torch.save(self.gen.critic.state_dict(), "models/gen_critic")
                    torch.save(self.disc.state_dict(), "models/disc")
                    print(
                        "Save current model at episode %i:%i."
                        % (ep_count, step)
                    )

    def load_models(self, name, dev = 'cpu'):
        self.disc.load_state_dict(torch.load('models/'+name+'/disc', map_location=dev))
        self.gen.load_models(name, 'cpu')
    
    def visualize(self, ep=4):
        return self.gen.visualize(ep)