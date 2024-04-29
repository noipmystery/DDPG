import torch
import numpy as np
import models
import torch.nn.functional as F


class DDPG:
    def __init__(self, state_dim, action_dim, action_bound, hidden_1, hidden_2,
                 sigma, actor_lr, critic_lr, weight_decay, gamma, tau):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = models.Actor(state_dim, action_dim, action_bound, hidden_1, hidden_2).to(self.device)
        self.target_actor = models.Actor(state_dim, action_dim, action_bound, hidden_1, hidden_2).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic = models.Critic(state_dim, action_dim, hidden_1, hidden_2).to(self.device)
        self.target_critic = models.Critic(state_dim, action_dim, hidden_1, hidden_2).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay)
        self.action_dim = action_dim
        self.sigma = sigma
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        action = self.actor(state).item()
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, target, net):
        for target_param, param in zip(target.parameters(), net.parameters()):
            target_param.data.copy_(target_param * (1.0 - self.tau) + param.data * self.tau)

    def learn(self, states, actions, rewards, next_states, dones):
        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        default = rewards + self.gamma * next_q_values
        q_targets = torch.where(dones, rewards, default).detach()
        q_values = self.critic(states, actions)

        critic_loss = torch.mean(F.mse_loss(q_targets, q_values))
        loss = critic_loss.item()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, actions))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)
        return loss
