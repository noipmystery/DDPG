from pyparsing import actions

from utils.replay import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import torch


class Trainer:
    def __init__(self, agent, env, args, print_interval=1000):
        self.agent = agent
        self.buffer = ReplayBuffer(args.memory_size)
        self.print_interval = print_interval
        self.replay_start_size = args.replay_start_size
        self.batch_size = args.batch_size
        self.total = args.total_steps
        self.env = env
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y %m %d %H %M %S")
        log = f'./logs/{formatted_time}/{args.env_name}'
        self.writer = SummaryWriter(log_dir=log)

    def train(self):
        frame = self.env.reset()[0]
        total_reward = 0
        loss = 0
        rewards = []
        episodes = 0
        for _ in range(self.total):
            state = self.buffer.transform(frame)
            action = self.agent.select_action(state)
            next_frame, reward, done, info, tmp = self.env.step(action)
            self.buffer.push(frame, action, reward, next_frame, done)
            total_reward += reward
            frame = next_frame

            if len(self.buffer) > self.replay_start_size:
                states, actions, next_states, rewards, dones = self.buffer.sample(self.batch_size)
                print(actions.shape, states.shape)
                loss = self.agent.learn(states, actions, next_states, rewards, dones)

            if _ % self.print_interval == 0 and len(rewards) > 0:
                print('frame : {}, loss : {:.8f}, reward : {}'.format(_, loss, np.mean(rewards[-10:])))
                self.writer.add_scalar('loss', loss, _)
                self.writer.add_scalar('reward', np.mean(rewards[-10:]), _)
                # writer.add_scalar('epsilon', eps, _)

            if done:
                episodes += 1
                rewards.append(total_reward)
                print('episode {}: total reward {}'.format(episodes, total_reward))
                frame = self.env.reset()[0]
                total_reward = 0

            if _ % 100 == 0:
                torch.cuda.empty_cache()
