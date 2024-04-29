from pyparsing import actions

from utils.replay import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import torch


class Trainer:
    def __init__(self, agent, env, args):
        self.agent = agent
        self.buffer = ReplayBuffer(args.memory_size)
        self.episodes = args.episodes
        self.iterations = args.iterations
        self.replay_start_size = args.replay_start_size
        self.batch_size = args.batch_size
        self.env = env
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y %m %d %H %M %S")
        log = f'./logs/{args.env_name}/{formatted_time}'
        self.writer = SummaryWriter(log_dir=log)

    def train(self):
        for episode in range(self.episodes):
            frame = self.env.reset()[0]
            total_reward = 0
            for _ in range(self.iterations):
                state = self.buffer.transform(frame)
                action = self.agent.select_action(state)
                next_frame, reward, done, info, tmp = self.env.step(action)
                action = torch.tensor(action, dtype=torch.float)
                self.buffer.push(frame, action, reward, next_frame, done)
                total_reward += reward
                frame = next_frame

                if len(self.buffer) > self.replay_start_size:
                    states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
                    loss = self.agent.learn(states, actions, rewards, next_states, dones)
                    if _ % 50 == 0:
                        print(f'Episode {episode}, iteration {_}, loss {loss}')

                if _ % 100 == 0:
                    torch.cuda.empty_cache()

            self.writer.add_scalar('reward', total_reward, episode)
            print(f'Episode {episode}, Total Reward: {total_reward}')