import argparse
import torch
from ddpg import DDPG
from utils.environment import make_env
from trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Pendulum-v1', help='Environment name')
    parser.add_argument('--hidden1', type=int, default=400, help='Size of first hidden layer')
    parser.add_argument('--hidden2', type=int, default=300, help='Size of second hidden layer')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma hyperparameter')
    parser.add_argument('--tau', type=float, default=1e-3, help='Tau hyperparameter')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--actor_lr', type=float, default=1e-4, help='Actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='Critic learning rate')
    parser.add_argument('--init_w', type=float, default=3e-3, help='Initial final layer weights')
    parser.add_argument('--sigma', type=float, default=0.2, help='Sigma hyperparameter')
    parser.add_argument('--memory_size', type=int, default=1000000, help='Memory size')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--total_steps', type=int, default=2000000, help='Total timesteps')
    parser.add_argument('--replay_start_size', type=int, default=10000, help='Replay start')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = parser.parse_args()
    env = make_env(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    print(state_dim, action_dim, action_bound)
    agent = DDPG(state_dim, action_dim, action_bound, args.hidden1, args.hidden2,
                 args.sigma, args.actor_lr, args.critic_lr, args.weight_decay, args.gamma, args.tau)

    trainer = Trainer(agent, env, args)
    trainer.train()
