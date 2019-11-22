from comet_ml import Experiment
import os
from arguments import get_args
import gym
import numpy as np
import gym_ergojr
import torch
from common.agent.ppo_agent import PPO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

experiment = Experiment(
            api_key='ZfKpzyaedH6ajYSiKmvaSwyCs',
            project_name='nas-v2',
            workspace='fgolemo')
args = get_args()
env_name = args.env_name
render = False
solved_reward = 300         # stop training if avg_reward > solved_reward
log_interval = 20           # print avg reward in the interval
max_episodes = 5000        # max training episodes
max_timesteps = 1500        # max timesteps in one episode

update_timestep = 4000      # update policy every n timesteps
action_std = 0.5            # constant std for action distribution (Multivariate Normal)
K_epochs = 80               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr = 0.0003                 # parameters for Adam optimizer
betas = (0.9, 0.999)

# jobid = os.environ['SLURM_ARRAY_TASK_ID']
# random_seed = args.seed + int(jobid)

# Create Env Variable
os.environ['noise_type'] = args.noise_type
os.environ['approach'] = args.approach
os.environ['variant'] = args.variant


random_seed = args.seed
folder_path = os.getcwd() + '/{}/Variant-{}/'.format(args.approach, args.variant)
file_path = folder_path + '/ppo_{}_{}_{}_{}_{}.pth'.format(args.env_name,
                                                           args.noise_type,
                                                           args.variant,
                                                           args.approach,
                                                           random_seed)
if not os.path.isdir(folder_path):
    os.makedirs(folder_path)
#############################################

# creating environment
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

if random_seed:
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

memory = Memory()
ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
print(lr, betas)

# logging variables
running_reward = 0
avg_length = 0
time_step = 0

# training loop
for i_episode in range(1, max_episodes +1):
    state = env.reset()
    for t in range(max_timesteps):
        time_step += 1
        # Running policy_old:
        action = ppo.select_action(state, memory)
        state, reward, done, _ = env.step(action)
        # Saving reward:
        memory.rewards.append(reward)

        # update if its time
        if time_step % update_timestep == 0:
            ppo.update(memory)
            memory.clear_memory()
            time_step = 0
        running_reward += reward
        if render:
            env.render()
        if done:
            break

    avg_length += t

    # stop training if avg_reward > solved_reward
    if running_reward > (log_interval *solved_reward):
        print("########## Solved! ##########")
        torch.save(ppo.policy.state_dict(), file_path)
        break

    # save every 500 episodes
    if i_episode % 500 == 0:
        torch.save(ppo.policy.state_dict(), file_path)

    # logging
    if i_episode % log_interval == 0:
        avg_length = int(avg_length /log_interval)
        running_reward = int((running_reward /log_interval))
        experiment.log_metric("Reward Mean", np.mean(running_reward), step=i_episode)
        experiment.add_tag(f'{args.approach}')
        experiment.add_tag(f'{args.variant} - {args.env_name}')

        print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
        running_reward = 0
        avg_length = 0
