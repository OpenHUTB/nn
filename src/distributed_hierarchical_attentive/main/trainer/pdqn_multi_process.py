import logging
import torch
import datetime, time, os
import random, collections
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from copy import deepcopy
from collections import deque
from algs.pdqn import P_DQN
from tensorboardX import SummaryWriter
from multiprocessing import Process, Queue, Pipe, connection
from gym_carla.multi_lane.settings import ARGS
from gym_carla.multi_lane.carla_env import CarlaEnv
from main.util.process import start_process, kill_process
from gym_carla.multi_lane.util.wrapper import fill_action_param, recover_steer, Action

#
# neural network hyper parameters
SIGMA = 0.5
SIGMA_STEER = 0.3
SIGMA_ACC = 0.5
THETA = 0.05
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LR_ACTOR = 0.0002
LR_CRITIC = 0.0002
GAMMA = 0.9  # q值更新系数
TAU = 0.01  # 软更新参数
EPSILON = 0.5  # epsilon-greedy
BUFFER_SIZE = 160000
MINIMAL_SIZE = 10000
BATCH_SIZE = 128
REPLACE_A = 500
REPLACE_C = 300
TOTAL_EPISODE = 7000
SIGMA_DECAY = 0.9999
PER_FLAG = True
modify_change_steer = False
clip_grad = 10
zero_index_gradients = True
inverting_gradients = True
base_name = f'origin_NOCA'
time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
SAVE_PATH = f"./out/multi_agent/pdqn/{time}"


def main():
    Args = ARGS.parse_args()
    log_level = logging.DEBUG if Args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    # env=gym.make('CarlaEnv-v0')
    env = CarlaEnv(Args)
    globals()['modify_change_steer'] = Args.modify_change_steer

    done = False
    truncated = False

    random.seed(0)
    torch.manual_seed(16)
    s_dim = env.get_observation_space()
    a_bound = env.get_action_bound()
    a_dim = 2

    episode_writer = SummaryWriter(SAVE_PATH)
    n_run = 3
    rosiolling_window = 100  # 100 car following events, average score
    result = []

    for run in [base_name]:
        worker_agent = P_DQN(deepcopy(s_dim), a_dim, a_bound, GAMMA, TAU, SIGMA_STEER, SIGMA, SIGMA_ACC, THETA, EPSILON,
                             BUFFER_SIZE, BATCH_SIZE, LR_ACTOR,
                             LR_CRITIC, clip_grad, zero_index_gradients, inverting_gradients, PER_FLAG,
                             torch.device('cpu'))
        # learner_agent = P_DQN(deepcopy(s_dim), a_dim, a_bound, GAMMA, TAU, SIGMA_STEER, SIGMA, SIGMA_ACC, THETA, EPSILON, BUFFER_SIZE, BATCH_SIZE, LR_ACTOR,
        #              LR_CRITIC, clip_grad, zero_index_gradients, inverting_gradients,PER_FLAG, DEVICE)

        # multi-process training
        process = list()
        traj_q = Queue(maxsize=BUFFER_SIZE)
        agent_q = Queue(maxsize=3)
        learner_q = Queue(maxsize=1)
        traj_send, traj_recv = Pipe()
        agent_send, agent_recv = Pipe()
        mp.set_start_method(method='spawn', force=True)  # force all the multiprocessing to 'spawn' methods
        # process.append(mp.Process(target=learner_mp,args=(traj_recv,agent_send,(deepcopy(s_dim), a_dim, a_bound),Args.ego_num)))
        process.append(mp.Process(target=learner_mp, args=(traj_q, agent_q, (deepcopy(s_dim), a_dim, a_bound))))
        # process.append(mp.Process(target=learner_mp,args=(traj_q,agent_q,learner_q,(deepcopy(s_dim), a_dim, a_bound))))
        [p.start() for p in process]

        # training part
        max_rolling_score = np.float32('-5')
        max_score = np.float32('-30')
        var = 3
        collision_train = 0
        learn_time = 0
        episode_score = []
        rolling_score = []
        cum_collision_num = []

        score_safe = []
        score_efficiency = []
        score_comfort = []

        try:
            for i in range(10):
                with tqdm(total=TOTAL_EPISODE // 10, desc="Iteration %d" % i) as pbar:
                    for i_episode in range(TOTAL_EPISODE // 10):
                        state = env.reset()
                        worker_agent.reset_noise()
                        score = 0
                        ttc, efficiency, comfort, lcen, yaw, impact, lane_change_reward = 0, 0, 0, 0, 0, 0, 0  # part objective scores

                        while not done and not truncated:
                            # if agent_recv.poll():
                            #     a,a_t,c,c_t=agent_recv.recv()
                            #     worker_agent.actor.load_state_dict(a)
                            #     worker_agent.actor_target.load_state_dict(a_t)
                            #     worker_agent.critic.load_state_dict(c)
                            #     worker_agent.critic_target.load_state_dict(c_t)
                            if not agent_q.empty():
                                ac, actor_t, cr, critic_t = worker_agent.actor.state_dict(), worker_agent.actor_target.state_dict(), \
                                    worker_agent.critic.state_dict(), worker_agent.critic_target.state_dict()
                                # temp_agent,learn_time=agent_q.get()
                                # a,c=temp_agent.actor.state_dict(),temp_agent.critic.state_dict()
                                # worker_agent.actor.load_state_dict(temp_agent.actor.state_dict())
                                # worker_agent.critic.load_state_dict(temp_agent.critic.state_dict())
                                actor, critic, learn_time = agent_q.get()
                                worker_agent.actor = actor
                                worker_agent.critic = critic

                            action, action_param, all_action_param = worker_agent.take_action(state)
                            next_state, reward, truncated, done, info = env.step(action, action_param)
                            if env.is_effective_action() and not info['Abandon']:
                                throttle_brake = -info['Brake'] if info['Brake'] > 0 else info['Throttle']
                                if info['Change'] == Action.LANE_FOLLOW:
                                    action = 1
                                elif info['Change'] == Action.LANE_CHANGE_LEFT:
                                    action = 0
                                elif info['Change'] == Action.LANE_CHANGE_RIGHT:
                                    action = 2
                                # action_param = np.array([[info['Steer'], throttle_brake]])
                                saved_action_param = fill_action_param(action, info['Steer'], throttle_brake,
                                                                       all_action_param, modify_change_steer)
                                print(f"Control In Replay Buffer: {action}, {saved_action_param}")
                                # traj_send.send((j,states[j],next_states[j],all_action_params[j],
                                #      rewards[j],truncateds[j],dones[j],infos[j]))
                                # if not traj_q.full():
                                traj_q.put((deepcopy(state), deepcopy(next_state), deepcopy(action),
                                            deepcopy(saved_action_param), deepcopy(reward),
                                            deepcopy(truncated), deepcopy(done), deepcopy(info)), block=True,
                                           timeout=None)

                                print(
                                    f"state -- vehicle_info:{state['vehicle_info']}\n"
                                    # f"waypoints:{state['left_waypoints']}, \n"
                                    # f"waypoints:{state['center_waypoints']}, \n"
                                    # f"waypoints:{state['right_waypoints']}, \n"
                                    f"ego_vehicle:{state['ego_vehicle']}, \n"
                                    f"light info: {state['light']}\n"
                                    f"next_state -- vehicle_info:{next_state['vehicle_info']}\n"
                                    # f"waypoints:{next_state['left_waypoints']}, \n"
                                    # f"waypoints:{next_state['center_waypoints']}, \n"
                                    # f"waypoints:{next_state['right_waypoints']}, \n"
                                    f"ego_vehicle:{next_state['ego_vehicle']}\n"
                                    f"light info: {next_state['light']}\n"
                                    f"action:{action}, action_param:{action_param} \n"
                                    f"all_action_param:{all_action_param}, saved_action_param:{saved_action_param}\n"
                                    f"reward:{reward}, truncated:{truncated}, done:{done}, learn_time:{learn_time}")
                                print()

                            state = next_state

                            # only record the first vehicle reward
                            if env.total_step == Args.pre_train_steps:
                                worker_agent.save_net(f"{SAVE_PATH}/pdqn_pre_trained.pth")
                            if env.is_effective_action() and not info['Abandon']:
                                score += reward
                                if not truncated:
                                    ttc += info['fTTC']
                                    efficiency += info['Efficiency']
                                    comfort += info['Comfort']
                                    lcen += info['Lane_center']
                                    yaw += info['Yaw']
                                    impact += info['impact']
                                    lane_change_reward += info['lane_changing_reward']

                            if env.rl_control_step > 10000 and env.is_effective_action() and \
                                    env.RL_switch and SIGMA_ACC > 0.1:
                                globals()['SIGMA'] *= SIGMA_DECAY
                                globals()['SIGMA_STEER'] *= SIGMA_DECAY
                                globals()['SIGMA_ACC'] *= SIGMA_DECAY
                                worker_agent.set_sigma(SIGMA_STEER, SIGMA_ACC)
                                logging.info("Agent Sigma %f %f", SIGMA_STEER, SIGMA_ACC)

                        if done or truncated:
                            # restart the training
                            done = False
                            truncated = False

                        # record episode results
                        if env.RL_switch:
                            episode_writer.add_scalar('Total_Reward', score, i * (TOTAL_EPISODE // 10) + i_episode)
                            score /= env.time_step + 1
                            episode_writer.add_scalar('Avg_Reward', score, i * (TOTAL_EPISODE // 10) + i_episode)
                            episode_writer.add_scalar('Time_Steps', env.time_step,
                                                      i * (TOTAL_EPISODE // 10) + i_episode)
                            episode_writer.add_scalar('TTC', ttc / (env.time_step + 1),
                                                      i * (TOTAL_EPISODE // 10) + i_episode)
                            episode_writer.add_scalar('Efficiency', efficiency / (env.time_step + 1),
                                                      i * (TOTAL_EPISODE // 10) + i_episode)
                            episode_writer.add_scalar('Comfort', comfort / (env.time_step + 1),
                                                      i * (TOTAL_EPISODE // 10) + i_episode)
                            episode_writer.add_scalar('Lcen', lcen / (env.time_step + 1),
                                                      i * (TOTAL_EPISODE // 10) + i_episode)
                            episode_writer.add_scalar('Yaw', yaw / (env.time_step + 1),
                                                      i * (TOTAL_EPISODE // 10) + i_episode)
                            episode_writer.add_scalar('Impact', impact / (env.time_step + 1),
                                                      i * (TOTAL_EPISODE // 10) + i_episode)
                            episode_writer.add_scalar('Lane_change_reward', lane_change_reward / (env.time_step + 1),
                                                      i * (TOTAL_EPISODE // 10) + i_episode)

                            episode_score.append(score)
                            score_safe.append(ttc)
                            score_efficiency.append(efficiency)
                            score_comfort.append(comfort)
                            # rolling_score.append(np.mean(episode_score[max]))
                            cum_collision_num.append(collision_train)

                            if max_score < score:
                                max_score = score
                                worker_agent.save_net(F"{SAVE_PATH}/pdqn_optimal.pth")

                        """ if rolling_score[rolling_score.__len__-1]>max_rolling_score:
                            max_rolling_score=rolling_score[rolling_score.__len__-1]
                            agent.save_net() """

                        # result.append([episode_score,rolling_score,cum_collision_num,score_safe,score_efficiency,score_comfort])
                        if (i_episode + 1) % 10 == 0:
                            pbar.set_postfix({
                                'episodes': '%d' % (TOTAL_EPISODE / 10 * i + i_episode + 1),
                                'score': '%.2f' % score
                            })
                        pbar.update(1)
                        worker_agent.save_net(f"{SAVE_PATH}/pdqn_final.pth")

            np.save(f"{SAVE_PATH}/result_{run}.npy", result)
        except KeyboardInterrupt:
            logging.info("Premature Terminated")
        # except BaseException as e:
        #      logging.info(e.args)
        finally:
            env.__del__()
            # process[-1].join() # waiting for learner
            episode_writer.close()
            worker_agent.save_net(f"{SAVE_PATH}/pdqn_final.pth")
            # process[-1].join()
            process_safely_terminate(process)
            logging.info('\nDone.')


def process_safely_terminate(process: list):
    for p in process:
        try:
            p.kill()
        except OSError as e:
            logging.SystemError(e)


# Pipe version multiprocess
# def learner_mp(traj_recv:connection.Connection, agent_send:connection.Connection, agent_param, ego_num):
#     learner_agent=P_DQN(agent_param[0], agent_param[1], agent_param[2], GAMMA, TAU, SIGMA_STEER, SIGMA, SIGMA_ACC, THETA, EPSILON, BUFFER_SIZE, BATCH_SIZE, LR_ACTOR,
#                      LR_CRITIC, clip_grad, zero_index_gradients, inverting_gradients,PER_FLAG, DEVICE)
#     impact_deques=[deque(maxlen=2) for _ in range(ego_num)]
#     while(True):
#         if traj_recv.poll(timeout=None):
#             trajectory=traj_recv.recv()
#             ego_id, state, next_state, all_action_param, reward, truncated, done, info=trajectory[0],trajectory[1],trajectory[2],trajectory[3],\
#                 trajectory[4],trajectory[5],trajectory[6],trajectory[7]
#             replay_buffer_adder(learner_agent,impact_deques[ego_id],state,next_state,all_action_param,reward,truncated,done,info)

#         if learner_agent.replay_buffer.size()>=MINIMAL_SIZE:
#             logging.info("LEARN BEGIN")
#             learner_agent.learn()
#             if learner_agent.learn_time!=0 and learner_agent.learn_time%2==0:
#                 actor,actor_t,critic,critic_t=learner_agent.actor.state_dict(),learner_agent.actor_target.state_dict(), \
#                     learner_agent.critic.state_dict(),learner_agent.critic_target.state_dict()
#                 a,a_t,c,c_t=deepcopy(learner_agent.actor.state_dict()),deepcopy(learner_agent.actor_target.state_dict()),\
#                     deepcopy(learner_agent.critic.state_dict()),deepcopy(learner_agent.critic_target.state_dict())
#                 agent_send.send((a,a_t,c,c_t))

# Queue vesion multiprocess
def learner_mp(traj_q: Queue, agent_q: Queue, agent_param):
    learner_agent = P_DQN(deepcopy(agent_param[0]), agent_param[1], agent_param[2], GAMMA, TAU, SIGMA_STEER, SIGMA,
                          SIGMA_ACC, THETA, EPSILON, BUFFER_SIZE, BATCH_SIZE, LR_ACTOR,
                          LR_CRITIC, clip_grad, zero_index_gradients, inverting_gradients, PER_FLAG, DEVICE)
    temp_agent = P_DQN(deepcopy(agent_param[0]), agent_param[1], agent_param[2], GAMMA, TAU, SIGMA_STEER, SIGMA,
                       SIGMA_ACC, THETA, EPSILON, BUFFER_SIZE, BATCH_SIZE, LR_ACTOR,
                       LR_CRITIC, clip_grad, zero_index_gradients, inverting_gradients, PER_FLAG, torch.device('cpu'))
    impact_deque = deque(maxlen=2)
    pid = os.getpid()
    actor, actor_t, critic, critic_t = None, None, None, None
    a, a_t, c, c_t = None, None, None, None
    while (True):
        # alter the batch_size and update times according to the replay buffer size:
        # reference: https://zhuanlan.zhihu.com/p/345353294, https://arxiv.org/abs/1711.00489
        k = max(learner_agent.replay_buffer.size() // MINIMAL_SIZE, 1)
        learner_agent.batch_size = k * BATCH_SIZE
        for _ in range(BATCH_SIZE):
            trajectory = traj_q.get(block=True, timeout=None)
            state, next_state, action, saved_action_param, reward, truncated, done, info = trajectory[0], trajectory[1], \
            trajectory[2], trajectory[3], \
                trajectory[4], trajectory[5], trajectory[6], trajectory[7]
            replay_buffer_adder(learner_agent, impact_deque, state, next_state, action, saved_action_param, reward,
                                truncated, done, info)
        if learner_agent.replay_buffer.size() >= MINIMAL_SIZE:
            logging.info("LEARN BEGIN")
            # print(f"LEARN TIME:{learner_agent.learn_time}")
            [learner_agent.learn() for _ in range(k)]
            if not agent_q.full():
                actor = deepcopy(learner_agent.actor).to('cpu')
                actor_t = deepcopy(learner_agent.actor_target).to('cpu')
                critic = deepcopy(learner_agent.critic).to('cpu')
                critic_t = deepcopy(learner_agent.critic_target).to('cpu')
                temp_agent.actor.load_state_dict(learner_agent.actor.state_dict())
                temp_agent.critic.load_state_dict(learner_agent.critic.state_dict())
                # actor,actor_t,critic,critic_t=learner_agent.actor.state_dict(),learner_agent.actor_target.state_dict(), \
                #     learner_agent.critic.state_dict(),learner_agent.critic_target.state_dict()
                # a,a_t,c,c_t=temp_agent.actor.state_dict(),temp_agent.actor_target.state_dict(), \
                #     temp_agent.critic.state_dict(),temp_agent.critic_target.state_dict()
                agent_q.put((actor, critic, learner_agent.learn_time), block=True, timeout=None)
                # agent_q.put((temp_agent,learner_agent.learn_time),block=True,timeout=None)


def replay_buffer_adder(agent, impact_deque, state, next_state, action, saved_action_param, reward, truncated, done,
                        info):
    """Input all the state info into agent's replay buffer"""
    if 'Throttle' in info:
        if info['control_state']:
            # under rl control
            if truncated:
                agent.store_transition(state, action, saved_action_param, reward, next_state,
                                       truncated, done, info)
            else:
                impact = info['impact'] / 9
                impact_deque.append([state, action, saved_action_param, reward, next_state,
                                     truncated, done, info])
                if len(impact_deque) == 2:
                    experience = impact_deque[0]
                    agent.store_transition(experience[0], experience[1], experience[2],
                                           experience[3] + impact, experience[4], experience[5],
                                           experience[6], experience[7])
                # agent.replay_buffer.add(state, action, saved_action_param, reward, next_state,
                #                         truncated, done, info)
        else:
            # Input the guided action to replay buffer
            if truncated:
                agent.store_transition(state, action, saved_action_param, reward, next_state,
                                       truncated, done, info)
            else:
                impact = info['impact'] / 9
                impact_deque.append([state, action, saved_action_param, reward, next_state,
                                     truncated, done, info])
                if len(impact_deque) == 2:
                    experience = impact_deque[0]
                    agent.store_transition(experience[0], experience[1], experience[2],
                                           experience[3] + impact, experience[4], experience[5],
                                           experience[6], experience[7])
                # agent.replay_buffer.add(state, action, saved_action_param, reward, next_state,
                #                         truncated, done, info)
    # else:
    #     # not work
    #     # Input the agent action to replay buffer
    #     agent.replay_buffer.add(state, action, all_action_param, reward, next_state, truncated, done, info)


if __name__ == '__main__':
    try:
        start_process()
        main()
    # except BaseException as e:
    #     logging.warning(e.args)
    finally:
        kill_process()
