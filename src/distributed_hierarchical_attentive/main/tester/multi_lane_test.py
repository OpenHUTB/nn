import logging
import torch
import datetime
import random, os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
from algs.pdqn import P_DQN
from collections import deque
from tensorboardX import SummaryWriter
from gym_carla.multi_lane.settings import ARGS
from gym_carla.multi_lane.carla_env import CarlaEnv, SpeedState
from main.util.process import start_process, kill_process

#
# neural network hyper parameters
SIGMA = 1
SIGMA_STEER = 0.3
SIGMA_ACC = 0.5
THETA = 0.001
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LR_ACTOR = 0.001
LR_CRITIC = 0.002
GAMMA = 0.99  # q值更新系数
TAU = 0.01  # 软更新参数
EPSILON = 0.5  # epsilon-greedy
POLICY_UPDATE_FREQ = 5
BUFFER_SIZE = 20000
MINIMAL_SIZE = 10000
BATCH_SIZE = 128
REPLACE_A = 500
REPLACE_C = 300
TOTAL_EPISODE = 3000
clip_grad = 10
zero_index_gradients = True
inverting_gradients = True
base_name = f'origin_NOCA'
time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
SAVE_PATH = f"./out/multi_lane/pdqn/test/{time}"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


def main():
    ARGS.set_defaults(train=False)
    ARGS.set_defaults(no_rendering=False)
    args = ARGS.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    # env=gym.make('CarlaEnv-v0')
    env = CarlaEnv(args)
    globals()['modify_change_steer'] = args.modify_change_steer

    done = False
    truncated = False

    random.seed(0)
    torch.manual_seed(8)
    s_dim = env.get_observation_space()
    a_bound = env.get_action_bound()
    a_dim = 2

    episode_writer = SummaryWriter(SAVE_PATH)
    result = []

    for run in [base_name]:
        param = torch.load('./out/pdqn_final_6.pth')
        agent = P_DQN(s_dim, a_dim, a_bound, GAMMA, TAU, SIGMA_STEER, SIGMA, SIGMA_ACC, THETA, EPSILON, BUFFER_SIZE,
                      BATCH_SIZE, LR_ACTOR,
                      LR_CRITIC, clip_grad, zero_index_gradients, inverting_gradients, False, DEVICE)
        agent.load_net(param)
        agent.train = False
        env.RL_switch = True
        agent.set_sigma(0, 0)

        VEL = []
        ACC = []
        JERK = []
        OFFLANE = []
        TTC = []
        REAR_VEL = []
        REAR_ACC = []

        try:
            for i in range(30):
                state = env.reset()

                score = 0
                ttc, efficiency, comfort, lcen, yaw, impact, lane_change_reward = 0, 0, 0, 0, 0, 0, 0  # part objective scores
                recover_time, lane_change_count, brake_count, global_brake_count, delay_index, avg_vel, rec_avg_vel, avg_jerk, avg_offlane = 0, 0, 0, 0, 0, 0, 0, 0, 0
                delay_i = deque(maxlen=100)
                rear_a = deque(maxlen=5)
                rear_v = deque(maxlen=200)
                recovery_mode = False
                ego_v = deque(maxlen=5000)
                ego_a = deque(maxlen=5000)
                ego_jerk = deque(maxlen=5000)
                ego_offlane = deque(maxlen=5000)
                ego_ttc = deque(maxlen=5000)
                ego_rear_a = deque(maxlen=5000)
                ego_rear_v = deque(maxlen=5000)
                recover_v = deque(maxlen=500)

                while not done and not truncated:
                    action, action_param, all_action_param = agent.take_action(state)
                    next_state, reward, truncated, done, info = env.step(action, action_param)
                    # if env.speed_state == SpeedState.REBOOT:
                    #     env.speed_state = SpeedState.RUNNING
                    state = next_state
                    print()

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
                    if not truncated and not done:
                        # with open(f"{SAVE_PATH}/ego_ttc.txt",'a') as f:
                        #     f.write(str(info['TTC'])+'\n')
                        # with open(f"{SAVE_PATH}/ego_vel.txt",'a') as f:
                        #     f.write(str(info['velocity'])+'\n')
                        # with open(f"{SAVE_PATH}/ego_acc.txt",'a') as f:
                        #     f.write(str(info['cur_acc'])+'\n')
                        # with open(f"{SAVE_PATH}/ego_jerk.txt",'a') as f:
                        #     f.write(str(abs(info['cur_acc']-info['last_acc'])/(1.0/args.fps))+'\n')
                        # with open(f"{SAVE_PATH}/ego_offlane.txt",'a') as f:
                        #     f.write(str(abs(info['offlane']))+'\n')
                        # with open(f"{SAVE_PATH}/ego_yaw.txt",'a') as f:
                        #     f.write(str(info['yaw_diff'])+'\n')

                        # macro index
                        if info['change_lane'] and info['rear_id'] != -1:
                            recovery_mode = True
                            lane_change_count += 1
                        if info['rear_id'] != -1:
                            if info['rear_a'] < 0:
                                global_brake_count += 1
                            ego_rear_a.append((info['rear_a'], recovery_mode))
                            # with open(f"{SAVE_PATH}/rear_a.txt",'a') as f:
                            #     f.write(str(recovery_mode)+'\t'+str(info['rear_a'])+'\n')
                        if recovery_mode == True:
                            rear_a.append(info['rear_a'])
                            if info['rear_id'] == -1:
                                avg_v = 0
                                for n in range(len(rear_v) - 1):
                                    avg_v += rear_v[n + 1] / (len(rear_v) - 1)
                                ind = rear_v[0] / (avg_v + 0.000001) if rear_v[0] / (avg_v + 0.000001) > 1.0 else 1.0
                                recover_v.append(avg_v)
                                delay_i.append(ind)

                                recovery_mode = False
                                rear_v.clear()
                                rear_a.clear()
                            elif len(rear_a) == rear_a.maxlen:
                                avg_a = 0
                                for a in rear_a:
                                    avg_a += a / rear_a.maxlen
                                if 0 <= avg_a < 0.001:
                                    rear_v.append(info['rear_v'])
                                    avg_v = 0
                                    for n in range(len(rear_v) - 1):
                                        avg_v += rear_v[n + 1] / (len(rear_v) - 1)
                                    ind = rear_v[0] / (avg_v + 0.000001) if rear_v[0] / (
                                                avg_v + 0.000001) > 1.0 else 1.0
                                    delay_i.append(ind)

                                    recovery_mode = False
                                    rear_v.clear()
                                    rear_a.clear()

                            if recovery_mode == True:
                                if info['rear_a'] < 0:
                                    brake_count += 1
                                rear_v.append(info['rear_v'])
                                recover_time += 1

                        # micro index
                        ego_v.append(info['velocity'])
                        ego_a.append(info['cur_acc'])
                        ego_jerk.append(abs(info['cur_acc'] - info['last_acc']) / (1.0 / args.fps))
                        ego_offlane.append(info['offlane'])
                        ego_ttc.append(info['TTC'])
                        ego_rear_v.append(info['rear_v'])

                if done:
                    episode_writer.add_scalar('Pass_Time_Steps', env.time_step, i)
                if done or truncated:
                    # restart the training
                    done = False
                    truncated = False

                VEL.append(np.array(ego_v, dtype=float))
                ACC.append(np.array(ego_a, dtype=float))
                JERK.append(np.array(ego_jerk, dtype=float))
                OFFLANE.append(np.array(ego_offlane, dtype=float))
                TTC.append(np.array(ego_ttc, dtype=float))
                REAR_ACC.append(np.array(ego_rear_a, dtype=float))
                REAR_VEL.append(np.array(ego_rear_v, dtype=float))

                episode_writer.add_scalar('Total_Reward', score, i)
                score /= env.time_step + 1
                episode_writer.add_scalar('Avg_Reward', score, i)
                episode_writer.add_scalar('Time_Steps', env.time_step, i)
                episode_writer.add_scalar('TTC', ttc / (env.time_step + 1), i)
                episode_writer.add_scalar('Efficiency', efficiency / (env.time_step + 1), i)
                episode_writer.add_scalar('Comfort', comfort / (env.time_step + 1), i)
                episode_writer.add_scalar('Lcen', lcen / (env.time_step + 1), i)
                episode_writer.add_scalar('Yaw', yaw / (env.time_step + 1), i)
                episode_writer.add_scalar('Impact', impact / (env.time_step + 1), i)
                episode_writer.add_scalar('Lane_change_reward', lane_change_reward / (env.time_step + 1), i)
                episode_writer.add_scalar('recover_time', recover_time, i)
                episode_writer.add_scalar('lane_change_count', lane_change_count, i)
                episode_writer.add_scalar('brake_count', brake_count, i)
                episode_writer.add_scalar('global_brake_count', brake_count, i)
                for index in delay_i:
                    delay_index += index / len(delay_i)
                delay_index = delay_index if delay_index > 1.0 else 1.0
                episode_writer.add_scalar('delay_index', delay_index, i)
                for vel in ego_v:
                    avg_vel += vel / len(ego_v)
                episode_writer.add_scalar('average_vel', avg_vel, i)
                for jerk in ego_jerk:
                    avg_jerk += jerk / len(ego_jerk)
                episode_writer.add_scalar('average_jerk', avg_jerk, i)
                for offlane in ego_offlane:
                    avg_offlane += abs(offlane) / len(ego_offlane)
                episode_writer.add_scalar('average_offlane', avg_offlane, i)
                if len(ego_ttc) > 0:
                    temp = []
                    for ttc in ego_ttc:
                        if ttc < 0:
                            temp.append(args.TTC_th)
                        else:
                            temp.append(ttc)
                    min_ttc = min(temp)
                else:
                    min_ttc = args.TTC_th
                episode_writer.add_scalar('min_ttc', min_ttc, i)
                for vel in recover_v:
                    rec_avg_vel += vel / len(recover_v)
                episode_writer.add_scalar('rec_avg_vel', rec_avg_vel, i)

            np.save(f"{SAVE_PATH}/ego_vel.npy", np.array(VEL))
            np.save(f"{SAVE_PATH}/ego_acc.npy", np.array(ACC))
            np.save(f"{SAVE_PATH}/ego_jerk.npy", np.array(JERK))
            np.save(f"{SAVE_PATH}/ego_offlane.npy", np.array(OFFLANE))
            np.save(f"{SAVE_PATH}/ego_ttc.npy", np.array(TTC))
            np.save(f"{SAVE_PATH}/rear_acc.npy", np.array(REAR_ACC))
            np.save(f"{SAVE_PATH}/rear_vel.npy", np.array(REAR_VEL))
        except KeyboardInterrupt:
            logging.info("Premature Terminated")
        except BaseException as e:
            logging.info(e.args)
        finally:
            env.__del__()
            episode_writer.close()
            logging.info('\nDone.')


if __name__ == '__main__':
    try:
        start_process()
        main()
    except BaseException as e:
        logging.warning(e.args)
    finally:
        kill_process()
