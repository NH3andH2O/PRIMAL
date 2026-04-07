#!/usr/bin/env python3
"""
改进版本的 PRIMAL 测试脚本 - 包含 GPU 使用验证
使用方法: python3 primal_testing_gpu.py
"""
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')  # INFO 级别日志
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_auto_jit=0')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from ACNet import ACNet
import numpy as np
import json
import mapf_gym_cap as mapf_gym
import time
import subprocess
import threading
from od_mstar3.col_set_addition import OutOfTimeError, NoSolutionError

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

results_root = "primal_results"
run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
results_path = os.path.join(results_root, run_timestamp)
environment_path = "saved_environments"
if not os.path.exists(results_root):
    os.makedirs(results_root)
if not os.path.exists(results_path):
    os.makedirs(results_path)


def print_gpu_status():
    """打印当前 GPU 使用状态"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("GPU 状态:")
            print("=" * 60)
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    gpu_idx, mem_used, mem_total, util = line.split(',')
                    print(f"GPU {gpu_idx}: 内存使用 {mem_used}/{mem_total}MB | 利用率 {util}%")
            print("=" * 60 + "\n")
    except Exception as e:
        print(f"无法获取 GPU 信息: {e}")


def monitor_gpu_in_background(stop_event, interval=2):
    """后台监控 GPU 使用情况"""
    while not stop_event.is_set():
        print_gpu_status()
        stop_event.wait(interval)


class PRIMAL(object):
    '''
    This class provides functionality for running multiple instances of the 
    trained network in a single environment
    '''
    def __init__(self, model_path, grid_size):
        self.grid_size = grid_size
        
        print("\n" + "=" * 60)
        print("初始化 PRIMAL 模型（使用 GPU）")
        print("=" * 60)
        
        # 配置 GPU
        config = tf.ConfigProto(allow_soft_placement=True)
        config.log_device_placement = True  # 显示设备分配日志
        config.gpu_options.allow_growth = True
        
        # 在 GPU 上创建会话和网络
        with tf.device('/gpu:0'):
            self.sess = tf.Session(config=config)
            self.network = ACNet("global", 5, None, False, grid_size, "global")
        
        # 加载模型权重
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.Saver()

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print(f"✓ 成功从 {ckpt.model_checkpoint_path} 加载模型")
        else:
            raise Exception(f"未找到检查点在: {model_path}")
        
        print("✓ 模型初始化完成\n")
        
    def set_env(self, gym):
        self.num_agents = gym.num_agents
        self.agent_states = []
        for i in range(self.num_agents):
            rnn_state = self.network.state_init
            self.agent_states.append(rnn_state)
        self.size = gym.SIZE
        self.env = gym
        
    def step_all_parallel(self):
        '''advances the state of the environment by a single step across all agents'''
        action_probs = [None for i in range(self.num_agents)]
        
        # 并行推理
        actions = []
        inputs = []
        goal_pos = []
        for agent in range(1, self.num_agents + 1):
            o = self.env._observe(agent)
            inputs.append(o[0])
            goal_pos.append(o[1])
        
        # 并行计算到 LSTM
        h3_vec = self.sess.run([self.network.h3], 
                               feed_dict={self.network.inputs: inputs,
                                         self.network.goal_pos: goal_pos})
        h3_vec = h3_vec[0]
        rnn_out = []
        
        # 通过 LSTM 顺序流动每个智能体的 RNN 状态
        for a in range(0, self.num_agents):
            rnn_state = self.agent_states[a]
            lstm_output, state = self.sess.run([self.network.rnn_out, self.network.state_out], 
                                          feed_dict={self.network.inputs: [inputs[a]],
                                                    self.network.h3: [h3_vec[a]],
                                                    self.network.state_in[0]: rnn_state[0],
                                                    self.network.state_in[1]: rnn_state[1]})
            rnn_out.append(lstm_output[0])
            self.agent_states[a] = state
        
        # 并行完成计算
        policy_vec = self.sess.run([self.network.policy], 
                                   feed_dict={self.network.rnn_out: rnn_out})
        policy_vec = policy_vec[0]
        for agent in range(1, self.num_agents + 1):
            action = np.argmax(policy_vec[agent - 1])
            self.env._step((agent, action))
          
    def find_path(self, max_step=256):
        '''run a full environment to completion, or until max_step steps'''
        solution = []
        step = 0
        while ((not self.env._complete()) and step < max_step):
            timestep = []
            for agent in range(1, self.env.num_agents + 1):
                timestep.append(self.env.world.getPos(agent))
            solution.append(np.array(timestep))
            self.step_all_parallel()
            step += 1
        if step == max_step:
            raise OutOfTimeError
        for agent in range(1, self.env.num_agents):
            timestep.append(self.env.world.getPos(agent))
        return np.array(solution)
    
    
def make_name(n, s, d, id, extension, dirname, extra=""):
    if extra == "":
        return dirname + '/' + "{}_agents_{}_size_{}_density_id_{}{}".format(n, s, d, id, extension)
    else:
        return dirname + '/' + "{}_agents_{}_size_{}_density_id_{}_{}{}".format(n, s, d, id, extra, extension)


def init_summary():
    return {
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total': 0,
        'finished': 0,
        'timeout': 0,
        'missing_environment': 0,
        'crashed': 0,
        'time_sum': 0.0,
        'time_count': 0,
        'length_sum': 0,
        'length_count': 0,
        'by_size_density': {}
    }


def update_summary(summary, n, s, d, result):
    density_key = '{:g}'.format(d)
    size_key = str(s)
    by_size = summary['by_size_density'].setdefault(size_key, {})
    bucket = by_size.setdefault(density_key, {
        'total': 0,
        'finished': 0,
        'timeout': 0,
        'missing_environment': 0,
        'crashed': 0,
        'time_sum': 0.0,
        'time_count': 0,
        'length_sum': 0,
        'length_count': 0
    })

    summary['total'] += 1
    bucket['total'] += 1

    if result.get('error') == 'missing_environment_file':
        summary['missing_environment'] += 1
        bucket['missing_environment'] += 1
    elif result.get('crashed'):
        summary['crashed'] += 1
        bucket['crashed'] += 1
    elif result.get('finished'):
        summary['finished'] += 1
        bucket['finished'] += 1
    else:
        summary['timeout'] += 1
        bucket['timeout'] += 1

    if 'time' in result:
        summary['time_sum'] += float(result['time'])
        summary['time_count'] += 1
        bucket['time_sum'] += float(result['time'])
        bucket['time_count'] += 1

    if 'length' in result:
        summary['length_sum'] += int(result['length'])
        summary['length_count'] += 1
        bucket['length_sum'] += int(result['length'])
        bucket['length_count'] += 1


def finalize_summary(summary):
    if summary['time_count'] > 0:
        summary['avg_time'] = summary['time_sum'] / summary['time_count']
    else:
        summary['avg_time'] = None

    if summary['length_count'] > 0:
        summary['avg_length'] = summary['length_sum'] / summary['length_count']
    else:
        summary['avg_length'] = None

    for size_key, density_map in summary['by_size_density'].items():
        for density_key, bucket in density_map.items():
            if bucket['time_count'] > 0:
                bucket['avg_time'] = bucket['time_sum'] / bucket['time_count']
            else:
                bucket['avg_time'] = None

            if bucket['length_count'] > 0:
                bucket['avg_length'] = bucket['length_sum'] / bucket['length_count']
            else:
                bucket['avg_length'] = None

    summary['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    

def run_simulations(next, primal):
    '''txt file: planning time, crash, nsteps, finished'''
    (n, s, d, id) = next
    environment_data_filename = make_name(n, s, d, id, ".npy", environment_path, extra="environment")
    if not os.path.exists(environment_data_filename):
        print(f"Skipping missing environment file: {environment_data_filename}")
        txt_filename = make_name(n, s, d, id, ".txt", results_path)
        results = {'finished': False, 'crashed': True, 'error': 'missing_environment_file', 'environment': environment_data_filename}
        f = open(txt_filename, 'w')
        f.write(json.dumps(results))
        f.close()
        return results
    world = np.load(environment_data_filename)
    gym = mapf_gym.MAPFEnv(num_agents=n, world0=world[0], goals0=world[1])
    primal.set_env(gym)
    solution_filename = make_name(n, s, d, id, ".npy", results_path, extra="solution")
    txt_filename = make_name(n, s, d, id, ".txt", results_path)
    world = gym.getObstacleMap()
    start_positions = tuple(gym.getPositions())
    goals = tuple(gym.getGoals())
    start_time = time.time()
    results = dict()
    start_time = time.time()
    try:
        path = primal.find_path(256 + 128 * int(s >= 80) + 128 * int(s >= 160))
        results['finished'] = True
        results['time'] = time.time() - start_time
        results['length'] = len(path)
        np.save(solution_filename, path)
    except OutOfTimeError:
        results['time'] = time.time() - start_time
        results['finished'] = False
    results['crashed'] = False
    f = open(txt_filename, 'w')
    f.write(json.dumps(results))
    f.close()
    return results


if __name__ == "__main__":
    # 打印 TensorFlow GPU 信息
    print("\n" + "=" * 60)
    print("TensorFlow GPU 信息")
    print("=" * 60)
    gpus = tf.config.list_physical_devices('GPU')
    print(f"检测到的 GPU 数量: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    print("=" * 60 + "\n")
    
    primal = PRIMAL('model_primal', 10)
    
    # 启动 GPU 监控
    gpu_monitor_stop = threading.Event()
    # 取消 GPU 监控线程（可选）
    # gpu_monitor_thread = threading.Thread(target=monitor_gpu_in_background, 
    #                                       args=(gpu_monitor_stop,), daemon=True)
    # gpu_monitor_thread.start()

    try:
        summary = init_summary()
        for num_agents in [4]:

            print("Starting tests for %d agents" % num_agents)
            valid_sizes = [size for size in [10, 20, 40, 80, 160]
                          if not (size == 10 and num_agents > 32)
                          and not (size == 20 and num_agents > 128)
                          and not (size == 40 and num_agents > 512)]
            total_runs = len(valid_sizes) * 4 * 100
            progress = tqdm(total=total_runs, desc="agents={}".format(num_agents), unit="sim")
            for size in [10, 20, 40, 80, 160]:
                if size == 10 and num_agents > 32:
                    continue
                if size == 20 and num_agents > 128:
                    continue
                if size == 40 and num_agents > 512:
                    continue
                for density in [0, .1, .2, .3]:
                    for iter in range(100):
                        result = run_simulations((num_agents, size, density, iter), primal)
                        update_summary(summary, num_agents, size, density, result)
                        progress.update(1)
            progress.close()

        finalize_summary(summary)
        summary_path = os.path.join(results_path, 'summary.json')
        with open(summary_path, 'w') as f:
            f.write(json.dumps(summary, indent=2, sort_keys=True))

        print("Summary written to {}".format(summary_path))
        print("Summary: total={}, finished={}, timeout={}, missing_environment={}, crashed={}".format(
            summary['total'], summary['finished'], summary['timeout'], summary['missing_environment'], summary['crashed']
        ))
        print("finished all tests!")
    finally:
        gpu_monitor_stop.set()
        print_gpu_status()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
