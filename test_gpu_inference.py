#!/usr/bin/env python3
"""
PRIMAL 模型 GPU 使用测试脚本
这个脚本会加载模型并运行少量推理，同时显示 GPU 使用情况
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from ACNet import ACNet
import numpy as np
import mapf_gym_cap as mapf_gym
import subprocess
import time
import threading

def print_gpu_info():
    """打印 GPU 使用信息"""
    try:
        cmd = [
            'nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            util, mem_used, mem_total = result.stdout.strip().split(',')
            print(f"  ├─ GPU 利用率: {util.strip()}%")
            print(f"  ├─ GPU 内存使用: {mem_used.strip()}/{mem_total.strip()} MB")
    except:
        pass

def test_primal_gpu():
    """测试 PRIMAL 模型在 GPU 上运行"""
    
    print("\n" + "="*70)
    print("PRIMAL GPU 使用测试")
    print("="*70)
    
    # 显示初始 GPU 状态
    print("\n[初始状态]")
    print_gpu_info()
    
    # 加载模型
    print("\n[步骤 1] 加载 PRIMAL 模型...")
    print("  配置: log_device_placement = True (显示设备信息)")
    
    # 创建 GPU 配置 - 注意这里启用了日志！
    config = tf.ConfigProto(allow_soft_placement=True)
    config.log_device_placement = True  # 启用此项以看到设备分配
    config.gpu_options.allow_growth = True
    
    print("\n[GPU 操作日志]")
    print("-" * 70)
    
    with tf.device('/gpu:0'):  # 显式指定 GPU
        sess = tf.Session(config=config)
        print("\n✓ 会话创建在 GPU 上")
        
        network = ACNet("global", 5, None, False, 10, "global")
        print("✓ 网络创建完成")
    
    print("-" * 70)
    
    # 检查模型加载
    print("\n[步骤 2] 加载模型权重...")
    try:
        ckpt = tf.train.get_checkpoint_state('model_primal')
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(f"✓ 成功加载: {ckpt.model_checkpoint_path}")
        else:
            print("✗ 未找到模型检查点")
            return
    except Exception as e:
        print(f"✗ 加载模型出错: {e}")
        return
    
    # 显示加载后的 GPU 状态
    print("\n[加载后状态]")
    print_gpu_info()
    
    # 测试推理
    print("\n[步骤 3] 运行推理测试...")
    
    # 创建虚拟数据
    num_agents = 4
    grid_size = 10
    
    dummy_inputs = np.random.randn(num_agents, 4, grid_size, grid_size).astype(np.float32)
    dummy_goals = np.random.randn(num_agents, 3).astype(np.float32)
    
    print(f"  输入形状: {dummy_inputs.shape}")
    print(f"  目标形状: {dummy_goals.shape}")
    
    print("\n[GPU 推理日志]")
    print("-" * 70)
    
    # 运行推理
    start_time = time.time()
    try:
        h3_output = sess.run(
            network.h3,
            feed_dict={
                network.inputs: dummy_inputs,
                network.goal_pos: dummy_goals
            }
        )
        inference_time = time.time() - start_time
        print(f"\n✓ 推理成功 (耗时: {inference_time:.3f}s)")
        print(f"  输出形状: {h3_output.shape}")
    except Exception as e:
        print(f"\n✗ 推理失败: {e}")
        return
    
    print("-" * 70)
    
    # 显示推理后的 GPU 状态
    print("\n[推理后状态]")
    print_gpu_info()
    
    # 多次推理以稳定 GPU 利用率
    print("\n[步骤 4] 运行 10 次推理测试...")
    times = []
    for i in range(10):
        start = time.time()
        _ = sess.run(
            network.h3,
            feed_dict={
                network.inputs: dummy_inputs,
                network.goal_pos: dummy_goals
            }
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  第 {i+1:2d} 次: {elapsed*1000:.2f}ms", end="")
        if (i + 1) % 5 == 0:
            print()
        else:
            print(" | ", end="")
    
    print("\n")
    avg_time = np.mean(times)
    print(f"✓ 平均推理时间: {avg_time*1000:.2f}ms")
    print(f"✓ 吞吐量: {1/avg_time:.1f} 推理/秒")
    
    # 最终 GPU 状态
    print("\n[最终状态]")
    print_gpu_info()
    
    sess.close()
    print("\n✓ 会话已关闭\n")
    
    print("="*70)
    print("✓ 测试完成！")
    print("="*70)
    print("""
如果上面显示了如下信息，说明 GPU 在工作:
✓ h3: (Identity): /device:GPU:0
✓ h3: (AvgPool): /device:GPU:0
✓ 等其他操作的 GPU 设备标记

或者 GPU 利用率不为 0%。
""")

if __name__ == "__main__":
    test_primal_gpu()
