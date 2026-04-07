#!/usr/bin/env python3
"""
快速验证脚本 - 测试改进的 PRIMAL GPU 配置
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print("\n" + "="*80)
print("PRIMAL GPU 配置验证")
print("="*80)

# 检查 GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"\n✓ 检测到 {len(gpus)} 个 GPU")

if len(gpus) == 0:
    print("✗ 未检测到 GPU！")
    exit(1)

# 测试原始配置
print("\n--- 测试：原始配置 ---")
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Graph().as_default():
    with tf.Session(config=config) as sess:
        a = tf.constant([[1.0, 2.0]], name='a')
        b = tf.constant([[1.0], [2.0]], name='b')
        c = tf.matmul(a, b, name='matmul')
        result = sess.run(c)
        print(f"✓ 原始配置运行成功，结果: {result}")

# 测试改进配置
print("\n--- 测试：改进配置（带设备分配日志）---")
config_improved = tf.ConfigProto(allow_soft_placement=True)
config_improved.log_device_placement = True
config_improved.gpu_options.allow_growth = True

print("\n会话日志（下面应该看到 GPU:0）:")
with tf.Graph().as_default():
    with tf.device('/gpu:0'):
        with tf.Session(config=config_improved) as sess:
            a = tf.constant([[1.0, 2.0]], name='a')
            b = tf.constant([[1.0], [2.0]], name='b')
            c = tf.matmul(a, b, name='matmul')
            result = sess.run(c)
            print(f"\n✓ 改进配置运行成功，结果: {result}")

print("\n" + "="*80)
print("✓ 所有验证通过！GPU 配置正确")
print("="*80)

# 检查 GPU 利用率
print("\n当前 GPU 内存使用:")
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                            '--format=csv,noheader,nounits'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(result.stdout)
except:
    print("(无法获取 GPU 信息)")

print("\n提示: 在实际运行 primal_testing.py 时，开另一个终端运行:")
print("  watch -n 1 nvidia-smi")
print("来实时监控 GPU 使用情况。\n")
