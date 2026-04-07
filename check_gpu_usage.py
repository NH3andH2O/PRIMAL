#!/usr/bin/env python3
"""
诊断脚本：检查 primal_testing.py 中的 GPU 使用情况
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 显示 TensorFlow 日志

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print("=" * 60)
print("TensorFlow GPU 诊断")
print("=" * 60)

# 检查可用设备
print("\n1. 检查可用的物理设备:")
gpus = tf.config.list_physical_devices('GPU')
print(f"   找到的 GPU 数量: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"   GPU {i}: {gpu}")

cpus = tf.config.list_physical_devices('CPU')
print(f"   找到的 CPU 数量: {len(cpus)}")

# 创建一个简单的图来测试 GPU
print("\n2. 创建测试图并在 GPU 上运行操作:")

# 使用 v1 API 的方式
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

# 添加更多日志信息
config.log_device_placement = True  # 这会显示每个操作在哪个设备上运行

with tf.variable_scope('test'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]], name='a')
    b = tf.constant([[1.0, 2.0], [3.0, 4.0]], name='b')
    c = tf.matmul(a, b, name='matmul')

sess = tf.Session(config=config)
print("\n3. 运行测试操作:")
result = sess.run(c)
print("   矩阵乘法结果:")
print(result)

print("\n4. ConfigProto 详细信息:")
print(f"   allow_soft_placement: {config.allow_soft_placement}")
print(f"   GPU allow_growth: {config.gpu_options.allow_growth}")
print(f"   GPU per_process_memory_fraction: {config.gpu_options.per_process_memory_fraction}")

sess.close()

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
print("\n建议:")
print("- 如果上面的操作出现在 '/job:localhost/replica:0/task:0/device:GPU:0'，说明 GPU 被正确使用")
print("- 如果操作出现在 '/job:localhost/replica:0/task:0/device:CPU:0'，说明没有使用 GPU")
