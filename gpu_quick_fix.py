#!/usr/bin/env python3
"""
快速参考：如何启用 PRIMAL 的 GPU 使用日志
直接运行: python3 gpu_quick_fix.py
"""

import sys
import os

def main():
    print("\n" + "="*70)
    print("PRIMAL GPU 快速诊断工具")
    print("="*70)
    
    # 1. 检查 GPU
    print("\n[步骤 1] 检查 GPU 硬件...")
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 找到 GPU:")
            print(result.stdout)
        else:
            print("✗ 未找到 GPU")
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False
    
    # 2. 检查 TensorFlow
    print("[步骤 2] 检查 TensorFlow...")
    try:
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ TensorFlow 检测到 {len(gpus)} 个 GPU:")
            for gpu in gpus:
                print(f"   - {gpu}")
        else:
            print("✗ TensorFlow 未检测到 GPU")
            return False
    except Exception as e:
        print(f"✗ TensorFlow 错误: {e}")
        return False
    
    # 3. 显示修复步骤
    print("\n[步骤 3] 启用 GPU 日志...")
    print("""
要使 primal_testing.py 显示 GPU 使用，请修改以下内容：

文件: primal_testing.py
找到行 (约第 30-35 行):
    config.log_device_placement = False

改为:
    config.log_device_placement = True

然后再运行脚本：
    python3 primal_testing.py

会看到类似输出：
    matmul: (MatMul): /device:GPU:0  ← 表示在 GPU 上运行
""")
    
    # 4. 建议
    print("[步骤 4] 实时监控 GPU（可选）")
    print("""
在另一个终端运行以下命令来实时监控 GPU 使用：

方法 A (推荐):
    watch -n 1 nvidia-smi

方法 B:
    nvidia-smi -l 1

方法 C:
    pip install gpustat
    gpustat -i 1
""")
    
    print("\n" + "="*70)
    print("✓ 诊断完成！GPU 已就绪")
    print("="*70 + "\n")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
