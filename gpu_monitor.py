#!/usr/bin/env python3
"""
稳定的 GPU 监控脚本
使用方法: python3 gpu_monitor.py
"""
import subprocess
import time
import sys
import os
from datetime import datetime

def get_gpu_stats():
    """获取当前 GPU 统计信息"""
    try:
        cmd = [
            'nvidia-smi',
            '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
            '--format=csv,noheader,nounits'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception as e:
        return None

def get_processes():
    """获取 GPU 上运行的进程"""
    try:
        cmd = [
            'nvidia-smi',
            '--query-compute-apps=pid,process_name,gpu_memory_usage',
            '--format=csv,noheader,nounits'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            output = result.stdout.strip()
            return output if output else "无进程"
        return None
    except Exception as e:
        return None

def clear_screen():
    """清屏"""
    os.system('clear' if os.name == 'posix' else 'cls')

def print_monitor():
    """打印监控信息"""
    while True:
        clear_screen()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("=" * 80)
        print(f"NVIDIA GPU 监控 - {timestamp}")
        print("=" * 80)
        
        # 显示 GPU 信息
        print("\n【GPU 状态】")
        print("-" * 80)
        stats = get_gpu_stats()
        if stats:
            lines = stats.split('\n')
            print(f"{'GPU':<5} {'名称':<25} {'内存(MB)':<15} {'利用率(%)':<12} {'温度(°C)':<10}")
            print("-" * 80)
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 6:
                        gpu_idx = parts[0].strip()
                        gpu_name = parts[1].strip()[:23]
                        mem_used = f"{parts[2].strip()}/{parts[3].strip()}"
                        util = f"{parts[4].strip()}%"
                        temp = f"{parts[5].strip()}°C"
                        print(f"{gpu_idx:<5} {gpu_name:<25} {mem_used:<15} {util:<12} {temp:<10}")
        else:
            print("✗ 无法获取 GPU 信息")
        
        # 显示进程
        print("\n【GPU 进程】")
        print("-" * 80)
        processes = get_processes()
        if processes:
            if processes != "无进程":
                print(f"{'PID':<10} {'进程名':<30} {'GPU内存(MB)':<15}")
                print("-" * 80)
                for line in processes.split('\n'):
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 3:
                            pid = parts[0].strip()
                            pname = parts[1].strip()[:28]
                            mem = parts[2].strip()
                            print(f"{pid:<10} {pname:<30} {mem:<15}")
            else:
                print("无运行进程")
        else:
            print("✗ 无法获取进程信息")
        
        print("\n" + "=" * 80)
        print("按 Ctrl+C 退出 | 每秒更新一次")
        print("=" * 80)
        
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("\n已停止监控")
            sys.exit(0)

if __name__ == "__main__":
    print_monitor()
