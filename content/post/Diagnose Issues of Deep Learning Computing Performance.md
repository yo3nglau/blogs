---
title: Diagnose Issues of Deep Learning Computing Performance
author: yo3nglau
date: '2025-11-07'
categories:
  - Computer Technology
  - Deep Learning
tags:
  - Guide
toc: true
---

**Last Updated: 2025-11-10**

## Preface

Training deep learning models can sometimes feel like waiting for water to boil—it tests your patience. When your model is training slower than expected, a systematic approach is key to identifying the bottleneck. Here is a step-by-step guide to diagnosing and fixing these performance issues.

## Step 0: The First Check - GPU Utilization

Your first command should always be to check if the GPU is even being used.

```
watch -n 1 nvidia-smi
```

This command provides a real-time view of your GPU's status. The critical metric is `GPU-Util`.

- **~0% Utilization:** This is the most common culprit for beginners. It typically means your code isn't actually using the GPU. **Fix:** Ensure your model and data tensors are moved to the CUDA device (e.g., `.cuda()`or `.to(device)`).
- **Sustained >90% Utilization:** Congratulations! Your GPU is the bottleneck. To speed things up, consider using multi-GPU training or a more powerful GPU.
- **Low & Fluctuating Utilization (e.g., peaks below 50%):** This indicates a problem where the GPU is waiting for work. The bottleneck is likely elsewhere, often the CPU or data pipeline.

## Step 1: Classify Your Training Task

Understanding the nature of your task helps you know what to expect. The performance characteristics can be categorized as follows:

| Scenario                                                     | CPU Usage                | GPU Usage                                                    | Processing Time & Fluctuation                           | Typical Bottleneck                                           |
| ------------------------------------------------------------ | ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------- | ------------------------------------------------------------ |
| **1. Small Model, Simple Data** (e.g., LeNet on MNIST)       | Low                      | Low, but stable with little fluctuation                      | Short and consistent                                    | The model itself is not computationally intensive. Little optimization room. |
| **2. Small Model, Complex Data** (e.g., ResNet-18 on ImageNet) | High during data loading | Low on average, with high, short peaks and long idle periods | GPU time short, but cycle time long due to data loading | **CPU (Data Preprocessing).** The GPU finishes quickly and waits for the next batch. |
| **3. Large Model, Simple Data**                              | Low                      | High and stable with little fluctuation                      | Long but consistent                                     | **GPU (Model Computation).** The GPU is constantly busy.     |
| **4. Large Model, Complex Data**                             | High                     | Variable, often unable to reach sustained high usage         | Long and potentially inconsistent                       | **Mixed (CPU & GPU).** Both components are stressed and can become the bottleneck. |

For Scenarios 1 and 2, optimization focus should be on choosing the right hardware (better CPU for #2) and code structure. For Scenarios 3 and 4, if GPU usage is low, you need to dig deeper.

## Step 2: Check CPU Utilization

If your GPU usage is low and fluctuating (characteristic of Scenario 2 or 4), the CPU is the prime suspect.

1. Check the CPU usage percentage (with `htop`, or `sysstat`). Note that 100% represents one core. If you has 5 cores, 500% usage means all cores are saturated.
2. **High CPU Usage (~100% * `core_count`):** Your CPU is the bottleneck. **Solution:** Migrate your project to a host with more CPU cores.
3. **Low CPU Usage:** Your code isn't efficiently leveraging the CPU for data loading. **Solution:** Increase the `num_workers`parameter in your PyTorch `DataLoader`. A good starting point is slightly less than the number of available CPU cores. Experiment to find the optimal value.

## Step 3: Code-Level Optimizations

If hardware isn't the clear issue, it's time to look at your code.

- **Avoid In-Iteration Overhead:** Are you saving sample images or logging excessively every batch? These I/O operations block the training loop. Do them less frequently (e.g., every N epochs).
- **Frequent Checkpointing:** Saving the entire model can be slow. Reduce the frequency of saving model checkpoints.
- **Use Official Guides:** [PyTorch](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) provides excellent performance tuning guides.

------

## Common Pitfalls and Advanced Tips

### 1. The PyTorch Threading Issue on Multi-Card Instances

**Symptoms:** You rent a multi-GPU instance to run different experiments on different cards, but everything runs extremely slowly, with low utilization on both CPU and GPU.

**Root Cause:** By default, PyTorch creates a number of threads equal to the number of CPU cores to accelerate operations. When multiple PyTorch processes (one per GPU experiment) each spawn this many threads, the system spends excessive time on thread scheduling instead of computation.

**Solution:** Limit the number of threads per process by adding `torch.set_num_threads(N)`to your code, where `N`is a number like 4 or 8, depending on your core count and the number of concurrent experiments.

### 2. The NumPy Performance Trap on Intel CPUs

**Symptoms:** Extremely high CPU usage (all cores maxed out) but low GPU usage, specifically on Intel CPUs.

**Root Cause:** NumPy uses a math library for acceleration. Intel CPUs achieve significantly higher performance with the **Intel Math Kernel Library (MKL)** than with the generic **OpenBLAS**. Some Conda mirrors (like Tsinghua's) default to installing the OpenBLAS version of NumPy.

**Solution:**

```
# 1. Uninstall current numpy
pip uninstall numpy
# or: conda uninstall numpy

# 2. Temporarily remove the Conda mirror to get the default channel
echo "" > /root/.condarc

# 3. Reinstall numpy, which will now fetch the MKL-optimized version
conda install numpy
```

#### **3. Pro Tips for Better Performance**

- **Prefer DDP over DP:** For PyTorch multi-GPU training, use `torch.nn.DistributedDataParallel`(DDP) instead of `torch.nn.DataParallel`(DP). The official docs state that DDP "offers much better performance and scaling to multiple-GPUs."
- **Update Your PyTorch Version:** If you are using Ampere architecture GPUs (e.g., RTX 3090, A100), upgrading PyTorch versions can yield significant performance improvements.
- **Isolate Heavy Experiments:** For resource-intensive hyperparameter tuning, it's better to run each experiment in a separate instance on different host machines rather than running them all on the same host/instance to avoid resource contention.

## Summary

By following this logical flow—from GPU monitoring to task classification, then to CPU and code inspection—you can efficiently pinpoint what's slowing down your training and take the correct action to get back to full speed.

## Resources

[AutoDL Docs](https://api.autodl.com/docs/perf/)
