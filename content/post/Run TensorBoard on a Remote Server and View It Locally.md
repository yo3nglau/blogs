---
title: Run TensorBoard on a Remote Server and View It Locally
author: yo3nglau
date: '2025-11-11'
categories:
  - Computer Technology
tags:
  - Guide
  - Remote Server
  - TensorBoard
toc: true
---

## Preface
When training machine learning models on remote servers, monitoring metrics such as loss, accuracy, and learning rate becomes essential. TensorBoard is a powerful visualization toolkit from TensorFlow that makes this possible. However, since training often takes place on remote machines without direct GUI access, developers must forward TensorBoard to their local browser. Fortunately, this is simple with secure SSH tunneling.

## Why Run TensorBoard Remotely?

Many training workloads:

- Require powerful GPUs only available on remote servers
- Run for long periods and need continuous observation
- Produce logs that are easiest to analyze visually

Instead of transferring logs back to your local machine, it is more efficient to stream the TensorBoard UI directly to your browser.

## Step 0: Create TensorBoard Log Directory

During training, ensure your script outputs logs:

```bash
python train.py --logdir ./logs  # or set in codes
```

You should see files populated inside the `logs` folder, including event traces.

## Step 1: Launch TensorBoard on the Remote Server

SSH into the server and run:

```bash
tensorboard --logdir ./logs --port 6006
```

You will see:

```
TensorBoard 2.x.x at http://localhost:6006/
```

This means TensorBoard is running, but only accessible locally on the server.

## Step 2: Forward the Port Using SSH

On your local machine (not on the server), run:

```bash
ssh -L 6006:localhost:6006 username@remote_server_ip -p remote_server_port
```

This command creates a secure tunnel that forwards the server’s `localhost:6006` to your computer’s `localhost:6006`.

Alternatively, specify a custom local port if 6006 is occupied:

```bash
ssh -L 16006:localhost:6006 username@remote_server_ip
```

## Step 3: Open TensorBoard Locally

Now, open your browser and visit:

```bash
http://localhost:6006  # or the custom port you chose
```

You should see TensorBoard’s interface displaying your logs in real time.


## Conclusion

Running TensorBoard on a remote server and visualizing it locally is a clean and secure workflow for tracking training progress. With SSH port forwarding, you retain full interactivity without transferring files or needing remote GUIs. Once set up, this workflow enhances productivity and provides deeper insights into model behavior during training.

## Resources

[TensorFlow/TensorBoard](https://www.tensorflow.org/tensorboard)
