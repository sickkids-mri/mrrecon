"""Tools for CuPy and PyTorch GPU memory management.

Related memory management documentation:
https://docs-cupy.chainer.org/en/stable/reference/memory.html
https://pytorch.org/docs/stable/notes/cuda.html
"""
import subprocess
import re


__all__ = ['nvidia_smi', 'nvidia_smi_memory', 'cupy_memory', 'torch_memory']


def nvidia_smi(processes_only=False):
    """Calls the command line utility nvidia-smi."""
    completedprocess = subprocess.run('nvidia-smi',
                                      universal_newlines=True,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
    out = completedprocess.stdout

    if processes_only:  # Excludes upper part of summary
        match = re.search(r'\+-+\+\n\| Processes.*', out, re.DOTALL)
        print(match.group())
    else:
        print(out)


def nvidia_smi_memory():
    """Concisely displays memory information using nvidia-smi."""
    # Gets list of memory used per GPU
    used = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                           '--format=csv,nounits,noheader'],
                          universal_newlines=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    used = used.stdout.strip().split('\n')

    # Gets list of free memory per GPU
    free = subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
                           '--format=csv,nounits,noheader'],
                          universal_newlines=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    free = free.stdout.strip().split('\n')

    # Gets list of total memory per GPU
    total = subprocess.run(['nvidia-smi', '--query-gpu=memory.total',
                            '--format=csv,nounits,noheader'],
                           universal_newlines=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    total = total.stdout.strip().split('\n')

    num_gpus = len(used)

    columns = f'{"GPU":>5}{"Used":>10}{"Free":>10}{"Total":>10}'
    gpus = []
    for a in range(num_gpus):
        line = f'{a:>5}'
        line += f'{used[a]:>7} MB'
        line += f'{free[a]:>7} MB'
        line += f'{total[a]:>7} MB'
        gpus.append(line)

    print('-----------------------------------')
    print('Memory Summary Through NVIDIA-SMI')
    print('-----------------------------------')
    print(columns)
    print('-----------------------------------')
    for line in gpus:
        print(line)
    print('-----------------------------------')


def cupy_memory(device=None):
    """Prints CuPy memory pool usage on selected device.

    Args:
        device (int): GPU device ID (e.g. 0, 1, 2, 3, ...). If no device is
            selected, gets memory usage on current CuPy device.
    """
    import cupy as cp

    if device is None:
        device = cp.cuda.runtime.getDevice()

    mempool = cp.get_default_memory_pool()

    with cp.cuda.Device(device):
        used, unit = readable_bytes(mempool.used_bytes())
        used = f'{used:.3f} {unit}'

        reserved, unit = readable_bytes(mempool.total_bytes())
        reserved = f'{reserved:.3f} {unit}'

    print('-------------------------')
    print(f'CuPy Memory Pool on GPU {device}')
    print('-------------------------')
    print(f'{used:>10} Used')
    print(f'{reserved:>10} Reserved')
    print('-------------------------')


def torch_memory(device=None):
    """Prints PyTorch memory usage on selected device.

    Args:
        device (int): GPU device ID (e.g. 0, 1, 2, 3, ...). If no device is
            selected, gets memory usage on current PyTorch device.
    """
    import torch

    if device is None:
        device = torch.cuda.current_device()

    used, unit = readable_bytes(torch.cuda.memory_allocated(device))
    used = f'{used:.3f} {unit}'

    reserved, unit = readable_bytes(torch.cuda.memory_cached(device))
    reserved = f'{reserved:.3f} {unit}'

    used_max, unit = readable_bytes(torch.cuda.max_memory_allocated(device))
    used_max = f'{used_max:.3f} {unit}'

    reserved_max, unit = readable_bytes(torch.cuda.max_memory_cached(device))
    reserved_max = f'{reserved_max:.3f} {unit}'

    print('-------------------------')
    print(f'PyTorch Memory on GPU {device}')
    print('-------------------------')
    print(f'{used:>10} Used')
    print(f'{reserved:>10} Reserved')
    print('-------------------------')
    print(f'{used_max:>10} Used Max')
    print(f'{reserved_max:>10} Reserved Max')
    print('-------------------------')


def readable_bytes(num_bytes):
    """Converts number of bytes to readable units.

    Args:
        num_bytes (int): Number of bytes.

    Returns:
        num_bytes (float): Number of bytes in the appropriate unit
            (B, KB, MB, or GB).
        unit (string): The appropriate unit.
    """
    units = ['B', 'KB', 'MB', 'GB']
    idx = 0
    while int(num_bytes / 1024):
        num_bytes /= 1024
        idx += 1
        if idx == (len(units) - 1):
            break

    unit = units[idx]
    return num_bytes, unit
