# Slurm Now

Tired of fiddling with Slurm settings to get your multi-GPU job to start ASAP? Slurm Now tells you how to request the largest number of GPUs in your `sbatch` or `srun` command such that _your job will start immediately_, allowing you to scale up as much as possible while avoiding queues.

This is geared towards distributed ML training workloads which require a certain world size (number of GPUs), but don't care about topology (the number of nodes or the number of GPUs per node).

For a world size `W`, system-memory-per-GPU `M`, and CPU-to-GPU ratio `C`, Slurm Now finds combinations of `N` and `G` where the set of `N` nodes each having `≥G` idle GPUs, `≥GxM` memory, and `≥GxC` idle CPUs are such that `NxG=W`.

In the case of multiple solutions, they will be sorted by `W` descending and then `G` descending.

## Installation

Slurm Now depends on `pyslurm`, whose version must be compatible with your Slurm. First check your slurm version with `sacct --version`, then install `pyslurm` from GitHub: `pip install git+https://github.com/PySlurm/pyslurm.git@${MY_SLURM_VERSION}`. This can take a few minutes.

Install Slurm Now with `pip`:

```sh
pip install git+https://github.com/schmidt-jake/slurm_now.git
```

Or add Slurm Now to your project with `uv`:

```sh
uv add git+https://github.com/schmidt-jake/slurm_now
```

Install the `table` extra to enable verbose CLI output:

```sh
pip install git+https://github.com/schmidt-jake/slurm_now.git[table]
```

```sh
uv add git+https://github.com/schmidt-jake/slurm_now --extra=table
```

## Getting Started

Use the CLI:

```sh
slurm_now --help
```

Example:

```sh
slurm_now --min-world-size=2 --gpu-type='a40|a100' --min-cpu-per-gpu=12
```

```console
Achieve up to world size 4 using --nodes=1 --ntasks-per-node=4 --gpus-per-node=a40:4
Achieve up to world size 4 using --nodes=4 --ntasks-per-node=1 --gpus-per-node=a40:1
Achieve up to world size 6 using --nodes=3 --ntasks-per-node=2 --gpus-per-node=a100:2
Achieve up to world size 6 using --nodes=6 --ntasks-per-node=1 --gpus-per-node=a100:1
Achieve up to world size 3 using --nodes=1 --ntasks-per-node=3 --gpus-per-node=a100:3
```

To show real-time cluster availability, combine `slurm_now` with `watch`:

```sh
watch slurm_now ...
```

Use the Python API:

```python
from slurm_now import run_search

for gpu_type, min_idle_gpus_per_node, world_size, node_group in run_search(min_world_size=8):
    print(
        f"Achieve up to world size {world_size} using "
        f"--nodes={len(node_group)} "
        f"--ntasks-per-node={min_idle_gpus_per_node} "
        f"--gpus-per-node={gpu_type}:{min_idle_gpus_per_node}"
    )
```

## Limitations

Slurm Now currently doesn't consider QoS constraints.
