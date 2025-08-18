from argparse import ArgumentParser
from importlib.util import find_spec

from slurm_now.api import run_search

if find_spec("tabulate"):
    from tabulate import tabulate

    PRINT_TABLE = True
else:
    PRINT_TABLE = False


def cli():
    parser = ArgumentParser(
        "Query the Slurm cluster for resource configuration options that will allow a job to start immediately."
    )
    parser.add_argument(
        "-w", "--min-world-size", type=int, default=1, help="The minimum world size (number of total GPUs) required."
    )
    parser.add_argument(
        "-g", "--gpu-type", type=str, default=r"^.*$", help="Regex pattern to limit the search to a specific GPU type."
    )
    parser.add_argument(
        "-c", "--min-cpu-per-gpu", type=float, default=1.0, help="The minimum ratio of CPUs to free GPUs required."
    )
    parser.add_argument(
        "-m",
        "--min-sys-mem-per-gpu-GB",
        type=float,
        default=0.1,
        help="The minimum system RAM per GPU required (in GB).",
    )
    parser.add_argument(
        "-p",
        "--partition-regex",
        type=str,
        default=r"^(?!.*(full|interactive)).*$",
        help="Regex pattern to limit the search to specific partitions.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    for gpu_type, min_idle_gpus_per_node, world_size, node_group, partitions in run_search(
        min_world_size=args.min_world_size,
        gpu_type=args.gpu_type,
        min_cpu_per_gpu=args.min_cpu_per_gpu,
        min_sys_mem_per_gpu_GB=args.min_sys_mem_per_gpu_GB,
        partition_regex=args.partition_regex,
    ):
        print(
            f"Achieve up to world size {world_size} using "
            f"--nodes={len(node_group)} "
            f"--ntasks-per-node={min_idle_gpus_per_node} "
            f"--gpus-per-node={gpu_type}:{min_idle_gpus_per_node} "
            f"--cpus-per-task={round(min(node.idle_cpu_per_gpu for node in node_group))} "
            f"--mem-per-gpu={round(min(node.idle_sys_mem_per_gpu_GB for node in node_group))} "
            f"--partition={','.join(partitions)}"
        )
        if PRINT_TABLE and args.verbose:
            print(tabulate(node_group, headers="keys"))
