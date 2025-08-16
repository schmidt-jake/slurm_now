import logging
import re
from collections import defaultdict
from collections.abc import Generator
from typing import NamedTuple

import pyslurm

logger = logging.getLogger(__name__)

GRES_REGEX = re.compile(r"^gpu:(\w+)$")
MiB_to_GB = 2**20 / 10**9


class NodeGPUs(NamedTuple):
    """A logical grouping of a set of homogeneous GPUs on a single node."""

    node_name: str
    gpu_type: str
    idle_gpus: int
    idle_cpu_per_gpu: float
    idle_sys_mem_per_gpu_GB: float
    state: str
    partitions: set[str]

    def __hash__(self) -> int:
        return hash((self.node_name, self.gpu_type))


def find_nodes(
    gpu_type: str = r"^.*$",
    min_cpu_per_gpu: float = 0.0,
    min_sys_mem_per_gpu_GB: float = 0.0,
    partition_regex: str = r"^(?!.*(full|interactive)).*$",
) -> Generator[NodeGPUs]:
    return (
        NodeGPUs(
            node_name=node_name,
            gpu_type=gpu_match.group(1),
            idle_gpus=idle_gpus,
            idle_cpu_per_gpu=cpu_per_gpu,
            idle_sys_mem_per_gpu_GB=sys_mem_per_gpu_GB,
            state=node.state,
            partitions=partitions,
        )
        for node_name, node in pyslurm.Nodes.load().items()
        for gres_type, num_configured in node.configured_gres.items()
        if (gpu_match := GRES_REGEX.match(gres_type))
        and re.match(gpu_type, gpu_match.group(1))
        and node.state in ["IDLE", "MIXED"]
        and (idle_gpus := num_configured - node.allocated_gres.get("gpu", 0)) > 0
        and (cpu_per_gpu := node.idle_cpus / idle_gpus) >= min_cpu_per_gpu
        and (sys_mem_per_gpu_GB := min(node.free_memory, node.idle_memory) * MiB_to_GB / idle_gpus)
        >= min_sys_mem_per_gpu_GB
        and len(partitions := {p for p in node.partitions if re.match(partition_regex, p)}) > 0
    )


def organize_node_groups(nodes: set[NodeGPUs]):
    # first find the min num gpus per node per gpu type
    groups: defaultdict[str, set[int]] = defaultdict(set)
    for node in nodes:
        groups[node.gpu_type].add(node.idle_gpus)

    logger.debug(f"{groups=}")

    # assign nodes to groups with <= num idle gpus per node with matching gpu type
    node_groups: defaultdict[tuple[str, int], set[NodeGPUs]] = defaultdict(set)
    for node in nodes:
        for _min in groups[node.gpu_type]:
            if node.idle_gpus >= _min:
                node_groups[node.gpu_type, _min].add(node)
    return node_groups


def _sort_key(x: tuple[tuple[str, int], set[NodeGPUs]]) -> tuple[str, int, int]:
    """Sort by GPU type, then by achievable world size (desc), then by number of GPUs per node (desc)."""
    (gpu_type, idle_gpus_per_node), node_group = x
    return gpu_type, idle_gpus_per_node * len(node_group), idle_gpus_per_node


def run_search(min_world_size: int, *args, **kwargs) -> Generator[tuple[str, int, int, set[NodeGPUs]]]:
    nodes = set(find_nodes(*args, **kwargs))
    node_groups = organize_node_groups(nodes)
    for (gpu_type, min_idle_gpus_per_node), node_group in sorted(node_groups.items(), key=_sort_key, reverse=True):
        world_size = min_idle_gpus_per_node * len(node_group)
        if world_size >= min_world_size:
            yield gpu_type, min_idle_gpus_per_node, world_size, node_group
