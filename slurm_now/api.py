import json
import logging
import re
import subprocess
from collections import defaultdict
from collections.abc import Generator
from functools import reduce
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

GRES_REGEX = re.compile(r"^(?P<category>\w+):(?P<type>\w+):(?P<number>\d+).*$")
MiB_to_GB = 2**20 / 10**9


def get_node_info() -> list[dict[str, Any]]:
    cmd = ["scontrol", "show", "node", "--all", "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
    data = json.loads(result.stdout)
    if data["errors"]:
        raise RuntimeError(f"Failed to get node info: {data['errors']}")
    return data["nodes"]


class NodeGPUs(NamedTuple):
    """A logical grouping of a set of homogeneous GPUs on a single node."""

    node_name: str
    gpu_type: str
    idle_gpus: int
    idle_cpu_per_gpu: float
    idle_sys_mem_per_gpu_GB: float
    state: set[str]
    partitions: set[str]

    def __hash__(self) -> int:
        return hash((self.node_name, self.gpu_type))

    @classmethod
    def from_dict(cls: type["NodeGPUs"], node: dict[str, Any]) -> "NodeGPUs":
        # get configured GPUs
        gres_match = GRES_REGEX.match(node["gres"])
        if not gres_match:
            raise RuntimeError(f"Node {node['name']} has invalid GRES: {node['gres']}")
        gpu_type = gres_match.group("type")
        number = int(gres_match.group("number"))

        # get idle GPUs
        gres_used_match = GRES_REGEX.match(node["gres_used"])
        if not gres_used_match:
            raise RuntimeError(f"Node {node['name']} has invalid GRES: {node['gres_used']}")
        if gpu_type != gres_used_match.group("type"):
            raise ValueError(f"Node {node['name']} has {gpu_type=} but {gres_used_match.group('type')=}")
        used_gpus = int(gres_used_match.group("number"))

        idle_gpus = number - used_gpus
        idle_cpus = node["effective_cpus"] - node["alloc_cpus"]

        return cls(
            node_name=node["name"],
            gpu_type=gpu_type,
            idle_gpus=idle_gpus,
            idle_cpu_per_gpu=idle_cpus / idle_gpus if idle_gpus > 0 else -1,
            idle_sys_mem_per_gpu_GB=node["free_mem"]["number"] * MiB_to_GB / idle_gpus if idle_gpus > 0 else -1,
            state=set(node["state"]),
            partitions=set(node["partitions"]),
        )


def find_nodes(
    gpu_type: str = r"^.*$",
    min_cpu_per_gpu: float = 0.0,
    min_sys_mem_per_gpu_GB: float = 0.0,
    partition_regex: str = r"^(?!.*(full|interactive)).*$",
) -> set[NodeGPUs]:
    node_gpus_set: set[NodeGPUs] = set()
    for node_info in get_node_info():
        try:
            node_gpus = NodeGPUs.from_dict(node_info)
        except RuntimeError:
            logger.debug(f"Invalid node info: {node_info}")
        else:
            if (
                re.match(gpu_type, node_gpus.gpu_type)
                and node_gpus.state.issubset({"IDLE", "MIXED"})
                and node_gpus.idle_cpu_per_gpu >= min_cpu_per_gpu
                and node_gpus.idle_sys_mem_per_gpu_GB >= min_sys_mem_per_gpu_GB
                and {p for p in node_gpus.partitions if re.match(partition_regex, p)}
            ):
                node_gpus_set.add(node_gpus)
            else:
                logger.debug(f"Skipping node {node_gpus.node_name} because it does not match the criteria")
    return node_gpus_set


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


def run_search(min_world_size: int, *args, **kwargs) -> Generator[tuple[str, int, int, set[NodeGPUs], set[str]]]:
    nodes = find_nodes(*args, **kwargs)
    node_groups = organize_node_groups(nodes)
    for (gpu_type, min_idle_gpus_per_node), node_group in sorted(node_groups.items(), key=_sort_key, reverse=True):
        world_size = min_idle_gpus_per_node * len(node_group)
        if world_size >= min_world_size:
            yield (
                gpu_type,
                min_idle_gpus_per_node,
                world_size,
                node_group,
                reduce(set.intersection, (node.partitions for node in node_group)),
            )
