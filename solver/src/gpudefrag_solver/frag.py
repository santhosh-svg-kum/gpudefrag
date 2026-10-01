"""Pure-Python reference of FGD's node fragmentation (mirrors Go frag.NodeScore).

Used by tests to check the CP-SAT linearization; not used on the hot path.
"""
from dataclasses import dataclass


@dataclass
class Class:
    cpu: int
    gpu_milli: int
    gpu_num: int
    pct: float


@dataclass
class NodeState:
    cpu: int
    mem: int
    gpus: list
    access: list  # per class


def node_frag(n: NodeState, classes) -> float:
    total = sum(n.gpus)
    frag = 0.0
    for c, ok_type in zip(classes, n.access):
        if c.gpu_milli == 0 or not ok_type:
            frag += c.pct * total
            continue
        fitting = [l for l in n.gpus if l >= c.gpu_milli]
        if len(fitting) >= c.gpu_num and n.cpu >= c.cpu:
            frag += c.pct * (total - sum(fitting))
        else:
            frag += c.pct * total
    return frag
