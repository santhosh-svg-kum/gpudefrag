"""Tests for the demand-driven defrag model."""
import itertools

from gpudefrag_solver.defrag import solve_defrag
from gpudefrag_solver.pb.gpudefrag.v1 import solver_pb2 as pb


def node(i, gpus, cpu=64000):
    return pb.Node(id=i, cpu_left=cpu, mem_left=1 << 20, gpu_left=gpus)


def mv(i, n, gpus, milli, cost, num=1):
    return pb.Movable(id=i, node=n, gpus=gpus, cpu=1000, mem=1, gpu_num=num, gpu_milli=milli, cost=cost)


def blocked(num, milli, cands):
    return pb.Pod(id=0, cpu=1000, mem=1, gpu_num=num, gpu_milli=milli, candidates=cands)


def req(nodes, b, movable, k=4):
    return pb.DefragRequest(nodes=nodes, blocked=[b], movable=movable, max_moves=k, time_limit_s=5, workers=4, deterministic=True)


def brute(nodes, b, movable, k):
    """Minimum total cost of a feasible plan, or None."""
    ids = [n.id for n in nodes]
    best = None
    opts = []
    for q in movable:
        o = [None]
        for nid in ids:
            if nid == q.node:
                continue
            n = next(n for n in nodes if n.id == nid)
            for gs in itertools.combinations(range(len(n.gpu_left)), q.gpu_num):
                o.append((nid, gs))
        opts.append(o)
    for combo in itertools.product(*opts):
        if sum(c is not None for c in combo) > k:
            continue
        left = {n.id: list(n.gpu_left) for n in nodes}
        for q, c in zip(movable, combo):
            if c is not None:
                for g in q.gpus:
                    left[q.node][g] += q.gpu_milli
        ok = True
        for q, c in zip(movable, combo):
            if c is not None:
                for g in c[1]:
                    left[c[0]][g] -= q.gpu_milli
        if any(v < 0 for l in left.values() for v in l):
            continue
        for t in b.candidates:
            fits = sum(v >= b.gpu_milli for v in left[t]) >= b.gpu_num
            if fits:
                cost = sum(q.cost for q, c in zip(movable, combo) if c is not None)
                best = cost if best is None else min(best, cost)
    return best


def check_plan(nodes, b, movable, resp):
    left = {n.id: list(n.gpu_left) for n in nodes}
    by_id = {q.id: q for q in movable}
    for m in resp.moves:
        q = by_id[m.pod]
        for g in q.gpus:
            left[q.node][g] += q.gpu_milli
    for m in resp.moves:
        q = by_id[m.pod]
        assert m.to_node != q.node and len(m.gpus) == q.gpu_num
        for g in m.gpus:
            left[m.to_node][g] -= q.gpu_milli
    (p,) = resp.placements
    assert p.node in b.candidates and len(p.gpus) == b.gpu_num
    for g in p.gpus:
        left[p.node][g] -= b.gpu_milli
    assert all(v >= 0 for l in left.values() for v in l), left


def test_two_share_pods_move_to_free_a_node():
    # A: two 500m pods on separate GPUs; B: one GPU busy, one free.
    nodes = [node(0, [500, 500]), node(1, [0, 1000])]
    movable = [mv(1, 0, [0], 500, 10), mv(2, 0, [1], 500, 20), mv(3, 1, [0], 1000, 5)]
    b = blocked(2, 1000, [0, 1])
    resp = solve_defrag(req(nodes, b, movable))
    assert resp.status == "OPTIMAL"
    check_plan(nodes, b, movable, resp)
    assert resp.cost == brute(nodes, b, movable, 4) == 30
    assert sorted(m.pod for m in resp.moves) == [1, 2]


def test_respects_max_moves():
    nodes = [node(0, [500, 500]), node(1, [0, 1000])]
    movable = [mv(1, 0, [0], 500, 10), mv(2, 0, [1], 500, 20), mv(3, 1, [0], 1000, 5)]
    resp = solve_defrag(req(nodes, blocked(2, 1000, [0, 1]), movable, k=1))
    assert resp.status == "INFEASIBLE" and not resp.moves


def test_picks_cheapest_target():
    nodes = [node(0, [0, 1000]), node(1, [1000, 0]), node(2, [1000, 1000])]
    movable = [mv(1, 0, [0], 1000, 50), mv(2, 1, [1], 1000, 7)]
    b = blocked(2, 1000, [0, 1])
    resp = solve_defrag(req(nodes, b, movable))
    check_plan(nodes, b, movable, resp)
    assert resp.cost == brute(nodes, b, movable, 4) == 7


def test_respects_destinations():
    nodes = [node(0, [500, 500]), node(1, [0, 1000]), node(2, [1000, 1000])]
    movable = [mv(1, 0, [0], 500, 10), mv(2, 0, [1], 500, 20)]
    for q in movable:
        q.dests.append(1)  # node 2 has the wrong GPU type for them
    b = blocked(2, 1000, [0])
    resp = solve_defrag(req(nodes, b, movable))
    check_plan(nodes, b, movable, resp)
    assert all(m.to_node == 1 for m in resp.moves)
