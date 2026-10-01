"""Demand-driven defragmentation: the cheapest set of pod migrations (at most
max_moves) that lets one blocked pod fit on one of its candidate nodes.

Device-level model: every movable pod stays or moves to another node with
explicit devices; capacity on every device counts devices freed by pods that
leave, so moves can chain within one plan.
"""
import time

from ortools.sat.python import cp_model

from .pb.gpudefrag.v1 import solver_pb2 as pb


def solve_defrag(req: pb.DefragRequest) -> pb.DefragResponse:
    t0 = time.monotonic()
    m = cp_model.CpModel()
    nodes = {n.id: n for n in req.nodes}
    (b,) = req.blocked

    # Blocked pod: exactly one target node, with devices.
    tgt, yb = {}, {}
    for nid in b.candidates:
        n = nodes[nid]
        tgt[nid] = m.NewBoolVar(f"t{nid}")
        ys = []
        for g in range(len(n.gpu_left)):
            yb[nid, g] = m.NewBoolVar("")
            ys.append(yb[nid, g])
        m.Add(sum(ys) == b.gpu_num * tgt[nid])
    m.AddExactlyOne(tgt.values())

    # Movable pods: stay, or move to another node with devices.
    moved, to, y = {}, {}, {}
    for q in req.movable:
        moved[q.id] = m.NewBoolVar(f"m{q.id}")
        dests = []
        for nid, n in nodes.items():
            if nid == q.node or len(n.gpu_left) < q.gpu_num or (q.dests and nid not in q.dests):
                continue
            v = m.NewBoolVar("")
            to[q.id, nid] = v
            dests.append(v)
            ys = []
            for g in range(len(n.gpu_left)):
                y[q.id, nid, g] = m.NewBoolVar("")
                ys.append(y[q.id, nid, g])
            m.Add(sum(ys) == q.gpu_num * v)
        m.Add(sum(dests) == moved[q.id])
    m.Add(sum(moved.values()) <= req.max_moves)

    # Capacity after the plan, per device and per node.
    for nid, n in nodes.items():
        leaving = [q for q in req.movable if q.node == nid]
        for g, left in enumerate(n.gpu_left):
            freed = sum(q.gpu_milli * moved[q.id] for q in leaving if g in q.gpus)
            taken = sum(q.gpu_milli * y[q.id, nid, g] for q in req.movable if (q.id, nid, g) in y)
            if (nid, g) in yb:
                taken += b.gpu_milli * yb[nid, g]
            m.Add(left + freed - taken >= 0)
        cpu = n.cpu_left + sum(q.cpu * moved[q.id] for q in leaving) - sum(
            q.cpu * to[q.id, nid] for q in req.movable if (q.id, nid) in to)
        mem = n.mem_left + sum(q.mem * moved[q.id] for q in leaving) - sum(
            q.mem * to[q.id, nid] for q in req.movable if (q.id, nid) in to)
        if nid in tgt:
            cpu -= b.cpu * tgt[nid]
            mem -= b.mem * tgt[nid]
        m.Add(cpu >= 0)
        m.Add(mem >= 0)

    m.Minimize(sum(q.cost * moved[q.id] for q in req.movable))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = req.time_limit_s or 1.0
    s.parameters.num_workers = req.workers or 8
    if req.deterministic:
        s.parameters.interleave_search = True
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return pb.DefragResponse(status=s.StatusName(st), wall_s=time.monotonic() - t0)

    moves = []
    for q in req.movable:
        if s.BooleanValue(moved[q.id]):
            nid = next(n for (qi, n), v in to.items() if qi == q.id and s.BooleanValue(v))
            gs = [g for g in range(len(nodes[nid].gpu_left)) if s.BooleanValue(y[q.id, nid, g])]
            moves.append(pb.Move(pod=q.id, to_node=nid, gpus=gs))
    t = next(n for n, v in tgt.items() if s.BooleanValue(v))
    gs = [g for g in range(len(nodes[t].gpu_left)) if s.BooleanValue(yb[t, g])]
    return pb.DefragResponse(
        moves=moves, placements=[pb.Assignment(pod=b.id, node=t, gpus=gs)],
        status=s.StatusName(st), wall_s=time.monotonic() - t0,
        cost=int(round(s.ObjectiveValue())))
