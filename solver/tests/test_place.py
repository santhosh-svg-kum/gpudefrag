"""Tests for the CP-SAT placement model."""
import itertools

from hypothesis import given, settings, strategies as st

from gpupack_solver.frag import Class, NodeState, node_frag
from gpupack_solver.pb.gpupack.v1 import solver_pb2 as pb
from gpupack_solver.place import solve_place

CLASSES = [
    pb.TypicalClass(cpu=1000, gpu_milli=500, gpu_num=1, pct=0.5),
    pb.TypicalClass(cpu=2000, gpu_milli=1000, gpu_num=1, pct=0.3),
    pb.TypicalClass(cpu=4000, gpu_milli=1000, gpu_num=2, pct=0.1),
    pb.TypicalClass(cpu=1000, gpu_milli=0, gpu_num=0, pct=0.1),
]


def node(i, gpus, cpu=16000):
    return pb.Node(id=i, cpu_left=cpu, mem_left=1 << 20, gpu_left=gpus, class_access=[True] * len(CLASSES))


def py_classes():
    return [Class(c.cpu, c.gpu_milli, c.gpu_num, c.pct) for c in CLASSES]


def apply(nodes, pods, assignment):
    """Return post-state dict id->NodeState, or None if infeasible."""
    st_ = {n.id: NodeState(n.cpu_left, n.mem_left, list(n.gpu_left), list(n.class_access)) for n in nodes}
    for a in assignment:
        p = pods[a.pod]
        s = st_[a.node]
        if len(a.gpus) != p.gpu_num or len(set(a.gpus)) != len(a.gpus):
            return None
        s.cpu -= p.cpu
        s.mem -= p.mem
        for g in a.gpus:
            s.gpus[g] -= p.gpu_milli
    if any(s.cpu < 0 or s.mem < 0 or min(s.gpus, default=0) < 0 for s in st_.values()):
        return None
    return st_


def objective(nodes, pods, assignment):
    post = apply(nodes, pods, assignment)
    placed = sum(pods[a.pod].weight for a in assignment)
    frag = sum(node_frag(s, py_classes()) for s in post.values())
    return placed, frag


def brute_force(nodes, pods):
    """Best (placed, -frag) over every assignment, GPU devices included."""
    options = []
    for p in pods:
        opts = [None]
        for nid in p.candidates:
            n = next(n for n in nodes if n.id == nid)
            for gs in itertools.combinations(range(len(n.gpu_left)), p.gpu_num):
                opts.append(pb.Assignment(pod=p.id, node=nid, gpus=list(gs)))
        options.append(opts)
    best = None
    for combo in itertools.product(*options):
        asg = [a for a in combo if a is not None]
        if apply(nodes, pods, asg) is None:
            continue
        placed, frag = objective(nodes, pods, asg)
        key = (placed, -round(frag, 6))
        if best is None or key > best:
            best = key
    return best


def pod(i, cpu, num, milli, cands):
    return pb.Pod(id=i, cpu=cpu, mem=1, gpu_num=num, gpu_milli=milli, candidates=cands, weight=num * milli + 1)


def request(nodes, pods, hint=()):
    return pb.PlaceRequest(nodes=nodes, pods=pods, classes=CLASSES, hint=list(hint),
                           time_limit_s=5, deterministic=True, workers=4)


def test_frag_reference_matches_fgd_example():
    # FGD-style hand check: one 1-GPU class at 100%: frag = milli on GPUs < 1000.
    cls = [Class(1000, 1000, 1, 1.0)]
    assert node_frag(NodeState(4000, 1, [1000, 500], [True]), cls) == 500
    assert node_frag(NodeState(4000, 1, [500, 500], [True]), cls) == 1000
    assert node_frag(NodeState(0, 1, [1000, 500], [True]), cls) == 1500  # lacks CPU


def test_matches_brute_force_small():
    nodes = [node(0, [1000, 600]), node(1, [1000, 1000]), node(2, [300, 1000])]
    pods = [pod(0, 1000, 1, 400, [0, 1, 2]), pod(1, 1000, 1, 600, [0, 1, 2]),
            pod(2, 2000, 1, 1000, [0, 1, 2]), pod(3, 500, 1, 300, [0, 2])]
    resp = solve_place(request(nodes, pods))
    assert resp.status == "OPTIMAL"
    placed, frag = objective(nodes, pods, resp.assignments)
    best = brute_force(nodes, pods)
    assert placed == best[0]
    assert abs(frag + best[1]) < 1e-3, (frag, best)
    assert abs(resp.frag - frag) < 1.0  # solver's own objective agrees (int-scaled weights)


def test_respects_candidates_and_capacity():
    nodes = [node(0, [1000]), node(1, [1000])]
    pods = [pod(0, 1000, 1, 1000, [1]), pod(1, 1000, 1, 1000, [1])]
    resp = solve_place(request(nodes, pods))
    assert len(resp.assignments) == 1 and resp.assignments[0].node == 1


@settings(max_examples=25, deadline=None)
@given(st.data())
def test_random_instances_feasible_and_not_worse_than_hint(data):
    n_nodes = data.draw(st.integers(1, 3))
    nodes = [node(i, data.draw(st.lists(st.sampled_from([0, 200, 500, 1000]), min_size=1, max_size=2)),
                  cpu=data.draw(st.sampled_from([2000, 8000]))) for i in range(n_nodes)]
    pods = []
    for i in range(data.draw(st.integers(1, 4))):
        num = data.draw(st.sampled_from([0, 1, 1, 2]))
        milli = 0 if num == 0 else (1000 if num == 2 else data.draw(st.sampled_from([250, 500, 1000])))
        pods.append(pod(i, data.draw(st.sampled_from([500, 1000, 3000])), num, milli, list(range(n_nodes))))
    # greedy hint: first fit
    hint, post = [], apply(nodes, pods, [])
    for p in pods:
        for n in nodes:
            s = post[n.id]
            free = [g for g, l in enumerate(s.gpus) if l >= p.gpu_milli][: p.gpu_num] if p.gpu_num else []
            if s.cpu >= p.cpu and s.mem >= p.mem and len(free) == p.gpu_num:
                a = pb.Assignment(pod=p.id, node=n.id, gpus=free)
                hint.append(a)
                post = apply(nodes, pods, hint)
                break
    resp = solve_place(request(nodes, pods, hint))
    assert apply(nodes, pods, resp.assignments) is not None
    got, frag = objective(nodes, pods, resp.assignments)
    want, hfrag = objective(nodes, pods, hint)
    assert got >= want
    if got == want:
        assert frag <= hfrag + 1.0


def all_patterns(nodes, pods):
    """Every feasible (subset, GPU assignment) per node with its exact frag."""
    out = []
    for n in nodes:
        base = NodeState(n.cpu_left, n.mem_left, list(n.gpu_left), list(n.class_access))
        cands = [p for p in pods if n.id in p.candidates]

        def rec(i, state, asg):
            if i == len(cands):
                out.append(pb.Pattern(node=n.id, assignments=list(asg), frag=node_frag(state, py_classes())))
                return
            rec(i + 1, state, asg)  # skip pod i
            p = cands[i]
            if state.cpu < p.cpu or state.mem < p.mem:
                return
            for gs in itertools.combinations(range(len(state.gpus)), p.gpu_num):
                if all(state.gpus[g] >= p.gpu_milli for g in gs):
                    nxt = NodeState(state.cpu - p.cpu, state.mem - p.mem, list(state.gpus), state.access)
                    for g in gs:
                        nxt.gpus[g] -= p.gpu_milli
                    rec(i + 1, nxt, asg + [pb.Assignment(pod=p.id, node=n.id, gpus=list(gs))])

        rec(0, base, [])
    return out


def test_pattern_mode_matches_brute_force():
    nodes = [node(0, [1000, 600]), node(1, [1000, 1000]), node(2, [300, 1000])]
    pods = [pod(0, 1000, 1, 400, [0, 1, 2]), pod(1, 1000, 1, 600, [0, 1, 2]),
            pod(2, 2000, 1, 1000, [0, 1, 2]), pod(3, 500, 1, 300, [0, 2])]
    req = request(nodes, pods)
    req.patterns.extend(all_patterns(nodes, pods))
    resp = solve_place(req)
    assert resp.status == "OPTIMAL"
    placed, frag = objective(nodes, pods, resp.assignments)
    best = brute_force(nodes, pods)
    assert placed == best[0] and abs(frag + best[1]) < 1e-3


def test_pattern_mode_uses_one_pattern_per_node_and_each_pod_once():
    nodes = [node(0, [1000]), node(1, [1000])]
    pods = [pod(0, 1000, 1, 1000, [0, 1]), pod(1, 1000, 1, 1000, [0, 1])]
    req = request(nodes, pods)
    req.patterns.extend(all_patterns(nodes, pods))
    resp = solve_place(req)
    assert sorted(a.pod for a in resp.assignments) == [0, 1]
    assert sorted(a.node for a in resp.assignments) == [0, 1]


def test_pattern_mode_only_uses_given_patterns():
    nodes = [node(0, [1000]), node(1, [1000])]
    pods = [pod(0, 1000, 1, 1000, [0, 1]), pod(1, 1000, 1, 1000, [0, 1])]
    req = request(nodes, pods)
    req.patterns.extend([
        pb.Pattern(node=0, frag=1000), pb.Pattern(node=1, frag=1000),
        pb.Pattern(node=1, frag=0, assignments=[pb.Assignment(pod=0, node=1, gpus=[0])]),
    ])
    resp = solve_place(req)
    assert [(a.pod, a.node) for a in resp.assignments] == [(0, 1)]
    assert abs(resp.frag - 1000) < 1e-6


def test_idle_weight_prefers_packing_over_fragmentation():
    # Two choices for one share pod: an idle node (frag 0) or a used node (frag 10).
    nodes = [node(0, [1000, 1000]), node(1, [500, 1000])]
    pods = [pod(0, 1000, 1, 500, [0, 1])]
    pats = [
        pb.Pattern(node=0, frag=0), pb.Pattern(node=1, frag=500),
        pb.Pattern(node=0, frag=0, opens_idle=True, assignments=[pb.Assignment(pod=0, node=0, gpus=[0])]),
        pb.Pattern(node=1, frag=600, assignments=[pb.Assignment(pod=0, node=1, gpus=[0])]),
    ]
    # totals: idle node 0 + 500 = 500; used node 0 + 600 = 600 -> frag alone prefers opening node 0.
    for w, want in ((0, 0), (-1, 1), (200, 1), (50, 0)):
        req = request(nodes, pods)
        req.patterns.extend(pats)
        req.idle_weight = w
        (a,) = solve_place(req).assignments
        assert a.node == want, (w, a.node)
