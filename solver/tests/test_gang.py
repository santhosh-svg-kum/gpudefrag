"""Gang (all-or-nothing) and topology-domain constraints in the pattern model."""
from gpupack_solver.pb.gpupack.v1 import solver_pb2 as pb
from gpupack_solver.place import solve_place

from test_place import CLASSES, all_patterns, apply


def dnode(i, dom, gpus):
    return pb.Node(id=i, cpu_left=64000, mem_left=1 << 20, gpu_left=gpus, class_access=[True] * len(CLASSES), domain=dom)


def gpod(i, gang, num, cands, local=True):
    return pb.Pod(id=i, cpu=1000, mem=1, gpu_num=num, gpu_milli=1000, candidates=cands,
                  weight=num * 1000 + 1, gang=gang, domain_local=local)


def solve(nodes, pods):
    req = pb.PlaceRequest(nodes=nodes, pods=pods, classes=CLASSES, time_limit_s=5, workers=4, deterministic=True)
    req.patterns.extend(all_patterns(nodes, pods))
    return solve_place(req)


def test_local_gang_stays_in_one_domain():
    # d1 has two nodes with 1 free GPU each; d2 has two nodes with 2 free each.
    nodes = [dnode(0, "d1", [1000, 0]), dnode(1, "d1", [0, 1000]), dnode(2, "d2", [1000, 1000]), dnode(3, "d2", [1000, 1000])]
    pods = [gpod(0, 7, 2, [0, 1, 2, 3]), gpod(1, 7, 2, [0, 1, 2, 3])]
    resp = solve(nodes, pods)
    assert sorted(a.pod for a in resp.assignments) == [0, 1]
    assert {a.node for a in resp.assignments} == {2, 3}
    assert apply(nodes, pods, resp.assignments) is not None


def test_gang_never_partial():
    # Only one node can take a 2-GPU pod: the 2-pod gang cannot run; the single can.
    nodes = [dnode(0, "d1", [1000, 1000]), dnode(1, "d1", [1000, 0])]
    pods = [gpod(0, 3, 2, [0, 1]), gpod(1, 3, 2, [0, 1]), gpod(2, 0, 1, [0, 1])]
    resp = solve(nodes, pods)
    placed = sorted(a.pod for a in resp.assignments)
    assert 0 not in placed and 1 not in placed and placed == [2]


def test_non_local_gang_may_span_domains():
    nodes = [dnode(0, "d1", [1000, 1000]), dnode(1, "d2", [1000, 1000])]
    pods = [gpod(0, 5, 2, [0, 1], local=False), gpod(1, 5, 2, [0, 1], local=False)]
    resp = solve(nodes, pods)
    assert sorted(a.pod for a in resp.assignments) == [0, 1]


def test_local_gang_blocked_when_split_is_only_option():
    nodes = [dnode(0, "d1", [1000, 1000]), dnode(1, "d2", [1000, 1000])]
    pods = [gpod(0, 5, 2, [0, 1]), gpod(1, 5, 2, [0, 1])]
    assert not solve(nodes, pods).assignments
