"""Batch placement as a CP-SAT model optimizing FGD's fragmentation measure.

frag(n) = sum_m pct_m * Total - sum_{m GPU class, accessible} pct_m * ok_m * U_mu(m)
  Total  = sum_g L_g                      (free milli-GPU after placement)
  U_mu   = sum_g [L_g >= mu] * L_g        (free milli on devices that fit mu)
  ok_m   = [#{g: L_g >= mu_m} >= num_m] and [C >= cpu_m]

Solved lexicographically: (1) maximize placed weight, (2) minimize total frag
of touched nodes subject to (1).
"""
import time

from ortools.sat.python import cp_model

from .pb.gpupack.v1 import solver_pb2 as pb

SCALE = 1_000_000  # pct -> integer weight


def _params(req, budget):
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = max(budget, 0.01)
    s.parameters.num_workers = req.workers or 8
    if req.deterministic:
        s.parameters.interleave_search = True
    return s


class _Model:
    def __init__(self, req):
        self.req = req
        self.m = cp_model.CpModel()
        self.nodes = {n.id: n for n in req.nodes}
        self.x = {}  # (pod, node) -> bool
        self.y = {}  # (pod, node, gpu) -> bool
        self.placed = {}  # pod -> bool
        self._build_assignment()
        self._build_frag()

    def _build_assignment(self):
        m = self.m
        for p in self.req.pods:
            xs = []
            for nid in p.candidates:
                n = self.nodes[nid]
                ys = [g for g, left in enumerate(n.gpu_left) if p.gpu_num and left >= p.gpu_milli]
                if len(ys) < p.gpu_num:
                    continue
                x = m.NewBoolVar(f"x{p.id}_{nid}")
                self.x[p.id, nid] = x
                xs.append(x)
                if p.gpu_num:
                    yv = []
                    for g in ys:
                        v = m.NewBoolVar(f"y{p.id}_{nid}_{g}")
                        self.y[p.id, nid, g] = v
                        yv.append(v)
                    m.Add(sum(yv) == p.gpu_num * x)
            s = m.NewBoolVar(f"s{p.id}")
            m.Add(sum(xs) == s)
            self.placed[p.id] = s

    def _build_frag(self):
        m, req = self.m, self.req
        pods = {p.id: p for p in req.pods}
        self.frag_terms = []
        touched = {nid for (_, nid) in self.x}
        for nid in sorted(touched):
            n = self.nodes[nid]
            here = [(pid, x) for (pid, k), x in self.x.items() if k == nid]
            cpu = n.cpu_left - sum(pods[pid].cpu * x for pid, x in here)
            m.Add(cpu >= 0)
            m.Add(n.mem_left - sum(pods[pid].mem * x for pid, x in here) >= 0)
            L = []
            for g, left in enumerate(n.gpu_left):
                use = [pods[pid].gpu_milli * v for (pid, k, gg), v in self.y.items() if k == nid and gg == g]
                lv = m.NewIntVar(0, left, f"L{nid}_{g}")
                m.Add(lv == left - sum(use))
                L.append(lv)
            total_w = sum(round(c.pct * SCALE) for c in req.classes)
            self.frag_terms.append(total_w * sum(L))
            if not L:
                continue
            # per distinct per-GPU threshold mu: b[g] <=> L_g >= mu, w[g] = b*L_g
            b, U = {}, {}
            for c, acc in zip(req.classes, n.class_access):
                if c.gpu_milli == 0 or not acc or c.gpu_milli in b:
                    continue
                mu = c.gpu_milli
                bs, ws = [], []
                for g, lv in enumerate(L):
                    bv = m.NewBoolVar("")
                    m.Add(lv >= mu).OnlyEnforceIf(bv)
                    m.Add(lv <= mu - 1).OnlyEnforceIf(bv.Not())
                    wv = m.NewIntVar(0, n.gpu_left[g], "")
                    m.Add(wv == lv).OnlyEnforceIf(bv)
                    m.Add(wv == 0).OnlyEnforceIf(bv.Not())
                    bs.append(bv)
                    ws.append(wv)
                b[mu], U[mu] = bs, sum(ws)
            cap = sum(n.gpu_left)
            for c, acc in zip(req.classes, n.class_access):
                if c.gpu_milli == 0 or not acc:
                    continue
                w = round(c.pct * SCALE)
                if w == 0:
                    continue
                host = m.NewBoolVar("")
                m.Add(sum(b[c.gpu_milli]) >= c.gpu_num).OnlyEnforceIf(host)
                m.Add(sum(b[c.gpu_milli]) <= c.gpu_num - 1).OnlyEnforceIf(host.Not())
                cpu_ok = m.NewBoolVar("")
                m.Add(cpu >= c.cpu).OnlyEnforceIf(cpu_ok)
                m.Add(cpu <= c.cpu - 1).OnlyEnforceIf(cpu_ok.Not())
                ok = m.NewBoolVar("")
                m.AddBoolAnd([host, cpu_ok]).OnlyEnforceIf(ok)
                m.AddBoolOr([host.Not(), cpu_ok.Not()]).OnlyEnforceIf(ok.Not())
                z = m.NewIntVar(0, cap, "")
                m.Add(z == U[c.gpu_milli]).OnlyEnforceIf(ok)
                m.Add(z == 0).OnlyEnforceIf(ok.Not())
                self.frag_terms.append(-w * z)

    def placed_expr(self):
        pods = {p.id: p for p in self.req.pods}
        return sum(pods[pid].weight * s for pid, s in self.placed.items())

    def add_hint(self, assignments):
        self.m.ClearHints()
        chosen = {(a.pod, a.node) for a in assignments}
        gpus = {(a.pod, a.node, g) for a in assignments for g in a.gpus}
        for (pid, nid), x in self.x.items():
            self.m.AddHint(x, (pid, nid) in chosen)
        for key, y in self.y.items():
            self.m.AddHint(y, key in gpus)
        for pid, s in self.placed.items():
            self.m.AddHint(s, any(a.pod == pid for a in assignments))

    def extract(self, solver):
        out = []
        for (pid, nid), x in sorted(self.x.items()):
            if solver.BooleanValue(x):
                gs = [g for (p, k, g), y in self.y.items() if p == pid and k == nid and solver.BooleanValue(y)]
                out.append(pb.Assignment(pod=pid, node=nid, gpus=sorted(gs)))
        return out


def solve_place(req: pb.PlaceRequest) -> pb.PlaceResponse:
    t0 = time.monotonic()
    mdl = _Model(req)
    m = mdl.m
    budget = req.time_limit_s or 0.5
    if req.hint:
        mdl.add_hint(req.hint)

    placed = mdl.placed_expr()
    m.Maximize(placed)
    s1 = _params(req, 0.4 * budget)
    st1 = s1.Solve(m)
    if st1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return pb.PlaceResponse(status=s1.StatusName(st1), wall_s=time.monotonic() - t0)
    best_placed = int(s1.ObjectiveValue())
    phase1 = mdl.extract(s1)

    m.Add(placed >= best_placed)
    mdl.add_hint(phase1)
    m.Minimize(sum(mdl.frag_terms))
    s2 = _params(req, budget - (time.monotonic() - t0))
    st2 = s2.Solve(m)
    if st2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        asg, status, frag = mdl.extract(s2), s2.StatusName(st2), s2.ObjectiveValue() / SCALE
        if st1 != cp_model.OPTIMAL and status == "OPTIMAL":
            status = "FEASIBLE"  # phase 2 optimal only given a non-optimal phase-1 bound
    else:
        asg, status, frag = phase1, "FEASIBLE", float("nan")
    return pb.PlaceResponse(
        assignments=asg, status=status, wall_s=time.monotonic() - t0,
        deterministic_time=s1.deterministic_time + s2.deterministic_time,
        frag=frag, placed_weight=best_placed)
