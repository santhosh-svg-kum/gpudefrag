# Elastic mode: merging defrag and node provisioning into one decision

Status: sketch for v2. No code, no numbers. RESULTS.md ("Defrag never
provisions nodes") names this as the open v2 item.

## Today

For one blocked pod, `mip.Defragmenter.Plan` asks the solver for at most
`MaxMoves` migrations that free a target node. The only alternatives it
weighs are "move pods" and "keep waiting". Adding a node is outside the
model, so the planner cannot say "a new node is cheaper than three moves".

## Proposed decision

One request, three kinds of action for the set of blocked pods:

1. Place on existing free capacity (no cost).
2. Migrate running pods, then place (cost: lost work + restart, as now).
3. Provision a node of type T, wait `ready_s`, then place (cost: node-seconds
   billed + extra wait for the blocked pod).

The solver picks the action set that minimizes one combined cost.

## Inputs

- Everything in `DefragRequest` today (nodes, blocked, movable, `max_moves`).
- Provisionable node templates: type, GPU count and model, CPU, memory,
  `ready_s` (provisioning latency), `price_per_s`, and a per-type cap.
- Expected wait for the blocked pods if nothing is done (queue-time proxy).
- A decommission rule: which provisioned nodes may be released and when.

## Objective (to be calibrated, not decided here)

    minimize  migration_cost
            + price_weight * provisioned_node_seconds
            + wait_weight  * sum(blocked wait until bind)

`wait_weight` relative to `price_weight` is the real policy knob. A migration
binds the blocked pod after the grace period plus restart; a provision binds
it after `ready_s`. The two weights decide which wins, and the break-even
report (restart sweep) is the existing tool for the migration side.

## Proto changes needed

- `DefragRequest`: `repeated NodeTemplate templates`, `double wait_weight`,
  `double price_weight`. Backward compatible (new optional fields).
- `DefragResponse`: `repeated Provision provisions` (template index, count,
  hosting pods), plus `provision_cost` next to `cost` and `benefit`.
- `Node` gains a flag for "virtual" nodes so templates can be modeled as
  extra candidate nodes with a fixed opening cost (a standard facility-
  location encoding in CP-SAT: boolean `open[t]` implies capacity).
- `validatePlan` must check provisioned placements against the template
  capacity and the per-type cap.

## Simulator changes needed

- `timed` needs a node-lifecycle event (provision requested, ready,
  released) and billing accounting; today the node set is fixed.
- A trace with node-count pressure (the published suites run at fixed size,
  so they cannot show provisioning benefits).

## Open questions

- Is the one-blocked-pod-at-a-time structure still right? Provisioning
  decisions are naturally batch-level (one node may host several blocked
  pods).
- How to value waiting: per-pod, per-job, or per-queue? Gang jobs make
  partial waits meaningless.
- Release policy: without a rule for returning nodes, provisioning always
  looks free over a long enough horizon.
- Should migrations onto a still-provisioning node be allowed (pipelined
  plans)? This enlarges the model a lot.
- How to validate without a real autoscaler: KWOK plus a fake provisioner
  with configurable latency is the likely route.
