package timed

import (
	"math"
	"math/rand"

	"gpupack/mip"
	"gpupack/model"
	"gpupack/sim"
)

// Planner finds migrations that unblock a pending pod (mip.Defragmenter).
type Planner interface {
	Plan(c *sim.Cluster, blocked model.PodRes, running []mip.Running) (*mip.Plan, string)
}

// DefragConfig enables demand-driven defragmentation.
type DefragConfig struct {
	Planner            Planner
	Interval           float64 // min seconds between defrag ticks
	CheckpointInterval float64 // seconds of work between checkpoints
	Restart            float64 // seconds a migrated pod spends restarting
	BenefitHorizon     float64 // benefit = blocked milli-GPU x this many seconds
	Margin             float64 // execute only if cost <= benefit / (1 + Margin)
	Seed               int64
}

type DefragStats struct {
	Ticks      int
	Plans      int
	Migrations int
	LostGpuSec float64 // GPU-seconds of work lost to migration (incl. restart)
	Rejected   map[string]int
}

// phase is job i's checkpoint offset, drawn per job from the seed so that
// checkpoints are not synchronized across pods.
func (d *DefragConfig) phase(i int) float64 {
	return rand.New(rand.NewSource(d.Seed*1_000_003+int64(i))).Float64() * d.CheckpointInterval
}

// lastCheckpoint is the work offset of the latest checkpoint at or before done.
func lastCheckpoint(done, phase, interval float64) float64 {
	if done < phase {
		return 0
	}
	return phase + math.Floor((done-phase)/interval)*interval
}

func (s *state) maybeDefrag() {
	d := s.cfg.Defrag
	now := s.eng.Now()
	if now-s.lastDefrag < d.Interval {
		return
	}
	var blocked *job
	for _, j := range s.pending {
		if j.Pod.Res.GpuNum > 0 && len(s.c.Feasible(j.Pod.Res)) == 0 {
			blocked = j
			break
		}
	}
	if blocked == nil {
		return
	}
	s.lastDefrag = now
	if s.res.Defrag.Rejected == nil {
		s.res.Defrag.Rejected = map[string]int{}
	}
	s.res.Defrag.Ticks++

	var running []mip.Running
	lost := map[int]float64{}
	for _, j := range s.jobs {
		if !j.running || j.Pod.Res.GpuNum == 0 {
			continue
		}
		done := j.workDone + math.Max(0, now-j.runStart)
		l := done - lastCheckpoint(done, j.ckptPhase, d.CheckpointInterval)
		lost[j.id] = l
		running = append(running, mip.Running{ID: j.id, Node: j.node, GPUs: j.gpus, Res: j.Pod.Res,
			Cost: float64(j.Pod.Res.TotalMilliGpu()) * (l + d.Restart)})
	}
	plan, why := d.Planner.Plan(s.c, blocked.Pod.Res, running)
	if why != "" {
		s.res.Defrag.Rejected[why]++
		return
	}
	benefit := float64(blocked.Pod.Res.TotalMilliGpu()) * d.BenefitHorizon
	if plan.Cost > benefit/(1+d.Margin) {
		s.res.Defrag.Rejected["hysteresis"]++
		return
	}

	s.account()
	for _, m := range plan.Moves {
		j := s.jobs[m.ID]
		s.c.Unbind(j.node, j.Pod.Res, j.gpus)
	}
	for _, m := range plan.Moves {
		j := s.jobs[m.ID]
		done := j.workDone + math.Max(0, now-j.runStart)
		j.workDone = done - lost[j.id]
		j.running = false
		s.start(j, m.To, m.GPUs, now+d.Restart)
		s.res.Defrag.Migrations++
		s.res.Defrag.LostGpuSec += float64(j.Pod.Res.TotalMilliGpu()) / model.Milli * (lost[j.id] + d.Restart)
	}
	s.start(blocked, plan.Target, plan.TargetGPUs, now)
	rest := s.pending[:0]
	for _, j := range s.pending {
		if j != blocked {
			rest = append(rest, j)
		}
	}
	s.pending = rest
	s.res.Defrag.Plans++
	s.dirty = true
}
