package timed

// DefragConfig enables demand-driven defragmentation (filled in by the
// defrag task).
type DefragConfig struct {
	CheckpointInterval float64
	Seed               int64
}

type DefragStats struct {
	Ticks      int
	Plans      int
	Migrations int
	LostGpuSec float64
	Rejected   map[string]int
}

func (d *DefragConfig) phase(i int) float64 { return 0 }

func (s *state) maybeDefrag() {}
