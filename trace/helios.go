package trace

import (
	"fmt"
	"sort"
	"time"

	"github.com/santhosh-svg-kum/gpudefrag/model"
)

// HeliosStats counts rows dropped while loading a Helios trace.
type HeliosStats struct {
	Start        time.Time // earliest submit time (Arrive is seconds after it)
	CPUOnly      int
	Uneven       int // gpu_num not divisible by node_num
	ZeroDuration int
}

const heliosTime = "2006-01-02 15:04:05"

// LoadHelios reads a HeliosData cluster_log.csv (SenseTime, SC '21). A job
// on node_num > 1 nodes becomes a gang of node_num pods with gpu_num/node_num
// whole GPUs each. Jobs are returned sorted by submit time; Job.Pod carries
// the job id and the first pod's resources, Job.Gang all pods.
func LoadHelios(path string) ([]Job, HeliosStats, error) {
	var st HeliosStats
	rows, err := readCSV(path)
	if err != nil {
		return nil, st, err
	}
	type raw struct {
		job    Job
		submit time.Time
	}
	var rs []raw
	for _, r := range rows {
		gpu, e1 := r.int("gpu_num")
		cpu, e2 := r.int("cpu_num")
		nodes, e3 := r.int("node_num")
		dur, e4 := r.int("duration")
		if err := firstErr(e1, e2, e3, e4); err != nil {
			return nil, st, err
		}
		switch {
		case gpu == 0:
			st.CPUOnly++
			continue
		case dur <= 0:
			st.ZeroDuration++
			continue
		case nodes <= 0 || gpu%nodes != 0:
			st.Uneven++
			continue
		}
		submit, err := time.Parse(heliosTime, r.str("submit_time"))
		if err != nil {
			return nil, st, fmt.Errorf("%s row %d: %w", path, r.line, err)
		}
		id := r.str("job_id")
		perCPU := cpu * 1000 / nodes
		if perCPU == 0 {
			perCPU = defaultMilliCPU
		}
		j := Job{Duration: float64(dur)}
		for k := int64(0); k < nodes; k++ {
			j.Gang = append(j.Gang, model.Pod{Name: fmt.Sprintf("%s-%d", id, k),
				Res: model.PodRes{MilliCPU: perCPU, GpuNum: int(gpu / nodes), GpuMilli: model.Milli}})
		}
		j.Pod = model.Pod{Name: id, Res: j.Gang[0].Res}
		rs = append(rs, raw{j, submit})
	}
	sort.SliceStable(rs, func(a, b int) bool { return rs[a].submit.Before(rs[b].submit) })
	out := make([]Job, len(rs))
	if len(rs) > 0 {
		st.Start = rs[0].submit
	}
	for i, r := range rs {
		r.job.Arrive = r.submit.Sub(st.Start).Seconds()
		out[i] = r.job
	}
	return out, st, nil
}
