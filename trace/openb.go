// Package trace loads workload traces into the common model.
package trace

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/santhosh-svg-kum/gpudefrag/model"
)

// Kubernetes' non-zero request defaults, which FGD's simulator applies when a
// pod requests no CPU or memory.
const (
	defaultMilliCPU = 100
	defaultMemMiB   = 200
)

// LoadOpenbNodes reads an Alibaba openb node list (sn,cpu_milli,memory_mib,gpu,model).
func LoadOpenbNodes(path string) ([]*model.NodeRes, error) {
	rows, err := readCSV(path)
	if err != nil {
		return nil, err
	}
	var out []*model.NodeRes
	for _, r := range rows {
		cpu, e1 := r.int("cpu_milli")
		mem, e2 := r.int("memory_mib")
		gpu, e3 := r.int("gpu")
		if err := firstErr(e1, e2, e3); err != nil {
			return nil, err
		}
		out = append(out, model.NewNode(r.str("sn"), cpu, mem, int(gpu), r.str("model")))
	}
	return out, nil
}

// LoadOpenbPods reads an Alibaba openb pod list, applying the same
// conversions as FGD's pod_csv_to_yaml.py: gpu_milli above 1000 is clamped to
// 1000 and CPU-only pods carry no GPU milli.
func LoadOpenbPods(path string) ([]model.Pod, error) {
	rows, err := readCSV(path)
	if err != nil {
		return nil, err
	}
	var out []model.Pod
	for _, r := range rows {
		cpu, e1 := r.int("cpu_milli")
		mem, e2 := r.int("memory_mib")
		num, e3 := r.int("num_gpu")
		milli, e4 := r.int("gpu_milli")
		if err := firstErr(e1, e2, e3, e4); err != nil {
			return nil, err
		}
		res := model.PodRes{MilliCPU: cpu, MemMiB: mem, GpuNum: int(num), GpuType: r.str("gpu_spec")}
		if res.MilliCPU == 0 {
			res.MilliCPU = defaultMilliCPU
		}
		if res.MemMiB == 0 {
			res.MemMiB = defaultMemMiB
		}
		if num > 0 {
			res.GpuMilli = min(milli, model.Milli)
			if res.GpuMilli <= 0 {
				return nil, fmt.Errorf("%s row %d: num_gpu=%d but gpu_milli=%d", path, r.line, num, milli)
			}
			if num > 1 && res.GpuMilli < model.Milli {
				return nil, fmt.Errorf("%s row %d: multi-GPU share pods are unsupported", path, r.line)
			}
		}
		out = append(out, model.Pod{Name: r.str("name"), Res: res})
	}
	return out, nil
}

type row struct {
	path string
	line int
	cols map[string]int
	vals []string
}

func (r row) str(col string) string {
	i, ok := r.cols[col]
	if !ok || i >= len(r.vals) {
		return ""
	}
	v := strings.TrimSpace(r.vals[i])
	if strings.EqualFold(v, "nan") {
		return ""
	}
	return v
}

func (r row) int(col string) (int64, error) {
	s := r.str(col)
	if s == "" {
		return 0, nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("%s row %d: column %s: %w", r.path, r.line, col, err)
	}
	return int64(f), nil
}

func readCSV(path string) ([]row, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	recs, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if len(recs) == 0 {
		return nil, fmt.Errorf("%s: empty file", path)
	}
	cols := map[string]int{}
	for i, h := range recs[0] {
		cols[strings.TrimSpace(h)] = i
	}
	rows := make([]row, 0, len(recs)-1)
	for i, rec := range recs[1:] {
		rows = append(rows, row{path: path, line: i + 2, cols: cols, vals: rec})
	}
	return rows, nil
}

func firstErr(errs ...error) error {
	for _, e := range errs {
		if e != nil {
			return e
		}
	}
	return nil
}

// Job is a pod with a run duration (seconds) and, once synthesized, an
// arrival time (seconds from the start of the run).
type Job struct {
	Pod      model.Pod
	Arrive   float64
	Duration float64
	// Gang lists every pod of a multi-pod job (all-or-nothing). Empty or
	// one pod means a single-pod job described by Pod.
	Gang []model.Pod
}

// LoadOpenbJobs is LoadOpenbPods plus duration = deletion_time - creation_time.
func LoadOpenbJobs(path string) ([]Job, error) {
	pods, err := LoadOpenbPods(path)
	if err != nil {
		return nil, err
	}
	rows, err := readCSV(path)
	if err != nil {
		return nil, err
	}
	jobs := make([]Job, len(pods))
	for i, r := range rows {
		c, e1 := r.int("creation_time")
		d, e2 := r.int("deletion_time")
		if err := firstErr(e1, e2); err != nil {
			return nil, err
		}
		jobs[i] = Job{Pod: pods[i], Duration: float64(d - c)}
	}
	return jobs, nil
}
