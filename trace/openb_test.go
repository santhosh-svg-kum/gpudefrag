package trace

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/santhosh-svg-kum/gpudefrag/model"
)

func TestLoadOpenbNodes(t *testing.T) {
	nodes, err := LoadOpenbNodes("testdata/nodes.csv")
	if err != nil {
		t.Fatal(err)
	}
	if len(nodes) != 3 {
		t.Fatalf("got %d nodes", len(nodes))
	}
	n := nodes[1]
	if n.Name != "openb-node-0002" || n.CPUCap != 96000 || n.MemCap != 786432 || n.GpuNum() != 8 || n.GpuType != "V100M32" {
		t.Fatalf("bad node %+v", n)
	}
	if nodes[2].GpuType != "" || nodes[2].GpuNum() != 0 {
		t.Fatalf("cpu node %+v", nodes[2])
	}
}

func TestLoadOpenbPods(t *testing.T) {
	pods, err := LoadOpenbPods("testdata/pods.csv")
	if err != nil {
		t.Fatal(err)
	}
	want := []model.Pod{
		{Name: "openb-pod-0001", Res: model.PodRes{MilliCPU: 12000, MemMiB: 16384, GpuNum: 1, GpuMilli: 1000}},
		{Name: "openb-pod-0002", Res: model.PodRes{MilliCPU: 6000, MemMiB: 12288, GpuNum: 1, GpuMilli: 460}},
		{Name: "openb-pod-0003", Res: model.PodRes{MilliCPU: 100, MemMiB: 200}},                                                            // kube nonzero defaults
		{Name: "openb-pod-0004", Res: model.PodRes{MilliCPU: 16000, MemMiB: 32768, GpuNum: 1, GpuMilli: 1000, GpuType: "V100M16|V100M32"}}, // clamped
		{Name: "openb-pod-0005", Res: model.PodRes{MilliCPU: 88000, MemMiB: 327680, GpuNum: 8, GpuMilli: 1000}},
	}
	if len(pods) != len(want) {
		t.Fatalf("got %d pods", len(pods))
	}
	for i := range want {
		if pods[i] != want[i] {
			t.Errorf("pod %d: got %+v want %+v", i, pods[i], want[i])
		}
	}
}

func TestLoadOpenbPodsRejectsUnsupportedRows(t *testing.T) {
	head := "name,cpu_milli,memory_mib,num_gpu,gpu_milli,gpu_spec\n"
	for name, row := range map[string]string{
		"share multi-gpu":   "p,1000,10,2,500,\n",
		"gpu without milli": "p,1000,10,1,0,\n",
		"non-numeric":       "p,abc,10,0,0,\n",
	} {
		path := filepath.Join(t.TempDir(), "p.csv")
		if err := os.WriteFile(path, []byte(head+row), 0o644); err != nil {
			t.Fatal(err)
		}
		_, err := LoadOpenbPods(path)
		if err == nil || !strings.Contains(err.Error(), "row 2") {
			t.Errorf("%s: want error naming row 2, got %v", name, err)
		}
	}
}

func TestRealOpenbData(t *testing.T) {
	dir := filepath.Join("..", "data", "openb")
	if _, err := os.Stat(dir); err != nil {
		t.Skip("run make data first")
	}
	nodes, err := LoadOpenbNodes(filepath.Join(dir, "openb_node_list_gpu_node.csv"))
	if err != nil {
		t.Fatal(err)
	}
	gpus := 0
	for _, n := range nodes {
		gpus += n.GpuNum()
	}
	if len(nodes) != 1213 || gpus != 6212 {
		t.Fatalf("nodes=%d gpus=%d", len(nodes), gpus)
	}
	pods, err := LoadOpenbPods(filepath.Join(dir, "openb_pod_list_default.csv"))
	if err != nil {
		t.Fatal(err)
	}
	if len(pods) != 8152 {
		t.Fatalf("pods=%d", len(pods))
	}
}

func TestLoadOpenbJobsDurations(t *testing.T) {
	jobs, err := LoadOpenbJobs("testdata/pods.csv")
	if err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 5 || jobs[0].Duration != 12537496 || jobs[2].Duration != 5 {
		t.Fatalf("%+v", jobs)
	}
	if jobs[1].Pod.Res.GpuMilli != 460 {
		t.Fatal("pod resources must match LoadOpenbPods")
	}
}
