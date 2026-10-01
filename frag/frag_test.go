package frag

import (
	"math"
	"testing"

	"github.com/santhosh-svg-kum/gpudefrag/model"
)

// fgdTypicalPods is TestingGenerateGetTypicalPods from FGD's pkg/utils/frag_test.go.
var fgdTypicalPods = []TargetPod{
	{model.PodRes{MilliCPU: 6000, GpuMilli: 465, GpuNum: 1, GpuType: ""}, 9.33 / 100},
	{model.PodRes{MilliCPU: 8000, GpuMilli: 440, GpuNum: 1, GpuType: "2080"}, 9.15 / 100},
	{model.PodRes{MilliCPU: 8000, GpuMilli: 475, GpuNum: 1, GpuType: "T4"}, 8.76 / 100},
	{model.PodRes{MilliCPU: 8000, GpuMilli: 440, GpuNum: 1, GpuType: "P100"}, 8.72 / 100},
	{model.PodRes{MilliCPU: 2000, GpuMilli: 465, GpuNum: 1, GpuType: ""}, 8.68 / 100},
	{model.PodRes{MilliCPU: 12000, GpuMilli: 900, GpuNum: 1, GpuType: ""}, 8.65 / 100},
	{model.PodRes{MilliCPU: 4000, GpuMilli: 900, GpuNum: 1, GpuType: ""}, 8.43 / 100},
	{model.PodRes{MilliCPU: 16000, GpuMilli: 678, GpuNum: 1, GpuType: "T4"}, 8.36 / 100},
	{model.PodRes{MilliCPU: 8000, GpuMilli: 500, GpuNum: 1, GpuType: ""}, 8.29 / 100},
	{model.PodRes{MilliCPU: 6000, GpuMilli: 511, GpuNum: 1, GpuType: ""}, 8.11 / 100},
	{model.PodRes{MilliCPU: 14000, GpuMilli: 1000, GpuNum: 2, GpuType: "2080"}, 0.54 / 100},
	{model.PodRes{MilliCPU: 4000, GpuMilli: 1000, GpuNum: 1, GpuType: "2080"}, 0.43 / 100},
	{model.PodRes{MilliCPU: 32000, GpuMilli: 1000, GpuNum: 2, GpuType: "T4"}, 0.43 / 100},
	{model.PodRes{MilliCPU: 16000, GpuMilli: 1000, GpuNum: 1, GpuType: "V100M16"}, 0.40 / 100},
	{model.PodRes{MilliCPU: 64000, GpuMilli: 1000, GpuNum: 2, GpuType: ""}, 0.40 / 100},
	{model.PodRes{MilliCPU: 10000, GpuMilli: 1000, GpuNum: 2, GpuType: ""}, 0.40 / 100},
	{model.PodRes{MilliCPU: 11400, GpuMilli: 1000, GpuNum: 1, GpuType: "T4"}, 0.36 / 100},
	{model.PodRes{MilliCPU: 16000, GpuMilli: 1000, GpuNum: 1, GpuType: "T4"}, 0.36 / 100},
	{model.PodRes{MilliCPU: 4000, GpuMilli: 1000, GpuNum: 2, GpuType: ""}, 0.36 / 100},
	{model.PodRes{MilliCPU: 14000, GpuMilli: 1000, GpuNum: 2, GpuType: "V100M16"}, 0.36 / 100},
	{model.PodRes{MilliCPU: 8000, GpuMilli: 1000, GpuNum: 4, GpuType: ""}, 0.36 / 100},
	{model.PodRes{MilliCPU: 16000, GpuMilli: 1000, GpuNum: 2, GpuType: ""}, 0.32 / 100},
	{model.PodRes{MilliCPU: 2000, GpuMilli: 1000, GpuNum: 1, GpuType: "T4"}, 0.32 / 100},
	{model.PodRes{MilliCPU: 6000, GpuMilli: 1000, GpuNum: 1, GpuType: ""}, 0.32 / 100},
	{model.PodRes{MilliCPU: 4000, GpuMilli: 1000, GpuNum: 1, GpuType: ""}, 0.32 / 100},
	{model.PodRes{MilliCPU: 5000, GpuMilli: 1000, GpuNum: 1, GpuType: ""}, 0.32 / 100},
	{model.PodRes{MilliCPU: 32000, GpuMilli: 1000, GpuNum: 4, GpuType: "V100M16"}, 0.32 / 100},
	{model.PodRes{MilliCPU: 32000, GpuMilli: 1000, GpuNum: 2, GpuType: ""}, 0.32 / 100},
	{model.PodRes{MilliCPU: 24000, GpuMilli: 1000, GpuNum: 8, GpuType: "2080"}, 0.32 / 100},
	{model.PodRes{MilliCPU: 40000, GpuMilli: 1000, GpuNum: 4, GpuType: ""}, 0.29 / 100},
	{model.PodRes{MilliCPU: 32000, GpuMilli: 1000, GpuNum: 8, GpuType: ""}, 0.29 / 100},
	{model.PodRes{MilliCPU: 32000, GpuMilli: 1000, GpuNum: 1, GpuType: "T4"}, 0.29 / 100},
	{model.PodRes{MilliCPU: 16000, GpuMilli: 1000, GpuNum: 1, GpuType: ""}, 0.25 / 100},
	{model.PodRes{MilliCPU: 7000, GpuMilli: 1000, GpuNum: 1, GpuType: "V100M16"}, 0.25 / 100},
	{model.PodRes{MilliCPU: 24000, GpuMilli: 1000, GpuNum: 1, GpuType: "T4"}, 0.25 / 100},
}

func node1080(cpuLeft int64, gpuLeft ...int64) *model.NodeRes {
	return &model.NodeRes{Name: "n", CPUCap: 64000, CPULeft: cpuLeft, GpuType: "1080", GpuLeft: gpuLeft}
}

func near(a, b float64) bool { return math.Abs(a-b) < 0.01 }

// Expected values are FGD's own (TestNodeGpuShareFragAmountScore).
func TestNodeScoreMatchesFGD(t *testing.T) {
	cases := []struct {
		node *model.NodeRes
		want float64
	}{
		{node1080(1000, 200, 1000, 1000, 500), 2566.62},
		{node1080(1000, 1000, 1000, 1000, 1000), 3802.40},
		{node1080(1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000), 7604.80},
	}
	for i, c := range cases {
		if got := NodeScore(c.node, fgdTypicalPods); !near(got, c.want) {
			t.Errorf("case %d: got %.2f want %.2f", i, got, c.want)
		}
	}

	one := []TargetPod{{model.PodRes{MilliCPU: 6000, GpuMilli: 465, GpuNum: 1}, 9.33 / 100}}
	n := node1080(1000, 200, 1000, 1000, 500)
	if k := NodePodKind(n, one[0].Res); k != Q4LackCPU {
		t.Fatalf("kind %v", k)
	}
	if got := NodeScore(n, one); !near(got, 251.91) {
		t.Fatalf("lack-cpu: got %.2f", got)
	}
}

func TestGpuFragMilliMatchesFGD(t *testing.T) {
	whole2 := model.PodRes{MilliCPU: 100, GpuMilli: 1000, GpuNum: 2, GpuType: "1080"}
	cases := []struct {
		node *model.NodeRes
		pod  model.PodRes
		want int64
	}{
		{node1080(1000, 200, 1000, 1000, 500), whole2, 700},
		{node1080(1000, 1000, 1000, 1000, 1000), whole2, 0},
		{node1080(1000, 200, 1000, 1000, 500), model.PodRes{MilliCPU: 100, GpuMilli: 200, GpuNum: 2}, 0},
	}
	for i, c := range cases {
		if got := GpuFragMilli(c.node, c.pod); got != c.want {
			t.Errorf("case %d: got %d want %d", i, got, c.want)
		}
	}
}

func TestKinds(t *testing.T) {
	n := node1080(4000, 1000, 300)
	cpuPod := model.PodRes{MilliCPU: 1000}
	if NodePodKind(n, cpuPod) != XLSatisfied || NodePodKind(n, model.PodRes{MilliCPU: 9000}) != XRLackCPU {
		t.Fatal("cpu pod kinds")
	}
	if NodePodKind(n, model.PodRes{MilliCPU: 1, GpuNum: 1, GpuMilli: 500, GpuType: "T4"}) != NoAccess {
		t.Fatal("no access")
	}
	if NodePodKind(n, model.PodRes{MilliCPU: 1, GpuNum: 2, GpuMilli: 1000}) != Q2LackGPU {
		t.Fatal("lack gpu")
	}
	if NodePodKind(n, model.PodRes{MilliCPU: 9000, GpuNum: 2, GpuMilli: 1000}) != Q1LackBoth {
		t.Fatal("lack both")
	}
	if NodePodKind(n, model.PodRes{MilliCPU: 1, GpuNum: 1, GpuMilli: 300}) != Q3Satisfied {
		t.Fatal("satisfied")
	}
}
