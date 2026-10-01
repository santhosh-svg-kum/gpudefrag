// Command gpupack-sim runs gpupack experiments.
//
//	gpupack-sim calib -data data -out results/calib
package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	var err error
	switch os.Args[1] {
	case "calib":
		err = runCalib(os.Args[2:])
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: gpupack-sim calib [flags]")
	os.Exit(2)
}
