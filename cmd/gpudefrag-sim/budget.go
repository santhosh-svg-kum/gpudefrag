package main

import "fmt"

// defaultBudgetMs is the per-batch CP-SAT time limit used when
// -solver-budget-ms is not given.
const defaultBudgetMs = 500

// budgetNote is the report-header sentence recording the solver budget, so a
// report says which budget produced its numbers.
func budgetNote(ms int) string {
	return fmt.Sprintf(". Solver budget: %d ms per batch", ms)
}
