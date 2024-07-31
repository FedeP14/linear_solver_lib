package utils

import (
	"fmt"
	"linear_system_lib/params"
	"time"

	"gonum.org/v1/gonum/mat"
)

// Convergence verifica se la soluzione attuale x è sufficientemente vicina alla soluzione esatta
func Convergence(A *mat.Dense, x *mat.VecDense, b *mat.VecDense, tol float64) bool {
	var r mat.VecDense
	r.MulVec(A, x)
	r.SubVec(&r, b)
	normB := mat.Norm(b, 2)
	normR := mat.Norm(&r, 2)
	return normR/normB < tol
}

// CalculateRelativeError calcola l'errore relativo tra la soluzione attuale e la soluzione esatta
func CalculateRelativeError(x *mat.VecDense, x_solution *mat.VecDense) float64 {
	var diff mat.VecDense
	diff.SubVec(x, x_solution)
	return mat.Norm(&diff, 2) / mat.Norm(x_solution, 2)
}

func CreateSolutionVector(size int) *mat.VecDense {
	xSolution := mat.NewVecDense(size, nil)
	for i := 0; i < size; i++ {
		xSolution.SetVec(i, 1.0)
	}
	return xSolution
}

func CreateBVector(matrix *mat.Dense, xSolution *mat.VecDense) *mat.VecDense {
	n, _ := matrix.Dims()
	b := mat.NewVecDense(n, nil)
	b.MulVec(matrix, xSolution)
	return b
}

func RunSolver(name string, solverFunc func(*mat.Dense, *mat.VecDense, *mat.VecDense, float64, int) (*mat.VecDense, int, bool), matrix *mat.Dense, b, xApprox, xSolution *mat.VecDense, tol float64) {
	fmt.Println()
	fmt.Println(name + ":")
	start := time.Now()
	xResult, iterations, converged := solverFunc(matrix, b, xApprox, tol, params.MaxIter)
	elapsed := time.Since(start)
	if converged {
		fmt.Printf("Time taken: %v\n", elapsed)
		fmt.Printf("Number of iterations: %d\n", iterations)
		fmt.Printf("Relative error: %e\n", CalculateRelativeError(xResult, xSolution))
	} else {
		fmt.Println(name + " did not converge")
	}
}
