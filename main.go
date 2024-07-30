package main

import (
	"fmt"
	"time"

	"gonum.org/v1/gonum/mat"

	"linear_system_lib/menu"
	"linear_system_lib/params"
	"linear_system_lib/solver"
	"linear_system_lib/utils"
)

func main() {
	matrixPath := "resources"
	matrixFile, err := menu.DisplayMatrices(matrixPath)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	matrix, err := utils.ReadMatrixFromFile(matrixPath + "/" + matrixFile)
	if err != nil {
		fmt.Println("Error reading the matrix", err)
		return
	}

	fmt.Print("Matrix A size: " + fmt.Sprint(matrix.Dims()) + "\n")
	n, _ := matrix.Dims()
	x_solution := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		x_solution.SetVec(i, 1.0)
	}
	x_approx := mat.NewVecDense(n, nil)

	b := mat.NewVecDense(n, nil)
	b.MulVec(matrix, x_solution)

	tol, err := menu.SelectTolerance(params.Tolerances)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	// Jacobi Method
	fmt.Println()
	fmt.Println("Jacobi Method:")
	start := time.Now()
	x_jacobi, iterations, converged := solver.Jacobi(matrix, b, x_approx, tol, params.MaxIter)
	elapsed := time.Since(start)
	if converged {
		fmt.Printf("Time taken: %v\n", elapsed)
		fmt.Printf("Number of iterations: %d\n", iterations)
		fmt.Printf("Relative error: %e\n", utils.CalculateRelativeError(x_jacobi, x_solution))
	} else {
		fmt.Println("Jacobi did not converge")
	}

	fmt.Println()
	// Gauss-Seidel Method
	fmt.Println("Gauss-Seidel Method:")
	start = time.Now()
	x_gs, iterations, converged := solver.GaussSeidel(matrix, b, x_approx, tol, params.MaxIter)
	elapsed = time.Since(start)
	if converged {
		fmt.Printf("Time taken: %v\n", elapsed)
		fmt.Printf("Number of iterations: %d\n", iterations)
		fmt.Printf("Relative error: %e\n", utils.CalculateRelativeError(x_gs, x_solution))
	} else {
		fmt.Println("Gauss-Seidel did not converge")
	}

	fmt.Println()
	fmt.Println("Gradient Method:")
	start = time.Now()
	x_gradient, iterations, converged := solver.GradientMethod(matrix, b, x_approx, tol, params.MaxIter)
	elapsed = time.Since(start)
	if converged {
		fmt.Printf("Time taken: %v\n", elapsed)
		fmt.Printf("Number of iterations: %d\n", iterations)
		fmt.Printf("Relative error: %e\n", utils.CalculateRelativeError(x_gradient, x_solution))
	} else {
		fmt.Println("Gradient method did not converge")
	}

	fmt.Println()
	// Conjugate Gradient Method
	fmt.Println("Conjugate Gradient Method:")
	start = time.Now()
	x_cg, iterations, converged := solver.ConjugateGradient(matrix, b, x_approx, tol, params.MaxIter)
	elapsed = time.Since(start)
	if converged {
		fmt.Printf("Time taken: %v\n", elapsed)
		fmt.Printf("Number of iterations: %d\n", iterations)
		fmt.Printf("Relative error: %e\n", utils.CalculateRelativeError(x_cg, x_solution))
	} else {
		fmt.Println("Conjugate Gradient method did not converge")
	}
}
