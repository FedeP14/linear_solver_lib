package main

import (
	"fmt"

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
		fmt.Println("Error reading the matrix:", err)
		return
	}

	r, c := matrix.Dims()
	fmt.Printf("Matrix A size: %v x %v\n", r, c)
	n, _ := matrix.Dims()
	xSolution := utils.CreateSolutionVector(n)
	xApprox := mat.NewVecDense(n, nil)
	b := utils.CreateBVector(matrix, xSolution)

	tol, err := menu.SelectTolerance(params.Tolerances)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	utils.RunSolver("Jacobi Method", solver.Jacobi, matrix, b, xApprox, xSolution, tol)
	utils.RunSolver("Gauss-Seidel Method", solver.GaussSeidel, matrix, b, xApprox, xSolution, tol)
	utils.RunSolver("Gradient Method", solver.GradientMethod, matrix, b, xApprox, xSolution, tol)
	utils.RunSolver("Conjugate Gradient Method", solver.ConjugateGradient, matrix, b, xApprox, xSolution, tol)
}
