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
		fmt.Println("Errore:", err)
		return
	}

	matrix, err := utils.ReadMatrixFromFile(matrixPath + "/" + matrixFile)
	if err != nil {
		fmt.Println("Errore nella lettura della matrice", err)
		return
	}

	fmt.Print("Matrice A dimensione: " + fmt.Sprint(matrix.Dims()) + "\n")
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
		fmt.Println("Errore:", err)
		return
	}

	// Metodo di Jacobi
	fmt.Println()
	fmt.Println("Metodo di Jacobi:")
	start := time.Now()
	x_jacobi, iterations, converged := solver.Jacobi(matrix, b, x_approx, tol, params.MaxIter)
	elapsed := time.Since(start)
	if converged {
		fmt.Printf("Tempo impiegato: %v\n", elapsed)
		fmt.Printf("Numero di iterazioni: %d\n", iterations)
		fmt.Printf("Errore relativo: %e\n", utils.CalculateRelativeError(x_jacobi, x_solution))
	} else {
		fmt.Println("Jacobi non ha converguto")
	}

	fmt.Println()
	// Metodo di Gauss-Seidel
	fmt.Println("Metodo di Gauss-Seidel:")
	start = time.Now()
	x_gs, iterations, converged := solver.GaussSeidel(matrix, b, x_approx, tol, params.MaxIter)
	elapsed = time.Since(start)
	if converged {
		fmt.Printf("Tempo impiegato: %v\n", elapsed)
		fmt.Printf("Numero di iterazioni: %d\n", iterations)
		fmt.Printf("Errore relativo: %e\n", utils.CalculateRelativeError(x_gs, x_solution))
	} else {
		fmt.Println("Gauss-Seidel non ha converguto")
	}

	fmt.Println()
	fmt.Println("Metodo del gradiente:")
	start = time.Now()
	x_gradient, iterations, converged := solver.GradientMethod(matrix, b, x_approx, tol, params.MaxIter)
	elapsed = time.Since(start)
	if converged {
		fmt.Printf("Tempo impiegato: %v\n", elapsed)
		fmt.Printf("Numero di iterazioni: %d\n", iterations)
		fmt.Printf("Errore relativo: %e\n", utils.CalculateRelativeError(x_gradient, x_solution))
	} else {
		fmt.Println("Il metodo del gradiente non ha converguto")
	}

	fmt.Println()
	// Metodo del gradiente coniugato
	fmt.Println("Metodo del gradiente coniugato:")
	start = time.Now()
	x_cg, iterations, converged := solver.ConjugateGradient(matrix, b, x_approx, tol, params.MaxIter)
	elapsed = time.Since(start)
	if converged {
		fmt.Printf("Tempo impiegato: %v\n", elapsed)
		fmt.Printf("Numero di iterazioni: %d\n", iterations)
		fmt.Printf("Errore relativo: %e\n", utils.CalculateRelativeError(x_cg, x_solution))
	} else {
		fmt.Println("Il metodo del gradiente coniugato non ha converguto")
	}
}
