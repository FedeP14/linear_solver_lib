package utils

import (
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
