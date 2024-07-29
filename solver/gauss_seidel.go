package solver

import (
	"linear_system_lib/utils"

	"gonum.org/v1/gonum/mat"
)

func GaussSeidel(A *mat.Dense, b *mat.VecDense, x_approx *mat.VecDense, tol float64, maxIter int) (*mat.VecDense, int, bool) {
	n, _ := A.Dims()
	x := mat.NewVecDense(n, nil)
	x.CloneFromVec(x_approx)

	for k := 0; k < maxIter; k++ {
		for i := 0; i < n; i++ {
			sum := 0.0
			for j := 0; j < n; j++ {
				if i != j {
					sum += A.At(i, j) * x.AtVec(j)
				}
			}
			x.SetVec(i, (b.AtVec(i)-sum)/A.At(i, i))
		}
		if utils.Convergence(A, x, b, tol) {
			return x, k + 1, true
		}
	}
	return x, maxIter, false
}
