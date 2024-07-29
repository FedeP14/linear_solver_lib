package solver

import (
	"gonum.org/v1/gonum/mat"
)

func GradientMethod(A *mat.Dense, b *mat.VecDense, x_approx *mat.VecDense, tol float64, maxIter int) (*mat.VecDense, int, bool) {
	n, _ := A.Dims()
	x := mat.NewVecDense(n, nil)
	x.CloneFromVec(x_approx)
	r := mat.NewVecDense(n, nil)
	temp := mat.NewVecDense(n, nil)

	temp.MulVec(A, x)
	r.SubVec(b, temp)
	p := mat.NewVecDense(n, nil)
	p.CloneFromVec(r)

	for k := 0; k < maxIter; k++ {
		temp.MulVec(A, p)
		alpha := mat.Dot(r, r) / mat.Dot(p, temp)
		x.AddScaledVec(x, alpha, p)

		temp.MulVec(A, p)
		r.AddScaledVec(r, -alpha, temp)

		if mat.Norm(r, 2) < tol {
			return x, k + 1, true
		}

		beta := mat.Dot(r, r) / mat.Dot(p, p)
		p.ScaleVec(beta, p)
		p.AddVec(r, p)
	}

	return x, maxIter, false
}
