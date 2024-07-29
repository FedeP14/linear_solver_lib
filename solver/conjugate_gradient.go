package solver

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func ConjugateGradient(A *mat.Dense, b *mat.VecDense, x_approx *mat.VecDense, tol float64, maxIter int) (*mat.VecDense, int, bool) {
	n, _ := A.Dims()
	x := mat.NewVecDense(n, nil)
	x.CloneFromVec(x_approx)
	r := mat.NewVecDense(n, nil)
	temp := mat.NewVecDense(n, nil)
	temp.MulVec(A, x)
	r.SubVec(b, temp)
	p := mat.NewVecDense(n, nil)
	p.CloneFromVec(r)
	rsold := mat.Dot(r, r)

	for k := 0; k < maxIter; k++ {
		Ap := mat.NewVecDense(n, nil)
		Ap.MulVec(A, p)
		alpha := rsold / mat.Dot(p, Ap)
		x.AddScaledVec(x, alpha, p)
		scaledVec := mat.NewVecDense(n, nil)
		scaledVec.ScaleVec(alpha, Ap)
		r.SubVec(r, scaledVec)
		rsnew := mat.Dot(r, r)
		if math.Sqrt(rsnew) < tol {
			return x, k + 1, true
		}
		p.ScaleVec(rsnew/rsold, p)
		p.AddVec(r, p)
		rsold = rsnew
	}

	return x, maxIter, false
}
