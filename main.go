package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"
)

// Funzione per il metodo di Jacobi
func jacobi(A *mat.Dense, b *mat.VecDense, x_approx *mat.VecDense, tol float64, maxIter int) (*mat.VecDense, int, bool) {
	n, _ := A.Dims()
	x := mat.NewVecDense(n, nil)
	x.CloneFromVec(x_approx)
	xOld := mat.NewVecDense(n, nil)
	for k := 0; k < maxIter; k++ {
		xOld.CloneFromVec(x)
		for i := 0; i < n; i++ {
			sum := 0.0
			for j := 0; j < n; j++ {
				if i != j {
					sum += A.At(i, j) * xOld.AtVec(j)
				}
			}
			x.SetVec(i, (b.AtVec(i)-sum)/A.At(i, i))
		}
		// Controllo della convergenza
		if convergence(A, x, b, tol) {
			return x, k + 1, true
		}
	}
	return x, maxIter, false
}

// Funzione per il metodo di Gauss-Seidel
func gaussSeidel(A *mat.Dense, b *mat.VecDense, x_approx *mat.VecDense, tol float64, maxIter int) (*mat.VecDense, int, bool) {
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
		// Controllo della convergenza
		if convergence(A, x, b, tol) {
			return x, k + 1, true
		}
	}
	return x, maxIter, false
}

// Metodo del gradiente
func gradientMethod(A *mat.Dense, b *mat.VecDense, x_approx *mat.VecDense, tol float64, maxIter int) (*mat.VecDense, int, bool) {
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

// Metodo del gradiente coniugato
func conjugateGradient(A *mat.Dense, b *mat.VecDense, x_approx *mat.VecDense, tol float64, maxIter int) (*mat.VecDense, int, bool) {
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

// Funzione di controllo della convergenza
func convergence(A *mat.Dense, x *mat.VecDense, b *mat.VecDense, tol float64) bool {
	var r mat.VecDense
	r.MulVec(A, x)
	r.SubVec(&r, b)
	normB := mat.Norm(b, 2)
	normR := mat.Norm(&r, 2)
	return normR/normB < tol
}

// Funzione per calcolare l'errore relativo
func calculateRelativeError(x *mat.VecDense, x_solution *mat.VecDense) float64 {
	var diff mat.VecDense
	diff.SubVec(x, x_solution)
	return mat.Norm(&diff, 2) / mat.Norm(x_solution, 2)
}

// Funzione per leggere la matrice da un file
func readMatrixFromFile(filename string) (*mat.Dense, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var rows, cols int
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "%") {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) == 3 {
			rows, _ = strconv.Atoi(parts[0])
			cols, _ = strconv.Atoi(parts[1])
			break
		}
	}

	matrix := mat.NewDense(rows, cols, nil)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "%") {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) == 3 {
			row, _ := strconv.Atoi(parts[0])
			col, _ := strconv.Atoi(parts[1])
			val, _ := strconv.ParseFloat(parts[2], 64)
			matrix.Set(row-1, col-1, val)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return matrix, nil
}

func main() {
	reader := bufio.NewReader(os.Stdin)
	matrixPath := "resources"
	fmt.Println("Matrici disponibili:")
	files, err := os.ReadDir(matrixPath)
	if err != nil {
		fmt.Println("Errore nella lettura della directory", err)
		return
	}
	for i, file := range files {
		fmt.Printf("%d. %s\n", i+1, file.Name())
	}
	fmt.Println("Inserisci il numero della matrice da caricare:")
	matrixIndexInput, _ := reader.ReadString('\n')
	matrixIndex, err := strconv.Atoi(strings.TrimSpace(matrixIndexInput))
	if err != nil || matrixIndex < 1 || matrixIndex > len(files) {
		fmt.Println("Indice non valido, per favore riprova.")
		return
	}
	matrixFile := files[matrixIndex-1].Name()
	matrix, err := readMatrixFromFile(matrixPath + "/" + matrixFile)
	if err != nil {
		fmt.Println("Errore nella lettura della matrice", err)
		return
	}
	fmt.Print("Matrice A dimensione: " + fmt.Sprint(matrix.Dims()) + "\n")
	// Creazione di x_solution e x_approx
	n, _ := matrix.Dims()
	x_solution := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		x_solution.SetVec(i, 1.0)
	}
	x_approx := mat.NewVecDense(n, nil) // Questo non è usato nel calcolo di b, ma lo creiamo comunque

	// Calcolo di b come prodotto di a e x_solution
	b := mat.NewVecDense(n, nil)
	b.MulVec(matrix, x_solution)

	// l'utente può scegliere la tolleranza 10^−4, 10^−6, 10^−8, 10^−10 nello stesso modo in cui sceglie le matrici
	// tol è un vettore di tolleranze
	tol := []float64{1e-4, 1e-6, 1e-8, 1e-10}

	fmt.Print("Seleziona la tolleranza:\n")
	for t := range tol {
		fmt.Printf(" %d. %e", t+1, tol[t])
		fmt.Println()
	}

	// L'utente sceglie la tolleranza
	tolIndexInput, _ := reader.ReadString('\n')
	tolIndex, err := strconv.Atoi(strings.TrimSpace(tolIndexInput))
	if err != nil || tolIndex < 1 || tolIndex > len(tol) {
		fmt.Println("Indice non valido, per favore riprova.")
		return
	}
	tolIndex--
	fmt.Println("Tolleranza selezionata: ", tol[tolIndex])
	maxIter := 20000

	// Metodo di Jacobi

	fmt.Println()
	fmt.Println("Metodo di Jacobi:")
	start := time.Now()
	x_jacobi, iterations, converged := jacobi(matrix, b, x_approx, tol[tolIndex], maxIter)
	elapsed := time.Since(start)
	if converged {
		fmt.Printf("Tempo impiegato: %v\n", elapsed)
		fmt.Printf("Numero di iterazioni: %d\n", iterations)
		fmt.Printf("Errore relativo: %e\n", calculateRelativeError(x_jacobi, x_solution))
	} else {
		fmt.Println("Jacobi non ha converguto")
	}

	fmt.Println()
	// Metodo di Gauss-Seidel
	fmt.Println("Metodo di Gauss-Seidel:")
	start = time.Now()
	x_gs, iterations, converged := gaussSeidel(matrix, b, x_approx, tol[tolIndex], maxIter)
	elapsed = time.Since(start)
	if converged {
		fmt.Printf("Tempo impiegato: %v\n", elapsed)
		fmt.Printf("Numero di iterazioni: %d\n", iterations)
		fmt.Printf("Errore relativo: %e\n", calculateRelativeError(x_gs, x_solution))
	} else {
		fmt.Println("Gauss-Seidel non ha converguto")
	}

	fmt.Println()
	fmt.Println("Metodo del gradiente:")
	start = time.Now()
	x_gradient, iterations, converged := gradientMethod(matrix, b, x_approx, tol[tolIndex], maxIter)
	elapsed = time.Since(start)
	if converged {
		fmt.Printf("Tempo impiegato: %v\n", elapsed)
		fmt.Printf("Numero di iterazioni: %d\n", iterations)
		fmt.Printf("Errore relativo: %e\n", calculateRelativeError(x_gradient, x_solution))
	} else {
		fmt.Println("Il metodo del gradiente non ha converguto")
	}

	fmt.Println()
	// Metodo del gradiente coniugato
	fmt.Println("Metodo del gradiente coniugato:")
	start = time.Now()
	x_cg, iterations, converged := conjugateGradient(matrix, b, x_approx, tol[tolIndex], maxIter)
	elapsed = time.Since(start)
	if converged {
		fmt.Printf("Tempo impiegato: %v\n", elapsed)
		fmt.Printf("Numero di iterazioni: %d\n", iterations)
		fmt.Printf("Errore relativo: %e\n", calculateRelativeError(x_cg, x_solution))
	} else {
		fmt.Println("Il metodo del gradiente coniugato non ha converguto")
	}
}
