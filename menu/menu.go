package menu

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func DisplayMatrices(matrixPath string) (string, error) {
	reader := bufio.NewReader(os.Stdin)
	files, err := os.ReadDir(matrixPath)
	if err != nil {
		return "", err
	}
	fmt.Println("Available matrices:")
	for i, file := range files {
		fmt.Printf("%d. %s\n", i+1, file.Name())
	}
	fmt.Println("Enter the number of the matrix to load:")
	matrixIndexInput, _ := reader.ReadString('\n')
	matrixIndex, err := strconv.Atoi(strings.TrimSpace(matrixIndexInput))
	if err != nil || matrixIndex < 1 || matrixIndex > len(files) {
		return "", fmt.Errorf("invalid index, please try again")
	}
	return files[matrixIndex-1].Name(), nil
}

func SelectTolerance(tol []float64) (float64, error) {
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Select the tolerance:\n")
	for t := range tol {
		fmt.Printf(" %d. %e", t+1, tol[t])
		fmt.Println()
	}

	tolIndexInput, _ := reader.ReadString('\n')
	tolIndex, err := strconv.Atoi(strings.TrimSpace(tolIndexInput))
	if err != nil || tolIndex < 1 || tolIndex > len(tol) {
		return 0, fmt.Errorf("invalid index, please try again")
	}
	return tol[tolIndex-1], nil
}
