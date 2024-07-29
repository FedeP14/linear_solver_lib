package utils

import (
	"bufio"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

func ReadMatrixFromFile(filename string) (*mat.Dense, error) {
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
