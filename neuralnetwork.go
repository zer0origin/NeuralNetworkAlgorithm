package main

import (
	"errors"
	"math"
)

type Matrix [][]float64

var IncompatibleMatrix = errors.New("incompatible Matrix, operation cannot be performed")
var EmptyMatrix = errors.New("matrix is empty")

func NewMatrix(rows uint32, columns uint32) Matrix {
	return func() [][]float64 {
		zeroArr := make([][]float64, rows)
		for r := range rows {
			for range columns {
				zeroArr[r] = append(zeroArr[r], 0)
			}
		}

		return zeroArr
	}()
}

func NewMatrixFromValues(values ...[]float64) Matrix {
	return values
}

// GetMatrixRow row is 0 indexed.
func (a Matrix) GetMatrixRow(row uint32) []float64 {
	return a[row]
}

func (a Matrix) GetMatrixColumn(column uint32) []float64 {
	rowLen := len(a)
	resultArr := make([]float64, rowLen)

	for x := range rowLen {
		resultArr[x] = a[x][column]
	}

	return resultArr
}

func (a Matrix) GetMatrixColumnAsMatrix(column uint32) Matrix {
	rowLen := len(a)
	resultArr := make([][]float64, rowLen)

	for x := range rowLen {
		resultArr[x] = append(resultArr[x], a[x][column])
	}

	return resultArr
}

func (a Matrix) Multiply(b Matrix) (Matrix, error) {
	rowsInA := len(a)
	columnInA := len(a[0])
	columnInB := len(b[0])
	rowsInB := len(b)

	if rowsInA == 0 || columnInB == 0 {
		return Matrix{}, EmptyMatrix
	}

	if columnInA != rowsInB {
		return Matrix{}, IncompatibleMatrix
	}

	result := NewMatrix(uint32(rowsInA), uint32(columnInB))

	for r := range rowsInA {
		for c := range columnInB {
			for n := range columnInA {
				result[r][c] = result[r][c] + a[r][n]*b[n][c]
			}
		}
	}

	return result, nil
}

func (a Matrix) SumRows() Matrix {
	rowsInA := len(a)
	columnInA := len(a[0])

	result := NewMatrix(uint32(rowsInA), 1)

	for r := range rowsInA {
		for c := range columnInA {
			result[r][0] += a[r][c]
		}
	}

	return result
}

func (a Matrix) ApplyFunc(toApply func(float64) float64) Matrix {
	rowsOfA := len(a)
	colOfA := len(a[0])

	result := NewMatrix(uint32(rowsOfA), uint32(colOfA))

	for r := range rowsOfA {
		for c := range colOfA {
			result[r][c] = toApply(a[r][c])
		}
	}

	return result
}

func (a Matrix) ApplyWithIndexes(toApply func(value float64, rows int, columns int) float64) Matrix {
	rowsOfA := len(a)
	colOfA := len(a[0])

	result := NewMatrix(uint32(rowsOfA), uint32(colOfA))

	for r := range rowsOfA {
		for c := range colOfA {
			result[r][c] = toApply(a[r][c], r, c)
		}
	}

	return result
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}

func (a Matrix) Transpose() Matrix {
	colsOfA := len(a[0])
	result := make([][]float64, colsOfA)

	for c := range colsOfA {
		result[c] = a.GetMatrixColumn(uint32(c))
	}

	return NewMatrixFromValues(result...)
}

//https://www.jeremyjordan.me/intro-to-neural-networks/
