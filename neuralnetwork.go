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

func (a Matrix) UpdateWeights(gradientDecent Matrix, learningRate float64) Matrix {
	return a.ApplyWithIndexes(func(value float64, rows int, columns int) float64 {
		return value + (learningRate * gradientDecent[rows][columns])
	})
}

func (a Matrix) GradientDescent(errors, output_K, output_J Matrix) Matrix {
	outputT_J := output_J.Transpose()
	weightMatrixRows := len(a)
	weightMatrixColumns := len(a[0])
	DeltaW := NewMatrix(uint32(weightMatrixRows), uint32(weightMatrixColumns))
	for r := 0; r < weightMatrixRows; r++ {
		for c := 0; c < weightMatrixColumns; c++ {
			//fmt.Printf("Calculate W_%d_%d=%.2f corrections\n", r, c, a[r][c])

			E_k := errors[r][0]
			O_k := output_K[r][0]
			OT_J := outputT_J[0][c]

			//E_k := errors[c][0]
			//fmt.Printf("Values E_%d=%f\n", r, E_k)
			//O_k := output_K[c][0]
			//fmt.Printf("Values O_%d=%f\n", c, O_k)
			//OT_J := outputT_J[0][r]
			//fmt.Printf("Values O^T_%d=%f\n", r, OT_J)

			correction := E_k * O_k * (1 - O_k) * OT_J
			DeltaW[r][c] = correction

			//fmt.Printf("Delta W_%d_%d =%.10f\n", r, c, correction)
			//fmt.Printf("----------------------\n")
		}
	}

	return DeltaW
}

func (a Matrix) Sum() float64 {
	weightMatrixRows := len(a)
	weightMatrixColumns := len(a[0])

	var sum float64 = 0
	for r := 0; r < weightMatrixRows; r++ {
		for c := 0; c < weightMatrixColumns; c++ {
			sum = sum + a[r][c]
		}
	}

	return sum
}

func (a Matrix) Transpose() Matrix {
	colsOfA := len(a[0])
	result := make([][]float64, colsOfA)

	for c := range colsOfA {
		result[c] = a.GetMatrixColumn(uint32(c))
	}

	return NewMatrixFromValues(result...)
}

func (a Matrix) OuterProduct(b Matrix) Matrix {
	rowsOfA := len(a)
	rowsOfB := len(b)

	if rowsOfA == 0 || rowsOfB == 0 {
		return NewMatrix(0, 0)
	}

	colsOfB := len(b[0])
	result := NewMatrix(uint32(rowsOfA), uint32(colsOfB))
	for i := 0; i < rowsOfA; i++ {
		for j := 0; j < colsOfB; j++ {
			result[i][j] = a[i][0] * b[0][j]
		}
	}

	return result
}

//https://www.jeremyjordan.me/intro-to-neural-networks/
