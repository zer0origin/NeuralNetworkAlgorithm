package main

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMatrixProduct(t *testing.T) {
	matrixA := NewMatrixFromValues([]float64{1, 2, 3}, []float64{4, 5, 6})
	matrixB := NewMatrixFromValues([]float64{7, 8}, []float64{9, 10}, []float64{11, 12})

	product, err := matrixA.DotProduct(matrixB)
	require.NoError(t, err)

	expected := NewMatrixFromValues([]float64{58, 64}, []float64{139, 154})
	assert.Equal(t, expected, product)
}

func TestThreeLayerNeural(t *testing.T) {
	//weight * input + bias, -->> activation function
	input := NewMatrixFromValues([]float64{0.9}, []float64{0.1}, []float64{0.8})
	W_input_hidden := NewMatrixFromValues([]float64{0.9, 0.3, 0.4}, []float64{0.2, 0.8, 0.2}, []float64{0.1, 0.5, 0.6})
	W_hidden_output := NewMatrixFromValues([]float64{0.3, 0.7, 0.5}, []float64{0.6, 0.5, 0.2}, []float64{0.8, 0.1, 0.9})

	X_hidden, err := W_input_hidden.DotProduct(input)
	require.NoError(t, err)
	O_hidden := X_hidden.ApplyFunc(Sigmoid)
	X_output, err := W_hidden_output.DotProduct(O_hidden)
	require.NoError(t, err)
	O_output := X_output.ApplyFunc(Sigmoid)

	expected := NewMatrixFromValues([]float64{0.7263033450139793}, []float64{0.7085980724248232}, []float64{0.778097059561142})
	assert.Equal(t, expected, O_output)
}

func TestTranspose(t *testing.T) {
	matrix := NewMatrixFromValues([]float64{1, 2}, []float64{3, 4}, []float64{5, 6})
	tMatrix := matrix.Transpose()

	assert.Equal(t, matrix, tMatrix.Transpose())
	assert.Equal(t, tMatrix, NewMatrixFromValues([]float64{1, 3, 5}, []float64{2, 4, 6}))
}

func TestBackwardPropConcept(t *testing.T) {
	e1 := 0.8
	sumW_jkO_j := 2.3
	o1 := 0.4
	fmt.Printf("t1 - O1: %18.3f\n", e1)
	fmt.Printf("Sigmoid: %18.3f\n", Sigmoid(sumW_jkO_j))
	fmt.Printf("Sigmoid*1-Sigmoid: %8.3f\n", Sigmoid(sumW_jkO_j)*(1-Sigmoid(sumW_jkO_j)))
	fmt.Printf("Calculated: %15.4f\n", -e1*Sigmoid(sumW_jkO_j)*(1-Sigmoid(sumW_jkO_j))*o1)
	fmt.Printf("Sum: -(t-0) * Sigmoid(sumW_jkO_j) * (1-Sigmoid(sumW_jkO_j)) * oj\nCalc:-%.3f * Sigmoid(%.3f) * (1-Sigmoid(%.3f)) * %.3f\n", e1, sumW_jkO_j, sumW_jkO_j, o1)
}

func TestThreeLayerNeuralBackwardPropagation(t *testing.T) {
	//weight * input + bias, -->> activation function
	input := NewMatrixFromValues([]float64{0.9}, []float64{0.1}, []float64{0.8})
	W_input_hidden := NewMatrixFromValues(
		[]float64{0.9, 0.3, 0.4},
		[]float64{0.2, 0.8, 0.2},
		[]float64{0.1, 0.5, 0.6})

	W_hidden_output := NewMatrixFromValues(
		[]float64{0.3, 0.7, 0.5},
		[]float64{0.6, 0.5, 0.2},
		[]float64{0.8, 0.1, 0.9})

	X_hidden, err := W_input_hidden.DotProduct(input)
	require.NoError(t, err)
	O_hidden := X_hidden.ApplyFunc(Sigmoid)
	X_output, err := W_hidden_output.DotProduct(O_hidden)
	require.NoError(t, err)
	O_output := X_output.ApplyFunc(Sigmoid)

	expected := NewMatrixFromValues(
		[]float64{0.7263033450139793},
		[]float64{0.7085980724248232},
		[]float64{0.778097059561142})

	assert.Equal(t, expected, O_output)

	//Calculate decent.
	TrainingDataExample := NewMatrixFromValues(
		[]float64{0.8},
		[]float64{0.6},
		[]float64{0.778097059561142})
	e := TrainingDataExample.ApplyWithIndexes(func(value float64, rows int, columns int) float64 {
		return value - O_output[rows][columns]
	})
	fmt.Println(e)

	W_hidden_output_rows := len(W_hidden_output)
	W_hidden_output_columns := len(W_hidden_output[0])
	DeltaW_hidden_output := NewMatrix(uint32(W_hidden_output_rows), uint32(W_hidden_output_columns))
	for r := 0; r < W_hidden_output_rows; r++ {
		for c := 0; c < W_hidden_output_columns; c++ {
			fmt.Printf("Calculate W_%d_%d=%.2f corrections\n", r, c, W_hidden_output[r][c])

			E_k := e[r][0]
			fmt.Printf("Values E_%d=%f\n", r, E_k)
			O_k := O_output[c][0]
			fmt.Printf("Values O_%d=%f\n", c, O_k)
			OT_J := O_output.Transpose()[0][r]
			fmt.Printf("Values O^T_%d=%f\n", r, OT_J)

			correction := (-E_k) * O_k * (1 - O_k) * OT_J
			DeltaW_hidden_output[r][c] = correction

			fmt.Printf("Delta W_%d_%d =%.10f\n", r, c, correction)
			fmt.Printf("----------------------\n")
		}
	}

	learningRate := 0.1
	fmt.Println(DeltaW_hidden_output)
	updatedW_hidden_output := W_hidden_output.ApplyWithIndexes(func(value float64, rows int, columns int) float64 {
		return value - (learningRate * DeltaW_hidden_output[rows][columns])
	})
	fmt.Println(W_hidden_output)
	fmt.Println(updatedW_hidden_output)
}

func TestOuterProduct(t *testing.T) {
	a := NewMatrixFromValues([]float64{1, 0, -1})
	b := NewMatrixFromValues([]float64{-2, 1, 2})

	result := a.Transpose().OuterProduct(b)
	fmt.Printf("TestOuterProduct: %.0f\n", result)
}
