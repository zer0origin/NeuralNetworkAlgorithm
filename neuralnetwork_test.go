package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMatrixProduct(t *testing.T) {
	matrixA := NewMatrix(2, 3)
	rowZero := matrixA.GetMatrixRow(0)
	rowZero[0] = 1
	rowZero[1] = 2
	rowZero[2] = 3

	rowOne := matrixA.GetMatrixRow(1)
	rowOne[0] = 4
	rowOne[1] = 5
	rowOne[2] = 6

	matrixB := NewMatrix(3, 2)
	browZero := matrixB.GetMatrixRow(0)
	browZero[0] = 7
	browZero[1] = 8

	browOne := matrixB.GetMatrixRow(1)
	browOne[0] = 9
	browOne[1] = 10

	browTwo := matrixB.GetMatrixRow(2)
	browTwo[0] = 11
	browTwo[1] = 12

	product, err := matrixA.DotProduct(matrixB)
	require.NoError(t, err)

	expected := NewMatrixFromValues([]float64{58, 64}, []float64{139, 154})
	assert.Equal(t, expected, product)
}

func TestThreeLayerNeural(t *testing.T) {
	//weight * input + bias, -->> activation function
	i := NewMatrixFromValues([]float64{0.9}, []float64{0.1}, []float64{0.8})
	inputHidden := NewMatrixFromValues([]float64{0.9, 0.3, 0.4}, []float64{0.2, 0.8, 0.2}, []float64{0.1, 0.5, 0.6})
	hiddenOutput := NewMatrixFromValues([]float64{0.3, 0.7, 0.5}, []float64{0.6, 0.5, 0.2}, []float64{0.8, 0.1, 0.9})

	product, err := inputHidden.DotProduct(i)
	require.NoError(t, err)
	product.ApplyFunc(Sigmoid)

	dotProduct, err := hiddenOutput.DotProduct(product)
	require.NoError(t, err)
	dotProduct.ApplyFunc(Sigmoid)

	expected := NewMatrixFromValues([]float64{0.7263033450139793}, []float64{0.7085980724248232}, []float64{0.778097059561142})
	assert.Equal(t, expected, dotProduct)
}
