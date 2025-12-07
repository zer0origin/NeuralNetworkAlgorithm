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

	product, err := matrixA.Multiply(matrixB)
	require.NoError(t, err)

	expected := NewMatrixFromValues([]float64{58, 64}, []float64{139, 154})
	assert.Equal(t, expected, product)
}

func TestThreeLayerNeural(t *testing.T) {
	//weight * input + bias, -->> activation function
	i := NewMatrixFromValues([]float64{0.9}, []float64{0.1}, []float64{0.8})
	inputHidden := NewMatrixFromValues([]float64{0.9, 0.3, 0.4}, []float64{0.2, 0.8, 0.2}, []float64{0.1, 0.5, 0.6})
	hiddenOutput := NewMatrixFromValues([]float64{0.3, 0.7, 0.5}, []float64{0.6, 0.5, 0.2}, []float64{0.8, 0.1, 0.9})

	product, err := inputHidden.Multiply(i)
	require.NoError(t, err)
	inputLayerOutput := product.ApplyFunc(Sigmoid)
	dotProduct, err := hiddenOutput.Multiply(inputLayerOutput)
	require.NoError(t, err)
	OutputLayerOutput := dotProduct.ApplyFunc(Sigmoid)

	expected := NewMatrixFromValues([]float64{0.7263033450139793}, []float64{0.7085980724248232}, []float64{0.778097059561142})
	assert.Equal(t, expected, OutputLayerOutput)
}

func TestTranspose(t *testing.T) {
	matrix := NewMatrixFromValues([]float64{1, 2}, []float64{3, 4}, []float64{5, 6})
	tMatrix := matrix.Transpose()

	assert.Equal(t, matrix, tMatrix.Transpose())
	assert.Equal(t, tMatrix, NewMatrixFromValues([]float64{1, 3, 5}, []float64{2, 4, 6}))
}

func TestThreeLayerNeuralBackwardPropagation(t *testing.T) {
	//weight * input + bias, -->> activation function
	i := NewMatrixFromValues([]float64{0.9}, []float64{0.1}, []float64{0.8})
	trainingData := NewMatrixFromValues([]float64{0.7263033450139793}, []float64{0.7085980724248232}, []float64{0.778097059561142})
	inputLayer := NewMatrixFromValues([]float64{0.9, 0.3, 0.4}, []float64{0.2, 0.8, 0.2}, []float64{0.1, 0.5, 0.6})
	hiddenOneLayer := NewMatrixFromValues([]float64{0.31, 0.7, 0.1}, []float64{0.6, 0.5, 0.1}, []float64{0.8, 0.2, 0.9})

	product, err := inputLayer.Multiply(i)
	require.NoError(t, err)
	inputLayerOutput := product.ApplyFunc(Sigmoid)
	dotProduct, err := hiddenOneLayer.Multiply(inputLayerOutput)
	require.NoError(t, err)
	OutputLayerOutput := dotProduct.ApplyFunc(Sigmoid)

	TrainingDifferenceMatrix := OutputLayerOutput.ApplyWithIndexes(func(value float64, rows int, columns int) float64 {
		return trainingData[rows][columns] - value
	})

	hiddenLayerTransposed := hiddenOneLayer.Transpose()
	outputLaterError, err := hiddenLayerTransposed.Multiply(TrainingDifferenceMatrix)
	require.NoError(t, err)

	fmt.Printf("OutputLayerOutput: %.5f\n", OutputLayerOutput)
	fmt.Printf("TrainingDifferenceMatrix: %.5f\n", TrainingDifferenceMatrix)
	fmt.Printf("hiddenLayerTransposed: %.5f\n", hiddenLayerTransposed)
	fmt.Printf("outputLaterError: %.5f\n", outputLaterError)
}
