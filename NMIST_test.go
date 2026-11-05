package main

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
)

//go:embed data/mnist_train.csv
var mnist_train string

func loadPreviousData(filename string) (data, error) {
	bytes, err := os.ReadFile(filename)
	if err != nil {
		return data{}, err
	}

	save := &data{}
	err = json.Unmarshal(bytes, save)
	if err != nil {
		return data{}, err
	}

	return *save, err
}

type data struct {
	Loss            float64 `json:"loss"`
	Inputs          Matrix  `json:"inputs"`
	Results         Matrix  `json:"results"`
	LearningRate    float64 `json:"learningRate"`
	W_input_hidden  Matrix  `json:"w_Input_Hidden"`
	W_hidden_hidden Matrix  `json:"w_Hidden_Hidden"`
	W_hidden_output Matrix  `json:"w_Hidden_Output"`
	O_output        Matrix  `json:"o_output"`
}

func TestMnistTraining(t *testing.T) {
	TestMnistTrainingInputs := strings.Split(mnist_train, "\n")
	trainingDatasetSize := len(TestMnistTrainingInputs) - 1

	fmt.Printf("Size of dataset: %d\n", trainingDatasetSize)

	W_input_hidden := NewMatrixFromValueFunction(128, 784, func() float64 {
		return GenerateRandomWeight(784)
	})
	W_hidden_hidden := NewMatrixFromValueFunction(64, 128, func() float64 {
		return GenerateRandomWeight(128)
	})
	W_hidden_output := NewMatrixFromValueFunction(10, 64, func() float64 {
		return GenerateRandomWeight(64)
	})

	learningRate := 0.1

	for epoch := 0; epoch < 10_000; epoch++ {
		totalLoss := 0.0

		for x := 0; x < trainingDatasetSize; x++ {
			line := TestMnistTrainingInputs[x]
			if len(line) < 3 {
				continue
			}

			expectedValue, _ := strconv.ParseInt(line[:1], 10, 32)
			targets := NewMatrix(10, 1)
			for i := 0; i < 10; i++ {
				if i == int(expectedValue) {
					targets[i][0] = 0.99
				} else {
					targets[i][0] = 0.01
				}
			}

			stringData := strings.Split(line[2:], ",")
			inputs := NewMatrix(784, 1)
			for i := 0; i < len(stringData); i++ {
				val, _ := strconv.ParseFloat(stringData[i], 64)
				inputs[i][0] = (val / 255 * 0.99) + 0.01
			}

			X_input_hidden, _ := W_input_hidden.Multiply(inputs)
			O_hidden_1 := X_input_hidden.ApplyFunc(Sigmoid)

			X_hidden_hidden, _ := W_hidden_hidden.Multiply(O_hidden_1)
			O_hidden_2 := X_hidden_hidden.ApplyFunc(Sigmoid)

			X_hidden_output, _ := W_hidden_output.Multiply(O_hidden_2)
			O_output := X_hidden_output.ApplyFunc(Sigmoid)

			for i := 0; i < 10; i++ {
				totalLoss += math.Pow(targets[i][0]-O_output[i][0], 2)
			}

			errorFunction := func(T float64, rows int, columns int) float64 {
				return T - O_output[rows][columns]
			}
			E_Output := targets.ApplyWithIndexes(errorFunction)

			E_Hidden_2, _ := W_hidden_output.Transpose().Multiply(E_Output)
			E_Hidden_1, _ := W_hidden_hidden.Transpose().Multiply(E_Hidden_2)

			DeltaW_hidden_output := W_hidden_output.GradientDescent(E_Output, O_output, O_hidden_2)
			W_hidden_output = W_hidden_output.UpdateWeights(DeltaW_hidden_output, learningRate)

			DeltaW_hidden_hidden := W_hidden_hidden.GradientDescent(E_Hidden_2, O_hidden_2, O_hidden_1)
			W_hidden_hidden = W_hidden_hidden.UpdateWeights(DeltaW_hidden_hidden, learningRate)

			DeltaW_input_hidden := W_input_hidden.GradientDescent(E_Hidden_1, O_hidden_1, inputs)
			W_input_hidden = W_input_hidden.UpdateWeights(DeltaW_input_hidden, learningRate)
		}

		if math.IsNaN(totalLoss) {
			fmt.Println("NAN DETECTED, BREAKING!")
			break
		}

		fmt.Printf("Epoch %d complete. Total Loss: %.5f\n", epoch, totalLoss)
	}
}

func TestCreate3Layers(t *testing.T) {
	data := Create3LayerNeuralNetwork(3, 784)
	fmt.Println(len(data))
}

func Create3LayerNeuralNetwork(layers int, nodes uint32) []Matrix {
	result := make([]Matrix, layers)
	for i := 0; i < layers; i++ {
		result[i] = NewMatrixFromValueFunction(nodes, nodes, func() float64 {
			return GenerateRandomWeight(layers * int(nodes))
		})
	}

	return result
}
