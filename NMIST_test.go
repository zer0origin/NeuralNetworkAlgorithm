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
	"time"
)

//go:embed data/mnist_train.csv
var mnist_train string

func TestMnistTraining(t *testing.T) {
	loadWeights := true
	now := time.Now().Unix()

	learningRate := 0.0001
	var W_input_hidden Matrix
	var W_hidden_hidden Matrix
	var W_hidden_output Matrix

	TestMnistTrainingInputs := strings.Split(mnist_train, "\n")
	trainingDatasetSize := len(TestMnistTrainingInputs) - 1

	fmt.Printf("Size of dataset: %d\n", trainingDatasetSize)

	if !loadWeights {
		W_input_hidden = NewMatrixFromValueFunction(128, 784, func() float64 {
			return GenerateRandomWeight(784)
		})
		W_hidden_hidden = NewMatrixFromValueFunction(64, 128, func() float64 {
			return GenerateRandomWeight(128)
		})
		W_hidden_output = NewMatrixFromValueFunction(10, 64, func() float64 {
			return GenerateRandomWeight(64)
		})
	} else {
		weights, err := loadPreviousData("1793886262-NeuralWeights.json ")
		if err != nil {
			panic(err)
		}

		W_input_hidden = weights.W_input_hidden
		W_hidden_hidden = weights.W_hidden_hidden
		W_hidden_output = weights.W_hidden_output

		fmt.Printf("Weights loaded from file! Loss: %.5f\n", weights.Loss)
	}

	fmt.Printf("Learning Rate: %f\n", learningRate)

	for epoch := 0; epoch < 10_000; epoch++ {
		totalLoss := 0.0

		for x := 0; x < trainingDatasetSize; x++ {
			line := TestMnistTrainingInputs[x]
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

		weights := data{
			Loss:            totalLoss,
			LearningRate:    learningRate,
			W_input_hidden:  W_input_hidden,
			W_hidden_hidden: W_hidden_hidden,
			W_hidden_output: W_hidden_output,
		}

		bytes, err := json.Marshal(weights)
		if err != nil {
			panic(err)
		}

		fileName := fmt.Sprintf("%d-NeuralWeights.json", now)
		fmt.Printf("Creating save file called %s.\n", fileName)
		err = os.WriteFile(fileName, bytes, 0644)
		if err != nil {
			fmt.Println(fmt.Errorf("couldnt save file: %e", err))
		} else {
			fmt.Printf("Created save file.\n")
		}
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
