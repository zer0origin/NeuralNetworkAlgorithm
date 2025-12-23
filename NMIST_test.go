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
	//Load data

	dataStruct, err := loadPreviousData("1766459574-First-60k-NeuralNetworkSettings.json")

	TestMnistTrainingInputs := strings.Split(mnist_train, "\n")
	trainingDatasetSize := len(TestMnistTrainingInputs) - 1
	var inputs = NewMatrix(784, uint32(trainingDatasetSize))
	var results = NewMatrix(10, uint32(trainingDatasetSize))

	W_input_hidden := NewMatrixFromValueFunction(128, 784, func() float64 {
		return GenerateRandomWeight(128 * 784)
	})

	W_hidden_hidden := NewMatrixFromValueFunction(64, 128, func() float64 {
		return GenerateRandomWeight(64 * 128)
	})

	W_hidden_output := NewMatrixFromValueFunction(10, 64, func() float64 {
		return GenerateRandomWeight(10 * 64)
	})

	if err == nil {
		W_input_hidden = dataStruct.W_input_hidden
		W_hidden_hidden = dataStruct.W_hidden_hidden
		W_hidden_output = dataStruct.W_hidden_output
	}

	//Generate Input data based on CSV
	for x := 0; x < trainingDatasetSize; x++ {
		stringData := strings.Split(TestMnistTrainingInputs[x][2:], ",")

		for i := 0; i < len(stringData); i++ {
			if value, err := strconv.ParseFloat(stringData[i], 64); err != nil {
				panic(err)
			} else {
				inputs[i][x] = (value / 255 * 0.99) + 0.01
			}
		}
	}

	//Generate Result data based on CSV
	for x := 0; x < trainingDatasetSize; x++ {
		expectedValue, err := strconv.ParseInt(TestMnistTrainingInputs[x][:1], 10, 32)
		if err != nil {
			panic(err)
		}

		for i := 0; i < len(results); i++ {
			if i == int(expectedValue) {
				results[i][x] = 0.99
			} else {
				results[i][x] = 0.01
			}
		}
	}

	learningRate := 0.0000000001
	lastLoss := math.MaxFloat64
	for i := 0; ; i++ {
		X_input_hidden, err := W_input_hidden.Multiply(inputs)
		if err != nil {
			panic(err)
		}
		O_hidden_1 := X_input_hidden.ApplyFunc(Sigmoid)

		X_hidden_hidden, err := W_hidden_hidden.Multiply(O_hidden_1)
		if err != nil {
			panic(err)
		}
		O_hidden_2 := X_hidden_hidden.ApplyFunc(Sigmoid)

		X_hidden_output, err := W_hidden_output.Multiply(O_hidden_2)
		if err != nil {
			panic(err)
		}
		O_output := X_hidden_output.ApplyFunc(Sigmoid)

		errorFunction := func(T float64, rows int, columns int) float64 {
			return T - O_output[rows][columns]
		}

		lossFunction := func(T float64, rows int, columns int) float64 {
			return math.Pow(T-O_output[rows][columns], 2)
		}

		loss := results.ApplyWithIndexes(lossFunction).Sum()
		if lastLoss < loss {
			dataToSave := data{
				Loss:            loss,
				Inputs:          inputs,
				Results:         results,
				LearningRate:    learningRate,
				W_input_hidden:  W_input_hidden,
				W_hidden_hidden: W_hidden_hidden,
				W_hidden_output: W_hidden_output,
				O_output:        O_output,
			}

			bytes, err := json.Marshal(dataToSave)
			if err != nil {
				panic(err)
			}

			now := time.Now().Unix()
			fileName := fmt.Sprintf("%d-NeuralNetworkSettings.json", now)
			fmt.Printf("%s - Creating save file called %s.\n", time.Now().Format(time.UnixDate), fileName)
			err = os.WriteFile(fileName, bytes, 0644)
			if err != nil {
				panic(err)
			}

			break
		} else {
			lastLoss = loss
		}

		// Calculate Error
		E_Output := results.ApplyWithIndexes(errorFunction)
		E_Hidden_2, err := W_hidden_output.Transpose().Multiply(E_Output)
		E_Hidden_1, err := W_hidden_hidden.Transpose().Multiply(E_Hidden_2)

		//Update Weights
		DeltaW_hidden_output := W_hidden_output.GradientDescent(E_Output, O_output, O_hidden_2)
		W_hidden_output = W_hidden_output.UpdateWeights(DeltaW_hidden_output, learningRate)

		DeltaW_hidden_hidden := W_hidden_hidden.GradientDescent(E_Hidden_2, O_hidden_2, O_hidden_1)
		W_hidden_hidden = W_hidden_hidden.UpdateWeights(DeltaW_hidden_hidden, learningRate)

		DeltaW_input_hidden := W_input_hidden.GradientDescent(E_Hidden_1, O_hidden_1, inputs)
		W_input_hidden = W_input_hidden.UpdateWeights(DeltaW_input_hidden, learningRate)

		fmt.Printf("%s - %d: Network Loss: %.10f\n", time.Now().Format(time.UnixDate)[:19], i, loss)
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
