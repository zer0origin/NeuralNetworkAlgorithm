package main

import (
	"encoding/json"
	"fmt"
	"os"
)

type data struct {
	Loss            float64 `json:"loss"`
	LearningRate    float64 `json:"learningRate"`
	W_input_hidden  Matrix  `json:"w_Input_Hidden"`
	W_hidden_hidden Matrix  `json:"w_Hidden_Hidden"`
	W_hidden_output Matrix  `json:"w_Hidden_Output"`
}

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

func calculate(inputs Matrix) Matrix {
	var W_input_hidden Matrix
	var W_hidden_hidden Matrix
	var W_hidden_output Matrix

	weights, err := loadPreviousData("1793888030-NeuralWeights.json")
	if err != nil {
		panic(err)
	}

	W_input_hidden = weights.W_input_hidden
	W_hidden_hidden = weights.W_hidden_hidden
	W_hidden_output = weights.W_hidden_output

	fmt.Println("Weights loaded from file!")

	inputs = inputs.ApplyFunc(func(val float64) float64 {
		return (val / 255 * 0.99) + 0.01
	})

	X_input_hidden, _ := W_input_hidden.Multiply(inputs)
	O_hidden_1 := X_input_hidden.ApplyFunc(Sigmoid)

	X_hidden_hidden, _ := W_hidden_hidden.Multiply(O_hidden_1)
	O_hidden_2 := X_hidden_hidden.ApplyFunc(Sigmoid)

	X_hidden_output, _ := W_hidden_output.Multiply(O_hidden_2)
	O_output := X_hidden_output.ApplyFunc(Sigmoid)

	return O_output
}
