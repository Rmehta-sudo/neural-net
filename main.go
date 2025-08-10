package main

import (
	"fmt"
	"github.com/Rmehta-sudo/neural-net/engine" // Assuming this is your module path
	"math/rand"
	// "time"
)

func main() {
	fmt.Println("Welcome to Rmehta-sudo's Micrograd in Go!")
	fmt.Println("This program demonstrates a minimal autograd engine and a simple neural network built with it.")
	fmt.Println("You'll see examples of individual value operations, neuron, layer, and multi-layer perceptron (MLP) usage.")
	fmt.Println("----------------------------------------------------------------------------------------------------\n")

	// Initialize random seed for reproducible results in examples
	// For real-world applications, consider cryptographically secure randomness or `time.Now().UnixNano()`
	rand.Seed(42)

	// --- 1. Demonstrate Value Operations and Automatic Differentiation ---
	// The `engine.Value` type is the fundamental building block.
	// It allows tracking data, gradients, and the computational graph.
	engine.TestValue()

	// --- 2. Demonstrate a Single Neuron ---
	// A `engine.Neuron` takes `Value` inputs, computes a weighted sum,
	// adds a bias, and applies an activation function (Tanh).
	engine.TestNeuron()

	// --- 3. Demonstrate a Neural Network Layer ---
	// An `engine.Layer` is a collection of `engine.Neuron`s.
	// It processes inputs and produces outputs for the next layer.
	engine.TestLayer()

	// --- 4. Demonstrate a Multi-Layer Perceptron (MLP) and Training ---
	// The `engine.MLP` combines multiple `engine.Layer`s to form a deep neural network.
	// This example shows how to build, perform forward passes,
	// calculate loss, and update parameters using gradient descent.
	engine.TestMLP()

	fmt.Println("----------------------------------------------------------------------------------------------------")
	fmt.Println("All demonstrations complete! You can now explore the `engine` package files to understand the implementation.")
	fmt.Println("Refer to the README.md for more details on building and training your own networks.")
}