package engine

import (
	"fmt"
	"math/rand"
)

// Neuron represents a single neuron in a neural network layer.
// It contains a slice of weights and a bias, both as Value objects.
type Neuron struct {
	Weights []*Value
	Bias    *Value
}

// String provides a formatted string representation of a Neuron.
func (n *Neuron) String() string {
	// Showing only the data and grad for weights and bias for conciseness
	weightData := make([]float64, len(n.Weights))
	weightGrad := make([]float64, len(n.Weights))
	for i, w := range n.Weights {
		weightData[i] = w.Data
		weightGrad[i] = w.Grad
	}
	return fmt.Sprintf("Neuron(\n  Weights Data: %.4f,\n  Weights Grad: %.4f,\n  Bias Data: %.4f,\n  Bias Grad: %.4f\n)",
		weightData, weightGrad, n.Bias.Data, n.Bias.Grad)
}

// NewNeuron creates and returns a new Neuron with 'numIn' input connections.
// Weights and bias are initialized with random values between -1 and 1.
func NewNeuron(numIn int) *Neuron {
	neur := Neuron{
		Weights: make([]*Value, numIn),
		Bias:    NewValue(rand.Float64()*2-1, "b"), // Bias initialized randomly
	}

	for i := 0; i < numIn; i++ {
		neur.Weights[i] = NewValue(rand.Float64()*2-1, fmt.Sprintf("w%d", i+1)) // Weights initialized randomly
	}

	return &neur
}

// Output computes the output of the neuron given a slice of input Values.
// It calculates the weighted sum of inputs plus bias, then applies the Tanh activation.
func (neur *Neuron) Output(inputs []*Value) *Value {
	// Note: Input validation (checking len(inputs) == len(neur.Weights)) is omitted here as per instructions,
	// but would typically be added for robustness.

	out := neur.Bias // Start with bias

	// Compute weighted sum
	for i := range neur.Weights {
		out = out.Add(neur.Weights[i].Mul(inputs[i]))
	}
	out.Label = "neuron_raw_output" // Label the raw sum before activation

	// Apply Tanh activation
	out = out.Tanh()
	out.Label = "neuron_output" // Label the final activated output
	return out
}

// Parameters returns a slice containing all trainable parameters (weights and bias) of the neuron.
func (neur *Neuron) Parameters() []*Value {
	return append(neur.Weights, neur.Bias)
}

// TestNeuron demonstrates the usage of the Neuron type.
// It creates a neuron, feeds it inputs, calculates the output,
// and then performs backpropagation to compute gradients for its parameters.
func TestNeuron() {
	fmt.Println("--- Testing Neuron ---")
	xs := make([]*Value, 5)
	for i := range xs {
		xs[i] = NewValue(float64(2*i), fmt.Sprintf("x%d", i+1))
	}
	n := NewNeuron(5)

	fmt.Println("Initial Neuron State:")
	fmt.Println(n)
	fmt.Printf("Inputs: %v\n\n", xs)

	res := n.Output(xs)
	res.Label = "neuron_final_result"
	fmt.Printf("Result (before backprop): %v\n\n", res)

	res.FullBackward()

	fmt.Printf("Result (after backprop): %v\n", res)
	fmt.Println("Neuron State (gradients computed):")
	fmt.Println(n) // This will now show updated gradients for weights and bias
	fmt.Printf("Inputs (gradients computed): %v\n", xs)
	fmt.Println("--- End TestNeuron ---")
	fmt.Println()
}

