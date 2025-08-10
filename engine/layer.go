package engine

import (
	"fmt"
	"math/rand"
)

// Layer represents a single layer in a neural network.
// It contains a slice of Neuron objects.
type Layer struct {
	Neurons []*Neuron
}

// String provides a formatted string representation of a Layer,
// detailing each neuron within it.
func (l *Layer) String() string {
	s := fmt.Sprintf("Layer with %d neurons:\n", len(l.Neurons))
	for i, neuron := range l.Neurons {
		s += fmt.Sprintf("  Neuron %d:\n%s\n", i+1, neuron.String())
	}
	return s
}

// NewLayer creates and returns a new Layer with 'outs' number of neurons,
// each having 'ins' input connections.
func NewLayer(ins, outs int) *Layer {
	l := Layer{
		Neurons: make([]*Neuron, outs),
	}

	for i := range l.Neurons {
		l.Neurons[i] = NewNeuron(ins) // Create each neuron in the layer
	}
	return &l
}

// Output computes the outputs of all neurons in the layer given a slice of input Values.
// It returns a slice of Value objects, one for each neuron's output.
func (l *Layer) Output(inputs []*Value) []*Value {
	out := make([]*Value, len(l.Neurons))

	for i := range l.Neurons {
		out[i] = l.Neurons[i].Output(inputs) // Get output from each neuron
		out[i].Label = fmt.Sprintf("layer_neuron_%d_output", i+1)
	}
	return out
}

// Parameters returns a slice containing all trainable parameters from all neurons within the layer.
func (l *Layer) Parameters() []*Value {
	var p []*Value
	for i := range l.Neurons {
		p = append(p, l.Neurons[i].Parameters()...) // Collect parameters from each neuron
	}
	return p
}

// TestLayer demonstrates the usage of the Layer type.
// It creates a layer, feeds it inputs, and prints the outputs and the layer's structure.
func TestLayer() {
	fmt.Println("--- Testing Layer ---")
	// For reproducibility in this example
	rand.Seed(42)

	l := NewLayer(3, 2) // A layer with 3 inputs and 2 output neurons
	xs := []*Value{
		NewValue(1.0, "x1"),
		NewValue(-2.0, "x2"),
		NewValue(3.0, "x3"),
	}

	fmt.Println("Inputs to Layer:")
	for _, x := range xs {
		fmt.Println(x)
	}
	fmt.Println()

	outs := l.Output(xs)

	fmt.Println("Layer Outputs (before backprop):")
	for _, o := range outs {
		fmt.Println(o)
	}
	fmt.Println()

	// To demonstrate backprop through a layer, let's create a dummy loss
	// and backpropagate from it.
	// For example, sum all outputs and backpropagate from the sum.
	if len(outs) > 0 {
		loss := outs[0]
		for i := 1; i < len(outs); i++ {
			loss = loss.Add(outs[i])
		}
		loss.Label = "layer_total_loss"

		fmt.Printf("Dummy Loss from Layer Outputs: %v\n\n", loss)
		loss.FullBackward()
		fmt.Printf("Dummy Loss after FullBackward: %v\n\n", loss)

		fmt.Println("Layer Outputs (after backprop - gradients computed):")
		for _, o := range outs {
			fmt.Println(o)
		}
		fmt.Println("\nLayer State (gradients computed for all neurons):")
		fmt.Println(l) // This will show updated gradients for all neuron parameters

		fmt.Println("\nInputs (gradients computed):")
		for _, x := range xs {
			fmt.Println(x)
		}
	}

	fmt.Println("--- End TestLayer ---")
	fmt.Println()
}

