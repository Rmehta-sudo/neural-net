package engine

import "fmt"

// MLP represents a Multi-Layer Perceptron neural network.
// It consists of a slice of Layer objects.
type MLP struct {
	Layers []*Layer
}

// NewMLP creates and returns a new MLP (Multi-Layer Perceptron) network.
// numOuts specifies the number of neurons in each hidden and output layer.
// numIn specifies the number of input features for the first layer.
func NewMLP(numOuts []int, numIn int) *MLP {
	mlp := MLP{
		Layers: make([]*Layer, len(numOuts)),
	}

	for i := range numOuts {
		if i == 0 {
			mlp.Layers[i] = NewLayer(numIn, numOuts[0]) // First layer connects to input features
		} else {
			// Subsequent layers connect to the output of the previous layer
			mlp.Layers[i] = NewLayer(numOuts[i-1], numOuts[i])
		}
	}
	return &mlp
}

// String provides a formatted string representation of an MLP,
// detailing each layer and its neurons.
func (mlp *MLP) String() string {
	s := fmt.Sprintf("MLP with %d layers:\n", len(mlp.Layers))
	for lIdx, layer := range mlp.Layers {
		s += fmt.Sprintf("  Layer %d:\n", lIdx+1)
		s += layer.String() // Use the Layer's String method
	}
	return s
}

// Output computes the output of the MLP for a given slice of input Values.
// It performs a forward pass through all layers.
func (mlp *MLP) Output(ins []*Value) []*Value {
	var result []*Value
	for i := range mlp.Layers {
		if i == 0 {
			result = mlp.Layers[i].Output(ins) // First layer processes initial inputs
		} else {
			result = mlp.Layers[i].Output(result) // Subsequent layers process previous layer's output
		}
	}
	return result
}

// Parameters returns a slice containing all trainable parameters (weights and biases)
// from all layers within the MLP.
func (mlp *MLP) Parameters() []*Value {
	var p []*Value
	for i := range mlp.Layers {
		p = append(p, mlp.Layers[i].Parameters()...) // Collect parameters from each layer
	}
	return p
}

// ToValue2D converts a 2D slice of float64 to a 2D slice of Value pointers.
func ToValue2D(data [][]float64) [][]*Value {
	out := make([][]*Value, len(data))
	for i := range data {
		out[i] = make([]*Value, len(data[i]))
		for j := range data[i] {
			out[i][j] = NewValue(data[i][j], fmt.Sprintf("x_%d_%d", i, j))
		}
	}
	return out
}

// ToValue1D converts a 1D slice of float64 to a 1D slice of Value pointers.
func ToValue1D(data []float64) []*Value {
	out := make([]*Value, len(data))
	for i := range data {
		out[i] = NewValue(data[i], fmt.Sprintf("y_%d", i))
	}
	return out
}
/*
TestMLP demonstrates the usage and training of an MLP network.
It sets up a binary classification problem, trains the MLP using gradient descent,
and prints the loss and predictions over iterations.
Input features (xs) and target labels (ys)
Create an MLP with:
	- 3 input features (from xs)
	- First hidden layer with 4 neurons
	- Second hidden layer with 4 neurons
	- Output layer with 1 neuron (for binary classification, typically one output before thresholding)

*/
func TestMLP() {
	fmt.Println("--- Testing MLP (Multi-Layer Perceptron) Training ---")
	fmt.Println("This example demonstrates a simple binary classification task.")
	fmt.Println("The goal is to train an MLP to map 3-dimensional inputs to a single output (-1.0 or 1.0).")

	xs := [][]float64{
		{2.0, 3.0, -1.0},
		{3.0, -1.0, 0.5},
		{0.5, 1.0, 1.0},
		{1.0, 1.0, -1.0},
	}

	ys := []float64{1.0, -1.0, -1.0, 1.0} // Target values

	fmt.Printf("\nDataset:\n  Inputs (xs): %v\n  Targets (ys): %v\n", xs, ys)

	mlp := NewMLP([]int{4, 4, 1}, 3)

	fmt.Printf("\nMLP Architecture:\n%s\n", mlp.String())

	// Convert raw float64 data to Value objects
	ysVal := ToValue1D(ys)
	xsVal := ToValue2D(xs)

	// Prepare slices to hold network outputs and loss components for each data point
	mlpOut := make([]*Value, len(xs))
	ydiffSquared := make([]*Value, len(xs))

	learningRate := 0.05 // Learning rate for gradient descent
	numIterations := 100 // Number of training iterations

	// Get all trainable parameters of the MLP
	params := mlp.Parameters()

	fmt.Printf("\nStarting Training for %d Iterations...\n", numIterations)
	fmt.Printf("Learning Rate: %.4f\n\n", learningRate)

	// Training Loop
	for c := 0; c < numIterations; c++ {
		// --- Forward Pass ---
		// Calculate the network's output for each input example
		for i := 0; i < len(xs); i++ {
			// mlp.Output returns a slice, but for a single output neuron, we take the first element
			mlpOut[i] = mlp.Output(xsVal[i])[0]
		}

		// --- Calculate Loss (Mean Squared Error) ---
		// For each example, calculate (predicted_output - target_output)^2
		totalLoss := NewValue(0.0, "total_loss") // Initialize total loss for the batch
		for i := 0; i < len(xs); i++ {
			// Calculate (predicted - target)
			diff := mlpOut[i].Sub(ysVal[i])
			// Calculate (predicted - target)^2
			ydiffSquared[i] = diff.Mul(diff) // (x - y)^2
			// Sum up the squared differences to get the total loss for the batch
			totalLoss = totalLoss.Add(ydiffSquared[i])
		}
		totalLoss.Label = "total_loss_sum"

		// --- Backward Pass (Backpropagation) ---
		// Compute gradients for all parameters with respect to the total loss
		totalLoss.FullBackward()

		// --- Parameter Update (Gradient Descent) ---
		// Adjust parameters based on their gradients and the learning rate
		for _, p := range params {
			p.Data -= learningRate * p.Grad // Update parameter data
		}

		// --- Reset Gradients ---
		// Set all gradients back to zero for the next iteration's backpropagation
		for _, p := range params {
			p.Grad = 0
		}

		// --- Print Training Progress ---
		if c%5 == 0 || c == numIterations-1 { // Print every 50 iterations and at the end
			fmt.Printf("Iteration %d:\n", c)
			fmt.Printf("  Loss: %.6f\n", totalLoss.Data)
			fmt.Printf("  Target Ys: [")
			for i, y := range ysVal {
				fmt.Printf("%.4f", y.Data)
				if i < len(ysVal)-1 {
					fmt.Print(", ")
				}
			}
			fmt.Println("]")
			fmt.Printf("  Current Ys: [")
			for i, out := range mlpOut {
				fmt.Printf("%.4f", out.Data)
				if i < len(mlpOut)-1 {
					fmt.Print(", ")
				}
			}
			fmt.Println("]\n")
		}
	}

	fmt.Println("--- Training Complete ---")
	fmt.Println("To verify learned parameters, you can inspect 'mlp.Parameters()'")
	fmt.Println("--- End TestMLP ---")
	fmt.Println()
}

