package engine

import (
	"fmt"
	"math"
)

// Value represents a scalar value in the computational graph.
// It stores its data, gradient, previous values in the graph,
// the operation that created it, a label for debugging,
// and a backward function for gradient propagation.
type Value struct {
	Data     float64
	Grad     float64
	Prev     []*Value
	Op       string
	Label    string
	Backward func()
}

// String provides a formatted string representation of a Value.
func (val *Value) String() string {
	return fmt.Sprintf("Value(label='%s', data=%.4f, grad=%.4f, op='%s')", val.Label, val.Data, val.Grad, val.Op)
}

// NewValue creates and returns a new Value instance.
// It initializes the data and label, and sets up a no-op backward function by default.
func NewValue(val float64, label string) *Value {
	return &Value{
		Data:     val,
		Label:    label,
		Backward: func() {}, // Default no-op backward function
	}
}

// Add performs element-wise addition between two Values.
// It returns a new Value representing the sum and sets up its backward function.
func (a *Value) Add(b *Value) *Value {
	out := &Value{
		Data:  a.Data + b.Data,
		Grad:  0,
		Prev:  []*Value{a, b},
		Op:    "+",
		Label: "",
	}

	out.Backward = func() {
		a.Grad += out.Grad
		b.Grad += out.Grad
	}

	return out
}

// Mul performs element-wise multiplication between two Values.
// It returns a new Value representing the product and sets up its backward function.
func (a *Value) Mul(b *Value) *Value {
	out := &Value{
		Data:  a.Data * b.Data,
		Grad:  0,
		Prev:  []*Value{a, b},
		Op:    "*",
		Label: "",
	}

	out.Backward = func() {
		a.Grad += b.Data * out.Grad
		b.Grad += a.Data * out.Grad
	}

	return out
}

// Sub performs element-wise subtraction between two Values (a - b).
// It returns a new Value representing the difference and sets up its backward function.
func (a *Value) Sub(b *Value) *Value {
	// a - b can be seen as a + (-1 * b)
	negB := b.Mul(NewValue(-1.0, "")) // Create a temporary Value for -1
	negB.Label = fmt.Sprintf("-%s", b.Label)
	out := a.Add(negB)
	out.Op = "-"
	out.Label = "" // Label will be set by user or derived during graph ops
	return out
}

// Pow calculates a Value raised to a given power (a^power).
// It returns a new Value representing the result and sets up its backward function.
func (a *Value) Pow(power float64) *Value {
	out := &Value{
		Data:  math.Pow(a.Data, power),
		Grad:  0,
		Prev:  []*Value{a},
		Op:    fmt.Sprintf("**%.4f", power), // Example: **2.0 for square
		Label: "",
	}

	out.Backward = func() {
		a.Grad += out.Grad * power * math.Pow(a.Data, power-1)
	}
	return out
}

// Div performs element-wise division between two Values (a / b).
// It returns a new Value representing the quotient and sets up its backward function.
func (a *Value) Div(b *Value) *Value {
	// a / b can be seen as a * b^(-1)
	invB := b.Pow(-1.0)
	invB.Label = fmt.Sprintf("1/%s", b.Label)
	out := a.Mul(invB)
	out.Op = "/"
	out.Label = "" // Label will be set by user or derived during graph ops
	return out
}

// Tanh applies the hyperbolic tangent activation function to a Value.
// It returns a new Value representing the tanh result and sets up its backward function.
func (a *Value) Tanh() *Value {
	out := &Value{
		Data:  math.Tanh(a.Data),
		Grad:  0,
		Prev:  []*Value{a},
		Op:    "tanh",
		Label: "",
	}

	out.Backward = func() {
		a.Grad += out.Grad * (1 - out.Data*out.Data)
	}

	return out
}

// reversedCopy creates a new slice with elements copied in reverse order.
func reversedCopy[T any](s []T) []T {
	n := len(s)
	reversed := make([]T, n)
	for i := 0; i < n; i++ {
		reversed[i] = s[n-1-i]
	}
	return reversed
}

// createTopoNet performs a topological sort of the computational graph
// starting from the given Value node. It returns a slice of Value pointers
// in topological order, where each node appears after all its dependencies.
func createTopoNet(L *Value) []*Value {
	var topoNet []*Value

	var buildTopo func(*Value)
	visited := map[*Value]bool{}
	buildTopo = func(node *Value) {
		if !visited[node] {
			visited[node] = true
			for _, child := range node.Prev {
				buildTopo(child)
			}
			topoNet = append(topoNet, node)
		}
	}
	buildTopo(L)
	return reversedCopy(topoNet)
}

// FullBackward initiates the backpropagation process from the current Value node.
// It computes gradients for all preceding nodes in the computational graph.
// The gradient of the current Value is initialized to 1.0 before backpropagation.
func (v *Value) FullBackward() {
	topo := createTopoNet(v)

	// Reset all gradients in the graph to zero before starting new backprop
	for _, node := range topo {
		node.Grad = 0
	}
	v.Grad = 1.0 // Gradient of the loss with respect to itself is 1

	// Iterate in topological order and call backward functions
	for _, node := range topo {
		node.Backward()
	}
}

// TestValue demonstrates the usage of the Value type and its operations.
// It creates a simple computational graph and performs forward and backward passes.
func TestValue() {
	fmt.Println("--- Testing Value Operations and Backpropagation ---")

	a := NewValue(5, "a")
	b := NewValue(10, "b")
	c := NewValue(12, "c")
	d := NewValue(20, "d")

	// Demonstrate basic operations
	fmt.Println("Initial Values:")
	fmt.Println(a)
	fmt.Println(b)
	fmt.Println(c)
	fmt.Println(d)
	fmt.Println()

	ab := a.Mul(b)
	ab.Label = "ab"
	cd := c.Mul(d)
	cd.Label = "cd"
	acd := a.Mul(c).Mul(d)
	acd.Label = "acd"

	apcd := a.Add(cd)
	apcd.Label = "apcd"
	m := ab.Add(cd)
	m.Label = "m"
	L := m.Add(apcd).Add(acd)
	L.Label = "L"

	fmt.Println("Computational Graph Nodes (before backprop):")
	for _, node := range createTopoNet(L) {
		fmt.Println(node)
	}
	fmt.Println()

	// Perform backpropagation
	L.FullBackward()

	fmt.Println("Computational Graph Nodes (after backprop - gradients computed):")
	for _, node := range createTopoNet(L) {
		fmt.Println(node)
	}
	fmt.Println("--- End TestValue ---")
	fmt.Println()
}