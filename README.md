# 🚀 Go Micrograd: A Tiny Autograd Engine and Neural Network Library

A minimalist implementation of a scalar-valued automatic differentiation engine and small neural network library in Go, inspired by Andrej Karpathy’s [micrograd](https://github.com/karpathy/micrograd). Designed to be **simple, educational, and fully self-contained**, showing the core mechanics of backpropagation and neural network training from scratch.

---

## ✨ Features
- **Scalar Autograd Engine** — Tracks data, gradients, and builds a dynamic computation graph.
- **Basic Operations & Activations** — `+`, `-`, `*`, `/`, `^`, `tanh` with automatic gradient calculation.
- **Neural Network Components**:
  - **Neuron** — Single perceptron with weights, bias, and `tanh` activation.
  - **Layer** — A collection of neurons.
  - **MLP** — Multi-Layer Perceptron with multiple layers.
- **Gradient Descent Training** — Simple loop for updating parameters.
- **Readable Code** — Easy to follow and modify.

---

## 🏁 Getting Started

### Prerequisites
- Go **1.18+**

### Installation
```bash
git clone https://github.com/Rmehta-sudo/neural-net.git
cd neural-net
go run main.go
```
`main.go` contains demonstrations of:
- Scalar autograd (`TestValue`)
- Single neuron (`TestNeuron`)
- Layer (`TestLayer`)
- MLP (`TestMLP` — full training example)

---

## 📂 Project Structure
```
neural-net/
├── main.go               # Entry point with usage examples
└── engine/
    ├── value.go          # Core Value type (data, gradient, autograd logic)
    ├── neuron.go         # Neuron implementation
    ├── layer.go          # Layer of neurons
    └── mlp.go            # Multi-Layer Perceptron
```

---

## 🔹 `engine/value.go` (Core Autograd)
`Value` represents a scalar in the computation graph with:
- **Data** — The numeric value.
- **Grad** — Gradient w.r.t. final loss.
- **Prev** — Previous `Value` objects (graph links).
- **Op** — Operation type (e.g., `+`, `*`, `tanh`).

Key methods:
- `NewValue(val, label)` — Create a new value.
- `Add`, `Sub`, `Mul`, `Div`, `Pow` — Arithmetic ops.
- `Tanh()` — Activation.
- `FullBackward()` — Backprop through the graph.

---

## 🛠 Usage
Example from `TestMLP` — binary classification:
```go
xs := [][]float64{
    {2.0, 3.0, -1.0},
    {3.0, -1.0, 0.5},
    {0.5, 1.0, 1.0},
    {1.0, 1.0, -1.0},
}
ys := []float64{1.0, -1.0, -1.0, 1.0}
mlp := engine.NewMLP([]int{4, 4, 1}, 3) // 3 inputs → 4 → 4 → 1
```
Training loop:
1. **Forward Pass** — `mlp.Output(inputs)`
2. **Loss Calculation** — Mean Squared Error (MSE)
3. **Backward Pass** — `loss.FullBackward()`
4. **Gradient Descent** — Update parameters
5. **Reset Gradients** — Set `p.Grad = 0`

---

## 📋 Custom Training Example (XOR)
```go
package main

import (
    "fmt"
    "github.com/Rmehta-sudo/neural-net/engine"
    "math/rand"
    "time"
)

func main() {
    rand.Seed(time.Now().UnixNano())

    inputs := [][]float64{
        {0, 0}, {0, 1}, {1, 0}, {1, 1},
    }
    targets := []float64{0, 1, 1, 0}

    mlp := engine.NewMLP([]int{4, 1}, 2)
    xVals := engine.ToValue2D(inputs)
    yVals := engine.ToValue1D(targets)

    lr := 0.03
    iters := 10000
    params := mlp.Parameters()

    for i := 0; i < iters; i++ {
        var preds []*engine.Value
        for _, x := range xVals {
            preds = append(preds, mlp.Output(x)[0])
        }

        loss := engine.NewValue(0.0, "loss")
        for j, p := range preds {
            diff := p.Sub(yVals[j])
            loss = loss.Add(diff.Mul(diff))
        }

        loss.FullBackward()

        for _, p := range params {
            p.Data -= lr * p.Grad
            p.Grad = 0
        }

        if i%(iters/10) == 0 {
            fmt.Printf("Iter %d, Loss: %.6f\n", i, loss.Data)
        }
    }

    for i, x := range xVals {
        fmt.Printf("Input: %v, Pred: %.4f\n", inputs[i], mlp.Output(x)[0].Data)
    }
}
```

---

## 📜 License
MIT License — see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgements
- Inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd)
- His YouTube lectures on neural networks and backpropagation
