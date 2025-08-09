package engine

import "fmt"
import "math/rand"

type layer struct{
	neurons []*neuron
}

func (l *layer) String() string {
    s := ""
    for i := range l.neurons {
        s += fmt.Sprintf("Neuron %d: %v\n", i+1, l.neurons[i])
    }
    return s
}

func createLayer(ins , outs int) * layer{
	var l layer 
	l.neurons = make([]*neuron, outs)

	for i := range l.neurons{
		l.neurons[i] = create_neuron(ins)
	}
	return &l
}

func (l * layer) layer_out (inputs []*Value) []*Value{
	out := make([]*Value , len(l.neurons))
	
	for i := range l.neurons{
		out[i] = l.neurons[i].neuron_out(inputs) 
	}
	return out
}
func (l * layer) params () []*Value{
	var p []*Value
	for i := range l.neurons{
		p = append(p, l.neurons[i].params()...)
	}
	return p
}

func TestLayer(){
	rand.Seed(42)
	l := createLayer(3 , 2)
	xs := []*Value{
		newVal(1.0, ""),
        newVal(-2.0, ""),
        newVal(3.0, ""),
	}

	outs := l.layer_out(xs)
	fmt.Printf("%v\n",outs)
	fmt.Printf("%v",l.neurons)


}