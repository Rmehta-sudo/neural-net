package engine

import (
	"fmt"
	"math/rand"
)

type neuron struct{
	weights []*Value
	bias *Value
}

func (n * neuron) String()string{
	return fmt.Sprintf("weights : %v ,\nbias:%v\n",n.weights,n.bias);
}

func create_neuron(num_in int) * neuron {
	neur := neuron{
		weights : make([]*Value, num_in),
		bias : newVal(rand.Float64()*2 - 1,"b"),
	} 

	for i := 0; i < num_in; i++ {
		neur.weights[i] = newVal(rand.Float64()*2-1, "w" + fmt.Sprintf("%d",i+1))
	}

	return &neur
}

func (neur * neuron) neuron_out (inputs []*Value) *Value{
	out := neur.bias
	
	for i := range neur.weights{
		out = out.add(neur.weights[i].mul(inputs[i]))
	}
	out = out.tanh()
	return out
}

func (neur * neuron) params () []*Value{
	return append(neur.weights, neur.bias)
}

func TestNeuron(){
	xs := make([]*Value , 5)
	for i := range xs{
		xs[i] = newVal(float64 (2*i) , "")
	}
	n := create_neuron(5)

	fmt.Println(n)
	fmt.Printf("inputs : %v\n\n",xs)

	res := n.neuron_out(xs)
	fmt.Printf("result: %v\n\n" , res);

	res.full_back()
	fmt.Printf("result: %v\n" , res);
	fmt.Println(n)
	fmt.Printf("inputs : %v\n",xs)


	

}