package engine

import "fmt"

type mlp struct {
	layers []*layer
}

func createMLP(num_outs []int , num_in int) * mlp{
	var MLP mlp
	MLP.layers = make([]*layer , len(num_outs))

	for i := range num_outs{
		if (i == 0){
			MLP.layers[i] = createLayer(num_in,num_outs[0])
		}else{
			MLP.layers[i] = createLayer(num_outs[i-1],num_outs[i])
		}
	}
	return &MLP
}

func (MLP * mlp) String() string{
	s := ""
	for l := range MLP.layers{
		s += fmt.Sprintf("Layer %d :\n%v\n",l+1,MLP.layers[l])
	}
	return s
}

func (MLP * mlp) mlp_out(ins []*Value) []*Value{
	var result []*Value
	for i := range MLP.layers{
		if (i==0){
			result = MLP.layers[i].layer_out(ins)
		}else{
			result = MLP.layers[i].layer_out(result)
		}
	}
	return result
}

func (MLP * mlp) params () []*Value{
	var p []*Value
	for i := range MLP.layers{
		p = append(p, MLP.layers[i].params()...)
	}
	return p
}

func toValue2D(data [][]float64) [][]*Value {
	out := make([][]*Value, len(data))
	for i := range data {
		out[i] = make([]*Value, len(data[i]))
		for j := range data[i] {
			out[i][j] = newVal(data[i][j], "")
		}
	}
	return out
}

func toValue1D(data []float64) []*Value {
	out := make([]*Value, len(data))
	for i := range data {
		out[i] = newVal(data[i], "")
	}
	return out
}

func TestMLP(){
	xs := [][]float64{
		{2.0, 3.0, -1.0},
		{3.0, -1.0, 0.5},
		{0.5, 1.0, 1.0},
		{1.0, 1.0, -1.0},
	}

	ys := []float64{1.0, -1.0, -1.0, 1.0}

	mlp := createMLP([]int{4,4,1} ,3 )
	 
	ys_val := toValue1D(ys)
	xs_val := toValue2D(xs)
	mlp_out := make([]*Value , 4)
	ydiff := make([]*Value , 4)

	for i := 0 ; i < 4;i++ {
		mlp_out[i] = mlp.mlp_out(xs_val[i])[0]
	}

	for i := 0 ; i < 4;i++ {
		ydiff[i] = mlp_out[i].add(ys_val[i].mul(newVal(-1 , "")))
		ydiff[i] = ydiff[i].mul(ydiff[i])
	}

	final_out := ydiff[0].add(ydiff[1]).add(ydiff[2]).add(ydiff[3])
	final_out.backward()

	step := 0.01

	params := mlp.params()

	for c := 0;c<30 ; c++{

		// forward pass 
		for i := 0 ; i < 4;i++ {
			mlp_out[i] = mlp.mlp_out(xs_val[i])[0]
		}

		for i := 0 ; i < 4;i++ {
			ydiff[i] = mlp_out[i].add(ys_val[i].mul(newVal(-1 , "")))
			ydiff[i] = ydiff[i].mul(ydiff[i])
		}

		// calculate the loss

		final_out := ydiff[0].add(ydiff[1]).add(ydiff[2]).add(ydiff[3])
		final_out.full_back()

		// print it 

		fmt.Printf("\n\nLoss : %v\n" ,final_out )
		fmt.Printf("Target ys : %v\n" ,ys_val )
		fmt.Printf("Current ys : %v\n" ,mlp_out )

		// reset gradients
		for i := range params{
			params[i].data -= step * params[i].grad 
		}
		for i := range params{
			params[i].grad = 0
		}	
	}
}