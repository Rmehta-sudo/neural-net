package engine

import "fmt"
import "math"

type Value struct{
	data float64
	grad float64
	prev []*Value
	op string
	label string
	backward func()
}

func (val *Value) String() string{
	return fmt.Sprintf("%s = %f | grad = %f | op = %s",val.label,val.data,val.grad,val.op)
}

func newVal(val float64,label string) *Value{
	return &Value{
		data : val,
		label : label,
		backward: func() {},
	}
}

func (a * Value) add(b * Value) *Value{
	out := &Value{
		data : a.data + b.data,
		grad : 0,
		prev : []*Value{a,b},
		op   : "+",
		label:"",
	}

	out.backward = func(){
		a.grad += out.grad
		b.grad += out.grad
	}

	return out
}

func (a * Value) mul(b * Value) *Value{
	out := &Value{
		data : a.data * b.data,
		grad : 0,
		prev : []*Value{a,b},
		op   : "*",
		label:"",
	}

	out.backward = func(){
		a.grad += b.data * out.grad
		b.grad += a.data * out.grad
	}

	return out
}

func (a * Value) tanh() *Value{
	out := &Value{
		data : math.Tanh(a.data) ,
		grad : 0,
		prev : []*Value{a},
		op   : "tanh",
		label:"",
	}

	out.backward = func(){
		a.grad += out.grad * (1 - out.data*out.data)
	}

	return out
}
/*
b = tanh(a)
L = f(b)

dL/da = dL/db * db/da = f'(x) * (1 - tanh^2(x)) = out.grad * (1 - out.data*out.data)
*/

/*
topoSort
a
b  r
	     L
c  s
d
*/

func reversedCopy[T any](s []T) []T {
    n := len(s)
    reversed := make([]T, n)
    for i := 0; i < n; i++ {
        reversed[i] = s[n-1-i]
    }
    return reversed
}

func create_topo_net(L * Value)[]*Value{
	var topo_net []*Value

	var build_topo func(*Value)
	visited := map[*Value]bool{}
	build_topo = func (node *Value){
		if(!visited[node]){
			// fmt.Println(node)
			visited[node] = true
			for _,child := range node.prev{
				build_topo(child)
			}
			topo_net = append(topo_net,node)
		}
	}
	build_topo(L)
	return reversedCopy(topo_net)
}

func (v *Value) full_back() {
	topo := create_topo_net(v)

	for _,node := range topo{
		node.grad = 0
	}
	v.grad = 1

	for _,node := range(topo){
		node.backward()
	}
}

func TestValue(){
	a := newVal( 5,"a")
	b := newVal(10,"b")
	c := newVal(12,"c")
	d := newVal(20,"d")

	ab := a.mul(b);ab.label = "ab"
	cd := c.mul(d);cd.label = "cd"
	acd := a.mul(c).mul(d);acd.label = "acd"

	apcd := a.add(cd);apcd.label="apcd"
	m  := ab.add(cd);m.label="m"
	L := m.add(apcd).add(acd);L.label="L"

	for _,node:= range create_topo_net(L){
		fmt.Println(node)
	}


}