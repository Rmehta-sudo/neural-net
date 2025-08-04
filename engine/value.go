package engine

import "fmt"

type Value struct{
	data float64
	grad float64
	prev []*Value
	op string
	label string
}

func (val *Value) String() string{
	return fmt.Sprintf("%s = %f | grad = %f | op = %s",val.label,val.data,val.grad,val.op)
}

func NewVal(val float64,label string) *Value{
	return &Value{
		data : val,
		label : label,
	}
}

func (a * Value) Add(b * Value)