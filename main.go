// Copyright 2023 The DFA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
)

// Matrix is a matrix
type Matrix struct {
	Cols int
	Rows int
	Data []float64
}

// NewMatrix creates a new matrix
func NewMatrix(cols, rows int) Matrix {
	return Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]float64, 0, cols*rows),
	}
}

// NewRandMatrix creates a new random matrix
func NewRandMatrix(rnd *rand.Rand, cols, rows int) Matrix {
	m := Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]float64, 0, cols*rows),
	}
	factor := math.Sqrt(2.0 / float64(cols))
	for i := 0; i < cols*rows; i++ {
		m.Data = append(m.Data, rnd.NormFloat64()*factor)
	}
	return m
}

// Size is the size of the matrix
func (m Matrix) Size() int {
	return m.Cols * m.Rows
}

// Mul multiplies two matrices
func Mul(m Matrix, n Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]float64, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm, sum := m.Data[j:j+columns], 0.0
			for k, value := range mm {
				sum += value * nn[k]
			}
			o.Data = append(o.Data, sum)
		}
	}
	return o
}

// H element wise multiplies two matrices
func H(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value*n.Data[i%lenb])
	}
	return o
}

// Add adds two matrices
func Add(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
}

// Sub subtracts two matrices
func Sub(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value-n.Data[i%lenb])
	}
	return o
}

// Neg negates a matrix
func Neg(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, -value)
	}
	return o
}

// Logis computes the logis of a matrix
func Logis(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, 1/(1+math.Exp(-value)))
	}
	return o
}

func logis(value float64) float64 {
	return 1 / (1 + math.Exp(-value))
}

// DLogis computes the dlogis of a matrix
func DLogis(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, logis(value)*(1-logis(value)))
	}
	return o
}

// T tramsposes a matrix
func T(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Rows,
		Rows: m.Cols,
		Data: make([]float64, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

// AppendOne appends 1 to each row of a matrix
func AppendOne(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols + 1,
		Rows: m.Rows,
		Data: make([]float64, 0, (m.Cols+1)*m.Rows),
	}
	length := len(m.Data)
	for i := 0; i < length; i += m.Cols {
		o.Data = append(o.Data, m.Data[i:i+m.Cols]...)
		o.Data = append(o.Data, 1.0)
	}
	return o
}

const (
	// Hidden is the number of hidden neurons
	Hidden = 10
)

// https://arxiv.org/abs/1609.01596
// https://github.com/dbehrlich/directFeedbackAlignment
func main() {
	rnd := rand.New(rand.NewSource(0))
	w1 := NewRandMatrix(rnd, 2+1, Hidden)
	w2 := NewRandMatrix(rnd, Hidden+1, Hidden)
	w3 := NewRandMatrix(rnd, Hidden+1, 1)
	b1 := NewRandMatrix(rnd, 1, Hidden)
	b2 := NewRandMatrix(rnd, 1, Hidden)
	input := NewMatrix(3, 1)
	input.Data = append(input.Data, 0.0, 0.0, 1.0)
	output := NewMatrix(1, 1)
	output.Data = append(output.Data, 0.0)
	forward := func() (y, a1, z1, a2, z2 Matrix) {
		a1 = Mul(w1, input)
		z1 = AppendOne(Logis(a1))
		a2 = Mul(w2, z1)
		z2 = AppendOne(Logis(a2))
		y = Logis(Mul(w3, z2))
		return
	}

	data := [][]float64{
		{0.0, 0.0, 0.0},
		{0.0, 1.0, 1.0},
		{1.0, 0.0, 1.0},
		{1.0, 1.0, 0.0},
	}

	for i := 0; i < 1024; i++ {
		example := data[rnd.Intn(len(data))]
		input.Data[0] = example[0]
		input.Data[1] = example[1]
		output.Data[0] = example[2]

		y, a1, z1, a2, z2 := forward()
		e := Sub(y, output)
		fmt.Println(e.Data)
		a1 = DLogis(a1)
		d_a1 := H(T(Mul(b1, e)), a1)
		a2 = DLogis(a2)
		d_a2 := H(T(Mul(b2, e)), a2)
		dw1 := Mul(d_a1, T(input))
		dw2 := Mul(d_a2, T(z1))
		dw3 := Mul(e, T(z2))
		w1 = Sub(w1, T(dw1))
		w2 = Sub(w2, T(dw2))
		w3 = Sub(w3, T(dw3))
	}
}
