// Copyright 2023 The DFA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

// Matrix is a matrix
type Matrix struct {
	Cols int
	Rows int
	Data []float64
}

// Mul multiplies two matrices
func Mul(m *Matrix, n *Matrix) *Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]float64, 0, m.Rows*n.Rows),
	}
	for i := 0; i < len(n.Data); i += m.Cols {
		for j := 0; j < len(m.Data); j += m.Cols {
			sum := 0.0
			for k := 0; k < m.Cols; k++ {
				sum += m.Data[j+k] * n.Data[i+k]
			}
			o.Data = append(o.Data, sum)
		}
	}
	return &o
}

func main() {

}
