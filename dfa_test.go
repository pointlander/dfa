// Copyright 2023 The DFA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "testing"

func TestMul(t *testing.T) {
	a := Matrix{
		Cols: 2,
		Rows: 2,
		Data: []float64{1, 2, 3, 4},
	}
	b := Matrix{
		Cols: 2,
		Rows: 1,
		Data: []float64{1, 2},
	}
	c := Mul(a, b)
	if c.Data[0] != 5 || c.Data[1] != 11 {
		t.Fatal("mul failed", c.Data)
	}
	e := Matrix{
		Cols: 2,
		Rows: 2,
		Data: []float64{1, 2, 3, 4},
	}
	f := Mul(a, e)
	if f.Data[0] != 5 || f.Data[1] != 11 || f.Data[2] != 11 || f.Data[3] != 25 {
		t.Fatal("mul failed", f.Data)
	}
}
