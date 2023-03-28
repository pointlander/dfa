// Copyright 2023 The DFA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dfa

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"sort"
	"testing"
)

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

func TestDFA(t *testing.T) {
	rnd := rand.New(rand.NewSource(2))
	w1 := NewRandMatrix(rnd, StateTotal, 2+1, Hidden)
	w2 := NewRandMatrix(rnd, StateTotal, Hidden+1, Hidden)
	w3 := NewRandMatrix(rnd, StateTotal, Hidden+1, 1)
	b1 := NewRandMatrix(rnd, StateTotal, 1, Hidden)
	b2 := NewRandMatrix(rnd, StateTotal, 1, Hidden)
	input := NewMatrix(0, 2, 1)
	input.Data = append(input.Data, 0.0, 0.0)
	output := NewMatrix(0, 1, 1)
	output.Data = append(output.Data, 0.0)
	forward := func() (y, a1, z1, a2, z2 Matrix) {
		a1 = Mul(w1, AppendOne(input))
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

	for i := 1; i < 1024; i++ {
		example := data[rnd.Intn(len(data))]
		input.Data[0] = example[0]
		input.Data[1] = example[1]
		output.Data[0] = example[2]

		y, a1, z1, a2, z2 := forward()
		e := Sub(y, output)
		t.Log(e.Data)
		a1 = DLogis(a1)
		d_a1 := H(T(Mul(b1, e)), a1)
		a2 = DLogis(a2)
		d_a2 := H(T(Mul(b2, e)), a2)
		dw1 := T(Mul(d_a1, T(AppendOne(input))))
		dw2 := T(Mul(d_a2, T(z1)))
		dw3 := T(Mul(e, T(z2)))
		bb1, bb2 := math.Pow(B1, float64(i)), math.Pow(B2, float64(i))
		for j, value := range dw1.Data {
			m := B1*w1.States[StateM][j] + (1-B1)*value
			v := B2*w1.States[StateV][j] + (1-B2)*value*value
			w1.States[StateM][j] = m
			w1.States[StateV][j] = v
			mhat := m / (1 - bb1)
			vhat := v / (1 - bb2)
			w1.Data[j] -= Eta * mhat / (math.Sqrt(float64(vhat)) + 1e-8)
		}
		for j, value := range dw2.Data {
			m := B1*w2.States[StateM][j] + (1-B1)*value
			v := B2*w2.States[StateV][j] + (1-B2)*value*value
			w2.States[StateM][j] = m
			w2.States[StateV][j] = v
			mhat := m / (1 - bb1)
			vhat := v / (1 - bb2)
			w2.Data[j] -= Eta * mhat / (math.Sqrt(float64(vhat)) + 1e-8)
		}
		for j, value := range dw3.Data {
			m := B1*w3.States[StateM][j] + (1-B1)*value
			v := B2*w3.States[StateV][j] + (1-B2)*value*value
			w3.States[StateM][j] = m
			w3.States[StateV][j] = v
			mhat := m / (1 - bb1)
			vhat := v / (1 - bb2)
			w3.Data[j] -= Eta * mhat / (math.Sqrt(float64(vhat)) + 1e-8)
		}
	}

	for _, example := range data {
		input.Data[0] = example[0]
		input.Data[1] = example[1]
		y, _, _, _, _ := forward()
		if example[2] > 0.5 {
			if y.Data[0] < 0.5 {
				t.Fatal("failed", example, y.Data)
			}
		} else {
			if y.Data[0] > 0.5 {
				t.Fatal("failed", example, y.Data)
			}
		}
	}
}

func TestNeuron(t *testing.T) {
	rnd := rand.New(rand.NewSource(1))
	inputs := NewMatrix(0, 256, 1)
	inputs.Data = append(inputs.Data, 1)
	for i := 1; i < 256; i++ {
		inputs.Data = append(inputs.Data, 0)
	}
	output := NewMatrix(0, 1, 1)
	output.Data = append(output.Data, 1)
	weights := NewRandMatrix(rnd, StateTotal, 256, 1)
	forward := func() (y Matrix) {
		y = Logis(Mul(weights, inputs))
		return
	}
	for i := 1; i < 1024; i++ {
		y := forward()
		e := Sub(y, output)
		t.Log(e.Data)
		derivative := T(Mul(e, T(inputs)))
		bb1, bb2 := math.Pow(B1, float64(i)), math.Pow(B2, float64(i))
		for j, value := range derivative.Data {
			m := B1*weights.States[StateM][j] + (1-B1)*value
			v := B2*weights.States[StateV][j] + (1-B2)*value*value
			weights.States[StateM][j] = m
			weights.States[StateV][j] = v
			mhat := m / (1 - bb1)
			vhat := v / (1 - bb2)
			weights.Data[j] -= Eta * mhat / (math.Sqrt(float64(vhat)) + 1e-8)
		}
	}
	t.Log(weights.Data)
}

func TestComplex(t *testing.T) {
	const Eta = .1
	rnd := rand.New(rand.NewSource(1))
	weights := make([]complex64, 0, 256)
	factor := math.Sqrt(2.0 / float64(256))
	for i := 0; i < 256; i++ {
		weights = append(weights, complex(float32(rnd.NormFloat64()*factor), float32(rnd.NormFloat64()*factor)))
	}
	inputs := make([]complex128, 256)
	inputs[0] = cmplx.Exp(math.Pi * (1i / 4))
	for i := 0; i < 1024; i++ {
		y := complex128(0)
		for j, value := range inputs {
			y += value * complex128(weights[j])
		}
		y = (y - 1) * (y - 1)
		t.Log(cmplx.Abs(y), cmplx.Phase(y), y)
		for j, value := range inputs {
			weights[j] -= complex64(Eta * value * y)
		}
	}
	type Weight struct {
		Index int
		Value complex64
	}
	w := make([]Weight, 0, 256)
	for i, value := range weights {
		w = append(w, Weight{
			Index: i,
			Value: value,
		})
	}
	sort.Slice(w, func(i, j int) bool {
		return cmplx.Abs(complex128(w[i].Value)) > cmplx.Abs(complex128(w[j].Value))
	})
	for _, value := range w {
		fmt.Println(value.Index, cmplx.Abs(complex128(value.Value)), cmplx.Phase(complex128(value.Value)), value.Value)
	}
}
