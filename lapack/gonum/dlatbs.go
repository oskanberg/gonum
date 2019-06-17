// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dlatbs solves a triangular banded system of equations.
func (Implementation) Dlatbs(uplo blas.Uplo, trans blas.Transpose, diag blas.Diag, normin bool, n, kd int, ab []float64, ldab int, x []float64, cnorm []float64) (scale float64) {
	noTran := trans == blas.NoTrans
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(badUplo)
	case !noTran && trans != blas.Trans && trans != blas.ConjTrans:
		panic(badTrans)
	case diag != blas.NonUnit && diag != blas.Unit:
		panic(badDiag)
	case n < 0:
		panic(nLT0)
	case kd < 0:
		panic(kdLT0)
	case ldab < kd+1:
		panic(badLdA)
	}

	// Quick return if possible.
	if n == 0 {
		return 0
	}

	switch {
	case len(ab) < (n-1)*ldab+kd:
		panic(shortAB)
	case len(x) < n:
		panic(shortX)
	case len(cnorm) < n:
		panic(shortCNorm)
	}

	bi := blas64.Implementation()
	if !normin {
		// Compute the 1-norm of each column, not including the diagonal.
		if uplo == blas.Upper {
			cnorm[0] = 0
			for j := 1; j < n; j++ {
				jlen := min(kd, j)
				cnorm[j] = bi.Dasum(jlen, ab[(j-jlen)*ldab+jlen:], ldab-1)
			}
		} else {
			for j := 0; j < n-1; j++ {
				jlen := min(kd, n-j-1)
				cnorm[j] = bi.Dasum(jlen, ab[j*ldab+kd-1:], ldab-1)
			}
			cnorm[n-1] = 0
		}
	}

	smlnum := dlamchS / dlamchP
	bignum := 1 / smlnum
	scale = 1

	// Set up indices and increments for loops below.
	var (
		jFirst, jLast, jInc int
		maind               int
	)
	if noTran {
		if uplo == blas.Upper {
			jFirst = n - 1
			jLast = 0
			jInc = -1
			maind = 0
		} else {
			jFirst = 0
			jLast = n - 1
			jInc = 1
			maind = kd
		}
	} else {
		if uplo == blas.Upper {
			jFirst = 0
			jLast = n - 1
			jInc = 1
			maind = 0
		} else {
			jFirst = n - 1
			jLast = 0
			jInc = -1
			maind = kd
		}
	}

	// Scale the column norms by tscal if the maximum element in cnorm is
	// greater than bignum.
	tmax := cnorm[bi.Idamax(n, cnorm, 1)]
	tscal := 1.0
	if tmax > bignum {
		tscal = 1 / (smlnum * tmax)
		bi.Dscal(n, tscal, cnorm, 1)
	}

	// Compute a bound on the computed solution vector to see if the Level 2
	// BLAS routine Dtbsv can be used.

	xMax := math.Abs(x[bi.Idamax(n, x, 1)])
	xBnd := xMax
	grow := 0.0

	// Compute the growth only if the maximum element in cnorm is NOT greater
	// than bignum.
	if tscal == 1 {
		if noTran {
			// Compute the growth in A * x = b.
			if diag == blas.NonUnit {
				// A is non-unit triangular.
				// Compute grow = 1/G_j and xBnd = 1/M_j.
				// Initially, G_0 = max{x(i), i=1,...,n}.
				grow = 1 / math.Max(xBnd, smlnum)
				xBnd = grow
				for j := jFirst; j <= jLast; j += jInc {
					if grow <= smlnum {
						// Exit the loop because the growth factor is too small.
						break
					}
					// M_j = G_{j-1} / abs(A[j,j])
					tjj := math.Abs(ab[j*ldab+maind])
					xBnd = math.Min(xBnd, math.Min(1, tjj)*grow)
					if tjj+cnorm[j] >= smlnum {
						// G_j = G_{j-1}*( 1 + cnorm[j] / abs(A[j,j]) )
						grow *= tjj / (tjj + cnorm[j])
					} else {
						// G_j could overflow, set grow to 0.
						grow = 0
					}
				}
				grow = xBnd
			} else {
				// A is unit triangular.
				// Compute grow = 1/G_j, where G_0 = max{x(i), i=1,...,n}.
				grow = math.Min(1, 1/math.Max(xBnd, smlnum))
				for j := jFirst; j <= jLast; j += jInc {
					if grow <= smlnum {
						// Exit the loop because the growth factor is too small.
						break
					}
					// G_j = G_{j-1}*( 1 + cnorm[j] )
					grow /= 1 + cnorm[j]
				}
			}
		} else {
			// Compute the growth in A^T * x = b.
			if diag == blas.NonUnit {
			} else {
			}
		}
	}

	if grow*tscal > smlnum {
		// The reciprocal of the bound on elements of X is not too small, use
		// the Level 2 BLAS solve.
		bi.Dtbsv(uplo, trans, diag, n, kd, ab, ldab, x, 1)
		// Scale the column norms by 1/tscal for return.
		if tscal != 1 {
			bi.Dscal(n, 1/tscal, cnorm, 1)
		}
		return scale
	}

	// Use a Level 1 BLAS solve, scaling intermediate results.

	// Scale the column norms by 1/tscal for return.
	if tscal != 1 {
		bi.Dscal(n, 1/tscal, cnorm, 1)
	}
	return scale
}
