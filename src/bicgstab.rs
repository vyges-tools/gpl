// SPDX-License-Identifier: Apache-2.0
//! The initial placement's linear solver: a row-major sparse matrix of `float`s built from
//! triplets, and BiCGSTAB with the identity preconditioner — each operation in the order the
//! reference's linear-algebra library performs it, so every intermediate float is the same.
//!
//! The order is the library's, for its SSE2 build (4-float packets, no fused multiply-add):
//! - [`Csr::from_triplets`]: duplicates summed in triplet order (a column-major pass), each row's
//!   entries then in column order;
//! - [`Csr::mul`]: per row, `tmp += value × x[col]` in column order from 0, then `0 + 1 × tmp`;
//! - [`sum`] (`redux` over a dense vector): two packet accumulators over pairs of packets, a
//!   leftover packet, the packet reduced as `(a0 + a2) + (a1 + a3)`, then the scalar tail in order;
//!   fewer than 4 elements, a plain left-to-right sum;
//! - [`bicgstab`]: the textbook loop with its restart, written as the library writes it.

/// A row-major compressed sparse matrix.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Csr {
    pub n: usize,
    /// Row starts (n + 1), column indices and values.
    pub starts: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<f32>,
}

impl Csr {
    /// `setFromTriplets` (sum of duplicates): the triplets go into a column-major matrix in their
    /// order, duplicates in a column summed into the first one (in that order), then a transposed
    /// copy sorts each row by column.
    pub fn from_triplets(n: usize, triplets: &[(usize, usize, f32)]) -> Csr {
        // Column-major pass: per column, the entries in triplet order, duplicates collapsed.
        let mut by_col: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        let mut slot: Vec<Option<usize>> = vec![None; n];
        let mut seen_cols: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(r, c, v) in triplets {
            by_col[c].push((r, v));
        }
        let mut collapsed: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for (c, entries) in by_col.iter().enumerate() {
            for &(r, v) in entries {
                match slot[r] {
                    Some(k) => collapsed[c][k].1 += v,
                    None => {
                        slot[r] = Some(collapsed[c].len());
                        collapsed[c].push((r, v));
                        seen_cols[c].push(r);
                    }
                }
            }
            for &r in &seen_cols[c] {
                slot[r] = None;
            }
        }
        // Transposed copy: rows in column order.
        let mut rows: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for (c, entries) in collapsed.iter().enumerate() {
            for &(r, v) in entries {
                rows[r].push((c, v));
            }
        }
        let mut m = Csr { n, starts: vec![0], ..Csr::default() };
        for row in rows {
            for (c, v) in row {
                m.cols.push(c);
                m.vals.push(v);
            }
            m.starts.push(m.cols.len());
        }
        m
    }

    /// `mat * x` into a fresh vector.
    pub fn mul(&self, x: &[f32]) -> Vec<f32> {
        (0..self.n)
            .map(|i| {
                let mut tmp = 0.0f32;
                for k in self.starts[i]..self.starts[i + 1] {
                    tmp += self.vals[k] * x[self.cols[k]];
                }
                0.0f32 + 1.0f32 * tmp
            })
            .collect()
    }
}

/// `redux(sum)` over `f(i)`, `i < n`, in the library's vectorized order (packets of 4, data
/// aligned so the packets start at 0).
pub fn sum(n: usize, f: impl Fn(usize) -> f32) -> f32 {
    const P: usize = 4;
    let packet = |at: usize| [f(at), f(at + 1), f(at + 2), f(at + 3)];
    let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
    let aligned_size2 = (n / (2 * P)) * (2 * P);
    let aligned_size = (n / P) * P;
    if aligned_size == 0 {
        if n == 0 {
            return 0.0;
        }
        let mut res = f(0);
        for i in 1..n {
            res += f(i);
        }
        return res;
    }
    let mut p0 = packet(0);
    if aligned_size > P {
        let mut p1 = packet(P);
        let mut i = 2 * P;
        while i < aligned_size2 {
            p0 = add(p0, packet(i));
            p1 = add(p1, packet(i + P));
            i += 2 * P;
        }
        p0 = add(p0, p1);
        if aligned_size > aligned_size2 {
            p0 = add(p0, packet(aligned_size2));
        }
    }
    // predux<Packet4f>: (a0 + a2) + (a1 + a3).
    let mut res = (p0[0] + p0[2]) + (p0[1] + p0[3]);
    for i in aligned_size..n {
        res += f(i);
    }
    res
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    sum(a.len(), |i| a[i] * b[i])
}

fn squared_norm(a: &[f32]) -> f32 {
    sum(a.len(), |i| a[i] * a[i])
}

/// What one solve returned (`iterations()`, `error()`, `info()`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Solve {
    pub iterations: usize,
    pub error: f32,
    pub converged: bool,
}

/// The library's `bicgstab(mat, rhs, x, IdentityPreconditioner, iters, tol_error)` as
/// `solveWithGuess` calls it: tolerance = float epsilon; `x` is updated in place.
pub fn bicgstab(mat: &Csr, rhs: &[f32], x: &mut [f32], max_iters: usize) -> Solve {
    let tol = f32::EPSILON;
    let n = mat.n;
    let mx = mat.mul(x);
    let mut r: Vec<f32> = (0..n).map(|i| rhs[i] - mx[i]).collect();
    let mut r0 = r.clone();
    let mut r0_sqnorm = squared_norm(&r0);
    let rhs_sqnorm = squared_norm(rhs);
    if rhs_sqnorm == 0.0 {
        x.iter_mut().for_each(|v| *v = 0.0);
        // tol_error is left at the tolerance: a Success.
        return Solve { iterations: max_iters, error: tol, converged: true };
    }
    let (mut rho, mut alpha, mut w) = (1.0f32, 1.0f32, 1.0f32);
    let mut v = vec![0.0f32; n];
    let mut p = vec![0.0f32; n];
    let tol2 = tol * tol * rhs_sqnorm;
    let eps2 = f32::EPSILON * f32::EPSILON;
    let (mut i, mut restarts) = (0usize, 0usize);
    while squared_norm(&r) > tol2 && i < max_iters {
        let rho_old = rho;
        rho = dot(&r0, &r);
        if rho.abs() < eps2 * r0_sqnorm {
            // Too orthogonal to r0: restart from the current residual.
            let mx = mat.mul(x);
            r = (0..n).map(|k| rhs[k] - mx[k]).collect();
            r0 = r.clone();
            r0_sqnorm = squared_norm(&r);
            rho = r0_sqnorm;
            if restarts == 0 {
                i = 0;
            }
            restarts += 1;
        }
        let beta = (rho / rho_old) * (alpha / w);
        for k in 0..n {
            p[k] = r[k] + beta * (p[k] - w * v[k]);
        }
        let y = p.clone();
        v = mat.mul(&y);
        alpha = rho / dot(&r0, &v);
        let s: Vec<f32> = (0..n).map(|k| r[k] - alpha * v[k]).collect();
        let z = s.clone();
        let t = mat.mul(&z);
        let tmp = squared_norm(&t);
        w = if tmp > 0.0 { dot(&t, &s) / tmp } else { 0.0 };
        for k in 0..n {
            x[k] += alpha * y[k] + w * z[k];
        }
        r = (0..n).map(|k| s[k] - w * t[k]).collect();
        i += 1;
    }
    let error = (squared_norm(&r) / rhs_sqnorm).sqrt();
    Solve { iterations: i, error, converged: error <= tol }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Rule (redux, SSE2): two packet accumulators, (a0 + a2) + (a1 + a3), then the tail; under 4
    // elements a left-to-right sum. Large-and-small values show the order.
    #[test]
    fn sums_follow_the_packet_order() {
        let v = [1e8f32, 1.0, -1e8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        // p0 = v[0..4] + v[4..8] = [1e8+1, 2, -1e8+1, 2]; tail v[8..11] (packet at 8 not taken:
        // aligned_size = 8 = aligned_size2).
        let p0 = [1e8f32 + 1.0, 1.0 + 1.0, -1e8f32 + 1.0, 1.0 + 1.0];
        let expect = (p0[0] + p0[2]) + (p0[1] + p0[3]) + 1.0 + 1.0 + 1.0;
        assert_eq!(sum(v.len(), |i| v[i]), expect);
        assert_eq!(sum(3, |i| v[i]), (1e8f32 + 1.0) + -1e8);
    }

    // Rule (setFromTriplets): duplicates sum in triplet order; rows sorted by column.
    #[test]
    fn triplets_collapse_in_order() {
        let m = Csr::from_triplets(2, &[(1, 1, 1e8), (0, 1, 2.0), (1, 1, 1.0), (1, 0, 3.0), (1, 1, -1e8)]);
        assert_eq!(m.starts, [0, 1, 3]);
        assert_eq!(m.cols, [1, 0, 1]);
        assert_eq!(m.vals, [2.0, 3.0, (1e8f32 + 1.0) - 1e8]);
    }

    // A small symmetric system converges to its solution.
    #[test]
    fn solves_a_small_system() {
        let m = Csr::from_triplets(2, &[(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)]);
        let mut x = vec![0.0f32; 2];
        let s = bicgstab(&m, &[1.0, 2.0], &mut x, 100);
        assert!(s.converged, "{s:?}");
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-6 && (x[1] - 7.0 / 11.0).abs() < 1e-6, "{x:?}");
    }
}
