// SPDX-License-Identifier: Apache-2.0
//! The density solver's 2-D cosine/sine transforms: a transcription of Takuya Ooura's
//! general-purpose FFT package (`fftsg` for one dimension, `fftsg2d` for two; Copyright Takuya
//! Ooura 1996-2001, whose licence permits use, modification and distribution for any purpose,
//! asking only that the package be credited). The placer's copy computes in `float`, so this
//! module does too, and every operation is the original's, in its order — the outputs are the
//! same floats bit for bit.
//!
//! Only what the three 2-D entry points the placer calls can reach is here, and only on the
//! single-threaded path with the routine-owned work buffer:
//! - [`ddct2d`]: a cosine transform along the rows, then along the columns;
//! - [`ddsct2d`]: cosine along the rows, sine along the columns;
//! - [`ddcst2d`]: sine along the rows, cosine along the columns.
//!
//! `isgn` picks the direction as in the original (`-1`: the forward transform the placer runs on
//! the density, `+1`: the inverse ones it runs on the potential and the field). Sizes are powers
//! of two.
//!
//! The grid is `n1` rows of `n2` floats, row-major and flat: element `[j1][j2]` is
//! `a[j1 * n2 + j2]`.
//!
//! The tables are lazy and shared between calls, as the original's are: `ip` (bit-reversal work
//! area, length at least `2 + round(sqrt(max(n1, n2)))`) and `w` (cos/sin table, length
//! `max(n1, n2) * 3 / 2`) start zeroed; `ip[0]` holds the size the twiddle half of `w` was
//! built for and `ip[1]` the size of the cosine half, and a call rebuilds either only when it
//! needs a larger one. Pass the same two buffers to every call, as the caller does.
//!
//! Numeric rules carried over from the C:
//! - every intermediate is an `f32`, and `f32` with `f32` stays `f32` — no fused multiply-add,
//!   no reassociation;
//! - a `0.5` literal is a `double`, so `0.5 * x`, `0.5 / x` and `0.5 - x` are computed in `f64`
//!   and rounded back to `f32` on assignment;
//! - in the table builders, `atan(1.0) / k` is a `double` rounded into a `float` `delta`, and
//!   `cos`/`sin` of `delta * k` take the `float` overloads: the platform's `cosf`/`sinf`, each a
//!   call of its own (see `cosf` below) — so the tables, and everything after them, match the
//!   reference built on the same platform.
//!
//! The butterfly routines below are transcribed statement for statement: their locals are
//! declared up front as in the original, so each line here lines up with one line there.

// Locals declared before their first assignment are the transcription's point, not an oversight.
#![allow(clippy::needless_late_init)]

/// The work buffer each 2-D call allocates when the caller passes none: four columns of `n1`
/// (two when `n2 == 2`, one when `n2 < 2`).
fn work_len(n1: usize, n2: usize) -> usize {
    let mut nt = 4 * n1;
    if n2 == 2 {
        nt >>= 1;
    } else if n2 < 2 {
        nt >>= 2;
    }
    nt
}

/// Builds whichever half of the table the size `n = max(n1, n2)` outgrows, as each 2-D entry
/// point does first; returns the twiddle half's length (`nw`), where the cosine half starts.
fn ensure_tables(n: usize, ip: &mut [i32], w: &mut [f32]) -> usize {
    let mut nw = ip[0] as usize;
    if n > (nw << 2) {
        nw = n >> 2;
        makewt(nw, ip, w);
    }
    if n > ip[1] as usize {
        makect(n, ip, &mut w[nw..]);
    }
    nw
}

/// `ddcst2d`: the sine transform along each row, then the cosine transform along each column.
pub fn ddcst2d(n1: usize, n2: usize, isgn: i32, a: &mut [f32], ip: &mut [i32], w: &mut [f32]) {
    ensure_tables(n1.max(n2), ip, w);
    let mut t = vec![0.0f32; work_len(n1, n2)];
    for i in 0..n1 {
        ddst(n2, isgn, &mut a[i * n2..(i + 1) * n2], ip, w);
    }
    ddxt2d_sub(n1, n2, 0, isgn, a, &mut t, ip, w);
}

/// `ddsct2d`: the cosine transform along each row, then the sine transform along each column.
pub fn ddsct2d(n1: usize, n2: usize, isgn: i32, a: &mut [f32], ip: &mut [i32], w: &mut [f32]) {
    ensure_tables(n1.max(n2), ip, w);
    let mut t = vec![0.0f32; work_len(n1, n2)];
    for i in 0..n1 {
        ddct(n2, isgn, &mut a[i * n2..(i + 1) * n2], ip, w);
    }
    ddxt2d_sub(n1, n2, 1, isgn, a, &mut t, ip, w);
}

/// `ddct2d`: the cosine transform along each row, then along each column.
pub fn ddct2d(n1: usize, n2: usize, isgn: i32, a: &mut [f32], ip: &mut [i32], w: &mut [f32]) {
    ensure_tables(n1.max(n2), ip, w);
    let mut t = vec![0.0f32; work_len(n1, n2)];
    for i in 0..n1 {
        ddct(n2, isgn, &mut a[i * n2..(i + 1) * n2], ip, w);
    }
    ddxt2d_sub(n1, n2, 0, isgn, a, &mut t, ip, w);
}

/// The column pass: four columns at a time (two when `n2 == 2`) are gathered into `t`, each
/// transformed (`ics == 0`: cosine, else sine) and scattered back.
#[allow(clippy::too_many_arguments)]
fn ddxt2d_sub(
    n1: usize,
    n2: usize,
    ics: i32,
    isgn: i32,
    a: &mut [f32],
    t: &mut [f32],
    ip: &mut [i32],
    w: &mut [f32],
) {
    if n2 > 2 {
        for j in (0..n2).step_by(4) {
            for i in 0..n1 {
                t[i] = a[i * n2 + j];
                t[n1 + i] = a[i * n2 + j + 1];
                t[2 * n1 + i] = a[i * n2 + j + 2];
                t[3 * n1 + i] = a[i * n2 + j + 3];
            }
            if ics == 0 {
                ddct(n1, isgn, t, ip, w);
                ddct(n1, isgn, &mut t[n1..], ip, w);
                ddct(n1, isgn, &mut t[2 * n1..], ip, w);
                ddct(n1, isgn, &mut t[3 * n1..], ip, w);
            } else {
                ddst(n1, isgn, t, ip, w);
                ddst(n1, isgn, &mut t[n1..], ip, w);
                ddst(n1, isgn, &mut t[2 * n1..], ip, w);
                ddst(n1, isgn, &mut t[3 * n1..], ip, w);
            }
            for i in 0..n1 {
                a[i * n2 + j] = t[i];
                a[i * n2 + j + 1] = t[n1 + i];
                a[i * n2 + j + 2] = t[2 * n1 + i];
                a[i * n2 + j + 3] = t[3 * n1 + i];
            }
        }
    } else if n2 == 2 {
        for i in 0..n1 {
            t[i] = a[i * n2];
            t[n1 + i] = a[i * n2 + 1];
        }
        if ics == 0 {
            ddct(n1, isgn, t, ip, w);
            ddct(n1, isgn, &mut t[n1..], ip, w);
        } else {
            ddst(n1, isgn, t, ip, w);
            ddst(n1, isgn, &mut t[n1..], ip, w);
        }
        for i in 0..n1 {
            a[i * n2] = t[i];
            a[i * n2 + 1] = t[n1 + i];
        }
    }
}

// ---- one dimension (`fftsg`) ----

/// `ddct`: the 1-D cosine transform of `a[..n]`, via a half-length complex FFT. The table is
/// (re)built here too if `n` outgrows it.
fn ddct(n: usize, isgn: i32, a: &mut [f32], ip: &mut [i32], w: &mut [f32]) {
    let mut xr: f32;
    let mut nw = ip[0] as usize;
    if n > (nw << 2) {
        nw = n >> 2;
        makewt(nw, ip, w);
    }
    let mut nc = ip[1] as usize;
    if n > nc {
        nc = n;
        makect(nc, ip, &mut w[nw..]);
    }
    if isgn < 0 {
        xr = a[n - 1];
        let mut j = n - 2;
        while j >= 2 {
            a[j + 1] = a[j] - a[j - 1];
            a[j] += a[j - 1];
            j -= 2;
        }
        a[1] = a[0] - xr;
        a[0] += xr;
        if n > 4 {
            rftbsub(n, a, nc, &w[nw..]);
            cftbsub(n, a, ip, nw, w);
        } else if n == 4 {
            cftbsub(n, a, ip, nw, w);
        }
    }
    dctsub(n, a, nc, &w[nw..]);
    if isgn >= 0 {
        if n > 4 {
            cftfsub(n, a, ip, nw, w);
            rftfsub(n, a, nc, &w[nw..]);
        } else if n == 4 {
            cftfsub(n, a, ip, nw, w);
        }
        xr = a[0] - a[1];
        a[0] += a[1];
        for j in (2..n).step_by(2) {
            a[j - 1] = a[j] - a[j + 1];
            a[j] += a[j + 1];
        }
        a[n - 1] = xr;
    }
}

/// `ddst`: the 1-D sine transform of `a[..n]`, the cosine transform's mirror.
fn ddst(n: usize, isgn: i32, a: &mut [f32], ip: &mut [i32], w: &mut [f32]) {
    let mut xr: f32;
    let mut nw = ip[0] as usize;
    if n > (nw << 2) {
        nw = n >> 2;
        makewt(nw, ip, w);
    }
    let mut nc = ip[1] as usize;
    if n > nc {
        nc = n;
        makect(nc, ip, &mut w[nw..]);
    }
    if isgn < 0 {
        xr = a[n - 1];
        let mut j = n - 2;
        while j >= 2 {
            a[j + 1] = -a[j] - a[j - 1];
            a[j] -= a[j - 1];
            j -= 2;
        }
        a[1] = a[0] + xr;
        a[0] -= xr;
        if n > 4 {
            rftbsub(n, a, nc, &w[nw..]);
            cftbsub(n, a, ip, nw, w);
        } else if n == 4 {
            cftbsub(n, a, ip, nw, w);
        }
    }
    dstsub(n, a, nc, &w[nw..]);
    if isgn >= 0 {
        if n > 4 {
            cftfsub(n, a, ip, nw, w);
            rftfsub(n, a, nc, &w[nw..]);
        } else if n == 4 {
            cftfsub(n, a, ip, nw, w);
        }
        xr = a[0] - a[1];
        a[0] += a[1];
        for j in (2..n).step_by(2) {
            a[j - 1] = -a[j] - a[j + 1];
            a[j] -= a[j + 1];
        }
        a[n - 1] = -xr;
    }
}

/// `makewt`: the twiddle table for complex FFTs up to `4 * nw` points — the first level from
/// `cos`/`sin` (with the bit-reversal indices when it is large enough), each further level a
/// strided copy of the one before. Sets `ip[0] = nw`, `ip[1] = 1`.
fn makewt(nw: usize, ip: &mut [i32], w: &mut [f32]) {
    let mut nwh: usize;
    let mut nw0: usize;
    let mut nw1: usize;
    let delta: f32;
    let wn4r: f32;
    let mut wk1r: f32;
    let mut wk1i: f32;
    let mut wk3r: f32;
    let mut wk3i: f32;

    ip[0] = nw as i32;
    ip[1] = 1;
    if nw > 2 {
        nwh = nw >> 1;
        // double / int, rounded into the float delta.
        delta = (1.0f64.atan() / nwh as f64) as f32;
        wn4r = cosf(delta * nwh as f32);
        w[0] = 1.0;
        w[1] = wn4r;
        if nwh == 4 {
            w[2] = cosf(delta * 2.0);
            w[3] = sinf(delta * 2.0);
        } else if nwh > 4 {
            makeipt(nw, ip);
            w[2] = (0.5f64 / cosf(delta * 2.0) as f64) as f32;
            w[3] = (0.5f64 / cosf(delta * 6.0) as f64) as f32;
            for j in (4..nwh).step_by(4) {
                w[j] = cosf(delta * j as f32);
                w[j + 1] = sinf(delta * j as f32);
                w[j + 2] = cosf(3.0 * delta * j as f32);
                w[j + 3] = -sinf(3.0 * delta * j as f32);
            }
        }
        nw0 = 0;
        while nwh > 2 {
            nw1 = nw0 + nwh;
            nwh >>= 1;
            w[nw1] = 1.0;
            w[nw1 + 1] = wn4r;
            if nwh == 4 {
                wk1r = w[nw0 + 4];
                wk1i = w[nw0 + 5];
                w[nw1 + 2] = wk1r;
                w[nw1 + 3] = wk1i;
            } else if nwh > 4 {
                wk1r = w[nw0 + 4];
                wk3r = w[nw0 + 6];
                w[nw1 + 2] = (0.5f64 / wk1r as f64) as f32;
                w[nw1 + 3] = (0.5f64 / wk3r as f64) as f32;
                for j in (4..nwh).step_by(4) {
                    wk1r = w[nw0 + 2 * j];
                    wk1i = w[nw0 + 2 * j + 1];
                    wk3r = w[nw0 + 2 * j + 2];
                    wk3i = w[nw0 + 2 * j + 3];
                    w[nw1 + j] = wk1r;
                    w[nw1 + j + 1] = wk1i;
                    w[nw1 + j + 2] = wk3r;
                    w[nw1 + j + 3] = wk3i;
                }
            }
            nw0 = nw1;
        }
    }
}

/// `makeipt`: the bit-reversal index table in `ip[2..]` that `bitrv2`/`bitrv2conj` read.
fn makeipt(nw: usize, ip: &mut [i32]) {
    let mut l: usize;
    let mut m: usize;
    let mut m2: usize;
    let mut p: i32;
    let mut q: i32;

    ip[2] = 0;
    ip[3] = 16;
    m = 2;
    l = nw;
    while l > 32 {
        m2 = m << 1;
        q = (m2 << 3) as i32;
        for j in m..m2 {
            p = ip[j] << 2;
            ip[m + j] = p;
            ip[m2 + j] = p + q;
        }
        m = m2;
        l >>= 2;
    }
}

/// `makect`: the cosine/sine table of `nc` entries (half-scaled) the real-transform steps read.
/// Sets `ip[1] = nc`.
fn makect(nc: usize, ip: &mut [i32], c: &mut [f32]) {
    let nch: usize;
    let delta: f32;

    ip[1] = nc as i32;
    if nc > 1 {
        nch = nc >> 1;
        delta = (1.0f64.atan() / nch as f64) as f32;
        c[0] = cosf(delta * nch as f32);
        c[nch] = (0.5f64 * c[0] as f64) as f32;
        for j in 1..nch {
            c[j] = (0.5f64 * cosf(delta * j as f32) as f64) as f32;
            c[nc - j] = (0.5f64 * sinf(delta * j as f32) as f64) as f32;
        }
    }
}

/// `cosf`: the table builders' `cos` of a `float`, one library call per use. The reference
/// build keeps `errno` semantics on math calls, so its compiler leaves each `sin` and `cos` a
/// separate call; ours, free to, merges a `sin` and a `cos` of the same argument into one
/// `sincos` call, which need not round the same (it does not on macOS) — the opaque argument
/// keeps them apart.
fn cosf(x: f32) -> f32 {
    std::hint::black_box(x).cos()
}

/// `sinf`: as [`cosf`].
fn sinf(x: f32) -> f32 {
    std::hint::black_box(x).sin()
}

/// `cftfsub`: the forward complex FFT of `n / 2` points in `a[..n]`, picking the kernel by size.
fn cftfsub(n: usize, a: &mut [f32], ip: &[i32], nw: usize, w: &[f32]) {
    if n > 8 {
        if n > 32 {
            cftf1st(n, a, &w[nw - (n >> 2)..]);
            if n > 512 {
                cftrec4(n, a, nw, w);
            } else if n > 128 {
                cftleaf(n, 1, a, nw, w);
            } else {
                cftfx41(n, a, nw, w);
            }
            bitrv2(n, ip, a);
        } else if n == 32 {
            cftf161(a, &w[nw - 8..]);
            bitrv216(a);
        } else {
            cftf081(a, w);
            bitrv208(a);
        }
    } else if n == 8 {
        cftf040(a);
    } else if n == 4 {
        cftx020(a);
    }
}

/// `cftbsub`: the backward complex FFT — the forward kernels, then the conjugating reversal.
fn cftbsub(n: usize, a: &mut [f32], ip: &[i32], nw: usize, w: &[f32]) {
    if n > 8 {
        if n > 32 {
            cftb1st(n, a, &w[nw - (n >> 2)..]);
            if n > 512 {
                cftrec4(n, a, nw, w);
            } else if n > 128 {
                cftleaf(n, 1, a, nw, w);
            } else {
                cftfx41(n, a, nw, w);
            }
            bitrv2conj(n, ip, a);
        } else if n == 32 {
            cftf161(a, &w[nw - 8..]);
            bitrv216neg(a);
        } else {
            cftf081(a, w);
            bitrv208neg(a);
        }
    } else if n == 8 {
        cftb040(a);
    } else if n == 4 {
        cftx020(a);
    }
}

/// `bitrv2`: the bit-reversal permutation of the complex points, from the `ip` table.
fn bitrv2(n: usize, ip: &[i32], a: &mut [f32]) {
    let mut j1: usize;
    let mut k1: usize;
    let mut l: usize;
    let mut m: usize;
    let nh: usize;
    let nm: usize;
    let mut xr: f32;
    let mut xi: f32;
    let mut yr: f32;
    let mut yi: f32;

    m = 1;
    l = n >> 2;
    while l > 8 {
        m <<= 1;
        l >>= 2;
    }
    nh = n >> 1;
    nm = 4 * m;
    if l == 8 {
        for k in 0..m {
            for j in 0..k {
                j1 = 4 * j + 2 * (ip[m + k] as usize);
                k1 = 4 * k + 2 * (ip[m + j] as usize);
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 -= nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nh;
                k1 += 2;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 += nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += 2;
                k1 += nh;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 -= nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nh;
                k1 -= 2;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 += nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
            }
            k1 = 4 * k + 2 * (ip[m + k] as usize);
            j1 = k1 + 2;
            k1 += nh;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nm;
            k1 += 2 * nm;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nm;
            k1 -= nm;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 -= 2;
            k1 -= nh;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nh + 2;
            k1 += nh + 2;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 -= nh - nm;
            k1 += 2 * nm - 2;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
        }
    } else {
        for k in 0..m {
            for j in 0..k {
                j1 = 4 * j + (ip[m + k] as usize);
                k1 = 4 * k + (ip[m + j] as usize);
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nh;
                k1 += 2;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += 2;
                k1 += nh;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nh;
                k1 -= 2;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= nm;
                xr = a[j1];
                xi = a[j1 + 1];
                yr = a[k1];
                yi = a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
            }
            k1 = 4 * k + (ip[m + k] as usize);
            j1 = k1 + 2;
            k1 += nh;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nm;
            k1 += nm;
            xr = a[j1];
            xi = a[j1 + 1];
            yr = a[k1];
            yi = a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
        }
    }
}

/// `bitrv2conj`: the bit-reversal permutation, conjugating as it goes.
fn bitrv2conj(n: usize, ip: &[i32], a: &mut [f32]) {
    let mut j1: usize;
    let mut k1: usize;
    let mut l: usize;
    let mut m: usize;
    let nh: usize;
    let nm: usize;
    let mut xr: f32;
    let mut xi: f32;
    let mut yr: f32;
    let mut yi: f32;

    m = 1;
    l = n >> 2;
    while l > 8 {
        m <<= 1;
        l >>= 2;
    }
    nh = n >> 1;
    nm = 4 * m;
    if l == 8 {
        for k in 0..m {
            for j in 0..k {
                j1 = 4 * j + 2 * (ip[m + k] as usize);
                k1 = 4 * k + 2 * (ip[m + j] as usize);
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 -= nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nh;
                k1 += 2;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 += nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += 2;
                k1 += nh;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 -= nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nh;
                k1 -= 2;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 += nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= 2 * nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
            }
            k1 = 4 * k + 2 * (ip[m + k] as usize);
            j1 = k1 + 2;
            k1 += nh;
            a[j1 - 1] = -a[j1 - 1];
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            a[k1 + 3] = -a[k1 + 3];
            j1 += nm;
            k1 += 2 * nm;
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nm;
            k1 -= nm;
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 -= 2;
            k1 -= nh;
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 += nh + 2;
            k1 += nh + 2;
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            j1 -= nh - nm;
            k1 += 2 * nm - 2;
            a[j1 - 1] = -a[j1 - 1];
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            a[k1 + 3] = -a[k1 + 3];
        }
    } else {
        for k in 0..m {
            for j in 0..k {
                j1 = 4 * j + (ip[m + k] as usize);
                k1 = 4 * k + (ip[m + j] as usize);
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nh;
                k1 += 2;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += 2;
                k1 += nh;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 += nm;
                k1 += nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nh;
                k1 -= 2;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
                j1 -= nm;
                k1 -= nm;
                xr = a[j1];
                xi = -a[j1 + 1];
                yr = a[k1];
                yi = -a[k1 + 1];
                a[j1] = yr;
                a[j1 + 1] = yi;
                a[k1] = xr;
                a[k1 + 1] = xi;
            }
            k1 = 4 * k + (ip[m + k] as usize);
            j1 = k1 + 2;
            k1 += nh;
            a[j1 - 1] = -a[j1 - 1];
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            a[k1 + 3] = -a[k1 + 3];
            j1 += nm;
            k1 += nm;
            a[j1 - 1] = -a[j1 - 1];
            xr = a[j1];
            xi = -a[j1 + 1];
            yr = a[k1];
            yi = -a[k1 + 1];
            a[j1] = yr;
            a[j1 + 1] = yi;
            a[k1] = xr;
            a[k1 + 1] = xi;
            a[k1 + 3] = -a[k1 + 3];
        }
    }
}

/// `bitrv216`: the 16-point bit reversal, unrolled.
fn bitrv216(a: &mut [f32]) {
    let x1r: f32;
    let x1i: f32;
    let x2r: f32;
    let x2i: f32;
    let x3r: f32;
    let x3i: f32;
    let x4r: f32;
    let x4i: f32;
    let x5r: f32;
    let x5i: f32;
    let x7r: f32;
    let x7i: f32;
    let x8r: f32;
    let x8i: f32;
    let x10r: f32;
    let x10i: f32;
    let x11r: f32;
    let x11i: f32;
    let x12r: f32;
    let x12i: f32;
    let x13r: f32;
    let x13i: f32;
    let x14r: f32;
    let x14i: f32;

    x1r = a[2];
    x1i = a[3];
    x2r = a[4];
    x2i = a[5];
    x3r = a[6];
    x3i = a[7];
    x4r = a[8];
    x4i = a[9];
    x5r = a[10];
    x5i = a[11];
    x7r = a[14];
    x7i = a[15];
    x8r = a[16];
    x8i = a[17];
    x10r = a[20];
    x10i = a[21];
    x11r = a[22];
    x11i = a[23];
    x12r = a[24];
    x12i = a[25];
    x13r = a[26];
    x13i = a[27];
    x14r = a[28];
    x14i = a[29];
    a[2] = x8r;
    a[3] = x8i;
    a[4] = x4r;
    a[5] = x4i;
    a[6] = x12r;
    a[7] = x12i;
    a[8] = x2r;
    a[9] = x2i;
    a[10] = x10r;
    a[11] = x10i;
    a[14] = x14r;
    a[15] = x14i;
    a[16] = x1r;
    a[17] = x1i;
    a[20] = x5r;
    a[21] = x5i;
    a[22] = x13r;
    a[23] = x13i;
    a[24] = x3r;
    a[25] = x3i;
    a[26] = x11r;
    a[27] = x11i;
    a[28] = x7r;
    a[29] = x7i;
}

/// `bitrv216neg`: the 16-point bit reversal for the backward transform, unrolled.
fn bitrv216neg(a: &mut [f32]) {
    let x1r: f32;
    let x1i: f32;
    let x2r: f32;
    let x2i: f32;
    let x3r: f32;
    let x3i: f32;
    let x4r: f32;
    let x4i: f32;
    let x5r: f32;
    let x5i: f32;
    let x6r: f32;
    let x6i: f32;
    let x7r: f32;
    let x7i: f32;
    let x8r: f32;
    let x8i: f32;
    let x9r: f32;
    let x9i: f32;
    let x10r: f32;
    let x10i: f32;
    let x11r: f32;
    let x11i: f32;
    let x12r: f32;
    let x12i: f32;
    let x13r: f32;
    let x13i: f32;
    let x14r: f32;
    let x14i: f32;
    let x15r: f32;
    let x15i: f32;

    x1r = a[2];
    x1i = a[3];
    x2r = a[4];
    x2i = a[5];
    x3r = a[6];
    x3i = a[7];
    x4r = a[8];
    x4i = a[9];
    x5r = a[10];
    x5i = a[11];
    x6r = a[12];
    x6i = a[13];
    x7r = a[14];
    x7i = a[15];
    x8r = a[16];
    x8i = a[17];
    x9r = a[18];
    x9i = a[19];
    x10r = a[20];
    x10i = a[21];
    x11r = a[22];
    x11i = a[23];
    x12r = a[24];
    x12i = a[25];
    x13r = a[26];
    x13i = a[27];
    x14r = a[28];
    x14i = a[29];
    x15r = a[30];
    x15i = a[31];
    a[2] = x15r;
    a[3] = x15i;
    a[4] = x7r;
    a[5] = x7i;
    a[6] = x11r;
    a[7] = x11i;
    a[8] = x3r;
    a[9] = x3i;
    a[10] = x13r;
    a[11] = x13i;
    a[12] = x5r;
    a[13] = x5i;
    a[14] = x9r;
    a[15] = x9i;
    a[16] = x1r;
    a[17] = x1i;
    a[18] = x14r;
    a[19] = x14i;
    a[20] = x6r;
    a[21] = x6i;
    a[22] = x10r;
    a[23] = x10i;
    a[24] = x2r;
    a[25] = x2i;
    a[26] = x12r;
    a[27] = x12i;
    a[28] = x4r;
    a[29] = x4i;
    a[30] = x8r;
    a[31] = x8i;
}

/// `bitrv208`: the 8-point bit reversal, unrolled.
fn bitrv208(a: &mut [f32]) {
    let x1r: f32;
    let x1i: f32;
    let x3r: f32;
    let x3i: f32;
    let x4r: f32;
    let x4i: f32;
    let x6r: f32;
    let x6i: f32;

    x1r = a[2];
    x1i = a[3];
    x3r = a[6];
    x3i = a[7];
    x4r = a[8];
    x4i = a[9];
    x6r = a[12];
    x6i = a[13];
    a[2] = x4r;
    a[3] = x4i;
    a[6] = x6r;
    a[7] = x6i;
    a[8] = x1r;
    a[9] = x1i;
    a[12] = x3r;
    a[13] = x3i;
}

/// `bitrv208neg`: the 8-point bit reversal for the backward transform, unrolled.
fn bitrv208neg(a: &mut [f32]) {
    let x1r: f32;
    let x1i: f32;
    let x2r: f32;
    let x2i: f32;
    let x3r: f32;
    let x3i: f32;
    let x4r: f32;
    let x4i: f32;
    let x5r: f32;
    let x5i: f32;
    let x6r: f32;
    let x6i: f32;
    let x7r: f32;
    let x7i: f32;

    x1r = a[2];
    x1i = a[3];
    x2r = a[4];
    x2i = a[5];
    x3r = a[6];
    x3i = a[7];
    x4r = a[8];
    x4i = a[9];
    x5r = a[10];
    x5i = a[11];
    x6r = a[12];
    x6i = a[13];
    x7r = a[14];
    x7i = a[15];
    a[2] = x7r;
    a[3] = x7i;
    a[4] = x3r;
    a[5] = x3i;
    a[6] = x5r;
    a[7] = x5i;
    a[8] = x1r;
    a[9] = x1i;
    a[10] = x6r;
    a[11] = x6i;
    a[12] = x2r;
    a[13] = x2i;
    a[14] = x4r;
    a[15] = x4i;
}

/// `cftf1st`: the first radix-4 stage of the forward transform over all of `a[..n]`.
fn cftf1st(n: usize, a: &mut [f32], w: &[f32]) {
    let mut j0: usize;
    let mut j1: usize;
    let mut j2: usize;
    let mut j3: usize;
    let mut k: usize;
    let m: usize;
    let mh: usize;
    let wn4r: f32;
    let csc1: f32;
    let csc3: f32;
    let mut wk1r: f32;
    let mut wk1i: f32;
    let mut wk3r: f32;
    let mut wk3i: f32;
    let mut wd1r: f32;
    let mut wd1i: f32;
    let mut wd3r: f32;
    let mut wd3i: f32;
    let mut x0r: f32;
    let mut x0i: f32;
    let mut x1r: f32;
    let mut x1i: f32;
    let mut x2r: f32;
    let mut x2i: f32;
    let mut x3r: f32;
    let mut x3i: f32;
    let mut y0r: f32;
    let mut y0i: f32;
    let mut y1r: f32;
    let mut y1i: f32;
    let mut y2r: f32;
    let mut y2i: f32;
    let mut y3r: f32;
    let mut y3i: f32;

    mh = n >> 3;
    m = 2 * mh;
    j1 = m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[0] + a[j2];
    x0i = a[1] + a[j2 + 1];
    x1r = a[0] - a[j2];
    x1i = a[1] - a[j2 + 1];
    x2r = a[j1] + a[j3];
    x2i = a[j1 + 1] + a[j3 + 1];
    x3r = a[j1] - a[j3];
    x3i = a[j1 + 1] - a[j3 + 1];
    a[0] = x0r + x2r;
    a[1] = x0i + x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i - x2i;
    a[j2] = x1r - x3i;
    a[j2 + 1] = x1i + x3r;
    a[j3] = x1r + x3i;
    a[j3 + 1] = x1i - x3r;
    wn4r = w[1];
    csc1 = w[2];
    csc3 = w[3];
    wd1r = 1.0;
    wd1i = 0.0;
    wd3r = 1.0;
    wd3i = 0.0;
    k = 0;
    for j in (2..mh - 2).step_by(4) {
        k += 4;
        wk1r = csc1 * (wd1r + w[k]);
        wk1i = csc1 * (wd1i + w[k + 1]);
        wk3r = csc3 * (wd3r + w[k + 2]);
        wk3i = csc3 * (wd3i + w[k + 3]);
        wd1r = w[k];
        wd1i = w[k + 1];
        wd3r = w[k + 2];
        wd3i = w[k + 3];
        j1 = j + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j] + a[j2];
        x0i = a[j + 1] + a[j2 + 1];
        x1r = a[j] - a[j2];
        x1i = a[j + 1] - a[j2 + 1];
        y0r = a[j + 2] + a[j2 + 2];
        y0i = a[j + 3] + a[j2 + 3];
        y1r = a[j + 2] - a[j2 + 2];
        y1i = a[j + 3] - a[j2 + 3];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        y2r = a[j1 + 2] + a[j3 + 2];
        y2i = a[j1 + 3] + a[j3 + 3];
        y3r = a[j1 + 2] - a[j3 + 2];
        y3i = a[j1 + 3] - a[j3 + 3];
        a[j] = x0r + x2r;
        a[j + 1] = x0i + x2i;
        a[j + 2] = y0r + y2r;
        a[j + 3] = y0i + y2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i - x2i;
        a[j1 + 2] = y0r - y2r;
        a[j1 + 3] = y0i - y2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[j2] = wk1r * x0r - wk1i * x0i;
        a[j2 + 1] = wk1r * x0i + wk1i * x0r;
        x0r = y1r - y3i;
        x0i = y1i + y3r;
        a[j2 + 2] = wd1r * x0r - wd1i * x0i;
        a[j2 + 3] = wd1r * x0i + wd1i * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[j3] = wk3r * x0r + wk3i * x0i;
        a[j3 + 1] = wk3r * x0i - wk3i * x0r;
        x0r = y1r + y3i;
        x0i = y1i - y3r;
        a[j3 + 2] = wd3r * x0r + wd3i * x0i;
        a[j3 + 3] = wd3r * x0i - wd3i * x0r;
        j0 = m - j;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j0] + a[j2];
        x0i = a[j0 + 1] + a[j2 + 1];
        x1r = a[j0] - a[j2];
        x1i = a[j0 + 1] - a[j2 + 1];
        y0r = a[j0 - 2] + a[j2 - 2];
        y0i = a[j0 - 1] + a[j2 - 1];
        y1r = a[j0 - 2] - a[j2 - 2];
        y1i = a[j0 - 1] - a[j2 - 1];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        y2r = a[j1 - 2] + a[j3 - 2];
        y2i = a[j1 - 1] + a[j3 - 1];
        y3r = a[j1 - 2] - a[j3 - 2];
        y3i = a[j1 - 1] - a[j3 - 1];
        a[j0] = x0r + x2r;
        a[j0 + 1] = x0i + x2i;
        a[j0 - 2] = y0r + y2r;
        a[j0 - 1] = y0i + y2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i - x2i;
        a[j1 - 2] = y0r - y2r;
        a[j1 - 1] = y0i - y2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[j2] = wk1i * x0r - wk1r * x0i;
        a[j2 + 1] = wk1i * x0i + wk1r * x0r;
        x0r = y1r - y3i;
        x0i = y1i + y3r;
        a[j2 - 2] = wd1i * x0r - wd1r * x0i;
        a[j2 - 1] = wd1i * x0i + wd1r * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[j3] = wk3i * x0r + wk3r * x0i;
        a[j3 + 1] = wk3i * x0i - wk3r * x0r;
        x0r = y1r + y3i;
        x0i = y1i - y3r;
        a[j3 - 2] = wd3i * x0r + wd3r * x0i;
        a[j3 - 1] = wd3i * x0i - wd3r * x0r;
    }
    wk1r = csc1 * (wd1r + wn4r);
    wk1i = csc1 * (wd1i + wn4r);
    wk3r = csc3 * (wd3r - wn4r);
    wk3i = csc3 * (wd3i - wn4r);
    j0 = mh;
    j1 = j0 + m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[j0 - 2] + a[j2 - 2];
    x0i = a[j0 - 1] + a[j2 - 1];
    x1r = a[j0 - 2] - a[j2 - 2];
    x1i = a[j0 - 1] - a[j2 - 1];
    x2r = a[j1 - 2] + a[j3 - 2];
    x2i = a[j1 - 1] + a[j3 - 1];
    x3r = a[j1 - 2] - a[j3 - 2];
    x3i = a[j1 - 1] - a[j3 - 1];
    a[j0 - 2] = x0r + x2r;
    a[j0 - 1] = x0i + x2i;
    a[j1 - 2] = x0r - x2r;
    a[j1 - 1] = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    a[j2 - 2] = wk1r * x0r - wk1i * x0i;
    a[j2 - 1] = wk1r * x0i + wk1i * x0r;
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    a[j3 - 2] = wk3r * x0r + wk3i * x0i;
    a[j3 - 1] = wk3r * x0i - wk3i * x0r;
    x0r = a[j0] + a[j2];
    x0i = a[j0 + 1] + a[j2 + 1];
    x1r = a[j0] - a[j2];
    x1i = a[j0 + 1] - a[j2 + 1];
    x2r = a[j1] + a[j3];
    x2i = a[j1 + 1] + a[j3 + 1];
    x3r = a[j1] - a[j3];
    x3i = a[j1 + 1] - a[j3 + 1];
    a[j0] = x0r + x2r;
    a[j0 + 1] = x0i + x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    a[j2] = wn4r * (x0r - x0i);
    a[j2 + 1] = wn4r * (x0i + x0r);
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    a[j3] = -wn4r * (x0r + x0i);
    a[j3 + 1] = -wn4r * (x0i - x0r);
    x0r = a[j0 + 2] + a[j2 + 2];
    x0i = a[j0 + 3] + a[j2 + 3];
    x1r = a[j0 + 2] - a[j2 + 2];
    x1i = a[j0 + 3] - a[j2 + 3];
    x2r = a[j1 + 2] + a[j3 + 2];
    x2i = a[j1 + 3] + a[j3 + 3];
    x3r = a[j1 + 2] - a[j3 + 2];
    x3i = a[j1 + 3] - a[j3 + 3];
    a[j0 + 2] = x0r + x2r;
    a[j0 + 3] = x0i + x2i;
    a[j1 + 2] = x0r - x2r;
    a[j1 + 3] = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    a[j2 + 2] = wk1i * x0r - wk1r * x0i;
    a[j2 + 3] = wk1i * x0i + wk1r * x0r;
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    a[j3 + 2] = wk3i * x0r + wk3r * x0i;
    a[j3 + 3] = wk3i * x0i - wk3r * x0r;
}

/// `cftb1st`: the first radix-4 stage of the backward transform.
fn cftb1st(n: usize, a: &mut [f32], w: &[f32]) {
    let mut j0: usize;
    let mut j1: usize;
    let mut j2: usize;
    let mut j3: usize;
    let mut k: usize;
    let m: usize;
    let mh: usize;
    let wn4r: f32;
    let csc1: f32;
    let csc3: f32;
    let mut wk1r: f32;
    let mut wk1i: f32;
    let mut wk3r: f32;
    let mut wk3i: f32;
    let mut wd1r: f32;
    let mut wd1i: f32;
    let mut wd3r: f32;
    let mut wd3i: f32;
    let mut x0r: f32;
    let mut x0i: f32;
    let mut x1r: f32;
    let mut x1i: f32;
    let mut x2r: f32;
    let mut x2i: f32;
    let mut x3r: f32;
    let mut x3i: f32;
    let mut y0r: f32;
    let mut y0i: f32;
    let mut y1r: f32;
    let mut y1i: f32;
    let mut y2r: f32;
    let mut y2i: f32;
    let mut y3r: f32;
    let mut y3i: f32;

    mh = n >> 3;
    m = 2 * mh;
    j1 = m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[0] + a[j2];
    x0i = -a[1] - a[j2 + 1];
    x1r = a[0] - a[j2];
    x1i = -a[1] + a[j2 + 1];
    x2r = a[j1] + a[j3];
    x2i = a[j1 + 1] + a[j3 + 1];
    x3r = a[j1] - a[j3];
    x3i = a[j1 + 1] - a[j3 + 1];
    a[0] = x0r + x2r;
    a[1] = x0i - x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i + x2i;
    a[j2] = x1r + x3i;
    a[j2 + 1] = x1i + x3r;
    a[j3] = x1r - x3i;
    a[j3 + 1] = x1i - x3r;
    wn4r = w[1];
    csc1 = w[2];
    csc3 = w[3];
    wd1r = 1.0;
    wd1i = 0.0;
    wd3r = 1.0;
    wd3i = 0.0;
    k = 0;
    for j in (2..mh - 2).step_by(4) {
        k += 4;
        wk1r = csc1 * (wd1r + w[k]);
        wk1i = csc1 * (wd1i + w[k + 1]);
        wk3r = csc3 * (wd3r + w[k + 2]);
        wk3i = csc3 * (wd3i + w[k + 3]);
        wd1r = w[k];
        wd1i = w[k + 1];
        wd3r = w[k + 2];
        wd3i = w[k + 3];
        j1 = j + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j] + a[j2];
        x0i = -a[j + 1] - a[j2 + 1];
        x1r = a[j] - a[j2];
        x1i = -a[j + 1] + a[j2 + 1];
        y0r = a[j + 2] + a[j2 + 2];
        y0i = -a[j + 3] - a[j2 + 3];
        y1r = a[j + 2] - a[j2 + 2];
        y1i = -a[j + 3] + a[j2 + 3];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        y2r = a[j1 + 2] + a[j3 + 2];
        y2i = a[j1 + 3] + a[j3 + 3];
        y3r = a[j1 + 2] - a[j3 + 2];
        y3i = a[j1 + 3] - a[j3 + 3];
        a[j] = x0r + x2r;
        a[j + 1] = x0i - x2i;
        a[j + 2] = y0r + y2r;
        a[j + 3] = y0i - y2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i + x2i;
        a[j1 + 2] = y0r - y2r;
        a[j1 + 3] = y0i + y2i;
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a[j2] = wk1r * x0r - wk1i * x0i;
        a[j2 + 1] = wk1r * x0i + wk1i * x0r;
        x0r = y1r + y3i;
        x0i = y1i + y3r;
        a[j2 + 2] = wd1r * x0r - wd1i * x0i;
        a[j2 + 3] = wd1r * x0i + wd1i * x0r;
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a[j3] = wk3r * x0r + wk3i * x0i;
        a[j3 + 1] = wk3r * x0i - wk3i * x0r;
        x0r = y1r - y3i;
        x0i = y1i - y3r;
        a[j3 + 2] = wd3r * x0r + wd3i * x0i;
        a[j3 + 3] = wd3r * x0i - wd3i * x0r;
        j0 = m - j;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j0] + a[j2];
        x0i = -a[j0 + 1] - a[j2 + 1];
        x1r = a[j0] - a[j2];
        x1i = -a[j0 + 1] + a[j2 + 1];
        y0r = a[j0 - 2] + a[j2 - 2];
        y0i = -a[j0 - 1] - a[j2 - 1];
        y1r = a[j0 - 2] - a[j2 - 2];
        y1i = -a[j0 - 1] + a[j2 - 1];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        y2r = a[j1 - 2] + a[j3 - 2];
        y2i = a[j1 - 1] + a[j3 - 1];
        y3r = a[j1 - 2] - a[j3 - 2];
        y3i = a[j1 - 1] - a[j3 - 1];
        a[j0] = x0r + x2r;
        a[j0 + 1] = x0i - x2i;
        a[j0 - 2] = y0r + y2r;
        a[j0 - 1] = y0i - y2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i + x2i;
        a[j1 - 2] = y0r - y2r;
        a[j1 - 1] = y0i + y2i;
        x0r = x1r + x3i;
        x0i = x1i + x3r;
        a[j2] = wk1i * x0r - wk1r * x0i;
        a[j2 + 1] = wk1i * x0i + wk1r * x0r;
        x0r = y1r + y3i;
        x0i = y1i + y3r;
        a[j2 - 2] = wd1i * x0r - wd1r * x0i;
        a[j2 - 1] = wd1i * x0i + wd1r * x0r;
        x0r = x1r - x3i;
        x0i = x1i - x3r;
        a[j3] = wk3i * x0r + wk3r * x0i;
        a[j3 + 1] = wk3i * x0i - wk3r * x0r;
        x0r = y1r - y3i;
        x0i = y1i - y3r;
        a[j3 - 2] = wd3i * x0r + wd3r * x0i;
        a[j3 - 1] = wd3i * x0i - wd3r * x0r;
    }
    wk1r = csc1 * (wd1r + wn4r);
    wk1i = csc1 * (wd1i + wn4r);
    wk3r = csc3 * (wd3r - wn4r);
    wk3i = csc3 * (wd3i - wn4r);
    j0 = mh;
    j1 = j0 + m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[j0 - 2] + a[j2 - 2];
    x0i = -a[j0 - 1] - a[j2 - 1];
    x1r = a[j0 - 2] - a[j2 - 2];
    x1i = -a[j0 - 1] + a[j2 - 1];
    x2r = a[j1 - 2] + a[j3 - 2];
    x2i = a[j1 - 1] + a[j3 - 1];
    x3r = a[j1 - 2] - a[j3 - 2];
    x3i = a[j1 - 1] - a[j3 - 1];
    a[j0 - 2] = x0r + x2r;
    a[j0 - 1] = x0i - x2i;
    a[j1 - 2] = x0r - x2r;
    a[j1 - 1] = x0i + x2i;
    x0r = x1r + x3i;
    x0i = x1i + x3r;
    a[j2 - 2] = wk1r * x0r - wk1i * x0i;
    a[j2 - 1] = wk1r * x0i + wk1i * x0r;
    x0r = x1r - x3i;
    x0i = x1i - x3r;
    a[j3 - 2] = wk3r * x0r + wk3i * x0i;
    a[j3 - 1] = wk3r * x0i - wk3i * x0r;
    x0r = a[j0] + a[j2];
    x0i = -a[j0 + 1] - a[j2 + 1];
    x1r = a[j0] - a[j2];
    x1i = -a[j0 + 1] + a[j2 + 1];
    x2r = a[j1] + a[j3];
    x2i = a[j1 + 1] + a[j3 + 1];
    x3r = a[j1] - a[j3];
    x3i = a[j1 + 1] - a[j3 + 1];
    a[j0] = x0r + x2r;
    a[j0 + 1] = x0i - x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i + x2i;
    x0r = x1r + x3i;
    x0i = x1i + x3r;
    a[j2] = wn4r * (x0r - x0i);
    a[j2 + 1] = wn4r * (x0i + x0r);
    x0r = x1r - x3i;
    x0i = x1i - x3r;
    a[j3] = -wn4r * (x0r + x0i);
    a[j3 + 1] = -wn4r * (x0i - x0r);
    x0r = a[j0 + 2] + a[j2 + 2];
    x0i = -a[j0 + 3] - a[j2 + 3];
    x1r = a[j0 + 2] - a[j2 + 2];
    x1i = -a[j0 + 3] + a[j2 + 3];
    x2r = a[j1 + 2] + a[j3 + 2];
    x2i = a[j1 + 3] + a[j3 + 3];
    x3r = a[j1 + 2] - a[j3 + 2];
    x3i = a[j1 + 3] - a[j3 + 3];
    a[j0 + 2] = x0r + x2r;
    a[j0 + 3] = x0i - x2i;
    a[j1 + 2] = x0r - x2r;
    a[j1 + 3] = x0i + x2i;
    x0r = x1r + x3i;
    x0i = x1i + x3r;
    a[j2 + 2] = wk1i * x0r - wk1r * x0i;
    a[j2 + 3] = wk1i * x0i + wk1r * x0r;
    x0r = x1r - x3i;
    x0i = x1i - x3r;
    a[j3 + 2] = wk3i * x0r + wk3r * x0i;
    a[j3 + 3] = wk3i * x0i - wk3r * x0r;
}

/// `cftrec4`: the remaining stages for `n > 512`, depth first over blocks of at most 512.
fn cftrec4(n: usize, a: &mut [f32], nw: usize, w: &[f32]) {
    let mut isplt: usize;
    let mut j: usize;
    let mut k: usize;
    let mut m: usize;

    m = n;
    while m > 512 {
        m >>= 2;
        cftmdl1(m, &mut a[n - m..], &w[nw - (m >> 1)..]);
    }
    cftleaf(m, 1, &mut a[n - m..], nw, w);
    k = 0;
    j = n - m;
    while j > 0 {
        k += 1;
        isplt = cfttree(m, j, k, a, nw, w);
        cftleaf(m, isplt, &mut a[j - m..], nw, w);
        j -= m;
    }
}

/// `cfttree`: the middle stages above the block ending at `j`, chosen from the block counter
/// `k`; returns which leaf variant (`isplt`) the block takes.
fn cfttree(n: usize, j: usize, k: usize, a: &mut [f32], nw: usize, w: &[f32]) -> usize {
    let mut i: usize;
    let isplt: usize;
    let mut m: usize;

    if (k & 3) != 0 {
        isplt = k & 1;
        if isplt != 0 {
            cftmdl1(n, &mut a[j - n..], &w[nw - (n >> 1)..]);
        } else {
            cftmdl2(n, &mut a[j - n..], &w[nw - n..]);
        }
    } else {
        m = n;
        i = k;
        while (i & 3) == 0 {
            m <<= 2;
            i >>= 2;
        }
        isplt = i & 1;
        if isplt != 0 {
            while m > 128 {
                cftmdl1(m, &mut a[j - m..], &w[nw - (m >> 1)..]);
                m >>= 2;
            }
        } else {
            while m > 128 {
                cftmdl2(m, &mut a[j - m..], &w[nw - m..]);
                m >>= 2;
            }
        }
    }
    isplt
}

/// `cftleaf`: a 512- or 128-point block's last stages, as a fixed schedule of kernels.
fn cftleaf(n: usize, isplt: usize, a: &mut [f32], nw: usize, w: &[f32]) {
    if n == 512 {
        cftmdl1(128, a, &w[nw - 64..]);
        cftf161(a, &w[nw - 8..]);
        cftf162(&mut a[32..], &w[nw - 32..]);
        cftf161(&mut a[64..], &w[nw - 8..]);
        cftf161(&mut a[96..], &w[nw - 8..]);
        cftmdl2(128, &mut a[128..], &w[nw - 128..]);
        cftf161(&mut a[128..], &w[nw - 8..]);
        cftf162(&mut a[160..], &w[nw - 32..]);
        cftf161(&mut a[192..], &w[nw - 8..]);
        cftf162(&mut a[224..], &w[nw - 32..]);
        cftmdl1(128, &mut a[256..], &w[nw - 64..]);
        cftf161(&mut a[256..], &w[nw - 8..]);
        cftf162(&mut a[288..], &w[nw - 32..]);
        cftf161(&mut a[320..], &w[nw - 8..]);
        cftf161(&mut a[352..], &w[nw - 8..]);
        if isplt != 0 {
            cftmdl1(128, &mut a[384..], &w[nw - 64..]);
            cftf161(&mut a[480..], &w[nw - 8..]);
        } else {
            cftmdl2(128, &mut a[384..], &w[nw - 128..]);
            cftf162(&mut a[480..], &w[nw - 32..]);
        }
        cftf161(&mut a[384..], &w[nw - 8..]);
        cftf162(&mut a[416..], &w[nw - 32..]);
        cftf161(&mut a[448..], &w[nw - 8..]);
    } else {
        cftmdl1(64, a, &w[nw - 32..]);
        cftf081(a, &w[nw - 8..]);
        cftf082(&mut a[16..], &w[nw - 8..]);
        cftf081(&mut a[32..], &w[nw - 8..]);
        cftf081(&mut a[48..], &w[nw - 8..]);
        cftmdl2(64, &mut a[64..], &w[nw - 64..]);
        cftf081(&mut a[64..], &w[nw - 8..]);
        cftf082(&mut a[80..], &w[nw - 8..]);
        cftf081(&mut a[96..], &w[nw - 8..]);
        cftf082(&mut a[112..], &w[nw - 8..]);
        cftmdl1(64, &mut a[128..], &w[nw - 32..]);
        cftf081(&mut a[128..], &w[nw - 8..]);
        cftf082(&mut a[144..], &w[nw - 8..]);
        cftf081(&mut a[160..], &w[nw - 8..]);
        cftf081(&mut a[176..], &w[nw - 8..]);
        if isplt != 0 {
            cftmdl1(64, &mut a[192..], &w[nw - 32..]);
            cftf081(&mut a[240..], &w[nw - 8..]);
        } else {
            cftmdl2(64, &mut a[192..], &w[nw - 64..]);
            cftf082(&mut a[240..], &w[nw - 8..]);
        }
        cftf081(&mut a[192..], &w[nw - 8..]);
        cftf082(&mut a[208..], &w[nw - 8..]);
        cftf081(&mut a[224..], &w[nw - 8..]);
    }
}

/// `cftmdl1`: a radix-4 middle stage over `a[..n]` (the first of the two twiddle patterns).
fn cftmdl1(n: usize, a: &mut [f32], w: &[f32]) {
    let mut j0: usize;
    let mut j1: usize;
    let mut j2: usize;
    let mut j3: usize;
    let mut k: usize;
    let m: usize;
    let mh: usize;
    let wn4r: f32;
    let mut wk1r: f32;
    let mut wk1i: f32;
    let mut wk3r: f32;
    let mut wk3i: f32;
    let mut x0r: f32;
    let mut x0i: f32;
    let mut x1r: f32;
    let mut x1i: f32;
    let mut x2r: f32;
    let mut x2i: f32;
    let mut x3r: f32;
    let mut x3i: f32;

    mh = n >> 3;
    m = 2 * mh;
    j1 = m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[0] + a[j2];
    x0i = a[1] + a[j2 + 1];
    x1r = a[0] - a[j2];
    x1i = a[1] - a[j2 + 1];
    x2r = a[j1] + a[j3];
    x2i = a[j1 + 1] + a[j3 + 1];
    x3r = a[j1] - a[j3];
    x3i = a[j1 + 1] - a[j3 + 1];
    a[0] = x0r + x2r;
    a[1] = x0i + x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i - x2i;
    a[j2] = x1r - x3i;
    a[j2 + 1] = x1i + x3r;
    a[j3] = x1r + x3i;
    a[j3 + 1] = x1i - x3r;
    wn4r = w[1];
    k = 0;
    for j in (2..mh).step_by(2) {
        k += 4;
        wk1r = w[k];
        wk1i = w[k + 1];
        wk3r = w[k + 2];
        wk3i = w[k + 3];
        j1 = j + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j] + a[j2];
        x0i = a[j + 1] + a[j2 + 1];
        x1r = a[j] - a[j2];
        x1i = a[j + 1] - a[j2 + 1];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        a[j] = x0r + x2r;
        a[j + 1] = x0i + x2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[j2] = wk1r * x0r - wk1i * x0i;
        a[j2 + 1] = wk1r * x0i + wk1i * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[j3] = wk3r * x0r + wk3i * x0i;
        a[j3 + 1] = wk3r * x0i - wk3i * x0r;
        j0 = m - j;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j0] + a[j2];
        x0i = a[j0 + 1] + a[j2 + 1];
        x1r = a[j0] - a[j2];
        x1i = a[j0 + 1] - a[j2 + 1];
        x2r = a[j1] + a[j3];
        x2i = a[j1 + 1] + a[j3 + 1];
        x3r = a[j1] - a[j3];
        x3i = a[j1 + 1] - a[j3 + 1];
        a[j0] = x0r + x2r;
        a[j0 + 1] = x0i + x2i;
        a[j1] = x0r - x2r;
        a[j1 + 1] = x0i - x2i;
        x0r = x1r - x3i;
        x0i = x1i + x3r;
        a[j2] = wk1i * x0r - wk1r * x0i;
        a[j2 + 1] = wk1i * x0i + wk1r * x0r;
        x0r = x1r + x3i;
        x0i = x1i - x3r;
        a[j3] = wk3i * x0r + wk3r * x0i;
        a[j3 + 1] = wk3i * x0i - wk3r * x0r;
    }
    j0 = mh;
    j1 = j0 + m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[j0] + a[j2];
    x0i = a[j0 + 1] + a[j2 + 1];
    x1r = a[j0] - a[j2];
    x1i = a[j0 + 1] - a[j2 + 1];
    x2r = a[j1] + a[j3];
    x2i = a[j1 + 1] + a[j3 + 1];
    x3r = a[j1] - a[j3];
    x3i = a[j1 + 1] - a[j3 + 1];
    a[j0] = x0r + x2r;
    a[j0 + 1] = x0i + x2i;
    a[j1] = x0r - x2r;
    a[j1 + 1] = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    a[j2] = wn4r * (x0r - x0i);
    a[j2 + 1] = wn4r * (x0i + x0r);
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    a[j3] = -wn4r * (x0r + x0i);
    a[j3 + 1] = -wn4r * (x0i - x0r);
}

/// `cftmdl2`: a radix-4 middle stage over `a[..n]` (the second twiddle pattern).
fn cftmdl2(n: usize, a: &mut [f32], w: &[f32]) {
    let mut j0: usize;
    let mut j1: usize;
    let mut j2: usize;
    let mut j3: usize;
    let mut k: usize;
    let mut kr: usize;
    let m: usize;
    let mh: usize;
    let wn4r: f32;
    let mut wk1r: f32;
    let mut wk1i: f32;
    let mut wk3r: f32;
    let mut wk3i: f32;
    let mut wd1r: f32;
    let mut wd1i: f32;
    let mut wd3r: f32;
    let mut wd3i: f32;
    let mut x0r: f32;
    let mut x0i: f32;
    let mut x1r: f32;
    let mut x1i: f32;
    let mut x2r: f32;
    let mut x2i: f32;
    let mut x3r: f32;
    let mut x3i: f32;
    let mut y0r: f32;
    let mut y0i: f32;
    let mut y2r: f32;
    let mut y2i: f32;

    mh = n >> 3;
    m = 2 * mh;
    wn4r = w[1];
    j1 = m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[0] - a[j2 + 1];
    x0i = a[1] + a[j2];
    x1r = a[0] + a[j2 + 1];
    x1i = a[1] - a[j2];
    x2r = a[j1] - a[j3 + 1];
    x2i = a[j1 + 1] + a[j3];
    x3r = a[j1] + a[j3 + 1];
    x3i = a[j1 + 1] - a[j3];
    y0r = wn4r * (x2r - x2i);
    y0i = wn4r * (x2i + x2r);
    a[0] = x0r + y0r;
    a[1] = x0i + y0i;
    a[j1] = x0r - y0r;
    a[j1 + 1] = x0i - y0i;
    y0r = wn4r * (x3r - x3i);
    y0i = wn4r * (x3i + x3r);
    a[j2] = x1r - y0i;
    a[j2 + 1] = x1i + y0r;
    a[j3] = x1r + y0i;
    a[j3 + 1] = x1i - y0r;
    k = 0;
    kr = 2 * m;
    for j in (2..mh).step_by(2) {
        k += 4;
        wk1r = w[k];
        wk1i = w[k + 1];
        wk3r = w[k + 2];
        wk3i = w[k + 3];
        kr -= 4;
        wd1i = w[kr];
        wd1r = w[kr + 1];
        wd3i = w[kr + 2];
        wd3r = w[kr + 3];
        j1 = j + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j] - a[j2 + 1];
        x0i = a[j + 1] + a[j2];
        x1r = a[j] + a[j2 + 1];
        x1i = a[j + 1] - a[j2];
        x2r = a[j1] - a[j3 + 1];
        x2i = a[j1 + 1] + a[j3];
        x3r = a[j1] + a[j3 + 1];
        x3i = a[j1 + 1] - a[j3];
        y0r = wk1r * x0r - wk1i * x0i;
        y0i = wk1r * x0i + wk1i * x0r;
        y2r = wd1r * x2r - wd1i * x2i;
        y2i = wd1r * x2i + wd1i * x2r;
        a[j] = y0r + y2r;
        a[j + 1] = y0i + y2i;
        a[j1] = y0r - y2r;
        a[j1 + 1] = y0i - y2i;
        y0r = wk3r * x1r + wk3i * x1i;
        y0i = wk3r * x1i - wk3i * x1r;
        y2r = wd3r * x3r + wd3i * x3i;
        y2i = wd3r * x3i - wd3i * x3r;
        a[j2] = y0r + y2r;
        a[j2 + 1] = y0i + y2i;
        a[j3] = y0r - y2r;
        a[j3 + 1] = y0i - y2i;
        j0 = m - j;
        j1 = j0 + m;
        j2 = j1 + m;
        j3 = j2 + m;
        x0r = a[j0] - a[j2 + 1];
        x0i = a[j0 + 1] + a[j2];
        x1r = a[j0] + a[j2 + 1];
        x1i = a[j0 + 1] - a[j2];
        x2r = a[j1] - a[j3 + 1];
        x2i = a[j1 + 1] + a[j3];
        x3r = a[j1] + a[j3 + 1];
        x3i = a[j1 + 1] - a[j3];
        y0r = wd1i * x0r - wd1r * x0i;
        y0i = wd1i * x0i + wd1r * x0r;
        y2r = wk1i * x2r - wk1r * x2i;
        y2i = wk1i * x2i + wk1r * x2r;
        a[j0] = y0r + y2r;
        a[j0 + 1] = y0i + y2i;
        a[j1] = y0r - y2r;
        a[j1 + 1] = y0i - y2i;
        y0r = wd3i * x1r + wd3r * x1i;
        y0i = wd3i * x1i - wd3r * x1r;
        y2r = wk3i * x3r + wk3r * x3i;
        y2i = wk3i * x3i - wk3r * x3r;
        a[j2] = y0r + y2r;
        a[j2 + 1] = y0i + y2i;
        a[j3] = y0r - y2r;
        a[j3 + 1] = y0i - y2i;
    }
    wk1r = w[m];
    wk1i = w[m + 1];
    j0 = mh;
    j1 = j0 + m;
    j2 = j1 + m;
    j3 = j2 + m;
    x0r = a[j0] - a[j2 + 1];
    x0i = a[j0 + 1] + a[j2];
    x1r = a[j0] + a[j2 + 1];
    x1i = a[j0 + 1] - a[j2];
    x2r = a[j1] - a[j3 + 1];
    x2i = a[j1 + 1] + a[j3];
    x3r = a[j1] + a[j3 + 1];
    x3i = a[j1 + 1] - a[j3];
    y0r = wk1r * x0r - wk1i * x0i;
    y0i = wk1r * x0i + wk1i * x0r;
    y2r = wk1i * x2r - wk1r * x2i;
    y2i = wk1i * x2i + wk1r * x2r;
    a[j0] = y0r + y2r;
    a[j0 + 1] = y0i + y2i;
    a[j1] = y0r - y2r;
    a[j1 + 1] = y0i - y2i;
    y0r = wk1i * x1r - wk1r * x1i;
    y0i = wk1i * x1i + wk1r * x1r;
    y2r = wk1r * x3r - wk1i * x3i;
    y2i = wk1r * x3i + wk1i * x3r;
    a[j2] = y0r - y2r;
    a[j2 + 1] = y0i - y2i;
    a[j3] = y0r + y2r;
    a[j3 + 1] = y0i + y2i;
}

/// `cftfx41`: the last stages for `n <= 128`, as four fixed kernels.
fn cftfx41(n: usize, a: &mut [f32], nw: usize, w: &[f32]) {
    if n == 128 {
        cftf161(a, &w[nw - 8..]);
        cftf162(&mut a[32..], &w[nw - 32..]);
        cftf161(&mut a[64..], &w[nw - 8..]);
        cftf161(&mut a[96..], &w[nw - 8..]);
    } else {
        cftf081(a, &w[nw - 8..]);
        cftf082(&mut a[16..], &w[nw - 8..]);
        cftf081(&mut a[32..], &w[nw - 8..]);
        cftf081(&mut a[48..], &w[nw - 8..]);
    }
}

/// `cftf161`: a 16-point butterfly kernel, unrolled.
fn cftf161(a: &mut [f32], w: &[f32]) {
    let wn4r: f32;
    let wk1r: f32;
    let wk1i: f32;
    let mut x0r: f32;
    let mut x0i: f32;
    let mut x1r: f32;
    let mut x1i: f32;
    let mut x2r: f32;
    let mut x2i: f32;
    let mut x3r: f32;
    let mut x3i: f32;
    let y0r: f32;
    let y0i: f32;
    let y1r: f32;
    let y1i: f32;
    let y2r: f32;
    let y2i: f32;
    let y3r: f32;
    let y3i: f32;
    let y4r: f32;
    let y4i: f32;
    let y5r: f32;
    let y5i: f32;
    let y6r: f32;
    let y6i: f32;
    let y7r: f32;
    let y7i: f32;
    let y8r: f32;
    let y8i: f32;
    let y9r: f32;
    let y9i: f32;
    let y10r: f32;
    let y10i: f32;
    let y11r: f32;
    let y11i: f32;
    let y12r: f32;
    let y12i: f32;
    let y13r: f32;
    let y13i: f32;
    let y14r: f32;
    let y14i: f32;
    let y15r: f32;
    let y15i: f32;

    wn4r = w[1];
    wk1r = w[2];
    wk1i = w[3];
    x0r = a[0] + a[16];
    x0i = a[1] + a[17];
    x1r = a[0] - a[16];
    x1i = a[1] - a[17];
    x2r = a[8] + a[24];
    x2i = a[9] + a[25];
    x3r = a[8] - a[24];
    x3i = a[9] - a[25];
    y0r = x0r + x2r;
    y0i = x0i + x2i;
    y4r = x0r - x2r;
    y4i = x0i - x2i;
    y8r = x1r - x3i;
    y8i = x1i + x3r;
    y12r = x1r + x3i;
    y12i = x1i - x3r;
    x0r = a[2] + a[18];
    x0i = a[3] + a[19];
    x1r = a[2] - a[18];
    x1i = a[3] - a[19];
    x2r = a[10] + a[26];
    x2i = a[11] + a[27];
    x3r = a[10] - a[26];
    x3i = a[11] - a[27];
    y1r = x0r + x2r;
    y1i = x0i + x2i;
    y5r = x0r - x2r;
    y5i = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    y9r = wk1r * x0r - wk1i * x0i;
    y9i = wk1r * x0i + wk1i * x0r;
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    y13r = wk1i * x0r - wk1r * x0i;
    y13i = wk1i * x0i + wk1r * x0r;
    x0r = a[4] + a[20];
    x0i = a[5] + a[21];
    x1r = a[4] - a[20];
    x1i = a[5] - a[21];
    x2r = a[12] + a[28];
    x2i = a[13] + a[29];
    x3r = a[12] - a[28];
    x3i = a[13] - a[29];
    y2r = x0r + x2r;
    y2i = x0i + x2i;
    y6r = x0r - x2r;
    y6i = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    y10r = wn4r * (x0r - x0i);
    y10i = wn4r * (x0i + x0r);
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    y14r = wn4r * (x0r + x0i);
    y14i = wn4r * (x0i - x0r);
    x0r = a[6] + a[22];
    x0i = a[7] + a[23];
    x1r = a[6] - a[22];
    x1i = a[7] - a[23];
    x2r = a[14] + a[30];
    x2i = a[15] + a[31];
    x3r = a[14] - a[30];
    x3i = a[15] - a[31];
    y3r = x0r + x2r;
    y3i = x0i + x2i;
    y7r = x0r - x2r;
    y7i = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    y11r = wk1i * x0r - wk1r * x0i;
    y11i = wk1i * x0i + wk1r * x0r;
    x0r = x1r + x3i;
    x0i = x1i - x3r;
    y15r = wk1r * x0r - wk1i * x0i;
    y15i = wk1r * x0i + wk1i * x0r;
    x0r = y12r - y14r;
    x0i = y12i - y14i;
    x1r = y12r + y14r;
    x1i = y12i + y14i;
    x2r = y13r - y15r;
    x2i = y13i - y15i;
    x3r = y13r + y15r;
    x3i = y13i + y15i;
    a[24] = x0r + x2r;
    a[25] = x0i + x2i;
    a[26] = x0r - x2r;
    a[27] = x0i - x2i;
    a[28] = x1r - x3i;
    a[29] = x1i + x3r;
    a[30] = x1r + x3i;
    a[31] = x1i - x3r;
    x0r = y8r + y10r;
    x0i = y8i + y10i;
    x1r = y8r - y10r;
    x1i = y8i - y10i;
    x2r = y9r + y11r;
    x2i = y9i + y11i;
    x3r = y9r - y11r;
    x3i = y9i - y11i;
    a[16] = x0r + x2r;
    a[17] = x0i + x2i;
    a[18] = x0r - x2r;
    a[19] = x0i - x2i;
    a[20] = x1r - x3i;
    a[21] = x1i + x3r;
    a[22] = x1r + x3i;
    a[23] = x1i - x3r;
    x0r = y5r - y7i;
    x0i = y5i + y7r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    x0r = y5r + y7i;
    x0i = y5i - y7r;
    x3r = wn4r * (x0r - x0i);
    x3i = wn4r * (x0i + x0r);
    x0r = y4r - y6i;
    x0i = y4i + y6r;
    x1r = y4r + y6i;
    x1i = y4i - y6r;
    a[8] = x0r + x2r;
    a[9] = x0i + x2i;
    a[10] = x0r - x2r;
    a[11] = x0i - x2i;
    a[12] = x1r - x3i;
    a[13] = x1i + x3r;
    a[14] = x1r + x3i;
    a[15] = x1i - x3r;
    x0r = y0r + y2r;
    x0i = y0i + y2i;
    x1r = y0r - y2r;
    x1i = y0i - y2i;
    x2r = y1r + y3r;
    x2i = y1i + y3i;
    x3r = y1r - y3r;
    x3i = y1i - y3i;
    a[0] = x0r + x2r;
    a[1] = x0i + x2i;
    a[2] = x0r - x2r;
    a[3] = x0i - x2i;
    a[4] = x1r - x3i;
    a[5] = x1i + x3r;
    a[6] = x1r + x3i;
    a[7] = x1i - x3r;
}

/// `cftf162`: the other 16-point butterfly kernel, unrolled.
fn cftf162(a: &mut [f32], w: &[f32]) {
    let wn4r: f32;
    let wk1r: f32;
    let wk1i: f32;
    let wk2r: f32;
    let wk2i: f32;
    let wk3r: f32;
    let wk3i: f32;
    let mut x0r: f32;
    let mut x0i: f32;
    let mut x1r: f32;
    let mut x1i: f32;
    let mut x2r: f32;
    let mut x2i: f32;
    let y0r: f32;
    let y0i: f32;
    let y1r: f32;
    let y1i: f32;
    let y2r: f32;
    let y2i: f32;
    let y3r: f32;
    let y3i: f32;
    let y4r: f32;
    let y4i: f32;
    let y5r: f32;
    let y5i: f32;
    let y6r: f32;
    let y6i: f32;
    let y7r: f32;
    let y7i: f32;
    let y8r: f32;
    let y8i: f32;
    let y9r: f32;
    let y9i: f32;
    let y10r: f32;
    let y10i: f32;
    let y11r: f32;
    let y11i: f32;
    let y12r: f32;
    let y12i: f32;
    let y13r: f32;
    let y13i: f32;
    let y14r: f32;
    let y14i: f32;
    let y15r: f32;
    let y15i: f32;

    wn4r = w[1];
    wk1r = w[4];
    wk1i = w[5];
    wk3r = w[6];
    wk3i = -w[7];
    wk2r = w[8];
    wk2i = w[9];
    x1r = a[0] - a[17];
    x1i = a[1] + a[16];
    x0r = a[8] - a[25];
    x0i = a[9] + a[24];
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    y0r = x1r + x2r;
    y0i = x1i + x2i;
    y4r = x1r - x2r;
    y4i = x1i - x2i;
    x1r = a[0] + a[17];
    x1i = a[1] - a[16];
    x0r = a[8] + a[25];
    x0i = a[9] - a[24];
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    y8r = x1r - x2i;
    y8i = x1i + x2r;
    y12r = x1r + x2i;
    y12i = x1i - x2r;
    x0r = a[2] - a[19];
    x0i = a[3] + a[18];
    x1r = wk1r * x0r - wk1i * x0i;
    x1i = wk1r * x0i + wk1i * x0r;
    x0r = a[10] - a[27];
    x0i = a[11] + a[26];
    x2r = wk3i * x0r - wk3r * x0i;
    x2i = wk3i * x0i + wk3r * x0r;
    y1r = x1r + x2r;
    y1i = x1i + x2i;
    y5r = x1r - x2r;
    y5i = x1i - x2i;
    x0r = a[2] + a[19];
    x0i = a[3] - a[18];
    x1r = wk3r * x0r - wk3i * x0i;
    x1i = wk3r * x0i + wk3i * x0r;
    x0r = a[10] + a[27];
    x0i = a[11] - a[26];
    x2r = wk1r * x0r + wk1i * x0i;
    x2i = wk1r * x0i - wk1i * x0r;
    y9r = x1r - x2r;
    y9i = x1i - x2i;
    y13r = x1r + x2r;
    y13i = x1i + x2i;
    x0r = a[4] - a[21];
    x0i = a[5] + a[20];
    x1r = wk2r * x0r - wk2i * x0i;
    x1i = wk2r * x0i + wk2i * x0r;
    x0r = a[12] - a[29];
    x0i = a[13] + a[28];
    x2r = wk2i * x0r - wk2r * x0i;
    x2i = wk2i * x0i + wk2r * x0r;
    y2r = x1r + x2r;
    y2i = x1i + x2i;
    y6r = x1r - x2r;
    y6i = x1i - x2i;
    x0r = a[4] + a[21];
    x0i = a[5] - a[20];
    x1r = wk2i * x0r - wk2r * x0i;
    x1i = wk2i * x0i + wk2r * x0r;
    x0r = a[12] + a[29];
    x0i = a[13] - a[28];
    x2r = wk2r * x0r - wk2i * x0i;
    x2i = wk2r * x0i + wk2i * x0r;
    y10r = x1r - x2r;
    y10i = x1i - x2i;
    y14r = x1r + x2r;
    y14i = x1i + x2i;
    x0r = a[6] - a[23];
    x0i = a[7] + a[22];
    x1r = wk3r * x0r - wk3i * x0i;
    x1i = wk3r * x0i + wk3i * x0r;
    x0r = a[14] - a[31];
    x0i = a[15] + a[30];
    x2r = wk1i * x0r - wk1r * x0i;
    x2i = wk1i * x0i + wk1r * x0r;
    y3r = x1r + x2r;
    y3i = x1i + x2i;
    y7r = x1r - x2r;
    y7i = x1i - x2i;
    x0r = a[6] + a[23];
    x0i = a[7] - a[22];
    x1r = wk1i * x0r + wk1r * x0i;
    x1i = wk1i * x0i - wk1r * x0r;
    x0r = a[14] + a[31];
    x0i = a[15] - a[30];
    x2r = wk3i * x0r - wk3r * x0i;
    x2i = wk3i * x0i + wk3r * x0r;
    y11r = x1r + x2r;
    y11i = x1i + x2i;
    y15r = x1r - x2r;
    y15i = x1i - x2i;
    x1r = y0r + y2r;
    x1i = y0i + y2i;
    x2r = y1r + y3r;
    x2i = y1i + y3i;
    a[0] = x1r + x2r;
    a[1] = x1i + x2i;
    a[2] = x1r - x2r;
    a[3] = x1i - x2i;
    x1r = y0r - y2r;
    x1i = y0i - y2i;
    x2r = y1r - y3r;
    x2i = y1i - y3i;
    a[4] = x1r - x2i;
    a[5] = x1i + x2r;
    a[6] = x1r + x2i;
    a[7] = x1i - x2r;
    x1r = y4r - y6i;
    x1i = y4i + y6r;
    x0r = y5r - y7i;
    x0i = y5i + y7r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    a[8] = x1r + x2r;
    a[9] = x1i + x2i;
    a[10] = x1r - x2r;
    a[11] = x1i - x2i;
    x1r = y4r + y6i;
    x1i = y4i - y6r;
    x0r = y5r + y7i;
    x0i = y5i - y7r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    a[12] = x1r - x2i;
    a[13] = x1i + x2r;
    a[14] = x1r + x2i;
    a[15] = x1i - x2r;
    x1r = y8r + y10r;
    x1i = y8i + y10i;
    x2r = y9r - y11r;
    x2i = y9i - y11i;
    a[16] = x1r + x2r;
    a[17] = x1i + x2i;
    a[18] = x1r - x2r;
    a[19] = x1i - x2i;
    x1r = y8r - y10r;
    x1i = y8i - y10i;
    x2r = y9r + y11r;
    x2i = y9i + y11i;
    a[20] = x1r - x2i;
    a[21] = x1i + x2r;
    a[22] = x1r + x2i;
    a[23] = x1i - x2r;
    x1r = y12r - y14i;
    x1i = y12i + y14r;
    x0r = y13r + y15i;
    x0i = y13i - y15r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    a[24] = x1r + x2r;
    a[25] = x1i + x2i;
    a[26] = x1r - x2r;
    a[27] = x1i - x2i;
    x1r = y12r + y14i;
    x1i = y12i - y14r;
    x0r = y13r - y15i;
    x0i = y13i + y15r;
    x2r = wn4r * (x0r - x0i);
    x2i = wn4r * (x0i + x0r);
    a[28] = x1r - x2i;
    a[29] = x1i + x2r;
    a[30] = x1r + x2i;
    a[31] = x1i - x2r;
}

/// `cftf081`: an 8-point butterfly kernel, unrolled.
fn cftf081(a: &mut [f32], w: &[f32]) {
    let wn4r: f32;
    let mut x0r: f32;
    let mut x0i: f32;
    let mut x1r: f32;
    let mut x1i: f32;
    let mut x2r: f32;
    let mut x2i: f32;
    let mut x3r: f32;
    let mut x3i: f32;
    let y0r: f32;
    let y0i: f32;
    let y1r: f32;
    let y1i: f32;
    let y2r: f32;
    let y2i: f32;
    let y3r: f32;
    let y3i: f32;
    let y4r: f32;
    let y4i: f32;
    let y5r: f32;
    let y5i: f32;
    let y6r: f32;
    let y6i: f32;
    let y7r: f32;
    let y7i: f32;

    wn4r = w[1];
    x0r = a[0] + a[8];
    x0i = a[1] + a[9];
    x1r = a[0] - a[8];
    x1i = a[1] - a[9];
    x2r = a[4] + a[12];
    x2i = a[5] + a[13];
    x3r = a[4] - a[12];
    x3i = a[5] - a[13];
    y0r = x0r + x2r;
    y0i = x0i + x2i;
    y2r = x0r - x2r;
    y2i = x0i - x2i;
    y1r = x1r - x3i;
    y1i = x1i + x3r;
    y3r = x1r + x3i;
    y3i = x1i - x3r;
    x0r = a[2] + a[10];
    x0i = a[3] + a[11];
    x1r = a[2] - a[10];
    x1i = a[3] - a[11];
    x2r = a[6] + a[14];
    x2i = a[7] + a[15];
    x3r = a[6] - a[14];
    x3i = a[7] - a[15];
    y4r = x0r + x2r;
    y4i = x0i + x2i;
    y6r = x0r - x2r;
    y6i = x0i - x2i;
    x0r = x1r - x3i;
    x0i = x1i + x3r;
    x2r = x1r + x3i;
    x2i = x1i - x3r;
    y5r = wn4r * (x0r - x0i);
    y5i = wn4r * (x0r + x0i);
    y7r = wn4r * (x2r - x2i);
    y7i = wn4r * (x2r + x2i);
    a[8] = y1r + y5r;
    a[9] = y1i + y5i;
    a[10] = y1r - y5r;
    a[11] = y1i - y5i;
    a[12] = y3r - y7i;
    a[13] = y3i + y7r;
    a[14] = y3r + y7i;
    a[15] = y3i - y7r;
    a[0] = y0r + y4r;
    a[1] = y0i + y4i;
    a[2] = y0r - y4r;
    a[3] = y0i - y4i;
    a[4] = y2r - y6i;
    a[5] = y2i + y6r;
    a[6] = y2r + y6i;
    a[7] = y2i - y6r;
}

/// `cftf082`: the other 8-point butterfly kernel, unrolled.
fn cftf082(a: &mut [f32], w: &[f32]) {
    let wn4r: f32;
    let wk1r: f32;
    let wk1i: f32;
    let mut x0r: f32;
    let mut x0i: f32;
    let mut x1r: f32;
    let mut x1i: f32;
    let y0r: f32;
    let y0i: f32;
    let y1r: f32;
    let y1i: f32;
    let y2r: f32;
    let y2i: f32;
    let y3r: f32;
    let y3i: f32;
    let y4r: f32;
    let y4i: f32;
    let y5r: f32;
    let y5i: f32;
    let y6r: f32;
    let y6i: f32;
    let y7r: f32;
    let y7i: f32;

    wn4r = w[1];
    wk1r = w[2];
    wk1i = w[3];
    y0r = a[0] - a[9];
    y0i = a[1] + a[8];
    y1r = a[0] + a[9];
    y1i = a[1] - a[8];
    x0r = a[4] - a[13];
    x0i = a[5] + a[12];
    y2r = wn4r * (x0r - x0i);
    y2i = wn4r * (x0i + x0r);
    x0r = a[4] + a[13];
    x0i = a[5] - a[12];
    y3r = wn4r * (x0r - x0i);
    y3i = wn4r * (x0i + x0r);
    x0r = a[2] - a[11];
    x0i = a[3] + a[10];
    y4r = wk1r * x0r - wk1i * x0i;
    y4i = wk1r * x0i + wk1i * x0r;
    x0r = a[2] + a[11];
    x0i = a[3] - a[10];
    y5r = wk1i * x0r - wk1r * x0i;
    y5i = wk1i * x0i + wk1r * x0r;
    x0r = a[6] - a[15];
    x0i = a[7] + a[14];
    y6r = wk1i * x0r - wk1r * x0i;
    y6i = wk1i * x0i + wk1r * x0r;
    x0r = a[6] + a[15];
    x0i = a[7] - a[14];
    y7r = wk1r * x0r - wk1i * x0i;
    y7i = wk1r * x0i + wk1i * x0r;
    x0r = y0r + y2r;
    x0i = y0i + y2i;
    x1r = y4r + y6r;
    x1i = y4i + y6i;
    a[0] = x0r + x1r;
    a[1] = x0i + x1i;
    a[2] = x0r - x1r;
    a[3] = x0i - x1i;
    x0r = y0r - y2r;
    x0i = y0i - y2i;
    x1r = y4r - y6r;
    x1i = y4i - y6i;
    a[4] = x0r - x1i;
    a[5] = x0i + x1r;
    a[6] = x0r + x1i;
    a[7] = x0i - x1r;
    x0r = y1r - y3i;
    x0i = y1i + y3r;
    x1r = y5r - y7r;
    x1i = y5i - y7i;
    a[8] = x0r + x1r;
    a[9] = x0i + x1i;
    a[10] = x0r - x1r;
    a[11] = x0i - x1i;
    x0r = y1r + y3i;
    x0i = y1i - y3r;
    x1r = y5r + y7r;
    x1i = y5i + y7i;
    a[12] = x0r - x1i;
    a[13] = x0i + x1r;
    a[14] = x0r + x1i;
    a[15] = x0i - x1r;
}

/// `cftf040`: the 4-point forward FFT.
fn cftf040(a: &mut [f32]) {
    let x0r: f32;
    let x0i: f32;
    let x1r: f32;
    let x1i: f32;
    let x2r: f32;
    let x2i: f32;
    let x3r: f32;
    let x3i: f32;

    x0r = a[0] + a[4];
    x0i = a[1] + a[5];
    x1r = a[0] - a[4];
    x1i = a[1] - a[5];
    x2r = a[2] + a[6];
    x2i = a[3] + a[7];
    x3r = a[2] - a[6];
    x3i = a[3] - a[7];
    a[0] = x0r + x2r;
    a[1] = x0i + x2i;
    a[2] = x1r - x3i;
    a[3] = x1i + x3r;
    a[4] = x0r - x2r;
    a[5] = x0i - x2i;
    a[6] = x1r + x3i;
    a[7] = x1i - x3r;
}

/// `cftb040`: the 4-point backward FFT.
fn cftb040(a: &mut [f32]) {
    let x0r: f32;
    let x0i: f32;
    let x1r: f32;
    let x1i: f32;
    let x2r: f32;
    let x2i: f32;
    let x3r: f32;
    let x3i: f32;

    x0r = a[0] + a[4];
    x0i = a[1] + a[5];
    x1r = a[0] - a[4];
    x1i = a[1] - a[5];
    x2r = a[2] + a[6];
    x2i = a[3] + a[7];
    x3r = a[2] - a[6];
    x3i = a[3] - a[7];
    a[0] = x0r + x2r;
    a[1] = x0i + x2i;
    a[2] = x1r + x3i;
    a[3] = x1i - x3r;
    a[4] = x0r - x2r;
    a[5] = x0i - x2i;
    a[6] = x1r - x3i;
    a[7] = x1i + x3r;
}

/// `cftx020`: the 2-point FFT.
fn cftx020(a: &mut [f32]) {
    let x0r: f32;
    let x0i: f32;

    x0r = a[0] - a[2];
    x0i = a[1] - a[3];
    a[0] += a[2];
    a[1] += a[3];
    a[2] = x0r;
    a[3] = x0i;
}

/// `rftfsub`: the post-processing that turns the forward half-length complex FFT into a real one.
fn rftfsub(n: usize, a: &mut [f32], nc: usize, c: &[f32]) {
    let mut k: usize;
    let mut kk: usize;
    let ks: usize;
    let m: usize;
    let mut wkr: f32;
    let mut wki: f32;
    let mut xr: f32;
    let mut xi: f32;
    let mut yr: f32;
    let mut yi: f32;

    m = n >> 1;
    ks = 2 * nc / m;
    kk = 0;
    for j in (2..m).step_by(2) {
        k = n - j;
        kk += ks;
        wkr = (0.5f64 - c[nc - kk] as f64) as f32;
        wki = c[kk];
        xr = a[j] - a[k];
        xi = a[j + 1] + a[k + 1];
        yr = wkr * xr - wki * xi;
        yi = wkr * xi + wki * xr;
        a[j] -= yr;
        a[j + 1] -= yi;
        a[k] += yr;
        a[k + 1] -= yi;
    }
}

/// `rftbsub`: the matching pre-processing for the backward direction.
fn rftbsub(n: usize, a: &mut [f32], nc: usize, c: &[f32]) {
    let mut k: usize;
    let mut kk: usize;
    let ks: usize;
    let m: usize;
    let mut wkr: f32;
    let mut wki: f32;
    let mut xr: f32;
    let mut xi: f32;
    let mut yr: f32;
    let mut yi: f32;

    m = n >> 1;
    ks = 2 * nc / m;
    kk = 0;
    for j in (2..m).step_by(2) {
        k = n - j;
        kk += ks;
        wkr = (0.5f64 - c[nc - kk] as f64) as f32;
        wki = c[kk];
        xr = a[j] - a[k];
        xi = a[j + 1] + a[k + 1];
        yr = wkr * xr + wki * xi;
        yi = wkr * xi - wki * xr;
        a[j] -= yr;
        a[j + 1] -= yi;
        a[k] += yr;
        a[k + 1] -= yi;
    }
}

/// `dctsub`: the cosine transform's pre/post twiddle.
fn dctsub(n: usize, a: &mut [f32], nc: usize, c: &[f32]) {
    let mut k: usize;
    let mut kk: usize;
    let ks: usize;
    let m: usize;
    let mut wkr: f32;
    let mut wki: f32;
    let mut xr: f32;

    m = n >> 1;
    ks = nc / n;
    kk = 0;
    for j in 1..m {
        k = n - j;
        kk += ks;
        wkr = c[kk] - c[nc - kk];
        wki = c[kk] + c[nc - kk];
        xr = wki * a[j] - wkr * a[k];
        a[j] = wkr * a[j] + wki * a[k];
        a[k] = xr;
    }
    a[m] *= c[0];
}

/// `dstsub`: the sine transform's pre/post twiddle.
fn dstsub(n: usize, a: &mut [f32], nc: usize, c: &[f32]) {
    let mut k: usize;
    let mut kk: usize;
    let ks: usize;
    let m: usize;
    let mut wkr: f32;
    let mut wki: f32;
    let mut xr: f32;

    m = n >> 1;
    ks = nc / n;
    kk = 0;
    for j in 1..m {
        k = n - j;
        kk += ks;
        wkr = c[kk] - c[nc - kk];
        wki = c[kk] + c[nc - kk];
        xr = wki * a[k] - wkr * a[j];
        a[k] = wkr * a[k] + wki * a[j];
        a[j] = xr;
    }
    a[m] *= c[0];
}


#[cfg(test)]
mod tests {
    use super::*;

    /// The inputs the pins were taken with: a 32-bit LCG, each value `(s >> 8) - 2^23` over
    /// `2^23` (exact in `f32`).
    struct Lcg(u32);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            ((self.0 >> 8) as i32 - 8388608) as f32 / 8388608.0f32
        }
    }

    fn tables(n1: usize, n2: usize) -> (Vec<i32>, Vec<f32>) {
        let m = n1.max(n2);
        (vec![0; (m as f64).sqrt().round() as usize + 2], vec![0.0; m * 3 / 2])
    }

    /// The placer's four calls on one pair of tables: `ddct2d(-1)`, `ddct2d(+1)`, `ddsct2d(+1)`,
    /// `ddcst2d(+1)`, each on fresh inputs. Returns each call's output.
    fn four_calls(n1: usize, n2: usize, rng: &mut Lcg, ip: &mut [i32], w: &mut [f32]) -> Vec<Vec<f32>> {
        (0..4)
            .map(|op| {
                let mut a: Vec<f32> = (0..n1 * n2).map(|_| rng.next()).collect();
                match op {
                    0 => ddct2d(n1, n2, -1, &mut a, ip, w),
                    1 => ddct2d(n1, n2, 1, &mut a, ip, w),
                    2 => ddsct2d(n1, n2, 1, &mut a, ip, w),
                    _ => ddcst2d(n1, n2, 1, &mut a, ip, w),
                }
                a
            })
            .collect()
    }

    // Read only by the macOS bit-for-bit test below.
    #[cfg(target_os = "macos")]
    fn fnv1a(v: &[f32]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for f in v {
            for b in f.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        }
        h
    }

    /// Pins from the C package itself (`fftsg.cpp` + `fftsg2d.cpp` built with clang `-O2
    /// -ffp-contract=off -fmath-errno` as the reference build compiles it, same LCG inputs, one
    /// table pair per grid shared by the four calls):
    /// an FNV-1a hash of every output float's bits, plus the first two and the last words.
    /// The tables take `cosf`/`sinf` from the platform's libm, and libms disagree in the last
    /// bit on a few arguments, so these pins hold for the platform they were taken on (macOS);
    /// the full bit-for-bit sweep (4x4 to 2048x16 and 1024x1024, tables shared within and across
    /// grids) is run against the C on each platform outside the crate.
    #[test]
    #[cfg(target_os = "macos")]
    fn matches_the_c_package_bit_for_bit() {
        // (n1, n2, per call: (hash, a[0], a[1], a[last]) bits)
        type Pin = (usize, usize, [(u64, u32, u32, u32); 4]);
        #[rustfmt::skip]
        let pins: [Pin; 3] = [
            (4, 4, [
                (0xb9e53ec39e56c049, 0xbf361b3e, 0xbed32db4, 0xc02bacc6),
                (0xc63057ba51a29c82, 0x3f9c818c, 0x3f3dd054, 0xbfb1367b),
                (0xa308df0c5a922026, 0x3f82b9a9, 0xbffd522a, 0xbe2dc9e4),
                (0x40a8f8855e82dab0, 0x3ee6d837, 0x3fa1f9b4, 0x40045ba2),
            ]),
            (8, 16, [
                (0x03804e6426532993, 0xc003280b, 0x3f505a26, 0xc01c6380),
                (0x46d1db17aa578830, 0x4001bd62, 0x3fb31cdc, 0x40bf3bea),
                (0x99ee991f65393fff, 0xbd8dd970, 0xc08d171a, 0xbf93a3c0),
                (0x99ad707917fecc4d, 0x40a835a2, 0x40012f19, 0xc103be54),
            ]),
            (64, 32, [
                (0xe66fb44b58dffab0, 0xc1e029f2, 0xc0a5b41c, 0x410331a1),
                (0xc0d1d56fd4a76f74, 0x3f7450e0, 0x418ca055, 0x40029111),
                (0xa2375d954d7a499a, 0xc19b21f6, 0x40f94b0e, 0x410843b5),
                (0x45b18b04f357665a, 0x3fef7128, 0xc08bac49, 0x3ef002c0),
            ]),
        ];
        let mut rng = Lcg(12345);
        for (n1, n2, want) in pins {
            let (mut ip, mut w) = tables(n1, n2);
            let got = four_calls(n1, n2, &mut rng, &mut ip, &mut w);
            for (op, (a, (hash, first0, first1, last))) in got.iter().zip(want).enumerate() {
                let words = (a[0].to_bits(), a[1].to_bits(), a[a.len() - 1].to_bits());
                assert_eq!((fnv1a(a), words), (hash, (first0, first1, last)), "{n1}x{n2} call {op}");
            }
        }
    }

    /// The tables are built once and reused: a call on tables a larger grid already built (a
    /// bigger cosine half read at a stride) gives the same bits as on fresh ones — the C's
    /// `ip[0]`/`ip[1]` size checks.
    #[test]
    fn reused_tables_give_the_same_bits() {
        for (n1, n2) in [(4, 4), (8, 16), (16, 8), (64, 32)] {
            let (mut ip, mut w) = tables(n1, n2);
            let fresh = four_calls(n1, n2, &mut Lcg(7), &mut ip, &mut w);
            // Once more on the same tables (built by the first call, reused from then on).
            let again = four_calls(n1, n2, &mut Lcg(7), &mut ip, &mut w);
            assert_eq!(fresh, again, "{n1}x{n2} reused");
            // On tables sized and built for a larger grid first.
            let (mut ip, mut w) = tables(256, 256);
            four_calls(256, 256, &mut Lcg(1), &mut ip, &mut w);
            assert_eq!(ip[0], 64);
            let big = four_calls(n1, n2, &mut Lcg(7), &mut ip, &mut w);
            for (a, b) in fresh.iter().zip(&big) {
                let close = a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-4 * (1.0 + x.abs()));
                assert!(close, "{n1}x{n2} on larger tables");
            }
        }
    }

    /// `ddct2d(-1)`, halving row 0 and column 0 and scaling by `4 / (n1 n2)`, then `ddct2d(+1)`
    /// returns the input: the pair is a transform and its inverse.
    #[test]
    fn forward_then_inverse_is_the_identity() {
        let (n1, n2) = (16, 32);
        let (mut ip, mut w) = tables(n1, n2);
        let mut rng = Lcg(3);
        let x: Vec<f32> = (0..n1 * n2).map(|_| rng.next()).collect();
        let mut a = x.clone();
        ddct2d(n1, n2, -1, &mut a, &mut ip, &mut w);
        for v in a.iter_mut().take(n2) {
            *v *= 0.5;
        }
        for i in 0..n1 {
            a[i * n2] *= 0.5;
        }
        for v in a.iter_mut() {
            *v *= 4.0 / (n1 * n2) as f32;
        }
        ddct2d(n1, n2, 1, &mut a, &mut ip, &mut w);
        for (got, want) in a.iter().zip(&x) {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
    }
}
