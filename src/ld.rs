// SPDX-License-Identifier: Apache-2.0
//! x87 extended precision (`long double` on x86-64 Linux: a 64-bit significand, round to nearest
//! even), for the few expressions the placer evaluates in it: the density solver's wavenumbers
//! (`π · i / n`, π as the extended constant) and the √2 threshold that sizes a cell's density box.
//!
//! ⛔ Evaluating these in `f64` instead rounds twice — to 53 bits, then to 24 — and lands one
//! `f32` ulp off whenever the 53-bit value sits on a 24-bit rounding boundary that the 64-bit one
//! does not. Only the operations those expressions use are modelled: non-negative finite values,
//! multiply, divide, compare, and conversion to `f32`.

use std::cmp::Ordering;

/// A non-negative extended value: `m · 2^e`, `m` normalised (top bit set) or zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ld {
    m: u64,
    e: i32,
}

/// π as an extended constant (the 64-bit significand of π, rounded to nearest).
pub const PI: Ld = Ld { m: 0xC90F_DAA2_2168_C235, e: -62 };
/// √2 as the literal `1.414213562373095048801L` reads into an extended value.
pub const SQRT2: Ld = Ld { m: 0xB504_F333_F9DE_6484, e: -63 };

impl Ld {
    pub const ZERO: Ld = Ld { m: 0, e: 0 };

    fn normalise(m: u64, e: i32) -> Ld {
        if m == 0 {
            return Ld::ZERO;
        }
        let z = m.leading_zeros() as i32;
        Ld { m: m << z, e: e - z }
    }

    /// An exact non-negative integer.
    pub fn from_u64(v: u64) -> Ld {
        Ld::normalise(v, 0)
    }

    /// An exact non-negative `f64`.
    pub fn from_f64(v: f64) -> Ld {
        assert!(v >= 0.0 && v.is_finite(), "extended value out of the modelled range: {v}");
        if v == 0.0 {
            return Ld::ZERO;
        }
        let bits = v.to_bits();
        let exp = ((bits >> 52) & 0x7FF) as i32;
        let frac = bits & ((1u64 << 52) - 1);
        let (m, e) = if exp == 0 { (frac, -1074) } else { (frac | (1u64 << 52), exp - 1075) };
        Ld::normalise(m, e)
    }

    /// An exact non-negative `f32`.
    pub fn from_f32(v: f32) -> Ld {
        Ld::from_f64(f64::from(v))
    }

    /// Round a wide significand `p` (with `p != 0`) at exponent `e` to 64 bits, nearest even.
    fn round_u128(p: u128, e: i32) -> Ld {
        let len = 128 - p.leading_zeros() as i32;
        if len <= 64 {
            return Ld::normalise(p as u64, e);
        }
        let shift = len - 64;
        let mut keep = (p >> shift) as u64;
        let rem = p & ((1u128 << shift) - 1);
        let half = 1u128 << (shift - 1);
        let mut e = e + shift;
        if rem > half || (rem == half && keep & 1 == 1) {
            let (k, carry) = keep.overflowing_add(1);
            keep = k;
            if carry {
                keep = 1u64 << 63;
                e += 1;
            }
        }
        Ld { m: keep, e }
    }

    /// `a · b`, rounded once to the extended format.
    pub fn times(self, b: Ld) -> Ld {
        if self.m == 0 || b.m == 0 {
            return Ld::ZERO;
        }
        Ld::round_u128(u128::from(self.m) * u128::from(b.m), self.e + b.e)
    }

    /// `a / b`, rounded once to the extended format.
    pub fn over(self, b: Ld) -> Ld {
        assert!(b.m != 0, "division by zero");
        if self.m == 0 {
            return Ld::ZERO;
        }
        let n = u128::from(self.m) << 64;
        let d = u128::from(b.m);
        let (q, r) = (n / d, n % d);
        let e = self.e - b.e - 64;
        if q >> 64 != 0 {
            // 65 bits: the last one is the guard, the remainder the sticky.
            let guard = q & 1 == 1;
            let mut keep = (q >> 1) as u64;
            let mut e = e + 1;
            if guard && (r != 0 || keep & 1 == 1) {
                let (k, carry) = keep.overflowing_add(1);
                keep = k;
                if carry {
                    keep = 1u64 << 63;
                    e += 1;
                }
            }
            Ld { m: keep, e }
        } else {
            // Exactly 64 bits: the remainder against half the divisor decides.
            let mut keep = q as u64;
            let mut e = e;
            let twice = r << 1;
            if twice > d || (twice == d && keep & 1 == 1) {
                let (k, carry) = keep.overflowing_add(1);
                keep = k;
                if carry {
                    keep = 1u64 << 63;
                    e += 1;
                }
            }
            Ld::normalise(keep, e)
        }
    }

    /// Conversion to `f32`, nearest even (normal results only, which is every value modelled).
    pub fn to_f32(self) -> f32 {
        if self.m == 0 {
            return 0.0;
        }
        let mut m = self.m >> 40;
        let rem = self.m & ((1u64 << 40) - 1);
        let half = 1u64 << 39;
        let mut e = self.e + 40;
        if rem > half || (rem == half && m & 1 == 1) {
            m += 1;
            if m == 1 << 24 {
                m >>= 1;
                e += 1;
            }
        }
        let field = e + 23 + 127;
        assert!((1..255).contains(&field), "extended value outside the f32 normal range");
        f32::from_bits(((field as u32) << 23) | (m as u32 & 0x7F_FFFF))
    }

    /// The value's exponent of its leading bit (for ordering).
    fn top(self) -> i32 {
        self.e + 63
    }
}

impl PartialOrd for Ld {
    fn partial_cmp(&self, o: &Ld) -> Option<Ordering> {
        Some(match (self.m == 0, o.m == 0) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => self.top().cmp(&o.top()).then(self.m.cmp(&o.m)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The constants are the extended values of π and √2 (64-bit significands, nearest).
    #[test]
    fn the_constants_read_back_as_pi_and_root_two() {
        assert_eq!(PI.to_f32(), std::f32::consts::PI);
        assert_eq!(SQRT2.to_f32(), std::f32::consts::SQRT_2);
        let pi64 = (PI.m >> 11) as f64 * 2f64.powi(PI.e + 11);
        assert_eq!(pi64, std::f64::consts::PI, "the top 53 bits of the extended π are π's double");
    }

    /// Multiplication and division round ONCE, at 64 bits: (π·3)/3 is π again here, and products
    /// of exact integers stay exact.
    #[test]
    fn arithmetic_rounds_once_at_sixty_four_bits() {
        let three = Ld::from_u64(3);
        assert_eq!(Ld::from_u64(6).over(three), Ld::from_u64(2));
        assert_eq!(Ld::from_u64(1 << 40).times(Ld::from_u64(1 << 20)), Ld::from_u64(1 << 60));
        assert_eq!(Ld::from_u64(1).over(three).times(three), Ld::from_u64(1), "1/3·3 rounds back to 1 in 64 bits");
        assert!(Ld::from_u64(5) < Ld::from_f64(5.5) && Ld::from_f64(5.5) < Ld::from_u64(6));
        assert_eq!(Ld::from_f64(0.75).to_f32(), 0.75);
    }

    /// A value 2^-30 above an f32 halfway point rounds UP from 64 bits; via f64 it would too, but
    /// one 2^-60 above rounds up only from 64 bits (f64 drops it and ties to even, down).
    #[test]
    fn conversion_to_f32_sees_bits_a_double_would_drop() {
        // 1 + 2^-24 (a tie between 1 and 1+2^-23) plus 2^-60.
        let v = Ld { m: (1u64 << 63) | (1u64 << 39) | (1u64 << 3), e: -63 };
        assert_eq!(v.to_f32(), 1.0 + f32::EPSILON);
        let tie = Ld { m: (1u64 << 63) | (1u64 << 39), e: -63 };
        assert_eq!(tie.to_f32(), 1.0, "an exact tie goes to even");
    }
}
