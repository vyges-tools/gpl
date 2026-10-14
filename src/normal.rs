// SPDX-License-Identifier: Apache-2.0
//! Normal variates drawn exactly as `boost::random::normal_distribution<double>` (Boost 1.89)
//! draws them from a `boost::random::mt19937`.
//!
//! The placer perturbs each instance once with one shared engine and a freshly built distribution
//! per draw, so nothing carries between draws except the engine state. What has to match is the
//! number of 32-bit words each draw takes from the engine and the double arithmetic on them.
//!
//! Boost 1.89 samples the unit normal with a 128-layer ziggurat (Marsaglia and Tsang):
//! - one bucket+fraction pair costs TWO engine words: the low 8 bits of the first pick the layer
//!   (bit 0 the sign, bits 1..7 the layer), its high 24 bits and the low 29 bits of the second form
//!   a 53-bit fraction in [0, 1) ([`int_float_pair`]);
//! - a point inside the layer's rectangle core is returned at once (2 words);
//! - otherwise layer 0 goes to the tail, and any other layer draws ONE more word as a uniform
//!   height and accepts or rejects against cheap linear bounds, then the density itself; a rejected
//!   point restarts from a new pair;
//! - the tail takes two exponentials (rates `R` and 1, where `R` is the tail start) until
//!   `2y > x^2`, returning `R + x`. The exponentials are themselves a 256-layer ziggurat with the
//!   same pair generator, where the tail is just a restart with a shift.
//!
//! The tables are the library's `double` tables, written here as exact bit patterns.

/// A 32-bit Mersenne Twister, `boost::random::mt19937`.
///
/// Seeding from a 32-bit value is the reference `init_genrand` recurrence; the library then
/// rewrites the low 31 bits of word 0, which the first twist never reads, so its output equals
/// the standard `std::mt19937` for the same seed (checked against both on the reference build).
#[derive(Clone)]
pub struct Mt19937 {
    x: [u32; MT_N],
    i: usize,
}

const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_A: u32 = 0x9908_b0df;
const MT_UPPER: u32 = 0x8000_0000;
const MT_LOWER: u32 = 0x7fff_ffff;

impl Mt19937 {
    /// The engine constructed from `seed`; an `int` seed is taken modulo 2^32 as the library does.
    pub fn new(seed: u32) -> Self {
        let mut x = [0u32; MT_N];
        x[0] = seed;
        for i in 1..MT_N {
            x[i] = 1_812_433_253u32.wrapping_mul(x[i - 1] ^ (x[i - 1] >> 30)).wrapping_add(i as u32);
        }
        let mut mt = Mt19937 { x, i: MT_N };
        mt.normalize_state();
        mt
    }

    /// The library's post-seed fix-up: recompute word 0's low bits from words m-1 and n-1, and
    /// never leave an all-zero state.
    fn normalize_state(&mut self) {
        let mut y0 = self.x[MT_M - 1] ^ self.x[MT_N - 1];
        if y0 & MT_UPPER != 0 {
            y0 = ((y0 ^ MT_A) << 1) | 1;
        } else {
            y0 <<= 1;
        }
        self.x[0] = (self.x[0] & MT_UPPER) | (y0 & MT_LOWER);
        if self.x.iter().all(|&w| w == 0) {
            self.x[0] = MT_UPPER;
        }
    }

    fn twist(&mut self) {
        let x = &mut self.x;
        for j in 0..MT_N - MT_M {
            let y = (x[j] & MT_UPPER) | (x[j + 1] & MT_LOWER);
            x[j] = x[j + MT_M] ^ (y >> 1) ^ ((x[j + 1] & 1).wrapping_mul(MT_A));
        }
        for j in MT_N - MT_M..MT_N - 1 {
            let y = (x[j] & MT_UPPER) | (x[j + 1] & MT_LOWER);
            x[j] = x[j - (MT_N - MT_M)] ^ (y >> 1) ^ ((x[j + 1] & 1).wrapping_mul(MT_A));
        }
        let y = (x[MT_N - 1] & MT_UPPER) | (x[0] & MT_LOWER);
        x[MT_N - 1] = x[MT_M - 1] ^ (y >> 1) ^ ((x[0] & 1).wrapping_mul(MT_A));
        self.i = 0;
    }

    /// The next 32-bit output (the engine's `operator()`).
    pub fn next_u32(&mut self) -> u32 {
        if self.i == MT_N {
            self.twist();
        }
        let mut z = self.x[self.i];
        self.i += 1;
        z ^= z >> 11;
        z ^= (z << 7) & 0x9d2c_5680;
        z ^= (z << 15) & 0xefc6_0000;
        z ^= z >> 18;
        z
    }
}

/// One draw of a freshly constructed `normal_distribution<double>(mean, sigma)` from `eng`:
/// the unit normal scaled by `sigma`, then shifted by `mean`, in that order.
pub fn normal(eng: &mut Mt19937, mean: f64, sigma: f64) -> f64 {
    unit_normal(eng) * sigma + mean
}

/// The layer index (low 8 bits of word 1) and a 53-bit fraction in [0, 1): the high 24 bits of
/// word 1 scaled by 2^-24, plus the low 29 bits of word 2, all scaled by 2^-29. Two words.
fn int_float_pair(eng: &mut Mt19937) -> (f64, usize) {
    let u = eng.next_u32();
    let bucket = (u & 0xff) as usize;
    let mut r = f64::from(u >> 8) * (1.0 / f64::from(1u32 << 24));
    let u = eng.next_u32();
    r += f64::from(u & 0x1fff_ffff);
    r *= 0.5 / f64::from(1u32 << 28);
    (r, bucket)
}

/// `uniform_01<double>` on a 32-bit engine: one word times 2^-32 (never reaches 1, so one word).
fn uniform_01(eng: &mut Mt19937) -> f64 {
    let factor = 1.0 / (f64::from(u32::MAX) + 1.0);
    f64::from(eng.next_u32()) * factor
}

/// The unit normal ziggurat; see the module notes for the draw counts.
fn unit_normal(eng: &mut Mt19937) -> f64 {
    loop {
        let (r, bucket) = int_float_pair(eng);
        let sign = f64::from((bucket & 1) as i32 * 2 - 1);
        let i = bucket >> 1;
        let x = r * NORMAL_X[i];
        if x < NORMAL_X[i + 1] {
            return x * sign;
        }
        if i == 0 {
            return normal_tail(eng) * sign;
        }
        let y01 = uniform_01(eng);
        let y = NORMAL_Y[i] + y01 * (NORMAL_Y[i + 1] - NORMAL_Y[i]);
        // Signed distances of the point above the chord and above the tangent; which of the two
        // bounds the density from above flips at the inflection point (x = 1).
        let chord = (NORMAL_X[i] - NORMAL_X[i + 1]) * y01 - (NORMAL_X[i] - x);
        let tangent = y - (NORMAL_Y[i] + (NORMAL_X[i] - x) * NORMAL_Y[i] * NORMAL_X[i]);
        let (above_ubound, above_lbound) =
            if NORMAL_X[i] >= 1.0 { (chord, tangent) } else { (tangent, chord) };
        if above_ubound < 0.0 && (above_lbound < 0.0 || y < (-(x * x / 2.0)).exp()) {
            return x * sign;
        }
    }
}

/// The normal tail beyond `R = NORMAL_X[1]`: `x ~ Exp(R)`, `y ~ Exp(1)`, drawn in that order,
/// until `2y > x^2`; returns `R + x`.
fn normal_tail(eng: &mut Mt19937) -> f64 {
    let tail_start = NORMAL_X[1];
    loop {
        let x = unit_exponential(eng) / tail_start;
        let y = unit_exponential(eng) / 1.0;
        if 2.0 * y > x * x {
            return x + tail_start;
        }
    }
}

/// The unit exponential ziggurat (256 layers). Its tail is the body again, shifted.
fn unit_exponential(eng: &mut Mt19937) -> f64 {
    let mut shift = 0.0;
    loop {
        let (r, i) = int_float_pair(eng);
        let x = r * EXP_X[i];
        if x < EXP_X[i + 1] {
            return shift + x;
        }
        if i == 0 {
            shift += EXP_X[1];
        } else {
            let y01 = uniform_01(eng);
            let y = EXP_Y[i] + y01 * (EXP_Y[i + 1] - EXP_Y[i]);
            let above_ubound = (EXP_X[i] - EXP_X[i + 1]) * y01 - (EXP_X[i] - x);
            let above_lbound = y - (EXP_Y[i + 1] + (EXP_X[i + 1] - x) * EXP_Y[i + 1]);
            if above_ubound < 0.0 && (above_lbound < 0.0 || y < (-x).exp()) {
                return x + shift;
            }
        }
    }
}

const fn b(bits: u64) -> f64 {
    f64::from_bits(bits)
}

// The library's `double` layer tables (normal: 129 edges, exponential: 257), as exact bits.
const NORMAL_X: [f64; 129] = [
    b(0x400db4668fe7d167), b(0x400b8a7c476d1741), b(0x4009c8e0c7c7f35e), b(0x4008aa73e440e862),
    b(0x4007d45eb36e9ff4), b(0x4007279dd4ac2679), b(0x400695c2be68d3e4), b(0x400616dff7c8dab3),
    b(0x4005a61edf7e73f4), b(0x400540520129e8c8), b(0x4004e3456b0e1da8), b(0x40048d61806d430c),
    b(0x40043d75b60bac8d), b(0x4003f29848d395fe), b(0x4003ac11b8e1e839), b(0x4003694f3a3721ba),
    b(0x400329d9725e1358), b(0x4002ed4df8097554), b(0x4002b35aa5ebcda5), b(0x40027bba2b5d9b7d),
    b(0x400246317a6b3231), b(0x4002128dd36bbd01), b(0x4001e0a342cee675), b(0x4001b04b731f48d4),
    b(0x40018164be0bf8c9), b(0x400153d16d455057), b(0x4001277720181096), b(0x4000fc3e4d95cda5),
    b(0x4000d211dd288ac4), b(0x4000a8ded0ec1159), b(0x40008093fe3e1aa9), b(0x40005921d1c4b0b9),
    b(0x4000327a1cc4a836), b(0x40000c8fea16f933), b(0x3fffceaeb2ca0ee2), b(0x3fff858aff317ac8),
    b(0x3fff3da09745b605), b(0x3ffef6dcddc7807d), b(0x3ffeb12e914817af), b(0x3ffe6c85a8495b0d),
    b(0x3ffe28d331c61c36), b(0x3ffde609397db2b3), b(0x3ffda41aaf794b3c), b(0x3ffd62fb5257b279),
    b(0x3ffd229f9bfe95c7), b(0x3ffce2fcb05f3115), b(0x3ffca4084e08c207), b(0x3ffc65b8c04d5d84),
    b(0x3ffc2804d2c6531d), b(0x3ffbeae3c60c7179), b(0x3ffbae4d457e8092), b(0x3ffb72395df55593),
    b(0x3ffb36a075492a98), b(0x3ffafb7b428f83ac), b(0x3ffac0c2c6fbfe60), b(0x3ffa8670475107fb),
    b(0x3ffa4c7d45cfb2a5), b(0x3ffa12e37c97caa0), b(0x3ff9d99cd86aeea8), b(0x3ff9a0a373c6d3cc),
    b(0x3ff967f1924c0e62), b(0x3ff92f819c67bdfd), b(0x3ff8f74e1b375764), b(0x3ff8bf51b49e8281),
    b(0x3ff8878727879e86), b(0x3ff84fe948480027), b(0x3ff81872fd216669), b(0x3ff7e11f3ada7506),
    b(0x3ff7a9e9016840d7), b(0x3ff772cb58a3242a), b(0x3ff73bc14d01277f), b(0x3ff704c5ec504e8f),
    b(0x3ff6cdd4426b0a02), b(0x3ff696e755e0eb23), b(0x3ff65ffa248d7f43), b(0x3ff62907a016eac0),
    b(0x3ff5f20aaa4d7638), b(0x3ff5bafe1164c044), b(0x3ff583dc8bfea848), b(0x3ff54ca0b4ff476a),
    b(0x3ff5154507206658), b(0x3ff4ddc3d839cb58), b(0x3ff4a6175432745f), b(0x3ff46e39778d4ba1),
    b(0x3ff4362409821672), b(0x3ff3fdd0959138fb), b(0x3ff3c538647e5b53), b(0x3ff38c54749af146),
    b(0x3ff3531d71460289), b(0x3ff3198ba9823477), b(0x3ff2df97057dd75f), b(0x3ff2a536fae26375),
    b(0x3ff26a627fb9231d), b(0x3ff22f0ffba96ce9), b(0x3ff1f33537495bfa), b(0x3ff1b6c7492bde7a),
    b(0x3ff179ba80458345), b(0x3ff13c024b2bbdff), b(0x3ff0fd911b972d18), b(0x3ff0be58456f2afc),
    b(0x3ff07e47d879726e), b(0x3ff03d4e7390f210), b(0x3feff6b21ffe30ec), b(0x3fef70a5866ad189),
    b(0x3feee848e954b85c), b(0x3fee5d6909f34423), b(0x3fedcfccc51a7480), b(0x3fed3f340dd86c6b),
    b(0x3fecab56ac6833a5), b(0x3fec13e2b012d149), b(0x3feb787a7c4f44a4), b(0x3fead8b25067d385),
    b(0x3fea340d1bad0391), b(0x3fe989f85c72c985), b(0x3fe8d9c6a9d0cf67), b(0x3fe822a858ac5eca),
    b(0x3fe763a1600c1764), b(0x3fe69b7b213c3f64), b(0x3fe5c8afdbecef6e), b(0x3fe4e94c08bd4d78),
    b(0x3fe3fabee18d682f), b(0x3fe2f98d6bb0e73a), b(0x3fe1e0ce6b54ec53), b(0x3fe0a936da5942d2),
    b(0x3fde8e576e3830fa), b(0x3fdb4c8fecd63b02), b(0x3fd73949183add9d), b(0x3fd16db47dfb32bd),
    b(0x0000000000000000),
];

const NORMAL_Y: [f64; 129] = [
    b(0x0000000000000000), b(0x3f65de9e33733182), b(0x3f76ba8b0ffc2db8), b(0x3f81a9b6b3fcb829),
    b(0x3f883f4bed1a0f0b), b(0x3f8f100847656bf0), b(0x3f9309cee4e1477c), b(0x3f96a23fa9d6c22f),
    b(0x3f9a4f57a25e8f32), b(0x3f9e0f951d58f849), b(0x3fa0f0e539c938c0), b(0x3fa2e282b7255da2),
    b(0x3fa4dc3fcbda5a08), b(0x3fa6ddc9dd20b8c5), b(0x3fa8e6db483cac0f), b(0x3faaf738c17b4ea1),
    b(0x3fad0eaf633a6b8a), b(0x3faf2d13368cf93f), b(0x3fb0a91f0918dae5), b(0x3fb1bf075c21538a),
    b(0x3fb2d834113457cb), b(0x3fb3f49878976d30), b(0x3fb514297b246583), b(0x3fb636dd69e998c6),
    b(0x3fb75cabd60f402a), b(0x3fb8858d6f55ed84), b(0x3fb9b17be7e73957), b(0x3fbae071dc7bf93d),
    b(0x3fbc126ac0128a82), b(0x3fbd4762ca995a18), b(0x3fbe7f56ea118c48), b(0x3fbfba44b5c61816),
    b(0x3fc07c1531a357f8), b(0x3fc11c835e726135), b(0x3fc1be6c8cbe5a43), b(0x3fc261d0aaaf7624),
    b(0x3fc306afe619efed), b(0x3fc3ad0aa9de455d), b(0x3fc454e19baadb54), b(0x3fc4fe359a145658),
    b(0x3fc5a907bafba9e3), b(0x3fc655594a3a5050), b(0x3fc7032bc88e51fa), b(0x3fc7b280eac0c6f7),
    b(0x3fc8635a99025d7b), b(0x3fc915baee7a2ddd), b(0x3fc9c9a43903cae2), b(0x3fca7f18f91a0d6a),
    b(0x3fcb361be1ec9a67), b(0x3fcbeeafd99e93b6), b(0x3fcca8d7f9ad4b43), b(0x3fcd64978f7e2d92),
    b(0x3fce21f21d136fa3), b(0x3fcee0eb59e75db3), b(0x3fcfa18733ee75d5), b(0x3fd031e4e8606256),
    b(0x3fd093dbc775a1f7), b(0x3fd0f6aa83b52201), b(0x3fd15a5387a71a06), b(0x3fd1bed95cc633cb),
    b(0x3fd2243eac7ee400), b(0x3fd28a864146d917), b(0x3fd2f1b307cdcc47), b(0x3fd359c810492f8e),
    b(0x3fd3c2c88fdc65e7), b(0x3fd42cb7e21f69bf), b(0x3fd497998ac6017a), b(0x3fd503713769e39c),
    b(0x3fd57042c17a74d2), b(0x3fd5de1230551a9b), b(0x3fd64ce3bb89770e), b(0x3fd6bcbbcd4d4694),
    b(0x3fd72d9f052408dd), b(0x3fd79f923abf1d11), b(0x3fd8129a811b882e), b(0x3fd886bd29e33e65),
    b(0x3fd8fbffc918800b), b(0x3fd972683912ac18), b(0x3fd9e9fc9ed4d931), b(0x3fda62c36ec797ea),
    b(0x3fdadcc371e07b84), b(0x3fdb5803cb437071), b(0x3fdbd48bfe6b8a90), b(0x3fdc5263f5ead9fc),
    b(0x3fdcd1940ad30932), b(0x3fdd52250cdb191e), b(0x3fddd4204b59916b), b(0x3fde578f9f2e03a3),
    b(0x3fdedc7d75b8e9be), b(0x3fdf62f4dd05d60f), b(0x3fdfeb019151c56e), b(0x3fe03a58060f304a),
    b(0x3fe08006ca85ac6a), b(0x3fe0c6942a5c900f), b(0x3fe10e07b50236c1), b(0x3fe1566980fc6949),
    b(0x3fe19fc2397562a2), b(0x3fe1ea1b2d9fe534), b(0x3fe2357e62437dc2), b(0x3fe281f6a5d33891),
    b(0x3fe2cf8fa7868c02), b(0x3fe31e5612075dad), b(0x3fe36e57aa6a89b9), b(0x3fe3bfa3745495cd),
    b(0x3fe41249dc6579c8), b(0x3fe4665cea512cc7), b(0x3fe4bbf07c6d4684), b(0x3fe5131a8eff8ed9),
    b(0x3fe56bf3924ad864), b(0x3fe5c696d34a27fd), b(0x3fe62322fc5a83b3), b(0x3fe681bab4ed2ff3),
    b(0x3fe6e2856a01cb2a), b(0x3fe745b04d03ea40), b(0x3fe7ab6f9c66e43b), b(0x3fe81400521b52b5),
    b(0x3fe87faa61a8cfa0), b(0x3fe8eec3c5bda1f6), b(0x3fe961b4c1b19f30), b(0x3fe9d8fdfaee4af6),
    b(0x3fea55418112ba08), b(0x3fead750b7275dd0), b(0x3feb6042cf926211), b(0x3febf19b6813348b),
    b(0x3fec8d923fa0897b), b(0x3fed37a74ffe486a), b(0x3fedf6071937f4c9), b(0x3feed5cf061144de),
    b(0x3ff0000000000000),
];

const EXP_X: [f64; 257] = [
    b(0x402164ec94bf5dc1), b(0x401ec9d9297ebb83), b(0x401bc39e51da71fc), b(0x4019e9dc0d487b85),
    b(0x4018939fe6f2ed19), b(0x40178750d6eac62f), b(0x4016aa676d4bbf72), b(0x4015ee7ae17313d2),
    b(0x40154ad83ccf73f6), b(0x4014b9d7cd4751d1), b(0x4014379766e41362), b(0x4013c14ec7c8b861),
    b(0x401354ee27ccf75e), b(0x4012f0e38a4411f0), b(0x401293f5ae49aaa5), b(0x40123d2bb659919f),
    b(0x4011ebbca0c9fa7c), b(0x40119f03bcb3c2d6), b(0x401156786775442a), b(0x401111a8034392a6),
    b(0x4010d031785d48a0), b(0x401091c1cdcba54e), b(0x401056118bf58eef), b(0x40101ce2b362ec2e),
    b(0x400fcbfe43f6c6e5), b(0x400f626e9791f7a7), b(0x400efcc26750ea4a), b(0x400e9aaf2af383c1),
    b(0x400e3bf26e190960), b(0x400de050af4ef19f), b(0x400d87946fec3bec), b(0x400d318d6b2738c5),
    b(0x400cde0fecf2a97f), b(0x400c8cf442c8c8f4), b(0x400c3e1641c2e0a7), b(0x400bf154de4bef77),
    b(0x400ba691d276da5e), b(0x400b5db15091ea0f), b(0x400b1699c003b60a), b(0x400ad13382d845c4),
    b(0x400a8d68c2ad86ea), b(0x400a4b2543e84c3b), b(0x400a0a563e49f178), b(0x4009caea3a24d9ea),
    b(0x40098cd0f18d1ad8), b(0x40094ffb34fc2a0e), b(0x4009145ad2f37544), b(0x4008d9e2823b3695),
    b(0x4008a085ce695bab), b(0x4008683906687342), b(0x400830f12cc0bec3), b(0x4007faa3e96e1412),
    b(0x4007c5477d1476d3), b(0x400790d2b56b71f9), b(0x40075d3ce2bd71c3), b(0x40072a7dce5cd218),
    b(0x4006f88db1f42507), b(0x4006c7652f9a7b1e), b(0x400696fd4a9748ee), b(0x4006674f60c3f432),
    b(0x40063855247b2e94), b(0x40060a0897081879), b(0x4005dc640388bd9e), b(0x4005af61fa38e107),
    b(0x400582fd4c1b4461), b(0x4005573106f8a75a), b(0x40052bf871acaab2), b(0x4005014f08b99508),
    b(0x4004d7307b1cb127), b(0x4004ad98a75da14c), b(0x4004848398d39432), b(0x40045bed851bc92c),
    b(0x400433d2c9bd42f8), b(0x40040c2fe9f5eead), b(0x4003e5018cadded0), b(0x4003be447a8d8b83),
    b(0x400397f59c345143), b(0x40037211f88ca856), b(0x40034c96b33bc965), b(0x400327810b2aa7d0),
    b(0x400302ce59265965), b(0x4002de7c0e962d70), b(0x4002ba87b445db51), b(0x400296eee942532b),
    b(0x400273af61c7daa6), b(0x400250c6e6403bba), b(0x40022e33524fe550), b(0x40020bf293f0f4a2),
    b(0x4001ea02aa9b3370), b(0x4001c861a6782a5a), b(0x4001a70da7a27820), b(0x40018604dd6fae9e),
    b(0x4001654585c404c1), b(0x400144cdec6f3a2b), b(0x4001249c6a92154a), b(0x400104af660befce),
    b(0x4000e50550efcfb7), b(0x4000c59ca900946f), b(0x4000a673f733c819), b(0x40008789cf3aad0f),
    b(0x400068dccf1126db), b(0x40004a6b9e9224a3), b(0x40002c34ef11391b), b(0x40000e377af911d4),
    b(0x3fffe0e40add09d8), b(0x3fffa5c6b3efe1e5), b(0x3fff6b1498515ed0), b(0x3fff30cb6ea0bc7f),
    b(0x3ffef6e8fc5b9168), b(0x3ffebd6b154a7678), b(0x3ffe844f9af4237f), b(0x3ffe4b947c16a452),
    b(0x3ffe1337b426509b), b(0x3ffddb374ad2357f), b(0x3ffda391538da50a), b(0x3ffd6c43ed1ea3fe),
    b(0x3ffd354d4130f2ad), b(0x3ffcfeab83ed7180), b(0x3ffcc85cf395a56c), b(0x3ffc925fd82323fb),
    b(0x3ffc5cb282eab1a4), b(0x3ffc27534e42e02d), b(0x3ffbf2409d2dfd85), b(0x3ffbbd78db072610),
    b(0x3ffb88fa7b324fb6), b(0x3ffb54c3f8cf2542), b(0x3ffb20d3d66e8bb5), b(0x3ffaed289dcaacff),
    b(0x3ffab9c0df81657a), b(0x3ffa869b32d0f30f), b(0x3ffa53b63556c690), b(0x3ffa21108ad0592d),
    b(0x3ff9eea8dcdde951), b(0x3ff9bc7ddac7035d), b(0x3ff98a8e3940bbf4), b(0x3ff958d8b235828a),
    b(0x3ff9275c048e73e1), b(0x3ff8f616f3fe1513), b(0x3ff8c50848cc6094), b(0x3ff8942ecfa40f54),
    b(0x3ff86389596108e7), b(0x3ff83316badfe62a), b(0x3ff802d5ccce7277), b(0x3ff7d2c56b7d17f7),
    b(0x3ff7a2e476b1240a), b(0x3ff77331d177d130), b(0x3ff743ac61fa041c), b(0x3ff714531150a9fb),
    b(0x3ff6e524cb59a608), b(0x3ff6b6207e8d3cdf), b(0x3ff687451bd3ebee), b(0x3ff65891965c9b8c),
    b(0x3ff62a04e3731a2e), b(0x3ff5fb9dfa56cf26), b(0x3ff5cd5bd4119335), b(0x3ff59f3d6b4e9cf9),
    b(0x3ff57141bc316f27), b(0x3ff54367c42cb5f8), b(0x3ff515ae81d900fb), b(0x3ff4e814f4cb45ea),
    b(0x3ff4ba9a1d6b18a4), b(0x3ff48d3cfcc883c4), b(0x3ff45ffc94716ca7), b(0x3ff432d7e6466cd0),
    b(0x3ff405cdf44f09c4), b(0x3ff3d8ddc08d336d), b(0x3ff3ac064ccfeffc), b(0x3ff37f469a851af0),
    b(0x3ff3529daa8a1ba1), b(0x3ff3260a7cfb7611), b(0x3ff2f98c11031721), b(0x3ff2cd2164a53b5d),
    b(0x3ff2a0c9748bcdaa), b(0x3ff274833bd0189f), b(0x3ff2484db3c2a329), b(0x3ff21c27d3b10e05),
    b(0x3ff1f01090a9c4e2), b(0x3ff1c406dd3d5283), b(0x3ff19809a93d2396), b(0x3ff16c17e1777ffb),
    b(0x3ff140306f707dbe), b(0x3ff114523917ac15), b(0x3ff0e87c207a2f66), b(0x3ff0bcad03710137),
    b(0x3ff090e3bb4b0072), b(0x3ff0651f1c7276f8), b(0x3ff0395df60db162), b(0x3ff00d9f119a3cd9),
    b(0x3fefc3c26504a9a1), b(0x3fef6c462b57feb5), b(0x3fef14c6e202949f), b(0x3feebd41e5e21b62),
    b(0x3fee65b483cf1044), b(0x3fee0e1bf77c31fe), b(0x3fedb6756a429057), b(0x3fed5ebdf1d86b8d),
    b(0x3fed06f28ef0e6fb), b(0x3fecaf102bc25adb), b(0x3fec57139a70d29f), b(0x3febfef99359fe99),
    b(0x3feba6beb33f8f89), b(0x3feb4e5f794c979b), b(0x3feaf5d844f224c9), b(0x3fea9d255396d261),
    b(0x3fea4442be14884a), b(0x3fe9eb2c75ff03bf), b(0x3fe991de42ad1338), b(0x3fe93853bdfda244),
    b(0x3fe8de8850d0c52a), b(0x3fe884772f2be1ec), b(0x3fe82a1b53fed599), b(0x3fe7cf6f7c7e8172),
    b(0x3fe7746e23077973), b(0x3fe71911797990bb), b(0x3fe6bd5362faa944), b(0x3fe6612d6d0c68e0),
    b(0x3fe60498c7dd2ecf), b(0x3fe5a78e3db8befd), b(0x3fe54a0629786f4d), b(0x3fe4ebf86bcd0b93),
    b(0x3fe48d5c5f35e712), b(0x3fe42e28ca706748), b(0x3fe3ce53d12162a0), b(0x3fe36dd2e26d8202),
    b(0x3fe30c9aa526da4b), b(0x3fe2aa9ee123680b), b(0x3fe247d26538ff2e), b(0x3fe1e426e93e49e7),
    b(0x3fe17f8ceb4bdfa0), b(0x3fe119f38749f5af), b(0x3fe0b348479b80fc), b(0x3fe04b76ed6a7558),
    b(0x3fdfc4d25d683209), b(0x3fdef00ccf5f4faa), b(0x3fde186678f1735a), b(0x3fdd3da24df17c36),
    b(0x3fdc5f7bd78c3f89), b(0x3fdb7da5dddda3c4), b(0x3fda97c8be5d5203), b(0x3fd9ad80552237d2),
    b(0x3fd8be5954d3606f), b(0x3fd7c9cdda17d019), b(0x3fd6cf40f0a72bbd), b(0x3fd5cdf89d024ac3),
    b(0x3fd4c515c60bfe21), b(0x3fd3b388fe3d6eca), b(0x3fd2980290da2633), b(0x3fd170db24d6f670),
    b(0x3fd03bf049c65c3c), b(0x3fcdecd8b76dbd98), b(0x3fcb38d1ef79b7cc), b(0x3fc85090fbc27a80),
    b(0x3fc522e6e54a2a73), b(0x3fc19335a95b8dba), b(0x3fbad6b2495b4d2b), b(0x3fb0589d8b5d4119),
    b(0x0000000000000000),
];

const EXP_Y: [f64; 257] = [
    b(0x0000000000000000), b(0x3f3dc31c329f0b4b), b(0x3f4fb20af78dfcb9), b(0x3f592bb5540c3e25),
    b(0x3f61946ba8e1a324), b(0x3f66d888f3a1feff), b(0x3f6c58b381cd4b11), b(0x3f71073d69574043),
    b(0x3f73fa97cee322fd), b(0x3f77049f37ec3620), b(0x3f7a23e9d4974836), b(0x3f7d5751fa745dc5),
    b(0x3f804ef2295fd7f9), b(0x3f81fb69edb37671), b(0x3f83b0b8c1516f62), b(0x3f856e930be416cb),
    b(0x3f8734b6e6aa74f5), b(0x3f8902ea688fa7bd), b(0x3f8ad8fa5542c92d), b(0x3f8cb6b9146e2757),
    b(0x3f8e9bfdde89c7ce), b(0x3f904452091e02f0), b(0x3f913e4554725f5f), b(0x3f923bc9e1b93a32),
    b(0x3f933cd225315d84), b(0x3f944151ce87f0be), b(0x3f95493da6ab0251), b(0x3f96548b72a24077),
    b(0x3f976331da87fc96), b(0x3f98752853ec9967), b(0x3f998a670f132a48), b(0x3f9aa2e6e6924e9b),
    b(0x3f9bbea150fa5870), b(0x3f9cdd9054331b0c), b(0x3f9dffae7a517468), b(0x3f9f24f6c7af9890),
    b(0x3fa026b2590dfaee), b(0x3fa0bc7a0c7cd651), b(0x3fa153d09f19b3a1), b(0x3fa1ecb45ff312d4),
    b(0x3fa28723c956c00c), b(0x3fa3231d7e3f14ae), b(0x3fa3c0a047ff18ff), b(0x3fa45fab14266b19),
    b(0x3fa5003cf296c5eb), b(0x3fa5a25513c5d2ca), b(0x3fa645f2c726a041), b(0x3fa6eb1579b6af52),
    b(0x3fa791bcb4ab089e), b(0x3fa839e81c3a396b), b(0x3fa8e3976e80776d), b(0x3fa98eca827b7c4c),
    b(0x3faa3b81471bf138), b(0x3faae9bbc26a8084), b(0x3fab997a10bed985), b(0x3fac4abc640721e9),
    b(0x3facfd83031e794a), b(0x3fadb1ce49315810), b(0x3fae679ea52eb2e5), b(0x3faf1ef49944e834),
    b(0x3fafd7d0ba699676), b(0x3fb04919d7f5c817), b(0x3fb0a70f19871b3b), b(0x3fb105c88756ca50),
    b(0x3fb165468f755392), b(0x3fb1c589a86fa340), b(0x3fb22692512c9d8c), b(0x3fb2886110ce0570),
    b(0x3fb2eaf676948dd1), b(0x3fb34e5319c6e718), b(0x3fb3b277999b9f9e), b(0x3fb417649d25b10e),
    b(0x3fb47d1ad343985c), b(0x3fb4e39af290d929), b(0x3fb54ae5b959d036), b(0x3fb5b2fbed91bb3e),
    b(0x3fb61bde5ccadef7), b(0x3fb6858ddc30b620), b(0x3fb6f00b488416b6), b(0x3fb75b5786193c1e),
    b(0x3fb7c77380d7a6f3), b(0x3fb834602c3bc4ba), b(0x3fb8a21e835a533b), b(0x3fb910af88e574b9),
    b(0x3fb9801447336b70), b(0x3fb9f04dd046f428), b(0x3fba615d3dd938b7), b(0x3fbad343b1655465),
    b(0x3fbb460254356548), b(0x3fbbb99a5771268f), b(0x3fbc2e0cf42e10af), b(0x3fbca35b6b80fd57),
    b(0x3fbd198706914dd7), b(0x3fbd909116ad9398), b(0x3fbe087af561bafb), b(0x3fbe8146048eb9cc),
    b(0x3fbefaf3ae83c33c), b(0x3fbf758566190414), b(0x3fbff0fca6cbea8d), b(0x3fc036ad7a6e7f04),
    b(0x3fc07550eeb7a5be), b(0x3fc0b4697b54b62f), b(0x3fc0f3f7efec1720), b(0x3fc133fd20c9712f),
    b(0x3fc17479e6f0ae78), b(0x3fc1b56f2031d666), b(0x3fc1f6ddaf3dca65), b(0x3fc238c67bbbe878),
    b(0x3fc27b2a72609940), b(0x3fc2be0a8504cf34), b(0x3fc30167aabe7d6e), b(0x3fc34542dffa0caf),
    b(0x3fc3899d2694d5c9), b(0x3fc3ce7785f8a905), b(0x3fc413d30b386a9a), b(0x3fc459b0c92dccc6),
    b(0x3fc4a011d8983096), b(0x3fc4e6f7583cb6fa), b(0x3fc52e626d078c49), b(0x3fc57654422e78f5),
    b(0x3fc5bece0954c2b6), b(0x3fc607d0fab06a31), b(0x3fc6515e5530d1ac), b(0x3fc69b775ea6da28),
    b(0x3fc6e61d63ee84ea), b(0x3fc73151b91a2839), b(0x3fc77d15b99f46fe), b(0x3fc7c96ac8851bae),
    b(0x3fc816525094e7e6), b(0x3fc863cdc48c1af9), b(0x3fc8b1de9f5062d5), b(0x3fc900866425bb79),
    b(0x3fc94fc69ee692a1), b(0x3fc99fa0e43e1623), b(0x3fc9f016d1e4c512), b(0x3fca412a0edf5cbc),
    b(0x3fca92dc4bc03c49), b(0x3fcae52f42eb5b0b), b(0x3fcb3824b8dcef3e), b(0x3fcb8bbe7c72e4a5),
    b(0x3fcbdffe67394435), b(0x3fcc34e65db9afee), b(0x3fcc8a784fce1802), b(0x3fcce0b638f6d09f),
    b(0x3fcd37a220b431fd), b(0x3fcd8f3e1ae3eeb8), b(0x3fcde78c48224f39), b(0x3fce408ed62f83a7),
    b(0x3fce9a48005940f2), b(0x3fcef4ba0fe8e09b), b(0x3fcf4fe75c963e7e), b(0x3fcfabd24cff9354),
    b(0x3fd0043eab93476a), b(0x3fd032f580797c2c), b(0x3fd0620ef05d90d2), b(0x3fd0918c4ee93e13),
    b(0x3fd0c16ef88f5333), b(0x3fd0f1b852d9a66c), b(0x3fd12269ccba9fba), b(0x3fd15384dee291ef),
    b(0x3fd1850b0c191982), b(0x3fd1b6fde19abc5a), b(0x3fd1e95ef77b09db), b(0x3fd21c2ff10b7eff),
    b(0x3fd24f727d4776fd), b(0x3fd2832857457629), b(0x3fd2b75346ae2262), b(0x3fd2ebf520394270),
    b(0x3fd3210fc6312435), b(0x3fd356a528fcd0dd), b(0x3fd38cb747b17def), b(0x3fd3c34830abb285),
    b(0x3fd3fa5a0230a14e), b(0x3fd431eeeb1841e2), b(0x3fd46a092b80beef), b(0x3fd4a2ab158bdad3),
    b(0x3fd4dbd70e26f91d), b(0x3fd5158f8dde89f5), b(0x3fd54fd721bda3e7), b(0x3fd58ab06c3aa9ef),
    b(0x3fd5c61e2631ee6c), b(0x3fd602231fef5876), b(0x3fd63ec2424827e4), b(0x3fd67bfe8fc60d9f),
    b(0x3fd6b9db25e4e99c), b(0x3fd6f85b3e649e9d), b(0x3fd7378230b08dea), b(0x3fd77753735e72e3),
    b(0x3fd7b7d29dc6801e), b(0x3fd7f90369b6ce59), b(0x3fd83ae9b5446138), b(0x3fd87d8984bc3f8c),
    b(0x3fd8c0e704b75d39), b(0x3fd905068c545d04), b(0x3fd949ec9f9a8110), b(0x3fd98f9df2097ba8),
    b(0x3fd9d61f695a3792), b(0x3fda1d76207521f4), b(0x3fda65a76aa30140), b(0x3fdaaeb8d6fdf6e5),
    b(0x3fdaf8b03428ef5f), b(0x3fdb43939454806f), b(0x3fdb8f6951990b88), b(0x3fdbdc3812aeeeb5),
    b(0x3fdc2a06d00ea583), b(0x3fdc78dcd983fb60), b(0x3fdcc8c1dc40e092), b(0x3fdd19bde97e1a0b),
    b(0x3fdd6bd97db9ed7a), b(0x3fddbf1d88a7210c), b(0x3fde139375e137fc), b(0x3fde6945367dd351),
    b(0x3fdec03d4b969d90), b(0x3fdf1886d1eb424d), b(0x3fdf722d8ebfc5fa), b(0x3fdfcd3dfe214576),
    b(0x3fe014e2b160f324), b(0x3fe043e8ebd26548), b(0x3fe073b931ee3b7d), b(0x3fe0a45b8854d02a),
    b(0x3fe0d5d8812b1e2b), b(0x3fe108394a1cc38d), b(0x3fe13b87bc33169c), b(0x3fe16fce6dce6fee),
    b(0x3fe1a518c71e3b25), b(0x3fe1db7319877b89), b(0x3fe212eaba813ec8), b(0x3fe24b8e228c50a3),
    b(0x3fe2856d111132bd), b(0x3fe2c098b61f4f24), b(0x3fe2fd23e345da5e), b(0x3fe33b23450e6318),
    b(0x3fe37aada708ddd9), b(0x3fe3bbdc44e1d114), b(0x3fe3fecb2bb18b80), b(0x3fe44399afa8e125),
    b(0x3fe48a6afb8ee069), b(0x3fe4d366c151f8af), b(0x3fe51eba1578899a), b(0x3fe56c9882da8773),
    b(0x3fe5bd3d694cac75), b(0x3fe610edc1a7af66), b(0x3fe667fa6d4f5c06), b(0x3fe6c2c3498418c6),
    b(0x3fe721bb5ba94b63), b(0x3fe7856e9b09d47e), b(0x3fe7ee8a2d243126), b(0x3fe85de87806c5b8),
    b(0x3fe8d4a376d3d22f), b(0x3fe95431c455aa39), b(0x3fe9de9715556d9b), b(0x3fea76baa562fae7),
    b(0x3feb210f0ee67f2a), b(0x3febe5007beb7b27), b(0x3fecd0a65081fff1), b(0x3fee0545e5881137),
    b(0x3ff0000000000000),
];

#[cfg(test)]
mod tests {
    use super::*;

    const SIGMA: f64 = 0.3071963263271184 * 100.0;

    /// Produced by Boost 1.89 (`boost::random::mt19937` seeded with 1, a new
    /// `normal_distribution<double>(0.0, SIGMA)` per draw), g++ 13 -O2 -ffp-contract=off, glibc.
    /// None of these 16 reaches the density test or the tail, so no `exp`/`log` is involved and
    /// they hold on any libm. The 3rd draw takes the wedge (one extra engine word).
    const SEED1_BITS: [u64; 16] = [
        0x40518c4ce67c7ed9, 0xc0348a3c1bbd1e0e, 0x3fcac3e2cecdc9c8, 0x40201a940ed8a8f1,
        0xc038c91b40ffdea6, 0x403714b1b429fbeb, 0x40472237d8beb936, 0xbfff0f17004e6390,
        0x4038228ed386d403, 0xc03c5baa7e87d81f, 0x40399a8c90bdb7a4, 0xc03230d13803fe78,
        0xbfd1cadfe4430f1f, 0xc02219f791a2d9e8, 0xc0301359e19e9e5c, 0xc034ab10ff5885e3,
    ];

    #[test]
    fn first_draws_match_boost_bit_for_bit() {
        let mut eng = Mt19937::new(1);
        for (n, &want) in SEED1_BITS.iter().enumerate() {
            let got = normal(&mut eng, 0.0, SIGMA);
            assert_eq!(got.to_bits(), want, "draw {n}: {got}");
        }
    }

    /// The draw COUNT is the value: after 1000 draws (same run as above, Boost 1.89) the engine's
    /// next word is 350572321. A wrong number of words per draw anywhere shifts it.
    #[test]
    fn engine_position_after_1000_draws_matches_boost() {
        let mut eng = Mt19937::new(1);
        for _ in 0..1000 {
            normal(&mut eng, 0.0, SIGMA);
        }
        assert_eq!(eng.next_u32(), 350_572_321);
    }

    /// The reference 10000th output of mt19937 seeded with the default 5489.
    #[test]
    fn mt19937_reference_output() {
        let mut eng = Mt19937::new(5489);
        let mut last = 0;
        for _ in 0..10000 {
            last = eng.next_u32();
        }
        assert_eq!(last, 4_123_659_995);
    }

    /// C `std::round` rounds halves away from zero; so does `f64::round`.
    #[test]
    fn round_is_half_away_from_zero() {
        assert_eq!(0.5f64.round(), 1.0);
        assert_eq!((-0.5f64).round(), -1.0);
        assert_eq!(2.5f64.round(), 3.0);
        assert_eq!((-2.5f64).round(), -3.0);
        assert_eq!(0.49999999999999994f64.round(), 0.0);
    }
}
