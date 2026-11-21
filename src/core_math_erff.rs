// SPDX-License-Identifier: Apache-2.0 AND MIT
//! `cr_erff`: the correctly rounded single-precision error function of CORE-MATH
//! (<https://gitlab.inria.fr/core-math/core-math>, `src/binary32/erf/erff.c`, commit
//! `567e1efc`), which the reference placer evaluates its macro density with since the f11b4276 pin
//! (OpenROAD #12388, vendored there as `third-party/core-math` and compiled `-ffp-contract=off`).
//! Transcribed from that MIT-licensed C (Copyright (c) 2022-2025 Alexei Sibidanov; the notice is
//! in `THIRD_PARTY_LICENSES.md`). Double-precision arithmetic only, no fused multiply-add (Rust
//! never contracts), so the bits are the same on every host — and, the function being correctly
//! rounded, the same as any correctly rounded `erff`.
//!
//! The coefficients are the C file's hexadecimal literals, stored as their IEEE-754 bit patterns.

/// `C[i - 7]` for `7 <= i < 63`: a degree-7 polynomial approximating `erf(i/16 + 1/32 + x)` for
/// `-1/32 <= x <= 1/32`.
const C: [[u64; 8]; 56] = [
    [0x3fdf86faa9428f9c, 0x3fecfc41e36c7dfa, 0xbfdb2c7dc53508b9, 0xbfc5a9de93fa556e, 0x3fc731793dbb01b5, 0x3f9133e06426cf18, 0xbfaa12a6289cafd8, 0x3f6717d6f1d6f557],
    [0x3fe1855a5fd3dd50, 0x3feb3aafcc27502f, 0xbfdcee5ac8e92bb2, 0xbfbfa02983ca2d79, 0x3fc77cd746cb1922, 0xbf5fa6f277886487, 0xbfa8de75458db416, 0x3f800899c98551c9],
    [0x3fe32a54cb8db67a, 0x3fe96164fafd8de5, 0xbfde23a7ea0c9ad3, 0xbfb3f5ee15671cf4, 0x3fc70e468a3d72d9, 0xbf93da68037cfc99, 0xbfa69ed9ba1f9839, 0x3f88cab9244a4ff4],
    [0x3fe4b13713ad3513, 0x3fe7791b886e7405, 0xbfdecef423109bf5, 0xbfa15c3c5cec6847, 0x3fc5f688fc931ba6, 0xbfa1da63ed190037, 0xbfa38427ca63cca4, 0x3f8fa00e52525e17],
    [0x3fe61955607dd15d, 0x3fe58a445da7c74e, 0xbfdef6c246a0f66c, 0x3f7e83e0d9d61330, 0x3fc44cc65535bc9f, 0xbfa87d3c4860435d, 0xbf9f90b10501169b, 0x3f922295856d427a],
    [0x3fe762870f720c6f, 0x3fe39ccc1b136d5c, 0xbfdea4feea4e4744, 0x3fa715e5952ebfba, 0x3fc22cdbd83c75c4, 0xbfada50aa1d925b6, 0xbf9754dc0a29b4dd, 0x3f9350b6bef9392c],
    [0x3fe88d1cd474a2e0, 0x3fe1b7e98fe26219, 0xbfdde65a22ce1419, 0x3fb40686a3f16400, 0x3fbf6b0cbb216b2b, 0xbfb09c7c903edd57, 0xbf8da7529fde6410, 0x3f9362a7a0588eab],
    [0x3fe999d4192a5717, 0x3fdfc3ee5d1524b3, 0xbfdcc990045b55c8, 0x3fbb37338e68b37d, 0x3fba0d120c872ea7, 0xbfb19bb2b07ecff6, 0xbf7a110f5f593aaf, 0x3f9272c15a57720e],
    [0x3fea89c850b7d54d, 0x3fdc40b0729ed54a, 0xbfdb5eaaef0a2346, 0x3fc0847c7dacbae1, 0x3fb47de0ba6d18fb, 0xbfb1d9de77a4b648, 0x3f530ffbe56f0726, 0x3f90a9cb99feea01],
    [0x3feb5e62fce16096, 0x3fd8eed36b886d95, 0xbfd9b64a06e50705, 0x3fc2bb6e2c744df5, 0x3fadee3261ca61bc, 0xbfb16996004f7da5, 0x3f7fdff37bae983e, 0x3f8c750083e65f9a],
    [0x3fec194b1d49a184, 0x3fd5d4fd33729015, 0xbfd7e0f4f045addb, 0x3fc444bc66c31a1b, 0x3fa356dbf16ec8f1, 0xbfb0643de0906cd8, 0x3f8b281af7bd3a2c, 0x3f86b97eaa2c6abd],
    [0x3fecbc54b476248e, 0x3fd2f7cc3fe6f423, 0xbfd5ee8429e36de8, 0x3fc52a8395f96177, 0x3f9313761ba257dc, 0xbfadcf844d5fed8f, 0x3f91e1420f475fa9, 0x3f8091c7dc1e18b2],
    [0x3fed4970f9ce00d9, 0x3fd059f59af7a905, 0xbfd3eda354de36c3, 0x3fc57b85ad439779, 0x3f58e913b9778136, 0xbfaa2893bd3435f4, 0x3f94d3a90e37164a, 0x3f74ce7f6e19a902],
    [0x3fedc29fb60715b0, 0x3fcbf8e1b1ca2277, 0xbfd1eb7095e5d6d2, 0x3fc549ea6f7a64f4, 0xbf8b10f12f3877a3, 0xbfa61420c8f7156a, 0x3f9674f1f92a8812, 0x3f625543ffd74d52],
    [0x3fee29e22a89d767, 0x3fc7bd5c7df3fe99, 0xbfcfe674494077bf, 0x3fc4a9feacf86578, 0xbf9a008269076644, 0xbfa1cf0e8fb4f1cb, 0x3f96e0d2ef105fb3, 0xbf3367205fbd7876],
    [0x3fee812fc64db36a, 0x3fc3fda6bc016991, 0xbfcc1cb278627920, 0x3fc3b10512314f1e, 0xbfa1e6457bb1b9a9, 0xbf9b1f6474e2388c, 0x3f9640a5345f7ec7, 0xbf63dae5a997fdb0],
    [0x3feeca6ccd709544, 0x3fc0b3f52ce8c380, 0xbfc8885019f63c6d, 0x3fc274275fc91a05, 0xbfa57f73699a8372, 0xbf93076a305fc7ce, 0x3f94c6ae04843a41, 0xbf70be5fcf5ecc91],
    [0x3fef0762fde45ee7, 0x3fbbb1c972f23e4a, 0xbfc5341e3c01b58d, 0x3fc107929f6f0b60, 0xbfa7e1b34f976c02, 0xbf873b62589c234a, 0x3f92a97ee1876486, 0xbf7595f40a3150fe],
    [0x3fef39bc242e43e6, 0x3fb6c7e64e7281c5, 0xbfc2274b86835fd3, 0x3fbefb890e5c770d, 0xbfa92c7db16847e0, 0xbf745477db5e2dd4, 0x3f901fc6165fc866, 0xbf78845509030c2c],
    [0x3fef62fe80272419, 0x3fb297db960e4f5d, 0xbfbecb83b087c04f, 0x3fbbce18363ca3d1, 0xbfa985aaf776482c, 0x3f3cd953efdae886, 0x3f8ab9a0b89b54ff, 0xbf79b5e576ccc31c],
    [0x3fef848acb544e95, 0x3fae1d4cf1e24501, 0xbfb9e12e1fde5552, 0x3fb8a27806df3d1b, 0xbfa91674e5eb3319, 0x3f73bc75595b2db8, 0x3f851bc537ac61af, 0xbf796b23b19ea04d],
    [0x3fef9f9ba8d3c733, 0x3fa83298d7172108, 0xbfb58d101f905a75, 0x3fb58f1456f8639b, 0xbfa808d1850b8231, 0x3f80c1bd99c348a7, 0x3f7f61e9d7bc48ca, 0xbf77f07c13441774],
    [0x3fefb54641aebbc9, 0x3fa34ac36ad8dafa, 0xbfb1c8ec267f9405, 0x3fb2a52c5d841848, 0xbfa68541c02b3b6b, 0x3f85afe400196379, 0x3f7565b2d6eda3d6, 0xbf7596aaff29e739],
    [0x3fefc67bcf2d7b8f, 0x3f9e85c449e377ef, 0xbfad177f166c07c6, 0x3fafe23b7584b504, 0xbfa4b12109613313, 0x3f88d9905c0acf7d, 0x3f69265032a669da, 0xbf72ac4a6dbcbf3e],
    [0x3fefd40bd6d7a785, 0x3f97f5188610ddc7, 0xbfa7954423f7c998, 0x3faaf5baae33887f, 0xbfa2ad77c7cbc474, 0x3f8a7b8c47ec2a51, 0x3f546646ee094bcc, 0xbf6ef19d8db86730],
    [0x3fefdea6e062d0c9, 0x3f92a875b5ffab58, 0xbfa2f3178cd6dcd5, 0x3fa68d1c45b94182, 0xbfa09648ed3aeaef, 0x3f8ad8b150d38164, 0xbf2e9a6023d9429f, 0xbf68722d19ee2e8e],
    [0x3fefe6e1742f7cf5, 0x3f8cd5ec93c1243a, 0xbf9e2ff3aaacb386, 0x3fa2aa4e5823cc89, 0xbf9d049842dbe399, 0x3f8a34edb21ab302, 0xbf5676e5996c7f9b, 0xbf623b01a35140bf],
    [0x3fefed37386190fb, 0x3f861beae53b72c2, 0xbf97d6193f22c3c1, 0x3f9e947279e3bb7d, 0xbf9906031b97ca97, 0x3f88d14d62561755, 0xbf61f245e7178882, 0xbf59257d4eb47685],
    [0x3feff20e0a7ba8c2, 0x3f80d1d69569b839, 0xbf92a8ca0dc02752, 0x3f98cc071b709751, 0xbf954a149f1b070c, 0x3f86e9137b13412c, 0xbf66577ed3d8e83b, 0xbf4e9c1a5178a289],
    [0x3feff5b8fb26f5f6, 0x3f79646f35a7663c, 0xbf8cf68ed9311b0b, 0x3f93e8735b5a694f, 0xbf91e1612d026fdf, 0x3f84afd8e6ca636d, 0xbf68c375170ccb22, 0xbf3c799443c4fd3b],
    [0x3feff87b1913e853, 0x3f730499b5039596, 0xbf864964201ec8ba, 0x3f8fa73d7eafba98, 0xbf8daa3022141fbb, 0x3f82509444c063b7, 0xbf699482a2f8a0a1, 0xbf0403d1f76c9454],
    [0x3feffa89fe5b3625, 0x3f6c4412bf4b8f35, 0xbf8100f347126cf0, 0x3f88ebda07671d40, 0xbf8850c6a31c98c1, 0x3f7fdac860c67d21, 0xbf6927d03d2ba12c, 0x3f30ff620b4190fe],
    [0x3feffc10194fcb64, 0x3f64d78bba8ca621, 0xbf79ba107a443e02, 0x3f836f273fbc04cc, 0xbf83b38716ac7e6f, 0x3f7b3fe0181914ac, 0xbf67d3fe7de98c5c, 0x3f3ea31f8e5317f7],
    [0x3feffd2eae369a07, 0x3f5e7f232d9e266c, 0xbf734c7442dd48d9, 0x3f7e066bed070a0b, 0xbf7f914f3c42fc0d, 0x3f76f4664ed2260f, 0xbf65e59910761d24, 0x3f439cbb6e84c126],
    [0x3feffdff92db56e5, 0x3f56235fbd7a4373, 0xbf6cb5e029b9e56a, 0x3f76fa4c7ef274da, 0xbf7903a089a835f3, 0x3f730f12e0ca1901, 0xbf639d21b6957f99, 0x3f45d3f8495a703c],
    [0x3feffe96a78a04a9, 0x3f4fe41cd9bb4f2c, 0xbf652d7b28966c0c, 0x3f716c192d86a1a7, 0xbf739bfce951100c, 0x3f6f376a7869f9e3, 0xbf612e6cef999c4f, 0x3f466acd4d667b50],
    [0x3fefff0312b010b5, 0x3f46caa0d3583018, 0xbf5efb729f4cf75b, 0x3f6a2da7cebe12ac, 0xbf6e6c27a24bc759, 0x3f693b1f4d8ea650, 0xbf5d82050aa94a08, 0x3f45cd7dc75d6cba],
    [0x3fefff50456dab8c, 0x3f40295ef6591865, 0xbf5679880e95a4da, 0x3f637d38e3a5c8eb, 0xbf675b3708aebb8f, 0x3f64231c4b4b0296, 0xbf58e26476489318, 0x3f445c3b570dd924],
    [0x3fefff86cfd3e657, 0x3f36be02102b353d, 0xbf502b157780d6ae, 0x3f5cc1d886861133, 0xbf61bff6f12ec9ab, 0x3f5fc0f77bd9c736, 0xbf54a3320bd0959d, 0x3f4267f8b4f95d20],
    [0x3fefffad0b901755, 0x3f2fc0d55470cf5e, 0xbf47121aff5e820e, 0x3f5506d6992f7de5, 0xbf5ab595d3ecd0d6, 0x3f58bdd79daaf754, 0xbf50d9b090f997c1, 0x3f4031ab9fd1c7da],
    [0x3fefffc7a37857d2, 0x3f25feada379d8a5, 0xbf405304df58f3aa, 0x3f4e79c081b8600f, 0xbf53e5dbe33232e0, 0x3f530eb208200729, 0xbf4b1d493b147945, 0x3f3bd587bbc071be],
    [0x3fefffd9fdeabcce, 0x3f1e3bcf436a1a49, 0xbf36e953111ef0a1, 0x3f45e3edf6768654, 0xbf4d5be67c0547a4, 0x3f4d07d9ffa1d435, 0xbf458328f5f358ca, 0x3f376d42d95c42c4],
    [0x3fefffe68f4fa777, 0x3f149e17724f4cdd, 0xbf2fe48c44e229c1, 0x3f3f2bd95d76f188, 0xbf457388cb12d011, 0x3f45decc25c5c079, 0xbf40d7499d1b0d2d, 0x3f3359332c94ecdc],
    [0x3fefffef1960d85d, 0x3f0be6abbb10a4cd, 0xbf26040381a8c313, 0x3f35fff1dde9ee9d, 0xbf3f0c933efa9971, 0x3f404cbf4a5cd760, 0xbf3a07f150af6dad, 0x3f2f68dd183426ba],
    [0x3feffff4db27f146, 0x3f02bb5cc22e5cd8, 0xbf1e25894899f526, 0x3f2ec8a8e5a72757, 0xbf364256ae0a3cf9, 0x3f380a836c18c46c, 0xbf33dea401af6775, 0x3f2915ddff3fe0d1],
    [0x3feffff8b500e77c, 0x3ef8f4ccca7fc769, 0xbf1478cffe305946, 0x3f2559f04adde504, 0xbf2f9e1577d6961d, 0x3f318bda53c14716, 0xbf2df8634c35541c, 0x3f23bb5c6b616337],
    [0x3feffffb43555b5f, 0x3ef07ebd2a2d26c8, 0xbf0b93e442a37f2b, 0x3f1d5cf15159ce28, 0xbf263f5e1469c006, 0x3f295a03acebac18, 0xbf2656e5e2a1f8e2, 0x3f1e98c437189bde],
    [0x3feffffcf23ff5fc, 0x3ee5a2adfa0b492c, 0xbf026c88270759f0, 0x3f140473572b99a8, 0xbf1f057cbde578a5, 0x3f222178d1c3c948, 0xbf20765b61a0d859, 0x3f1765b3ea03ddbe],
    [0x3feffffe0bd3e852, 0x3edc282cd3957a72, 0xbef86ad6dfa44faa, 0x3f0b0f313f03a029, 0xbf156e44abecd255, 0x3f19ad1ecfe34a89, 0xbf17fe4033478618, 0x3f11a8184e049fbf],
    [0x3feffffec2641a9e, 0x3ed22df29821407e, 0xbef00c902a6cfd98, 0x3f022234eb88671f, 0xbf0d57a181c9e6e1, 0x3f1200c283b54a90, 0xbf114b4c3295a7d0, 0x3f0a4f966f713bde],
    [0x3fefffff37d63a36, 0x3ec74adc8f405eec, 0xbee4ed4228e44858, 0x3ef81918baea92ba, 0xbf03e81b17a0009c, 0x3f09004a36116436, 0xbf08aa1ba400e076, 0x3f035cd4e2340a9e],
    [0x3fefffff82cdcf1b, 0x3ebd9c73698fa87d, 0xbedb11017ec67115, 0x3eefc0dfadf653f8, 0xbefac4e03cd2dfc2, 0x3f0131806b5abbc5, 0xbf01672ef66fcaaf, 0x3efc2882c7debed7],
    [0x3fefffffb248c39d, 0x3eb2acee2f5ec66a, 0xbed15cc570408a36, 0x3ee4be757bbb75a3, 0xbef1d6aa5f8d2940, 0x3ef76c5937d5105e, 0xbef84dffc3ca9302, 0x3ef43c8315f2c30a],
    [0x3fefffffd01f36af, 0x3ea75fa8dbc840ba, 0xbec6186da0133f5a, 0x3edae023231e1af5, 0xbee790812f7ca394, 0x3eef9c25656d0ef2, 0xbef0cc66682e304c, 0x3eecc170a75d6f9c],
    [0x3fefffffe2ba0ea5, 0x3e9d06ad6ecde88e, 0xbebbe46aa8edc9a1, 0x3ed143860c7840b8, 0xbededaba78fb1260, 0x3ee52138a96ecee2, 0xbee6fca538c4e2ee, 0x3ee434040640bcef],
    [0x3fefffffee3cc32c, 0x3e91e1e857adb8dd, 0xbeb1769ce5f2a6e8, 0x3ec5fe5d479b0543, 0xbed405d865c94c2a, 0x3edbfc94feb96afc, 0xbedf245d5f3e8358, 0x3edc142456acf443],
];

/// The polynomial for `|x| < 0x1.cp-2`, in `x^2`.
const SMALL: [u64; 8] = [0x3ff20dd750429b6d, 0xbfd812746b0375fb, 0x3fbce2f219fd6f45, 0xbf9b82ce2cbf0838, 0x3f7565bb655adb85, 0xbf4c025bfc879c94, 0x3f1f81718f61309c, 0xbeecc67bd88f5867];

fn d(bits: u64) -> f64 {
    f64::from_bits(bits)
}

/// `cr_erff(x)`.
pub fn cr_erff(x: f32) -> f32 {
    let ax = x.abs();
    let ux = ax.to_bits();
    let s = f64::from(x);
    let mut z = f64::from(ax);
    // for |x| > 0x1.f5a888p+1, erf(x) rounds to +-1 to nearest
    if ux > 0x407a_d444 {
        let os = 1.0f32.copysign(x);
        if ux > (0xff << 23) {
            return x + x; // NaN
        }
        if ux == (0xff << 23) {
            return os; // +-inf
        }
        return os - f32::from_bits(0x3300_0000) * os; // 0x1p-25f
    }
    let v = (16.0 * z).floor();
    let i = (16.0f32 * ax) as u32;
    if ux < 0x3ee0_0000 {
        // |x| < 0x1.cp-2
        let c = SMALL.map(d);
        let z2 = s * s;
        let z4 = z2 * z2;
        let z8 = z4 * z4;
        let mut c0 = c[0] + z2 * c[1];
        let c2 = c[2] + z2 * c[3];
        let mut c4 = c[4] + z2 * c[5];
        let c6 = c[6] + z2 * c[7];
        c0 += z4 * c2;
        c4 += z4 * c6;
        c0 += z8 * c4;
        return (s * c0) as f32;
    }
    z = (z - 0.03125) - 0.0625 * v;
    let c = C[i as usize - 7].map(d);
    let z2 = z * z;
    let z4 = z2 * z2;
    let mut c0 = c[0] + z * c[1];
    let c2 = c[2] + z * c[3];
    let mut c4 = c[4] + z * c[5];
    let c6 = c[6] + z * c[7];
    c0 += z2 * c2;
    c4 += z2 * c6;
    c0 += z4 * c4;
    c0.copysign(s) as f32
}

#[cfg(test)]
mod tests {
    use super::cr_erff;

    /// Rule (OpenROAD #12388, `third-party/core-math` `cr_erff` at the f11b4276 pin): the placer's
    /// error function is the correctly rounded one. Expected bits from the vendored C compiled
    /// `-ffp-contract=off`; this transcription matched it on all 2^32 inputs (2026-11-21). Covers
    /// both polynomial ranges, the `|x| < 0x1.cp-2` boundary from both sides, the saturation edge
    /// and the infinities.
    #[test]
    fn matches_the_reference_bits() {
        let cases: [(u32, u32); 20] = [
            (0x00000000, 0x00000000), (0x80000000, 0x80000000), (0x0da24260, 0x0db71709), (0x3dcccccd, 0x3de652f5),
            (0x3ee00000, 0x3eed840e), (0x3f000000, 0x3f053f7b), (0x3f800000, 0x3f57bb3d), (0xbf800000, 0xbf57bb3d),
            (0x3fc00000, 0x3f7752ab), (0x40000000, 0x3f7ecd71), (0x40400000, 0x3f7ffe8d), (0x4079999a, 0x3f7fffff),
            (0x407ae148, 0x3f800000), (0x40800000, 0x3f800000), (0xc0f00000, 0xbf800000), (0x7f800000, 0x3f800000),
            (0xff800000, 0xbf800000), (0x3dfcd680, 0x3e0ded55), (0x402df851, 0x3f7ff813), (0x3edfffff, 0x3eed840e),
        ];
        for (x, want) in cases {
            assert_eq!(cr_erff(f32::from_bits(x)).to_bits(), want, "cr_erff({x:08x})");
        }
        assert!(cr_erff(f32::NAN).is_nan());
    }
}
