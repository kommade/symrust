use arpfloat::{FP128, Float};
use crate::utils::float_gcd::FloatGCD;

static RANDOM_INTEGERS: [i64; 20] = [
    -2_472_871_614_674_160_760,
    5_521_018_729_607_591_024,
    4_300_965_516_080_931_108,
    -4_578_745_523_759_869_611,
    7_748_889_266_673_084_649,
    6_172_766_956_285_176_222,
    -4_494_374_757_580_458_303,
    3_195_926_724_108_846_752,
    964_167_498_586_960_652,
    191_871_291_051_866_735,
    -2_926_512_688_656_657_048,
    3_162_464_865_393_524_369,
    4_884_015_281_603_812_250,
    4_071_626_853_863_341_261,
    5_553_576_954_470_466_762,
    -8_738_625_366_432_481_120,
    4_416_725_642_104_433_997,
    3_530_444_650_116_216_642,
    -8_191_126_007_653_466_011,
    -4_030_489_133_450_473_717,
];

static RANDOM_FLOATS: [f64; 20] = [
    -0.0,
    0.719_417_472_586_502_6,
    0.156_555_220_490_132_6,
    -0.559_996_400_670_714_9,
    0.048_267_793_231_280_5,
    0.385_869_857_716_574_1,
    0.153_336_894_933_717_38,
    -0.389_824_865_926_469_5,
    0.890_319_084_602_016_5,
    0.356_898_764_354_880_66,
    0.610_315_494_366_701_7,
    -0.886_906_194_842_212_1,
    0.367_724_014_369_424_37,
    0.553_724_986_858_105,
    0.193_503_864_900_720_02,
    -0.895_594_113_885_801,
    0.108_317_789_493_537_43,
    0.198_477_694_849_974_68,
    -0.169_762_193_883_428_4,
    0.479_838_321_254_475_56,
];

#[test]
fn try_is_integer() {
    super::setup();
    for i in &RANDOM_INTEGERS {
        let float = Float::from_i64(FP128, *i);
        assert!(float.is_integer());
    }
    for f in &RANDOM_FLOATS {
        let float = Float::from_f64(*f);
        assert!(!float.is_integer());
    }
}

#[test]
fn try_gcd() {
    super::setup();
    let gcd_euclid = |a, b| {
        let mut a = a;
        let mut b = b;
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a
    };

    for i in RANDOM_INTEGERS.windows(2) {
        let a = Float::from_i64(FP128, i[0]);
        let b = Float::from_i64(FP128, i[1]);
        let (gcd, a_div, b_div) = a.gcd(&b);
        let real_gcd = gcd_euclid(i[0], i[1]);
        assert_eq!(gcd, real_gcd.unsigned_abs());
        assert_eq!(a_div.abs(), (i[0] / real_gcd).abs());
        assert_eq!(b_div.abs(), (i[1] / real_gcd).abs());
    }
}

#[test]
fn try_is_one() {
    super::setup();
    let one = Float::from_i64(FP128, 1);
    assert!(one.is_one());
    for i in &RANDOM_INTEGERS {
        let float = Float::from_i64(FP128, *i);
        assert!(!float.is_one());
    }
    for f in &RANDOM_FLOATS {
        let float = Float::from_f64(*f);
        assert!(!float.is_one());
    }
}