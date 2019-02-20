// Copyright 2019 PingCAP, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// See the License for the specific language governing permissions and
// limitations under the License.


use std::{f64, i64};

#[inline]
pub fn i64_abs(x: i64) -> i64 {
    i64::abs(x)
}
#[inline]
pub fn f64_abs(x: f64) -> f64 {
    f64::abs(x)
}

#[inline]
pub fn i64_pow(x: i64, y: u32) -> i64 {
    x.pow(y)
}

#[inline]
pub fn f64_pow_f64(x: f64, y: f64) -> f64 {
    x.powf(y)
}



#[cfg(test)]
mod test{
    use super::*;

    #[test]
    fn test_pow() {
        assert_eq!(i64_pow(2,4), 16)
    }

    #[test]
    fn test_f64_pow_f64() {
        assert_eq!(f64_pow_f64(2.0,4.0), 16.0)
    }
}