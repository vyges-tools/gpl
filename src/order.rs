// SPDX-License-Identifier: Apache-2.0
//! The reference's sorts, where their order is a value.
//!
//! `std::sort` / `std::ranges::sort` are not stable. The reference links libc++ (22.1.8): below 24
//! elements its sort keeps equal elements in input order (the 3-to-5 networks swap only strictly
//! ordered pairs, insertion sort shifts only past greater ones); above that the partitioning
//! reorders them in a way that depends on the whole input. [`libcxx_sort_by`] is the sort itself, for the call sites
//! whose ties are values. The same transcription as the router's and the resizer's `order` modules.

/// The heap sort `std::sort` falls back to past its depth limit (`__partial_sort`): not
/// transcribed — refused where it would run.
#[derive(Debug, Clone, PartialEq)]
pub struct HeapFallback {
    pub len: usize,
}

/// libc++'s `std::sort` / `std::ranges::sort` with a comparator (`__sort_dispatch` →
/// `__introsort`, the non-branchless path a non-arithmetic element takes), element for element:
/// the same permutation of equal elements as the reference. `less` is the strict weak order.
pub fn libcxx_sort_by<T: Clone>(v: &mut [T], mut less: impl FnMut(&T, &T) -> bool) -> Result<(), HeapFallback> {
    if v.is_empty() {
        return Ok(());
    }
    // `2 * __bit_log2(len)`.
    let depth = 2 * (usize::BITS - 1 - v.len().leading_zeros()) as usize;
    let mut s = Sorter { v, less: &mut less };
    s.introsort(0, s.v.len(), depth, true)
}

struct Sorter<'a, T, F> {
    v: &'a mut [T],
    less: &'a mut F,
}

impl<T: Clone, F: FnMut(&T, &T) -> bool> Sorter<'_, T, F> {
    fn lt(&mut self, a: usize, b: usize) -> bool {
        (self.less)(&self.v[a], &self.v[b])
    }

    /// `__sort3`: the branching network.
    fn sort3(&mut self, x: usize, y: usize, z: usize) -> bool {
        if !self.lt(y, x) {
            if !self.lt(z, y) {
                return false;
            }
            self.v.swap(y, z);
            if self.lt(y, x) {
                self.v.swap(x, y);
            }
            return true;
        }
        if self.lt(z, y) {
            self.v.swap(x, z);
            return true;
        }
        self.v.swap(x, y);
        if self.lt(z, y) {
            self.v.swap(y, z);
        }
        true
    }

    fn sort4(&mut self, x1: usize, x2: usize, x3: usize, x4: usize) {
        self.sort3(x1, x2, x3);
        if self.lt(x4, x3) {
            self.v.swap(x3, x4);
            if self.lt(x3, x2) {
                self.v.swap(x2, x3);
                if self.lt(x2, x1) {
                    self.v.swap(x1, x2);
                }
            }
        }
    }

    fn sort5(&mut self, x1: usize, x2: usize, x3: usize, x4: usize, x5: usize) {
        self.sort4(x1, x2, x3, x4);
        if self.lt(x5, x4) {
            self.v.swap(x4, x5);
            if self.lt(x4, x3) {
                self.v.swap(x3, x4);
                if self.lt(x3, x2) {
                    self.v.swap(x2, x3);
                    if self.lt(x2, x1) {
                        self.v.swap(x1, x2);
                    }
                }
            }
        }
    }

    /// `__insertion_sort` (guarded) and `__insertion_sort_unguarded` (the element before
    /// `first` bounds the scan).
    fn insertion_sort(&mut self, first: usize, last: usize, guarded: bool) {
        if first == last {
            return;
        }
        for i in first + 1..last {
            if self.lt(i, i - 1) {
                let t = self.v[i].clone();
                let (mut j, mut k) = (i, i - 1);
                loop {
                    self.v[j] = self.v[k].clone();
                    j = k;
                    if guarded && j == first {
                        break;
                    }
                    k -= 1;
                    if !(self.less)(&t, &self.v[k]) {
                        break;
                    }
                }
                self.v[j] = t;
            }
        }
    }

    /// `__insertion_sort_incomplete`: sorts small ranges outright; else insertion-sorts until
    /// the eighth move, and says whether it finished.
    fn insertion_sort_incomplete(&mut self, first: usize, last: usize) -> bool {
        match last - first {
            0 | 1 => return true,
            2 => {
                if self.lt(last - 1, first) {
                    self.v.swap(first, last - 1);
                }
                return true;
            }
            3 => {
                self.sort3(first, first + 1, last - 1);
                return true;
            }
            4 => {
                self.sort4(first, first + 1, first + 2, last - 1);
                return true;
            }
            5 => {
                self.sort5(first, first + 1, first + 2, first + 3, last - 1);
                return true;
            }
            _ => {}
        }
        let mut j = first + 2;
        self.sort3(first, first + 1, j);
        let mut count = 0;
        for i in j + 1..last {
            if self.lt(i, j) {
                let t = self.v[i].clone();
                let mut k = j;
                j = i;
                loop {
                    self.v[j] = self.v[k].clone();
                    j = k;
                    if j == first {
                        break;
                    }
                    k -= 1;
                    if !(self.less)(&t, &self.v[k]) {
                        break;
                    }
                }
                self.v[j] = t;
                count += 1;
                if count == 8 {
                    return i + 1 == last;
                }
            }
            j = i;
        }
        true
    }

    /// `__partition_with_equals_on_right`: the pivot at `first`; returns its place and whether
    /// the range was already partitioned.
    fn partition_right(&mut self, mut first: usize, mut last: usize) -> (usize, bool) {
        let begin = first;
        let pivot = self.v[first].clone();
        loop {
            first += 1;
            if !(self.less)(&self.v[first], &pivot) {
                break;
            }
        }
        if begin == first - 1 {
            while first < last {
                last -= 1;
                if (self.less)(&self.v[last], &pivot) {
                    break;
                }
            }
        } else {
            loop {
                last -= 1;
                if (self.less)(&self.v[last], &pivot) {
                    break;
                }
            }
        }
        let already = first >= last;
        while first < last {
            self.v.swap(first, last);
            loop {
                first += 1;
                if !(self.less)(&self.v[first], &pivot) {
                    break;
                }
            }
            loop {
                last -= 1;
                if (self.less)(&self.v[last], &pivot) {
                    break;
                }
            }
        }
        let pivot_pos = first - 1;
        if begin != pivot_pos {
            self.v[begin] = self.v[pivot_pos].clone();
        }
        self.v[pivot_pos] = pivot;
        (pivot_pos, already)
    }

    /// `__partition_with_equals_on_left`: elements equal to the pivot go left of it; returns the
    /// start of what remains to sort.
    fn partition_left(&mut self, mut first: usize, mut last: usize) -> usize {
        let begin = first;
        let pivot = self.v[first].clone();
        if (self.less)(&pivot, &self.v[last - 1]) {
            loop {
                first += 1;
                if (self.less)(&pivot, &self.v[first]) {
                    break;
                }
            }
        } else {
            loop {
                first += 1;
                if !(first < last && !(self.less)(&pivot, &self.v[first])) {
                    break;
                }
            }
        }
        if first < last {
            loop {
                last -= 1;
                if !(self.less)(&pivot, &self.v[last]) {
                    break;
                }
            }
        }
        while first < last {
            self.v.swap(first, last);
            loop {
                first += 1;
                if (self.less)(&pivot, &self.v[first]) {
                    break;
                }
            }
            loop {
                last -= 1;
                if !(self.less)(&pivot, &self.v[last]) {
                    break;
                }
            }
        }
        let pivot_pos = first - 1;
        if begin != pivot_pos {
            self.v[begin] = self.v[pivot_pos].clone();
        }
        self.v[pivot_pos] = pivot;
        first
    }

    /// `__introsort`.
    fn introsort(&mut self, mut first: usize, mut last: usize, mut depth: usize, mut leftmost: bool) -> Result<(), HeapFallback> {
        const LIMIT: usize = 24;
        const NINTHER_THRESHOLD: usize = 128;
        loop {
            let len = last - first;
            match len {
                0 | 1 => return Ok(()),
                2 => {
                    if self.lt(last - 1, first) {
                        self.v.swap(first, last - 1);
                    }
                    return Ok(());
                }
                3 => {
                    self.sort3(first, first + 1, last - 1);
                    return Ok(());
                }
                4 => {
                    self.sort4(first, first + 1, first + 2, last - 1);
                    return Ok(());
                }
                5 => {
                    self.sort5(first, first + 1, first + 2, first + 3, last - 1);
                    return Ok(());
                }
                _ => {}
            }
            if len < LIMIT {
                self.insertion_sort(first, last, leftmost);
                return Ok(());
            }
            if depth == 0 {
                return Err(HeapFallback { len });
            }
            depth -= 1;
            let half = len / 2;
            if len > NINTHER_THRESHOLD {
                self.sort3(first, first + half, last - 1);
                self.sort3(first + 1, first + (half - 1), last - 2);
                self.sort3(first + 2, first + (half + 1), last - 3);
                self.sort3(first + (half - 1), first + half, first + (half + 1));
                self.v.swap(first, first + half);
            } else {
                self.sort3(first + half, first, last - 1);
            }
            if !leftmost && !self.lt(first - 1, first) {
                first = self.partition_left(first, last);
                continue;
            }
            let (i, already) = self.partition_right(first, last);
            if already {
                let fs = self.insertion_sort_incomplete(first, i);
                if self.insertion_sort_incomplete(i + 1, last) {
                    if fs {
                        return Ok(());
                    }
                    last = i;
                    continue;
                } else if fs {
                    first = i + 1;
                    continue;
                }
            }
            self.introsort(first, i, depth, leftmost)?;
            leftmost = false;
            first = i + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Rule (libc++ __introsort): below 24 elements equal keys keep input order; above, the
    // partition moves them. The 30-element
    // permutation was printed by the reference's own libc++ (instruments/rsz/sort-order).
    const LIBCXX_30: [u32; 30] = [8, 28, 24, 20, 4, 16, 0, 12, 15, 11, 7, 19, 3, 23, 27, 10, 14, 6, 18, 22, 2, 26, 9, 13, 5, 17, 21, 25, 1, 29];

    #[test]
    fn libcxx_sort_reorders_ties_as_the_reference_does() {
        let mut v: Vec<(u32, u32)> = (0..30).map(|i| (i * 7 % 4, i)).collect();
        libcxx_sort_by(&mut v, |a, b| a.0 < b.0).unwrap();
        assert_eq!(v.iter().map(|x| x.1).collect::<Vec<_>>(), LIBCXX_30);
        let mut w: Vec<(u32, u32)> = (0..23).map(|i| (i * 7 % 4, i)).collect();
        libcxx_sort_by(&mut w, |a, b| a.0 < b.0).unwrap();
        assert!(w.windows(2).all(|p| p[0].0 < p[1].0 || (p[0].0 == p[1].0 && p[0].1 < p[1].1)));
    }

    // Rule (libc++ `__sort3`/`__sort4`/`__sort5`): the networks swap only a strictly ordered
    // pair, so a tie keeps its input order at 3 to 5 elements too.
    #[test]
    fn the_small_networks_keep_ties_in_input_order() {
        let mut v = vec![(5, 'a'), (1, 'b'), (5, 'c')];
        libcxx_sort_by(&mut v, |a, b| a.0 < b.0).unwrap();
        assert_eq!(v, vec![(1, 'b'), (5, 'a'), (5, 'c')]);
        let mut w = vec![(2, 'a'), (2, 'b'), (1, 'c'), (2, 'd'), (1, 'e')];
        libcxx_sort_by(&mut w, |a, b| a.0 < b.0).unwrap();
        assert_eq!(w, vec![(1, 'c'), (1, 'e'), (2, 'a'), (2, 'b'), (2, 'd')]);
    }
}
