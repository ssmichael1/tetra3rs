//! Breadth-first combination iterator.
//!
//! Yields k-combinations from a list of items ordered by the sum of selected
//! indices. Since the input items are sorted brightest-first (lowest index =
//! brightest), this naturally prioritizes combinations involving the brightest
//! stars, matching tetra3's `breadth_first_combinations`.
//!
//! Implementation: a min-heap keyed by `(index sum, combination)`. Each
//! combination is pushed exactly once, by a unique parent (see
//! [`Iterator::next`]), so the heap holds only the search frontier and no
//! "already seen" set is needed. Uses fixed-size `[u32; K]` arrays instead of
//! `Vec<u32>` to avoid heap allocations in the hot inner loop.

use crate::prelude::*;
use alloc::collections::BinaryHeap;
use core::cmp::Reverse;

/// Iterator that yields k-combinations from `items` in order of increasing
/// sum of positional indices (i.e. brightest-first when items are brightness-sorted).
pub struct BreadthFirstCombinations<const K: usize> {
    items: Vec<usize>,
    heap: BinaryHeap<Reverse<(usize, [u32; K])>>,
}

impl<const K: usize> BreadthFirstCombinations<K> {
    /// Create a new iterator yielding `K`-combinations from `items`.
    ///
    /// `items` should be sorted in priority order (e.g. brightest star indices first).
    /// The iterator yields `[usize; K]`, each containing elements from `items`.
    pub fn new(items: &[usize]) -> Self {
        let mut heap = BinaryHeap::new();
        if items.len() >= K && K > 0 {
            // Seed with the first K indices: [0, 1, ..., K-1]
            let initial: [u32; K] = core::array::from_fn(|i| i as u32);
            heap.push(Reverse((K * (K - 1) / 2, initial)));
        }
        Self {
            items: items.to_vec(),
            heap,
        }
    }
}

impl<const K: usize> Iterator for BreadthFirstCombinations<K> {
    type Item = [usize; K];

    fn next(&mut self) -> Option<[usize; K]> {
        let Reverse((sum, combo)) = self.heap.pop()?;

        // Successors: increment one position by 1, keeping the combination
        // strictly increasing. Every combination other than the seed has
        // several such predecessors; to push each exactly once, its *parent*
        // is defined as the predecessor that decrements its lowest
        // decrementable position. Position `i` of `combo` is therefore
        // incremented only while every lower position sits at its minimum
        // (`combo[j] == j`, which nothing can decrement) — so the scan stops
        // at the first position that is not. A parent's sum is one less than
        // its child's, so it is always popped (and the child pushed) before
        // the child's turn: the yield order is exactly ascending
        // `(sum, combination)`.
        for i in 0..K {
            let next_val = combo[i] + 1;
            let upper = if i + 1 < K {
                combo[i + 1]
            } else {
                self.items.len() as u32
            };
            if next_val < upper {
                let mut child = combo;
                child[i] = next_val;
                self.heap.push(Reverse((sum + 1, child)));
            }
            if combo[i] != i as u32 {
                break;
            }
        }

        // Map positional indices to actual items.
        Some(core::array::from_fn(|j| self.items[combo[j] as usize]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfs_combinations_complete() {
        // All C(5,3) = 10 combinations should be yielded
        let items: Vec<usize> = vec![10, 20, 30, 40, 50];
        let combos: Vec<[usize; 3]> = BreadthFirstCombinations::<3>::new(&items).collect();
        assert_eq!(combos.len(), 10);
        // First combo should be the 3 "brightest" (lowest index)
        assert_eq!(combos[0], [10, 20, 30]);
    }

    #[test]
    fn test_bfs_combinations_order() {
        // Combinations should come in order of increasing index sum
        let items: Vec<usize> = (0..6).collect();
        let combos: Vec<[usize; 4]> = BreadthFirstCombinations::<4>::new(&items).collect();
        assert_eq!(combos.len(), 15); // C(6,4) = 15

        // Verify sum ordering
        let sums: Vec<usize> = combos.iter().map(|c| c.iter().sum()).collect();
        for w in sums.windows(2) {
            assert!(w[0] <= w[1], "sums not in order: {} > {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_bfs_combinations_early_stop() {
        // Should be able to take just a few without enumerating all
        let items: Vec<usize> = (0..50).collect();
        let first_10: Vec<[usize; 4]> = BreadthFirstCombinations::<4>::new(&items)
            .take(10)
            .collect();
        assert_eq!(first_10.len(), 10);
        assert_eq!(first_10[0], [0, 1, 2, 3]);
    }

    #[test]
    fn test_bfs_too_few_items() {
        let items: Vec<usize> = vec![1, 2];
        let combos: Vec<[usize; 4]> = BreadthFirstCombinations::<4>::new(&items).collect();
        assert!(combos.is_empty());
    }

    #[test]
    fn test_bfs_yields_every_combination_once_in_sum_order() {
        // Reference: all combinations, sorted by (index sum, combination) —
        // the order a heap with a dedup set produces.
        fn reference<const K: usize>(n: usize) -> Vec<[usize; K]> {
            let mut all: Vec<[usize; K]> = Vec::new();
            let mut c: [usize; K] = core::array::from_fn(|i| i);
            if n < K {
                return all;
            }
            loop {
                all.push(c);
                // Advance to the next combination in lexicographic order.
                let Some(i) = (0..K).rev().find(|&i| c[i] < n - K + i) else {
                    break;
                };
                c[i] += 1;
                for j in i + 1..K {
                    c[j] = c[j - 1] + 1;
                }
            }
            all.sort_by_key(|c| (c.iter().sum::<usize>(), *c));
            all
        }
        for n in 0..12 {
            let items: Vec<usize> = (0..n).collect();
            let got4: Vec<[usize; 4]> = BreadthFirstCombinations::<4>::new(&items).collect();
            assert_eq!(got4, reference::<4>(n), "K=4, n={n}");
            let got2: Vec<[usize; 2]> = BreadthFirstCombinations::<2>::new(&items).collect();
            assert_eq!(got2, reference::<2>(n), "K=2, n={n}");
        }
    }
}
