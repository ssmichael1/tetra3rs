//! Breadth-first combination iterator.
//!
//! Yields k-combinations from a list of items ordered by the sum of selected
//! indices. Since the input items are sorted brightest-first (lowest index =
//! brightest), this naturally prioritizes combinations involving the brightest
//! stars, matching tetra3's `breadth_first_combinations`.
//!
//! Implementation: min-heap keyed by index sum, with a HashSet for dedup.
//! Uses fixed-size `[u32; K]` arrays instead of `Vec<u32>` to avoid heap
//! allocations in the hot inner loop.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

/// Iterator that yields k-combinations from `items` in order of increasing
/// sum of positional indices (i.e. brightest-first when items are brightness-sorted).
pub struct BreadthFirstCombinations<const K: usize> {
    items: Vec<usize>,
    n: usize,
    heap: BinaryHeap<Reverse<(usize, [u32; K])>>,
    seen: HashSet<[u32; K]>,
}

impl<const K: usize> BreadthFirstCombinations<K> {
    /// Create a new iterator yielding `K`-combinations from `items`.
    ///
    /// `items` should be sorted in priority order (e.g. brightest star indices first).
    /// The iterator yields `Vec<usize>` of length `K`, each containing elements from `items`.
    pub fn new(items: &[usize]) -> Self {
        let n = items.len();
        let mut bfc = Self {
            items: items.to_vec(),
            n,
            heap: BinaryHeap::new(),
            seen: HashSet::new(),
        };
        if n >= K && K > 0 {
            // Seed with the first K indices: [0, 1, ..., K-1]
            let initial: [u32; K] = std::array::from_fn(|i| i as u32);
            let sum: usize = initial.iter().map(|&x| x as usize).sum();
            bfc.seen.insert(initial);
            bfc.heap.push(Reverse((sum, initial)));
        }
        bfc
    }
}

impl<const K: usize> Iterator for BreadthFirstCombinations<K> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Vec<usize>> {
        let Reverse((_, combo)) = self.heap.pop()?;

        // Generate successors: for each position, try incrementing by 1.
        // A combination [a, b, c, d] (strictly increasing) can produce:
        //   - increment d if d+1 < n
        //   - increment c if c+1 < d
        //   - increment b if b+1 < c
        //   - increment a if a+1 < b
        for i in 0..K {
            let next_val = combo[i] + 1;
            let upper = if i + 1 < K {
                combo[i + 1]
            } else {
                self.n as u32
            };
            if next_val < upper {
                let mut new_combo = combo;
                new_combo[i] = next_val;
                if self.seen.insert(new_combo) {
                    let sum: usize = new_combo.iter().map(|&x| x as usize).sum();
                    self.heap.push(Reverse((sum, new_combo)));
                }
            }
        }

        // Map positional indices to actual items.
        Some(combo.iter().map(|&i| self.items[i as usize]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfs_combinations_complete() {
        // All C(5,3) = 10 combinations should be yielded
        let items: Vec<usize> = vec![10, 20, 30, 40, 50];
        let combos: Vec<Vec<usize>> = BreadthFirstCombinations::<3>::new(&items).collect();
        assert_eq!(combos.len(), 10);
        // First combo should be the 3 "brightest" (lowest index)
        assert_eq!(combos[0], vec![10, 20, 30]);
    }

    #[test]
    fn test_bfs_combinations_order() {
        // Combinations should come in order of increasing index sum
        let items: Vec<usize> = (0..6).collect();
        let combos: Vec<Vec<usize>> = BreadthFirstCombinations::<4>::new(&items).collect();
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
        let first_10: Vec<Vec<usize>> =
            BreadthFirstCombinations::<4>::new(&items).take(10).collect();
        assert_eq!(first_10.len(), 10);
        assert_eq!(first_10[0], vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_bfs_too_few_items() {
        let items: Vec<usize> = vec![1, 2];
        let combos: Vec<Vec<usize>> = BreadthFirstCombinations::<4>::new(&items).collect();
        assert!(combos.is_empty());
    }
}
