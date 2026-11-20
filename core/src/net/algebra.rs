//! Sum-of-products normalisation of search attribute expressions.
//!
//! Python flattens an expression into a list of nodes. Each node is a kind tag
//! and the indices of its children. Children always come before their parent,
//! so the last node is the root.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A leaf holds one attribute. Its only child is that attribute's index.
const LEAF: u8 = 0;
/// A conjunction. Its children must all hold at once.
const AND: u8 = 1;
/// A disjunction. Any one of its children may hold.
const OR: u8 = 2;

const FLATTEN_ERROR: &str =
    "This query could not be flattened; combine the alternatives with | at the top level instead.";

/// Reduces an expression tree to a sum of products.
///
/// Returns one vector of leaf indices per alternative. The leaves are the
/// indices that Python stored with each leaf node.
#[pyfunction]
pub fn sum_of_products(nodes: Vec<(u8, Vec<usize>)>) -> PyResult<Vec<Vec<usize>>> {
    if nodes.is_empty() {
        return Ok(Vec::new());
    }
    terms(&nodes, nodes.len() - 1)
}

/// The sum of products of the node at `index`.
fn terms(nodes: &[(u8, Vec<usize>)], index: usize) -> PyResult<Vec<Vec<usize>>> {
    let (kind, children) = &nodes[index];
    match *kind {
        LEAF => Ok(vec![vec![children[0]]]),
        OR => {
            let mut alternatives = Vec::new();
            for &child in children {
                alternatives.extend(terms(nodes, child)?);
            }
            Ok(alternatives)
        }
        AND => {
            let mut flattened = Vec::new();
            for &child in children {
                if nodes[child].0 == LEAF {
                    flattened.push(nodes[child].1[0]);
                    continue;
                }
                let mut nested = terms(nodes, child)?;
                if nested.len() != 1 {
                    return Err(PyValueError::new_err(FLATTEN_ERROR));
                }
                flattened.extend(nested.swap_remove(0));
            }
            Ok(vec![flattened])
        }
        _ => unreachable!("the Python encoder only emits the kinds defined here"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(id: usize) -> (u8, Vec<usize>) {
        (LEAF, vec![id])
    }

    #[test]
    fn a_leaf_is_one_term() {
        assert_eq!(terms(&[leaf(7)], 0).unwrap(), vec![vec![7]]);
    }

    #[test]
    fn a_conjunction_of_leaves_is_one_term() {
        let nodes = vec![leaf(0), leaf(1), (AND, vec![0, 1])];
        assert_eq!(terms(&nodes, 2).unwrap(), vec![vec![0, 1]]);
    }

    #[test]
    fn nested_conjunctions_flatten() {
        // (a & b) & c
        let nodes = vec![
            leaf(0),
            leaf(1),
            (AND, vec![0, 1]),
            leaf(2),
            (AND, vec![2, 3]),
        ];
        assert_eq!(terms(&nodes, 4).unwrap(), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn a_disjunction_gives_one_term_per_alternative() {
        let nodes = vec![leaf(0), leaf(1), (OR, vec![0, 1])];
        assert_eq!(terms(&nodes, 2).unwrap(), vec![vec![0], vec![1]]);
    }

    #[test]
    fn a_conjunction_with_a_single_alternative_disjunction_flattens() {
        // a & (b) where the disjunction has one alternative
        let nodes = vec![leaf(0), leaf(1), (OR, vec![1]), (AND, vec![0, 2])];
        assert_eq!(terms(&nodes, 3).unwrap(), vec![vec![0, 1]]);
    }

    #[test]
    fn a_conjunction_with_several_alternatives_is_rejected() {
        // a & (b | c)
        let nodes = vec![
            leaf(0),
            leaf(1),
            leaf(2),
            (OR, vec![1, 2]),
            (AND, vec![0, 3]),
        ];
        assert!(terms(&nodes, 4).is_err());
    }
}
