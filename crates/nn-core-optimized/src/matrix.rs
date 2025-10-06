#![allow(dead_code)]

use std::ops::{Add, Sub};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A cache-friendly matrix representation backed by a contiguous `Vec<f64>`.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

#[cfg(feature = "parallel")]
const PARALLEL_WORK_THRESHOLD: usize = 512;

impl Matrix {
    /// Creates a matrix from a pre-allocated buffer.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len());
        Self { rows, cols, data }
    }

    /// Creates a zero-filled matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Creates a column matrix from a slice.
    pub fn from_vec(data: &[f64]) -> Self {
        Self::new(data.len(), 1, data.to_vec())
    }

    /// Returns the matrix length.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the matrix contains no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Retrieves an element by row and column.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[self.index(row, col)]
    }

    /// Sets an element by row and column.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        let idx = self.index(row, col);
        self.data[idx] = value;
    }

    #[inline]
    fn index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    #[inline]
    /// Applies a function element-wise.
    pub fn map<F>(&self, func: F) -> Matrix
    where
        F: Fn(f64) -> f64 + Send + Sync,
    {
        crate::profile_scope!("matrix.map");
        let mut data = self.data.clone();

        #[cfg(feature = "parallel")]
        {
            if Self::should_parallelize(self.len()) {
                data.par_iter_mut().for_each(|value| *value = func(*value));
                return Matrix::new(self.rows, self.cols, data);
            }
        }

        for value in &mut data {
            *value = func(*value);
        }

        Matrix::new(self.rows, self.cols, data)
    }

    /// Adds two matrices.
    pub fn add(&self, other: &Matrix) -> Matrix {
        crate::profile_scope!("matrix.add");
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let len = self.len();

        #[cfg(feature = "parallel")]
        {
            if Self::should_parallelize(len) {
                let data: Vec<f64> = self
                    .data
                    .par_iter()
                    .zip(other.data.par_iter())
                    .map(|(a, b)| a + b)
                    .collect();
                return Matrix::new(self.rows, self.cols, data);
            }
        }

        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            data.push(self.data[i] + other.data[i]);
        }
        Matrix::new(self.rows, self.cols, data)
    }

    /// Subtracts another matrix element-wise.
    pub fn subtract(&self, other: &Matrix) -> Matrix {
        crate::profile_scope!("matrix.subtract");
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let len = self.len();

        #[cfg(feature = "parallel")]
        {
            if Self::should_parallelize(len) {
                let data: Vec<f64> = self
                    .data
                    .par_iter()
                    .zip(other.data.par_iter())
                    .map(|(a, b)| a - b)
                    .collect();
                return Matrix::new(self.rows, self.cols, data);
            }
        }

        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            data.push(self.data[i] - other.data[i]);
        }
        Matrix::new(self.rows, self.cols, data)
    }

    /// Element-wise multiplication.
    pub fn dot(&self, other: &Matrix) -> Matrix {
        crate::profile_scope!("matrix.dot");
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let len = self.len();

        #[cfg(feature = "parallel")]
        {
            if Self::should_parallelize(len) {
                let data: Vec<f64> = self
                    .data
                    .par_iter()
                    .zip(other.data.par_iter())
                    .map(|(a, b)| a * b)
                    .collect();
                return Matrix::new(self.rows, self.cols, data);
            }
        }

        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            data.push(self.data[i] * other.data[i]);
        }
        Matrix::new(self.rows, self.cols, data)
    }

    /// Matrix multiplication.
    pub fn mul(&self, other: &Matrix) -> Matrix {
        crate::profile_scope!("matrix.mul");
        assert_eq!(self.cols, other.rows);

        let mut result = vec![0.0; self.rows * other.cols];

        #[cfg(feature = "parallel")]
        {
            if Self::should_parallelize(self.rows * other.cols) {
                result
                    .par_chunks_mut(other.cols)
                    .enumerate()
                    .for_each(|(row_idx, row)| {
                        let row_start = row_idx * self.cols;
                        for (col_idx, value) in row.iter_mut().enumerate() {
                            let mut sum = 0.0;
                            let mut k = 0;
                            while k < self.cols {
                                sum += self.data[row_start + k]
                                    * other.data[k * other.cols + col_idx];
                                k += 1;
                            }
                            *value = sum;
                        }
                    });
                return Matrix::new(self.rows, other.cols, result);
            }
        }

        for row in 0..self.rows {
            for col in 0..other.cols {
                let mut sum = 0.0;
                let row_start = row * self.cols;
                let mut k = 0;
                while k < self.cols {
                    sum += self.data[row_start + k] * other.data[k * other.cols + col];
                    k += 1;
                }
                result[row * other.cols + col] = sum;
            }
        }

        Matrix::new(self.rows, other.cols, result)
    }

    /// Multiplies by the transpose of another matrix without materialising it.
    pub fn mul_transpose(&self, other: &Matrix) -> Matrix {
        crate::profile_scope!("matrix.mul_transpose");
        assert_eq!(self.cols, other.cols);

        let mut result = vec![0.0; self.rows * other.rows];

        #[cfg(feature = "parallel")]
        {
            if Self::should_parallelize(self.rows * other.rows) {
                result
                    .par_chunks_mut(other.rows)
                    .enumerate()
                    .for_each(|(row_idx, row)| {
                        for (col_idx, value) in row.iter_mut().enumerate() {
                            let mut sum = 0.0;
                            let mut k = 0;
                            while k < self.cols {
                                sum += self.data[row_idx * self.cols + k]
                                    * other.data[col_idx * other.cols + k];
                                k += 1;
                            }
                            *value = sum;
                        }
                    });
                return Matrix::new(self.rows, other.rows, result);
            }
        }

        for row in 0..self.rows {
            for col in 0..other.rows {
                let mut sum = 0.0;
                let mut k = 0;
                while k < self.cols {
                    sum += self.data[row * self.cols + k]
                        * other.data[col * other.cols + k];
                    k += 1;
                }
                result[row * other.rows + col] = sum;
            }
        }
        Matrix::new(self.rows, other.rows, result)
    }

    /// Multiplies the transpose of this matrix by another matrix.
    pub fn transpose_mul(&self, other: &Matrix) -> Matrix {
        crate::profile_scope!("matrix.transpose_mul");
        assert_eq!(self.rows, other.rows);

        let mut result = vec![0.0; self.cols * other.cols];

        #[cfg(feature = "parallel")]
        {
            if Self::should_parallelize(self.cols * other.cols) {
                result
                    .par_chunks_mut(other.cols)
                    .enumerate()
                    .for_each(|(row_idx, row)| {
                        for (col_idx, value) in row.iter_mut().enumerate() {
                            let mut sum = 0.0;
                            let mut k = 0;
                            while k < self.rows {
                                sum += self.data[k * self.cols + row_idx]
                                    * other.data[k * other.cols + col_idx];
                                k += 1;
                            }
                            *value = sum;
                        }
                    });
                return Matrix::new(self.cols, other.cols, result);
            }
        }

        for row in 0..self.cols {
            for col in 0..other.cols {
                let mut sum = 0.0;
                let mut k = 0;
                while k < self.rows {
                    sum += self.data[k * self.cols + row]
                        * other.data[k * other.cols + col];
                    k += 1;
                }
                result[row * other.cols + col] = sum;
            }
        }

        Matrix::new(self.cols, other.cols, result)
    }

    /// Returns the transpose of the matrix.
    pub fn transpose(&self) -> Matrix {
        crate::profile_scope!("matrix.transpose");
        let mut result = vec![0.0; self.rows * self.cols];
        for row in 0..self.rows {
            for col in 0..self.cols {
                result[col * self.rows + row] = self.data[self.index(row, col)];
            }
        }
        Matrix::new(self.cols, self.rows, result)
    }

    /// Consumes the matrix returning a column vector.
    pub fn into_vec(self) -> Vec<f64> {
        self.data
    }

    /// Returns a column vector copy. Panics if the matrix is not a column.
    pub fn to_column_vec(&self) -> Vec<f64> {
        assert_eq!(self.cols, 1);
        self.data.clone()
    }

    #[cfg(feature = "parallel")]
    #[inline]
    fn should_parallelize(work: usize) -> bool {
        work >= PARALLEL_WORK_THRESHOLD
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Self::Output {
        self.add(rhs)
    }
}

impl Sub<&Matrix> for &Matrix {
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        self.subtract(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_get() {
        let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 1), 4.0);
    }

    #[test]
    fn test_add() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = a.add(&b);
        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_mul() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let result = a.mul(&b);
        assert_eq!(result.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_transpose_mul() {
        let a = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 1, vec![0.5, 1.5, 2.5]);
        let result = a.transpose_mul(&b);
        assert_eq!(result.data, vec![17.5, 22.0]);
    }

    #[test]
    fn test_mul_transpose() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(3, 2, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let result = a.mul_transpose(&b);
        assert_eq!(result.data, vec![17.0, 23.0, 29.0, 39.0, 53.0, 67.0]);
    }
}
