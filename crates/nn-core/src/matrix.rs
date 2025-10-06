#![allow(dead_code)]

/// A matrix of `f64` values.
#[derive(Debug, PartialEq, Clone)]
pub struct Matrix {
    /// The number of rows in the matrix.
    pub rows: usize,
    /// The number of columns in the matrix.
    pub cols: usize,
    /// The data of the matrix, stored as a vector of vectors.
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    /// Creates a new matrix.
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows.
    /// * `cols` - The number of columns.
    /// * `data` - The data of the matrix.
    pub fn new(rows: usize, cols: usize, data: Vec<Vec<f64>>) -> Self {
        assert_eq!(data.len(), rows);
        for row in &data {
            assert_eq!(row.len(), cols);
        }
        Self { rows, cols, data }
    }

    /// Creates a new matrix from a vector.
    /// The resulting matrix will have one column.
    ///
    /// # Arguments
    ///
    /// * `data` - The vector to create the matrix from.
    pub fn from_vec(data: &Vec<f64>) -> Matrix {
        Matrix::new(data.len(), 1, data.iter().map(|&x| vec![x]).collect())
    }

    /// Applies a function to each element of the matrix.
    pub fn map<F>(&self, func: F) -> Matrix
    where
        F: Fn(f64) -> f64,
    {
        crate::profile_scope!("matrix.map");
        let mut result_data = vec![vec![0.0; self.cols]; self.rows];

        for i in 0..self.rows {
            for j in 0..self.cols {
                result_data[i][j] = func(self.data[i][j]);
            }
        }
        Matrix::new(self.rows, self.cols, result_data)
    }

    /// Adds two matrices element-wise.
    pub fn add(&self, other: &Matrix) -> Matrix {
        crate::profile_scope!("matrix.add");
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut result_data = vec![vec![0.0; self.cols]; self.rows];

        for i in 0..self.rows {
            for j in 0..self.cols {
                result_data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        Matrix::new(self.rows, self.cols, result_data)
    }

    /// Subtracts another matrix element-wise.
    pub fn subtract(&self, other: &Matrix) -> Matrix {
        crate::profile_scope!("matrix.subtract");
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut result_data = vec![vec![0.0; self.cols]; self.rows];

        for i in 0..self.rows {
            for j in 0..self.cols {
                result_data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        Matrix::new(self.rows, self.cols, result_data)
    }

    /// Multiplies two matrices.
    pub fn mul(&self, other: &Matrix) -> Matrix {
        crate::profile_scope!("matrix.mul");
        assert_eq!(self.cols, other.rows);

        let mut result_data = vec![vec![0.0; other.cols]; self.rows];

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result_data[i][j] = sum;
            }
        }
        Matrix::new(self.rows, other.cols, result_data)
    }

    /// Multiplies this matrix by the transpose of another matrix without materialising the transpose.
    pub fn mul_transpose(&self, other: &Matrix) -> Matrix {
        crate::profile_scope!("matrix.mul_transpose");
        assert_eq!(self.cols, other.cols);

        let mut result_data = vec![vec![0.0; other.rows]; self.rows];

        for i in 0..self.rows {
            for j in 0..other.rows {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[j][k];
                }
                result_data[i][j] = sum;
            }
        }
        Matrix::new(self.rows, other.rows, result_data)
    }

    /// Multiplies the transpose of this matrix by another matrix without materialising the transpose.
    pub fn transpose_mul(&self, other: &Matrix) -> Matrix {
        crate::profile_scope!("matrix.transpose_mul");
        assert_eq!(self.rows, other.rows);

        let mut result_data = vec![vec![0.0; other.cols]; self.cols];

        for i in 0..self.cols {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.rows {
                    sum += self.data[k][i] * other.data[k][j];
                }
                result_data[i][j] = sum;
            }
        }
        Matrix::new(self.cols, other.cols, result_data)
    }

    /// Performs element-wise multiplication of two matrices.
    pub fn dot(&self, other: &Matrix) -> Matrix {
        crate::profile_scope!("matrix.dot");
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut result_data = vec![vec![0.0; self.cols]; self.rows];

        for i in 0..self.rows {
            for j in 0..self.cols {
                result_data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }
        Matrix::new(self.rows, self.cols, result_data)
    }

    /// Transposes the matrix.
    pub fn transpose(&self) -> Matrix {
        crate::profile_scope!("matrix.transpose");
        let mut result_data = vec![vec![0.0; self.rows]; self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result_data[j][i] = self.data[i][j];
            }
        }
        Matrix::new(self.cols, self.rows, result_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_matrix() {
        let m = Matrix::new(2, 2, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 2);
        assert_eq!(m.data, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn test_add_matrices() {
        let m1 = Matrix::new(2, 2, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let m2 = Matrix::new(2, 2, vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let result = m1.add(&m2);
        let expected = Matrix::new(2, 2, vec![vec![6.0, 8.0], vec![10.0, 12.0]]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtract_matrices() {
        let m1 = Matrix::new(2, 2, vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let m2 = Matrix::new(2, 2, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let result = m1.subtract(&m2);
        let expected = Matrix::new(2, 2, vec![vec![4.0, 4.0], vec![4.0, 4.0]]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mul_matrices() {
        let m1 = Matrix::new(2, 2, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let m2 = Matrix::new(2, 2, vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let result = m1.mul(&m2);
        let expected = Matrix::new(2, 2, vec![vec![19.0, 22.0], vec![43.0, 50.0]]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_dot_matrices() {
        let m1 = Matrix::new(2, 2, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let m2 = Matrix::new(2, 2, vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let result = m1.dot(&m2);
        let expected = Matrix::new(2, 2, vec![vec![5.0, 12.0], vec![21.0, 32.0]]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_transpose() {
        let m = Matrix::new(2, 3, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let result = m.transpose();
        let expected = Matrix::new(3, 2, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_from_vec() {
        let v = vec![1.0, 2.0, 3.0];
        let m = Matrix::from_vec(&v);
        let expected = Matrix::new(3, 1, vec![vec![1.0], vec![2.0], vec![3.0]]);
        assert_eq!(m, expected);
    }

    #[test]
    fn test_map() {
        let m = Matrix::new(2, 2, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let result = m.map(|x| x * 2.0);
        let expected = Matrix::new(2, 2, vec![vec![2.0, 4.0], vec![6.0, 8.0]]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mul_transpose() {
        let a = Matrix::new(2, 2, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::new(3, 2, vec![vec![5.0, 6.0], vec![7.0, 8.0], vec![9.0, 10.0]]);
        let result = a.mul_transpose(&b);
        let expected = Matrix::new(
            2,
            3,
            vec![vec![17.0, 23.0, 29.0], vec![39.0, 53.0, 67.0]],
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_transpose_mul() {
        let a = Matrix::new(3, 2, vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        let b = Matrix::new(3, 1, vec![vec![0.5], vec![1.5], vec![2.5]]);
        let result = a.transpose_mul(&b);
        let expected = Matrix::new(2, 1, vec![vec![17.5], vec![22.0]]);
        assert_eq!(result, expected);
    }
}
