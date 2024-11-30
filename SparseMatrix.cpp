#include "SparseMatrix.h"
#include <iostream>
#include <chrono>
#include <random>
SparseMatrix::SparseMatrix(int rows, int cols) : rows(rows), cols(cols), row_start(rows + 1, 0) {}

void SparseMatrix::addElement(int row, int col, double value) {
    if (value != 0 && unique_indices.find({ row, col }) == unique_indices.end()) {
        values.push_back(value);
        column_indices.push_back(col);
        row_start[row + 1]++;
        unique_indices.insert({ row, col });
    }
}

void SparseMatrix::finalize() {
    for (int i = 1; i <= rows; ++i) {
        row_start[i] += row_start[i - 1];
    }
}

SparseMatrix SparseMatrix::operator+(const SparseMatrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for addition.");
    }

    SparseMatrix result(rows, cols);
    std::map<std::pair<int, int>, double> temp_map;

    // Add elements from the first matrix
    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        for (int j = row_start[i]; j < row_start[i + 1]; ++j) {
            {
                temp_map[{i, column_indices[j]}] += values[j];
            }
        }
    }

    // Add elements from the second matrix
    #pragma omp parallel for
    for (int i = 0; i < other.rows; ++i) {
        for (int j = other.row_start[i]; j < other.row_start[i + 1]; ++j) {
            {
                temp_map[{i, other.column_indices[j]}] += other.values[j];
            }
        }
    }

    // Fill the result matrix
    for (const auto& elem : temp_map) {
        result.addElement(elem.first.first, elem.first.second, elem.second);
    }

    result.finalize();
    return result;
}

SparseMatrix SparseMatrix::operator*(const SparseMatrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Number of columns in the first matrix must match the number of rows in the second matrix.");
    }

    SparseMatrix result(rows, other.cols);
    std::vector<double> temp_values;
    std::vector<int> temp_column_indices;
    std::vector<int> temp_row_start(rows + 1, 0);


    std::vector<std::map<int, double>> local_row_maps(rows);

    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        for (int j = row_start[i]; j < row_start[i + 1]; ++j) {
            int col = column_indices[j];
            double value = values[j];
            for (int k = other.row_start[col]; k < other.row_start[col + 1]; ++k) {
                 int other_col = other.column_indices[k];
                 local_row_maps[i][other_col] += value * other.values[k];
            }
        }
    }

        
    for (int i = 0; i < rows; ++i) {
        for (const auto& elem : local_row_maps[i]) {
            temp_values.push_back(elem.second);
            temp_column_indices.push_back(elem.first);
            temp_row_start[i + 1]++;
        }
    }
    

    for (int i = 1; i <= rows; ++i) {
        temp_row_start[i] += temp_row_start[i - 1];
    }

    result.values = temp_values;
    result.column_indices = temp_column_indices;
    result.row_start = temp_row_start;

    return result;
}

void SparseMatrix::print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double value = 0.0;
            for (int k = row_start[i]; k < row_start[i + 1]; ++k) {
                if (column_indices[k] == j) {
                    value = values[k];
                    break;
                }
            }
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}

void measurePerformance(const SparseMatrix& mat1, const SparseMatrix& mat2, int num_threads) {
    omp_set_num_threads(num_threads);

    auto start = std::chrono::high_resolution_clock::now();
    SparseMatrix result_add = mat1 + mat2;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_add = end - start;

    start = std::chrono::high_resolution_clock::now();
    SparseMatrix result_mul = mat1 * mat2;
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_mul = end - start;

    std::cout << "Threads,Addition Time (s),Multiplication Time (s)" << std::endl;
    std::cout << num_threads << "," << duration_add.count() << "," << duration_mul.count() << std::endl;

}

int main() {
    SparseMatrix mat1(8, 8);
    mat1.addElement(0, 0, 1.0);
    mat1.addElement(0, 7, 2.0);
    mat1.addElement(1, 1, 3.0);
    mat1.addElement(2, 2, 4.0);
    mat1.addElement(3, 3, 5.0);
    mat1.addElement(4, 4, 6.0);
    mat1.addElement(5, 5, 7.0);
    mat1.addElement(6, 6, 8.0);
    mat1.addElement(7, 7, 9.0);
    mat1.finalize();

    SparseMatrix mat2(8, 8);
    mat2.addElement(0, 0, 10.0);
    mat2.addElement(1, 1, 11.0);
    mat2.addElement(2, 2, 12.0);
    mat2.addElement(3, 3, 13.0);
    mat2.addElement(4, 4, 14.0);
    mat2.addElement(5, 5, 15.0);
    mat2.addElement(6, 6, 16.0);
    mat2.addElement(7, 7, 17.0);
    mat2.finalize();

    std::cout << "Matrix 1:" << std::endl;
    mat1.print();

    std::cout << "Matrix 2:" << std::endl;
    mat2.print();

    SparseMatrix result = mat1 + mat2;
    std::cout << "Sum of matrix :" << std::endl;
    result.print();

    SparseMatrix resultMUL = mat1 * mat2;

    std::cout << "Result of matrix multiplication:" << std::endl;
    resultMUL.print();

    for (int num_threads = 1; num_threads <= 15; ++num_threads) {
        measurePerformance(mat1, mat2, num_threads);
    }

    return 0;
}