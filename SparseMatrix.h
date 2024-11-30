#include <vector>
#include <map>
#include <stdexcept>
#include <omp.h>
#include <set>

class SparseMatrix {
public:
    SparseMatrix(int rows, int cols);
    void addElement(int row, int col, double value);
    void finalize();
    SparseMatrix operator+(const SparseMatrix& other) const;
    SparseMatrix operator*(const SparseMatrix& other) const;
    void print() const;

private:
    int rows, cols;
    std::vector<double> values;
    std::vector<int> column_indices;
    std::vector<int> row_start;
    std::set<std::pair<int, int>> unique_indices;
};