class Matrix {
  constructor(cols = 3, rows = 3) {
    this.cols = cols;
    this.rows = rows;
    this.data = this.zerosMatrix(cols, rows);
  }

  // Function to create a matrix filled with zeros
  zerosMatrix(cols, rows) {
    const matrix = [];
    for (let i = 0; i < rows; i++) {
      matrix[i] = [];
      for (let j = 0; j < cols; j++) {
        matrix[i][j] = 0;
      }
    }
    return matrix;
  }

  copy() {
    const copyMatrix = new Matrix(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        copyMatrix.data[i][j] = this.data[i][j];
      }
    }
    return copyMatrix;
  }

    // Function to get the value at position (i, j) in the matrix
  at(i, j) {
    if (i < 0 || i >= this.rows || j < 0 || j >= this.cols) {
      throw new Error('Invalid position. Position is out of bounds.');
    }
    return this.data[i][j];
  }

  // Function to set the value at position (i, j) in the matrix
  setAt(i, j, value) {
    if (i < 0 || i >= this.rows || j < 0 || j >= this.cols) {
      throw new Error('Invalid position. Position is out of bounds.');
    }
    this.data[i][j] = value;
  }

  // Function to create a matrix filled with random floats in the interval [min, max]
  static random(n, m, min, max) {
    // Ensure min is strictly less than max
    if (min > max) {
      const temp = min;
      min = max;
      max = temp;
    }

    const randomMatrix = new Matrix(m, n);

    for (let i = 0; i < randomMatrix.rows; i++) {
      for (let j = 0; j < randomMatrix.cols; j++) {
        randomMatrix.data[i][j] = Math.random() * (max - min) + min;
      }
    }

    return randomMatrix;
  }

}