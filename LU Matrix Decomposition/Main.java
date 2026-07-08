import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ForkJoinPool;

public class Main {
    private int n;
    private double[][] A;
    private double[][] L;
    private double[][] U;
    private final String inputFile;
    private final String outputFile;
    // Execution: true for parallel, false o/w (sequential)
    private boolean parallel = true;

    public Main(String inputFile, String outputFile, boolean parallel) {
        this.inputFile = inputFile;
        this.outputFile = outputFile;
        this.parallel = parallel;
    }

    public static void main(String[] args) {
        // Determine input file from command-line arguments
        String inputFile = "input.txt";
        boolean defaultFile = true;

        // If user provides an argument, use it as the input file
        if (args.length > 0) {
            inputFile = args[0];
            defaultFile = false;
        }

        // Read configuration file to determine execution mode
        boolean parallel = readConfig("config.txt");

        // Create instance of LU factorization and run
        Main lu = new Main(inputFile, "output.txt", parallel);
        lu.run(defaultFile);
    }

    /**
     * Read config file to determine execution mode.
     * @param file = name of config file
     * @return true if parallel execution, false if sequential execution
     */
    private static boolean readConfig(String file) {
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;

            while ((line = br.readLine()) != null) {
                if (line.trim().startsWith("parallel_execution")) {
                    return Boolean.parseBoolean(line.split("=")[1].trim());
                }
            }
            return true; // Default to parallel execution if unspecified
        } catch (IOException e) {
            throw new RuntimeException("Error: config file could not be read;", e);
        }
    }

    /**
     * High level method executing LU decomposition
     * @param defaultFile - true if default input file, false o/w
     */
    public void run(boolean defaultFile) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
            if (defaultFile) {
                writer.println("No input file specified. Using default: input.txt");
            }
            writer.println("Input file: " + inputFile);
            writer.println("Output file: " + outputFile);
            writer.println("Execution mode: " + (parallel ? "parallel" : "sequential"));

            // Read matrix and perform decomposition
            try {
                readMatrix(inputFile); // read matrix
                writer.println();
                writer.println("Matrix A:");
                printMatrix(A, writer, false);

                decompose(); // LU decomposition

                // Print results
                writer.println();
                writer.println("Final Matrix L:");
                printMatrix(L, writer, false);
                writer.println();
                writer.println("Final Matrix U:");
                printMatrix(U, writer, false);

                // Verify decomposition & tolerance
                double tolerance = verifyDecomposition(writer);
                writer.println();
                writer.println(String.format("Tolerance (difference between A and LU): %.4f", tolerance));
                writer.println();
                writer.println("Decomposition complete. Results written to " + outputFile);

            } catch (IllegalArgumentException e) {
                // Handle non-square matrix errors
                writer.println();
                writer.println("Error: " + e.getMessage());
            } catch (ArithmeticException e) {
                // Handle singular matrix errors
                writer.println();
                writer.println("Error: " + e.getMessage());
            }
        } catch (IOException e) {
            // Handle file writing errors
            System.err.println("Error: " + e.getMessage());
        }
    }

    /**
     * Read matrix from file and validate
     * @param file = name of input file
     * @throws IOException if file cannot be read
     * @throws IllegalArgumentException if matrix is not square
     */
    private void readMatrix(String file) throws IOException {
        List<double[]> rows = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;

            // Extract values and parse to double
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) continue;
                String[] values = line.trim().split("\\s+");

                double[] row = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    row[i] = Double.parseDouble(values[i]);
                }
                rows.add(row);
            }
        }

        // Validate matrix is square
        n = rows.size();

        // Use try-with-resources to ensure the writer is closed,
        // Use writer.flush() to ensure error is written before exception is thrown
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("output.txt", true))) {
            // Check if matrix is empty
            if (n == 0) {
                writer.newLine();
                writer.write("Error: Matrix is empty.");
                // Crucial: Manually flush the buffer before throwing to guarantee write
                writer.flush();
                throw new IllegalArgumentException("Matrix is empty.");
            }

            // Check if matrix is square
            for (double[] row : rows) {
                if (row.length != n) {
                    writer.newLine();
                    writer.write("Error: Matrix must be square.");
                    // Crucial: Manually flush the buffer before throwing to guarantee write
                    writer.flush();
                    throw new IllegalArgumentException("Matrix must be square.");
                }
            }
        }

        // Initialize A, L, and U
        A = new double[n][n];
        L = new double[n][n];
        U = new double[n][n]; // Initialized to zeros by default

        for (int i = 0; i < n; i++) {
            A[i] = rows.get(i); // Fill with input data
        }
        for (int i = 0; i < n; i++) {
            L[i][i] = 1.0; // Create identity matrix
        }
    }

    // Intermediate method; decides execution mode
    private void decompose() throws IOException {
        if (parallel) {
            decomposeParallel();
        } else {
            decomposeSequential();
        }
    }

    /**
     * Parallel execution of Doolittle algorithm, via ForkJoinPool
     * Parallelize computation of U, L elements within rows
     */
    private void decomposeParallel() throws IOException {
        ForkJoinPool pool = new ForkJoinPool();

        // Calculate elements of U
        for (int row = 0; row < n; row++) {
            final int i = row;
            List<Callable<Void>> subtasks_U = new ArrayList<>();

            // Add subtask for each column in the current row (U)
            for (int col = i; col < n; col++) {
                final int j = col;

                subtasks_U.add(() -> {
                    double sum = 0.0;
                    // Summation(L[i][k] * U[k][j]) for k < i
                    for (int k = 0; k < i; k++) {
                        sum += L[i][k] * U[k][j];
                    }
                    // U[i][j] = A[i][j] - Summation()
                    U[i][j] = A[i][j] - sum;
                    return null;
                });
            }
            // Parallel execution of U subtasks
            try {
                pool.invokeAll(subtasks_U);
            } catch (InterruptedException e) {
                throw new RuntimeException("Parallel computation interrupted", e);
            }

            // Singularity error check
            if (U[i][i] == 0) {
                pool.shutdown();
                singularityErrorHandler(A);
                throw new ArithmeticException("Matrix is singular, cannot perform decomposition.");
            }

            // Calculate elements of L
            List<Callable<Void>> subtasks_L = new ArrayList<>();

            // Add subtask for each row in the current column (L)
            for (int row_j = i + 1; row_j < n; row_j++) {
                final int j = row_j;
                subtasks_L.add(() -> {
                    double sum = 0.0;
                    // Summation(L[j][k] * U[k][i]) for k < i
                    for (int k = 0; k < i; k++) {
                        sum += L[j][k] * U[k][i];
                    }
                    // L[j][i] = (A[j][i] - Summation() / U[i][i]
                    L[j][i] = (A[j][i] - sum) / U[i][i];
                    return null;
                });
            }
            // Parallel execution of L subtasks
            try {
                pool.invokeAll(subtasks_L);
            } catch (InterruptedException e) {
                throw new RuntimeException("Parallel computation interrupted", e);
            }
        }
        pool.shutdown(); // Shutdown thread pool
    }

    /**
     * Sequential execution of Doolittle algorithm
     * Processes matrices row by row
     */
    private void decomposeSequential() throws IOException {
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double sum = 0.0;
                // Summation(L[i][k] * U[k][j]) for k < i
                for (int k = 0; k < i; k++) {
                    sum += L[i][k] * U[k][j];
                }
                // U[i][j] = A[i][j] - Summation()
                U[i][j] = A[i][j] - sum;
            }

            // Singularity error check
            if (U[i][i] == 0) {
                singularityErrorHandler(A);
                throw new ArithmeticException("Matrix is singular, cannot perform decomposition.");
            }

            // Calculate elements of L
            for (int j = i + 1; j < n; j++) {
                double sum = 0.0;
                // Summation(L[j][k] * U[k][i]) for k < i
                for (int k = 0; k < i; k++) {
                    sum += L[j][k] * U[k][i];
                }
                // L[j][i] = (A[j][i] - Summation() / U[i][i]
                L[j][i] = (A[j][i] - sum) / U[i][i];
            }
        }
    }
    // Supplementary method; write singularity error to file and halt program
    private static void singularityErrorHandler (double[][] matrix) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("output.txt", true))) {
            // Write matrix A where applicable
            writer.newLine();
            writer.write("Matrix A: ");
            writer.newLine();
            for (double[] values : matrix) {
                writer.write(rowToString(values));
                writer.newLine();
            }
            // Write singularity error message
            writer.newLine();
            writer.write("Error: Matrix is singular, cannot perform decomposition.");
        }
    }
    // Supplementary method; convert matrix row to string
    static String rowToString(double[] row) {
       StringBuilder stringRow = new StringBuilder();
       for (Double value : row) {
           stringRow.append(String.format("%.1f ", value));
       }
       return stringRow.toString();
    }

    /**
     * Verify LU decomposition: compute A - L*U & calculate tolerance
     * @param writer - PrintWriter to output file
     * @return tolerance (Frobenius norm of difference matrix)
     */
    private double verifyDecomposition(PrintWriter writer) {

        double[][] LU = matrixMultiplication(L, U);
        double[][] D = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                D[i][j] = A[i][j] - LU[i][j];
            }
        }

        writer.println();
        writer.println("Difference Matrix (A - LU):");
        printMatrix(D, writer, true);

        double toleranceSquared = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                toleranceSquared += D[i][j] * D[i][j];
            }
        }
        return Math.sqrt(toleranceSquared);
    }
    // Supplementary method; multiply two matrices
    private double[][] matrixMultiplication(double[][] M1, double[][] M2) {
        double[][] matrix = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;

                for (int k = 0; k < n; k++) {
                    sum += M1[i][k] * M2[k][j];
                }
                matrix[i][j] = sum;
            }
        }
        return matrix;
    }

    /**
     * Prints properly formatted matrix to output file
     * @param matrix - matrix
     * @param writer - PrintWriter to output file
     * @param format - true for 4 decimal places, false for 1 decimal place
     */
    private void printMatrix(double[][] matrix, PrintWriter writer, boolean format) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (format) {
                    writer.print(String.format("%.4f", matrix[i][j]));
                } else {
                    writer.print(String.format("%.1f", matrix[i][j]));
                }
                if (j < n - 1) {
                    writer.print(" ");
                }
            }
            writer.println();
        }
    }

}