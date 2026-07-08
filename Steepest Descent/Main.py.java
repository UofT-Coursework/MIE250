import java.io.*;
import java.util.Scanner;

// Abstract superclass for objective functions
abstract class ObjectiveFunction {

    // Compute the objective function value
    abstract double compute(double[] variables);

    // Compute the gradient of the objective function
    abstract double[] computeGradient(double[] variables);

    // Return the bounds of the function
    abstract double[] getBounds();

    // Return the name of the objective function
    abstract String getName();
}
// Quadratic function subclass
class QuadraticFunction extends ObjectiveFunction {

    @Override
    // Compute the Quadratic function value
    double compute(double[] variables) {
        double sum = 0.0;
        for (double value : variables) {
            sum += Math.pow(SteepestDescentOptimizer.floor5(value), 2);
        }
        return sum;
    }

    @Override
    // Compute the gradient of the Quadratic function
    double[] computeGradient(double[] variables) {
        double[] gradient = new double[variables.length];
        for (int i = 0; i < variables.length; i++) {
            gradient[i] = 2 * SteepestDescentOptimizer.floor5(variables[i]);
        }
        return gradient;
    }

    @Override
    double[] getBounds() {
        return new double[]{-5.0, 5.0}; // return bounds
    }

    @Override
    public String getName() {
        return "Quadratic"; // return name of function
    }
}
// Rosenbrock function subclass
class RosenbrockFunction extends ObjectiveFunction {

    @Override
    // Compute the Rosenbrock function value
    double compute(double[] value) {
        double sum = 0.0;
        for (int i = 0; i < value.length - 1; i++) {
            double x1 = SteepestDescentOptimizer.floor5(value[i]);
            double x2 = SteepestDescentOptimizer.floor5(value[i + 1]);
            sum += 100 * Math.pow((x2 - (x1 * x1)), 2) + Math.pow((1 - x1), 2);
        }
        return sum;
    }

    @Override
    // Compute the gradient of the Rosenbrock function
    double[] computeGradient(double[] variables) {
        int d = variables.length;
        double[] gradient = new double[d];

        for (int i = 0; i < d; i++) {
            if (i == 0) {
                // For the first element
                double x1 = SteepestDescentOptimizer.floor5(variables[i]);
                double x2 = SteepestDescentOptimizer.floor5(variables[i + 1]);
                gradient[i] = -400 * x1 * (x2 - (x1 * x1)) - 2 * (1 - x1);
            } else if (i == d - 1) {
                // For the last element
                double x1 = SteepestDescentOptimizer.floor5(variables[i - 1]);
                double x2 = SteepestDescentOptimizer.floor5(variables[i]);
                gradient[i] = 200 * (x2 - (x1 * x1));
            } else {
                // For middle elements
                double x1 = SteepestDescentOptimizer.floor5(variables[i - 1]);
                double x2 = SteepestDescentOptimizer.floor5(variables[i]);
                double x3 = SteepestDescentOptimizer.floor5(variables[i + 1]);
                gradient[i] = 200 * (x2 - (x1 * x1)) - 400 * x2 * (x3 - (x2 * x2)) - 2 * (1 - x2);
            }
        }
        return gradient;
    }

    @Override
    double[] getBounds() {
        return new double[]{-5.0, 5.0}; // return bounds
    }

    @Override
    String getName() {
        return "Rosenbrock"; // return name of function
    }
}

class Input {
    String functionName;
    int dimensionality;
    int iterations;
    double tolerance;
    double stepSize;
    double[] initialPoint;
}

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        SteepestDescentOptimizer SDO = new SteepestDescentOptimizer();
        Input input;

        // Enter or exit program
        int enterChoice = SDO.getValidatedInput(scanner, "Press 0 to exit or 1 to enter the program:");
        if (enterChoice == 0) {
            System.out.println("Exiting program...");
            scanner.close();
            System.exit(0);
        }

        // Get input method
        int inputMethod = SDO.getValidatedInput(scanner, "Press 0 for .txt input or 1 for manual input:");

        // Get output method
        int outputMethod = SDO.getValidatedInput(scanner, "Press 0 for .txt output or 1 for console output:");

        // Get input values
        if (inputMethod == 0) {
            input = SDO.getFileInput(scanner);
        } else {
            input = SDO.getManualInput(scanner);
        }
        if (input == null) System.exit(0);

        // Setup output
        PrintWriter write;
        if (outputMethod == 0) { // file output
            System.out.println("Please provide the path for the output file:");
            String path = scanner.nextLine();
            try {
                write = new PrintWriter(new FileWriter(path));
            } catch (IOException e) {
                System.out.println("Error reading the file.");
                return;
            }
        } else { // console output
            write = new PrintWriter(System.out);
        }

        // Create corresponding function class for optimization
        ObjectiveFunction function = createObjFunc(input.functionName);

        // Print parameters
        SDO.printParameters(function, input, write);

        // Run optimization
        SDO.optimizeSteepestDescent(function, input.initialPoint, input.iterations, input.tolerance, input.stepSize, input.dimensionality, write);

        write.close();
        scanner.close();
    }
    static ObjectiveFunction createObjFunc (String function){
        if (function.equalsIgnoreCase("quadratic")) {
            return new QuadraticFunction();
        } else {
            return new RosenbrockFunction();
        }
    }
}

class SteepestDescentOptimizer {
    void optimizeSteepestDescent(ObjectiveFunction objectiveFunction, double[] variables, int iterations, double tolerance, double stepSize, int dimensionality, PrintWriter write) {
        // First iteration
        printIteration(1, objectiveFunction.compute(variables), variables, tolerance, write);

        // Iterate n times to optimize
        for (int n = 2; n <= iterations; n++) {
            double[] gradient = objectiveFunction.computeGradient(variables);

            // Must calculate variables differently for Rosenbrock / Quadratic functions to replicate example output
            if (objectiveFunction instanceof RosenbrockFunction) {
                for (int i = 0; i < dimensionality; i++) {
                    variables[i] = floor5(variables[i]) - stepSize * floor5(gradient[i]);
                }
            } else {
                for (int i = 0; i < dimensionality; i++) {
                    variables[i] = floor5(variables[i]) - stepSize * floor5(gradient[i]);
                    variables[i] = Math.nextDown(variables[i]); // Needed to match example output
                }
            }

            double objValue = objectiveFunction.compute(variables); // current objective function value
            double gradientMagnitude = computeGradientMagnitude(gradient); // gradient magnitude

            // Print iteration results
            printIteration(n, objValue, variables, gradientMagnitude, write);

            // Check for convergence
            if (gradientMagnitude < tolerance) {
                printResult(true, n, write);
                return;
            }
        }
        // Where max iterations are reached without convergence
        printResult(false, iterations, write);
    }

    // Compute gradient magnitude
    static double computeGradientMagnitude(double[] vector) {
        double magnitude = 0.0;
        for (double element : vector) {
            magnitude += element * element;
        }
        return Math.sqrt(magnitude);
    }

    // Method for printing parameters
    void printParameters(ObjectiveFunction function, Input input, PrintWriter write) {
        write.println("Objective Function: " + function.getName());
        write.println("Dimensionality: " + input.dimensionality);
        write.print("Initial Point:");
        for (double value : input.initialPoint) {
            // Floor value but display only one decimal place as per provided example
            value = Math.floor(value * 100000.0) / 100000.0;
            String stringValue = String.format("%.1f", value);

            write.print(" ");
            write.print(stringValue);
        }
        write.println();
        write.println("Iterations: " + input.iterations);
        write.println("Tolerance: " + format5(input.tolerance));
        write.println("Step Size: " + format5(input.stepSize));
        write.println();
        write.println("Optimization process:");
    }

    // Method for printing iterations
    void printIteration(int iteration, double objValue, double[] colVector, double tolerance, PrintWriter write) {
        write.println("Iteration " + iteration + ":");
        write.println("Objective Function Value: " + format5(objValue));
        write.print("x-values:");
        for (double v : colVector) {
            write.print(" ");
            write.print(format5(v));
        }
        write.println();
        if (iteration >= 2){
            write.println("Current Tolerance: "+ format5(tolerance));
        }
        write.println();
    }

    // Method to print final result
    void printResult(boolean result, int n, PrintWriter write) {
        if (result) {
            write.println("Convergence reached after " + n + " iterations.");
        } else {
            write.println("Maximum iterations reached without satisfying the tolerance.");
        }
        write.println();
        write.println("Optimization process completed.");
    }

    // Print menu prompts and validate input
    int getValidatedInput(Scanner scanner, String prompt) {
        while (true) {
            System.out.println(prompt);
            String input = scanner.nextLine();
            if (input.equals("0") || input.equals("1")) {
                return Integer.parseInt(input);
            }
            System.out.println("Please enter a valid input (0 or 1).");
        }
    }

    // Get manual input from user
    Input getManualInput(Scanner scanner) {
        Input input = new Input();
        QuadraticFunction qFunction = new QuadraticFunction();

        System.out.println("Enter the choice of objective function (quadratic or rosenbrock):");
        input.functionName = scanner.nextLine();

        System.out.println("Enter the dimensionality of the problem:");
        input.dimensionality = Integer.parseInt(scanner.nextLine());

        System.out.println("Enter the number of iterations:");
        input.iterations = Integer.parseInt(scanner.nextLine());

        System.out.println("Enter the tolerance:");
        input.tolerance = Double.parseDouble(scanner.nextLine());

        System.out.println("Enter the step size:");
        input.stepSize = Double.parseDouble(scanner.nextLine());

        // Validate function name
        if (!input.functionName.equalsIgnoreCase("quadratic") && !input.functionName.equalsIgnoreCase("rosenbrock")) {
            System.out.println("Error: Unknown objective function.");
            scanner.close();
            return null;
        }

        System.out.println("Enter the initial point as " + input.dimensionality + " space-separated values:");
        String[] stringPoint = scanner.nextLine().split(" ");
        // Check for dimensionality mismatch
        if (stringPoint.length != input.dimensionality) {
            System.out.println("Error: Initial point dimensionality mismatch.");
            scanner.close();
            return null;
        }

        // Parse initial point values into double array
        input.initialPoint = new double[input.dimensionality];
        for (int i = 0; i < input.dimensionality; i++) {
            input.initialPoint[i] = Double.parseDouble(stringPoint[i]);
        }

        // Check for out of bounds initial point
        if (!checkBounds(input.initialPoint, qFunction.getBounds())) {
            scanner.close();
            return null;
        }
        return input; // return input values
    }

    // Read file input
    Input getFileInput(Scanner scanner) {
        System.out.println("Please provide the path to the config file:");
        File file = new File(scanner.nextLine());

        try (Scanner fileScanner = new Scanner(file)) {
            Input input = new Input();
            QuadraticFunction qFunction = new QuadraticFunction();

            // Read and parse input values from file
            input.functionName = fileScanner.nextLine().split("//")[0].trim();
            input.dimensionality = Integer.parseInt(fileScanner.nextLine().split("//")[0].trim());
            input.iterations = Integer.parseInt(fileScanner.nextLine().split("//")[0].trim());
            input.tolerance = Double.parseDouble(fileScanner.nextLine().split("//")[0].trim());
            input.stepSize = Double.parseDouble(fileScanner.nextLine().split("//")[0].trim());

            // Validate function name
            if (!input.functionName.equalsIgnoreCase("quadratic") && !input.functionName.equalsIgnoreCase("rosenbrock")) {
                System.out.println("Error: Unknown objective function.");
                fileScanner.close();
                scanner.close();
                return null;
            }

            String[] stringPoint = fileScanner.nextLine().split("//")[0].trim().split(" ");
            // Check for dimensionality mismatch
            if (stringPoint.length != input.dimensionality) {
                System.out.println("Error: Initial point dimensionality mismatch.");
                fileScanner.close();
                scanner.close();
                return null;
            }

            // Parse initial point values into double array
            input.initialPoint = new double[input.dimensionality];
            for (int i = 0; i < input.dimensionality; i++) {
                input.initialPoint[i] = Double.parseDouble(stringPoint[i]);
            }

            // Check for out of bounds initial point
            if (!checkBounds(input.initialPoint, qFunction.getBounds())) {
                fileScanner.close();
                scanner.close();
                return null;
            }

            fileScanner.close();
            return input; // return input values

        } catch (FileNotFoundException e) {
            System.out.println("Error reading the file.");
            scanner.close();
            return null;
        }
    }

    // Check if initial point is within bounds
    boolean checkBounds(double[] variables, double[] bounds) {
        for (double variable : variables) {
            if (variable < bounds[0] || variable > bounds[1]) {
                System.out.println("Error: Initial point " + variable + " is outside the bounds [-5.0, 5.0].");
                return false;
            }
        }
        return true;
    }

    // Return floored and formatted String value
    static String format5(double value) {
        double floored = Math.floor(value * 100000.0) / 100000.0;
        return String.format("%.5f", floored);
    }

    // Return floored double value for calculations
    static double floor5(double value) {
        return Math.floor(value * 100000.0) / 100000.0;
    }
}
