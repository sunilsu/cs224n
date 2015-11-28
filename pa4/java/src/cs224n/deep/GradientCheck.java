package cs224n.deep;

import java.util.List;
import org.ejml.simple.SimpleMatrix;

/** 
 * Implements a numerical gradient check over a series
 * of matrices for a neural network.
 * 
 * @author Chris Billovits chrisb@cs.stanford.edu
 */

public class GradientCheck {
        
    public static final double eps = 1e-4;
    public static final double abs_threshold = 1e-7;
    public static final double rel_threshold = 1e-9;
    private static int attempt = 1;
    
    /** 
     * Given an ordered set of weight matrices and their corresponding
       derivatives,
     * calculates the relative and absolute error and logs them to file. 
     *
     * @param Y: The actual label vector for a given example.
     * 
     * @param weights: The weight matrices for the neural network.  By
     * convention, the last weight matrix is taken to be the input vector
     * X derived from vocabulary matrix L.
     * 
     * @param matrixDerivatives: The corresponding derivatives of the
     * objective function wrt each weight matrix.
     * 
     * @param objFn: The class that implements an objective function for
     * the network
     *
     * @return True if the L2 norm of (analytical derivatives -
     * functional derivatives) is less than the threshold.
     * 
     * Note that for large networks, correct implementations may exceed
     * threshold while still being correct, but they should generally be
     * the same order of magnitude as the threshold.  If you believe this
     * is the case, then the relative error should be similar with
     * different network dimensions.
     */

    public static boolean check(SimpleMatrix Y, 
				List<SimpleMatrix> weights, 
				List<SimpleMatrix> matrixDerivatives, 
				ObjectiveFunction objFn)
    {
	
	SimpleMatrix X = weights.get(weights.size() - 1); 
	
	int flops = 0; //Number of "weight" numbers for relative error
	
	double error = 0;
	if (weights.size() != matrixDerivatives.size()){
	    System.err.println("Error: Missed a weight or derivative matrix!");
	    return false;
	}
		
	for (int i = 0; i < weights.size(); i++){
	    SimpleMatrix w = weights.get(i);
	    SimpleMatrix dw = matrixDerivatives.get(i);
	    
	    // Make sure dimensions match                                                                                                                       
	    if (!((w.numRows() == dw.numRows())
		  && (w.numCols() == dw.numCols())) )
		{
		    System.err.println("Error: Matrix and its derivative " 
				       + "are not the same dimension!: ");
		    w.printDimensions();
		    dw.printDimensions();
		    return false;
		}
	     else {
		flops += (w.numRows() * w.numCols());
	    }
	    double err = errFromMatrix(Y, w, dw, objFn, X);
	    error += err;
	    //System.out.println("i=" + i + "E= " + err);
	}
	error = Math.sqrt(error);
	attempt++;
	if (error > abs_threshold)
		System.out.println("Trial " + attempt + " Abs | Rel : "
			   + error + " | " 
			   + (error / (flops * 1.0)));
	return error <= abs_threshold;
    }

 
    /*   Finds the error from the given matrix by deviating by eps in both
     *   directions.
     */
    private static double errFromMatrix(SimpleMatrix Y, SimpleMatrix matr, SimpleMatrix deriv,
					ObjectiveFunction objFn, SimpleMatrix X)
    {
        double error = 0;
        for (int c = 0; c < matr.numCols(); c++){
            for (int r = 0; r < matr.numRows(); r++){

                double prior = matr.get(r, c);

                matr.set(r, c, prior + eps);
                double higher = objFn.valueAt(Y, X);
                matr.set(r, c, prior - eps);
                double lower = objFn.valueAt(Y, X);
                matr.set(r, c, prior);

                double analytic_deriv = (higher - lower) / (2.0 * eps);

                double componentError = deriv.get(r, c) - analytic_deriv;
                error += (componentError * componentError);
            }
        }
        return error;
    }
}
