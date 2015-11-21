package cs224n.deep;

import java.lang.*;
import org.ejml.simple.SimpleMatrix;

/**
 * An objective function takes an input vector and computes the error for a given example.
 * @author Chris Billovits
 */
public interface ObjectiveFunction {

    /**
     * Evaluates an objective function on top of a Matrix vector input, 
     * against a label.
     */   
    public double valueAt(SimpleMatrix label, SimpleMatrix input);

}
