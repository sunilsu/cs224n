package cs224n.deep;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.ops.SpecializedOps;
import org.ejml.simple.*;

import java.text.*;

public class WindowModel_Batch implements ObjectiveFunction {

	protected SimpleMatrix L, W, U, rW, rU;
	//
	protected int windowSize,wordSize, hiddenSize, outputNodes;
	protected double lr;
	protected static Map<String, Integer> labelToIndex = new HashMap<String, Integer>();
	protected static Map<Integer, String> indexToLabel = new HashMap<Integer, String>();
	private double C = 0.001;
	private boolean checkGradient;
	private int Epochs;
        private int batchSize;
 	
	static {
		labelToIndex.put("O", 0);
		labelToIndex.put("LOC", 1);
		labelToIndex.put("MISC", 2);
		labelToIndex.put("ORG", 3);
		labelToIndex.put("PER", 4);
		indexToLabel.put(0, "O");
		indexToLabel.put(1, "LOC");
		indexToLabel.put(2, "MISC");
		indexToLabel.put(3, "ORG");
		indexToLabel.put(4, "PER");

	}
	
	public WindowModel_Batch(int _windowSize, int _hiddenSize, double _lr, double _C,  int epochs, int _batchSize, boolean check){
		//TODO
		windowSize = _windowSize; // assuming odd window size
		hiddenSize = _hiddenSize;
		wordSize = 50; // 50 dim word vector
		outputNodes = 5; // number of NER outputs
		lr = _lr;
		C = _C;
                batchSize = _batchSize;
		Epochs = epochs;
		checkGradient = check;
               
	}
	
	private SimpleMatrix initRandom(int rows, int cols) {
		double eInit = Math.sqrt(6.0) / Math.sqrt(cols + rows -1);// fanIn = cols, fanOut = rows
		return SimpleMatrix.random(rows, cols, -eInit, eInit, new Random(System.nanoTime()));
	}

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		// initialize with bias inside as the last column
	        // or separately
		//TODO: should bias coulumn be set to 0 ??
		W = initRandom(hiddenSize, windowSize*wordSize + 1); // first column is bias
		U = initRandom(outputNodes, hiddenSize+1); // first column is bias
		// change bias columns to 0
		double[] zeros = new double[W.numRows()];
		W.setColumn(0, 0, zeros);
		zeros = new double[U.numRows()];
		U.setColumn(0, 0, zeros);
		L = FeatureFactory.allVecs;
		
	}
	
	public List<String> getLabels(List<Datum> sentence) {
		List<String> labels = new ArrayList<String>();
		for (Datum datum: sentence) {
			labels.add(datum.label);
		}
		return labels;
	}
	/**
	 * 
	 * @param label
	 * @return one hot encoded vector
	 */
	public SimpleMatrix getLabelMatrix(String label) {
		int index = labelToIndex.get(label);
		SimpleMatrix lmat = new SimpleMatrix(outputNodes, 1);
		lmat.set(index, 1.0);
		return lmat;
	}
	
	/**
	 * 
	 * @param sentence List of words in a sentence that needs to be padded
	 * @return List of padded words
	 */
	public List<String> pad(List<Datum> sentence) {
		List<String> padded = new ArrayList<String>();
		int padLen = windowSize / 2;
		for (int i=0; i<padLen; i++) {
			padded.add("<s>");
		}
		for (Datum datum: sentence) {
			padded.add(datum.word);
		}
		for (int i=0; i<padLen; i++) {
			padded.add("</s>");
		}
		return padded;
	}
	/**
	 * 
	 * @param paddedSentence List of padded words in a sentence
	 * @return List of a windowed words, each window is a List of words
	 */
	public List<List<String>> window(List<String> paddedSentence) {
		List<List<String>> windows = new ArrayList<List<String>>();
		int padLen = windowSize / 2;
		for (int index = padLen; index < (paddedSentence.size()-padLen); index++) {
			List<String> window = new ArrayList<String>();
			for (int offset = -padLen; offset <= padLen; offset++) {
				window.add(paddedSentence.get(index+offset));
			}
			windows.add(window);
		}
		return windows;
	}
	/**
	 * 
	 * @param one window of sentence words
	 * @return vector representation of of concatenated words in the window
	 */
	public SimpleMatrix toInputVector(List<String> window) {
		SimpleMatrix X = new SimpleMatrix(windowSize*wordSize +1, 1); // The first row = 1, to deal with bias
		int row = 0;
		X.set(row++, 0, 1.0);
		for (String word : window) {
			String key = word.toLowerCase();
			if (!FeatureFactory.wordToNum.containsKey(key)) {
				key = "UUUNKKK";
			}
			int index = FeatureFactory.wordToNum.get(key);
			SimpleMatrix wordVec = L.extractVector(false,
					index); // extract column vector at row=index
			X.insertIntoThis(row, 0, wordVec);
			row += wordVec.getNumElements(); // increment row to point to next
												// slot to insert
		}
		return X;
	}
        
        public SimpleMatrix toInputVector2(List<String> window) {
		SimpleMatrix X = new SimpleMatrix(windowSize*wordSize +1, 1); // The first row = 1, to deal with bias
  		int row = 0;
		X.set(row++, 0, 1.0);
		for (String word : window) {
			String key = word.toLowerCase();
			if (!FeatureFactory.wordToNum.containsKey(key)) {
				key = "UUUNKKK";
			}
			int index = FeatureFactory.wordToNum.get(key);
			SimpleMatrix wordVec = L.extractVector(false,
					index); // extract column vector at row=index
			X.insertIntoThis(row, 0, wordVec);
			row += wordVec.getNumElements(); // increment row to point to next
												// slot to insert
		}
		return X;
	}
	/**
	 * 
	 * @param window input words in the window
	 * @param X updated word Vector for the words in window
	 * updates the allVecs vector with these word vectors
	 */
	public void updateWordVecInLookup(List<String> window, SimpleMatrix X) {
		for (int i=0; i< window.size(); i++) {
			String word = window.get(i);
			// get the wordVec for this word in X
			SimpleMatrix wordVec = X.extractMatrix(1+wordSize*i, 1+wordSize*(i+1), 0, 1);
			String key = word.toLowerCase();
			if (!FeatureFactory.wordToNum.containsKey(key)) {
				key = "UUUNKKK";
			}
			int index = FeatureFactory.wordToNum.get(key);
			// update allVecs matrix for this word
			L.insertIntoThis(0, index, wordVec);
		}
	}
	/**
	 * 
	 * @param mat
	 * @return tanh of elements of mat
	 */
	public static SimpleMatrix tanh(SimpleMatrix mat) {
		SimpleMatrix tanhMat = new SimpleMatrix(mat.numRows(), mat.numCols());
		for (int i=0; i<mat.numRows(); i++) {
			for (int j=0; j<mat.numCols(); j++) {
				tanhMat.set(i, j, Math.tanh(mat.get(i, j)));
			}
		}
		return tanhMat;
	}
	/**
	 * 
	 * @param mat input matrix
	 * @return derivative of tanh of each element
	 */
	public static SimpleMatrix tanhPrime(SimpleMatrix mat) {
		SimpleMatrix tanhPrimeMat = new SimpleMatrix(mat.numRows(), mat.numCols());
		for (int i=0; i<mat.numRows(); i++) {
			for (int j=0; j<mat.numCols(); j++) {
				double x = Math.tanh(mat.get(i, j));
				x = x*x;
				tanhPrimeMat.set(i, j, 1.0 - x);
			}
		}
		return tanhPrimeMat;
	}
	/**
	 * 
	 * @param mat input matrix
	 * @return softmax of mat elements
	 */
	public static SimpleMatrix softmax(SimpleMatrix mat) {
		SimpleMatrix softmaxMat = new SimpleMatrix(mat);
		double max = mat.get(0,0);
		for (int row=0; row < softmaxMat.numCols(); row++) {
			for (int col=0; row < softmaxMat.numRows(); row++) {
				if (softmaxMat.get(row, col) > max) {
					max = softmaxMat.get(row, col);
				}
			}
		}
		for (int row=0; row < softmaxMat.numCols(); row++) {
			for (int col=0; row < softmaxMat.numRows(); row++) {
				softmaxMat.set(row, col, Math.exp(softmaxMat.get(row, col)-max));
			}
			//System.out.println("sum = " + sum);
		}
		double sum = softmaxMat.elementSum();
		return softmaxMat.scale(1.0/sum);
	}
	/**
	 * 
	 * @param vec vector that will be extended [1 vec]
	 * @return extended vector
	 */
	public SimpleMatrix extendVector(SimpleMatrix vec) {
		SimpleMatrix extendedVec = new SimpleMatrix(vec.numRows() + 1, vec.numCols());
		extendedVec.set(0, 0, 1.0);
		extendedVec.insertIntoThis(1, 0, vec);
		return extendedVec;
	}
	/**
	 * 
	 * @param X input to NN
	 * @return output vector of NN from forward pass
	 */
	public SimpleMatrix feedForward(SimpleMatrix X) {
		SimpleMatrix z1 = W.mult(X);
		SimpleMatrix a = extendVector(tanh(z1));
		SimpleMatrix p = softmax(U.mult(a));
		return p;
	}
	/**
	 * 
	 * @param out output of NN
	 * @return label corresponding to the max value in out
	 */
	public String getOutput(SimpleMatrix out) {
		double max = -1.0;
		int index = -1;
		for (int i=0; i<out.numRows(); i++) {
			if (out.get(i, 0) > max) {
				max = out.get(i, 0);
				index = i;
			}
		}
		return indexToLabel.get(index);
	}
	/**
	 * 
	 * @param Y gold vector (one-hot)
	 * @param p NN output vector
	 * @return avg -ve log loss
	 */
	public double logloss(SimpleMatrix Y, SimpleMatrix p) {
		double loss = 0.0;
		int m = Y.numCols();
		for (int k=0; k<m; k++) {
			for (int i=0; i<Y.numRows(); i++) {
				loss -= Y.get(i, k) * Math.log(p.get(i, k));
			}
		}
		loss = (1.0 / m) * loss;
		return loss;
	}
	
	/**
	 * 
	 * @return regularized cost
	 */
	public double regularizedCost() {
		// add regularization cost
		rU = zeroFirstColumn(U);
		rW = zeroFirstColumn(W);
		double r = SpecializedOps.elementSumSq(rW.getMatrix());
		r += SpecializedOps.elementSumSq(rU.getMatrix());
		return C * r / 2.0;
	}
	
	/**
	 * 
	 * @param Y gold vector (one-hot)
	 * @param p NN output vector
	 * @return logloss + regularized cost
	 */
	public double cost(SimpleMatrix Y, SimpleMatrix p) {
		return logloss(Y, p) + regularizedCost();
	}
	
	/**
	 * 
	 * @param mat weight matrix with bias as first column
	 * @return matrix with the bias column zeroed out
	 */
	private SimpleMatrix zeroFirstColumn(SimpleMatrix mat) {
		SimpleMatrix retMat = new SimpleMatrix(mat);
		double[] zeros = new double[mat.numRows()];
		retMat.setColumn(0, 0, zeros);
		return retMat;
	}


	/**
	 * Simplest SGD training 
	 */
	public void train(List<List<Datum>> _trainData ) {
		List<List<String>> inputWindows = new ArrayList<List<String>>();
		List<String> labels = new ArrayList<String>();
		for (List<Datum> sentence: _trainData) {
			List<String> paddedSentence = pad(sentence);
			inputWindows.addAll(window(paddedSentence));
			labels.addAll(getLabels(sentence));
		}
		for (int epoch=0; epoch<Epochs; epoch++) {
			System.out.println("Epoch = " + epoch);
			Random seed = new Random(System.nanoTime());
                        //Random seed = new Random(10);
			Collections.shuffle(inputWindows, seed);
			Collections.shuffle(labels, seed);
			double j = 0.0;
			int TP = 0, FP = 0, FN = 0;
                        
                        int numWindows = inputWindows.size();
                        int numBatches = (int) Math.ceil(numWindows * 1.0/batchSize);
                        
                        System.out.println("numWindows = " + numWindows);
                        System.out.println("numBatches = " + numBatches);
                        
                        int windowId = 0;
                        int actBatchSize;
                        SimpleMatrix dJdUAvg = new SimpleMatrix(U.numRows(),U.numCols());
                        SimpleMatrix dJdWAvg = new SimpleMatrix(W.numRows(),W.numCols());
                        
                        for (int k = 0; k<numBatches;k++){
                            actBatchSize = batchSize;
                            if (k == (numBatches - 1) && (numWindows%batchSize)>0){
                                actBatchSize = numWindows % batchSize;
                            }
                                 for (int i=0; i<actBatchSize; i++) {
                                        String label = labels.get(windowId);
                                        List<String> window = inputWindows.get(windowId);
                                        SimpleMatrix Y = getLabelMatrix(label);
                                        SimpleMatrix X = toInputVector(window);
                                        // feed forward
                                        SimpleMatrix z1 = W.mult(X);
                                        SimpleMatrix a = extendVector(tanh(z1));
                                        SimpleMatrix p = softmax(U.mult(a));
                                        
                                        if ( k%(numBatches/10) == 0 && i < 10){
                                           System.out.println("windowId = "+ windowId);
                                           System.out.println("p = " + p);
                                           System.out.println("j = " + logloss(Y,p));
                                        }
                                        // calculate gradients
                                        SimpleMatrix delta2 = p.minus(Y);
                                        //System.out.println("delta2 = " + delta2);
                                        
                                        SimpleMatrix dJdU = delta2.mult(a.transpose());
                                        
                                        // calculate delta1
                                        SimpleMatrix delta1 = U.transpose().mult(delta2);
                                        delta1 = delta1.extractMatrix(1, SimpleMatrix.END, 0, SimpleMatrix.END); // remove row 0 (bias part)
                                        delta1 = delta1.elementMult(tanhPrime(z1)); // dJdZ1
                                        //
                                        SimpleMatrix dJdW = delta1.mult(X.transpose());
                                        
                                        //
                                        SimpleMatrix dJdX = W.transpose().mult(delta1);
                                        //average dJdU and dJdW
                                        dJdUAvg = dJdUAvg.plus(dJdU.scale(1.0/actBatchSize));
                                        dJdWAvg = dJdWAvg.plus(dJdW.scale(1.0/actBatchSize));
                                        // ***********************
                                        //gradient check
                                        if (checkGradient) {
                                                List<SimpleMatrix> weights = new ArrayList<SimpleMatrix>();
                                                weights.add(U);
                                                weights.add(W);
                                                weights.add(X);
                                                List<SimpleMatrix> matrixDerivatives = new ArrayList<SimpleMatrix>();
                                                matrixDerivatives.add(dJdU);
                                                matrixDerivatives.add(dJdW);
                                                matrixDerivatives.add(dJdX);
                                                boolean check =
                                                                GradientCheck.check(Y, weights, matrixDerivatives, this);
                                                System.out.println("check: " + check);
                                                //if (!check) System.exit(-1);
                                        }
                                        windowId++;
                                        //update weights for X
                                        X = X.minus(dJdX.scale(lr));
                                        updateWordVecInLookup(window, X);
                                }
                                        // ************************
                                         //regularize 
                                        rU = zeroFirstColumn(U);
                                        rW = zeroFirstColumn(W);
                                        dJdUAvg = dJdUAvg.plus(rU.scale(C));
                                        dJdWAvg = dJdWAvg.plus(rW.scale(C));
                                        //update weights for U and W
                                        U = U.minus(dJdUAvg.scale(lr));
                                        W = W.minus(dJdWAvg.scale(lr));
                                        
                        }
                         j = 0.0;   
			for (int i=0; i<inputWindows.size(); i++) {
				String label = labels.get(i);
				List<String> window = inputWindows.get(i);
				SimpleMatrix Y = getLabelMatrix(label);
				SimpleMatrix X = toInputVector(window);
				SimpleMatrix p = feedForward(X);
				String output = getOutput(p);
                                 if (!output.equals("O") && output.equals(label)) TP++;
                                if (!output.equals("O") && !output.equals(label)) FP++;
                                if (!output.equals(label) && label.equals("O")) FN++;

				j += logloss(Y, p)/inputWindows.size();
                                
                                if (i%(numWindows/10)==0){
                                    System.out.println("label = " + label);
                                    System.out.println("j = " + logloss(Y,p));
                                    p.print();
                                }
			}
                        
                        //evaluate performance for each epoch
			int TN = numWindows - TP - FP - FN;
			j += regularizedCost();
                        
                        //evaluate
                        double accuracy = (TP+TN) * 1.0/numWindows;
                        double precision = TP * 1.0/(TP+FP);
                        double recall = TP * 1.0/(TP + FN);
                        double f1 = 2*(precision * recall)/(precision * recall);
                      
			System.out.println("Total Cost = " + j);
			System.out.println("Accuracy = " +  accuracy +", Precision = " + precision +", Recall = " + recall + ", F1 = " + f1 );
                        System.out.println("actual phrases = " + (TP + FN) + ", found phrases = " + (TP+FP) + ", correct phrases = " + TP);

		}

	}

	
	public void test(List<List<Datum>> testData) throws IOException {
		FileWriter fw = new FileWriter("nn1.out");
		for (List<Datum> sentence : testData) {
			List<String> paddedSentence = pad(sentence);
			List<List<String>> windows = window(paddedSentence);
			for (int i=0; i<windows.size(); i++) {
				List<String> window = windows.get(i);
				SimpleMatrix X = toInputVector(window);
				String label = sentence.get(i).label;
				String word = sentence.get(i).word;
				SimpleMatrix p = feedForward(X);
				String output = getOutput(p);
				fw.write(word + "\t" + label + "\t" + output + "\n");
                                
                                if (i%(windows.size()/10) == 0){
                                  System.out.println("label = " + label);
                                  p.print();
                                  
                                }
			}
		}
		fw.close();
	}

	@Override
	public double valueAt(SimpleMatrix label, SimpleMatrix input) {
		SimpleMatrix p = feedForward(input);
		return logloss(label, p);
	}
	
}