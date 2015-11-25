package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;

import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, W, U;
	//
	protected int windowSize,wordSize, hiddenSize, outputNodes;
	protected double lr;
	protected static Map<String, Integer> labelToIndex = new HashMap<String, Integer>();
	private double C = 0.01;
	
	static {
		labelToIndex.put("O", 0);
		labelToIndex.put("LOC", 1);
		labelToIndex.put("MISC", 2);
		labelToIndex.put("ORG", 3);
		labelToIndex.put("PER", 4);
	}
	
	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
		//TODO
		windowSize = _windowSize; // assuming odd window size
		hiddenSize = _hiddenSize;
		wordSize = 50; // 50 dim word vector
		outputNodes = 5; // number of NER outputs
		lr = _lr;		
	}
	
	private SimpleMatrix initRandom(int rows, int cols) {
		double eInit = Math.sqrt(6.0) / Math.sqrt(cols + rows);// fanIn = cols, fanOut = rows
		return SimpleMatrix.random(rows, cols, -eInit, eInit, new Random());
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
			SimpleMatrix wordVec = FeatureFactory.allVecs.extractVector(false,
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
			FeatureFactory.allVecs.insertIntoThis(0, index, wordVec);
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
				double x = Math.pow(Math.tanh(mat.get(i, j)), 2.0);
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
		SimpleMatrix softmaxMat = new SimpleMatrix(mat.numRows(), mat.numCols());
		for (int col=0; col < mat.numCols(); col++) {
			double sum = 0.0;
			for (int row=0; row < mat.numRows(); row++) {
				softmaxMat.set(row, col, Math.exp(mat.get(row, col)));
				sum += softmaxMat.get(row, col);
			}
			for (int row=0; row < mat.numRows(); row++) {
				softmaxMat.set(row, col, softmaxMat.get(row, col) / sum);
			}
		}
		return softmaxMat;
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

	public SimpleMatrix feedForward(SimpleMatrix X) {
		SimpleMatrix a1 = tanh(W.mult(X));
		// a1 is a vector, extend by 1 row to account for bias
		SimpleMatrix a = extendVector(a1);
		return softmax(U.mult(a));		
	}
	
	public double logLoss(SimpleMatrix Y, SimpleMatrix p) {
		double loss = 0.0;
		for (int i=0; i<Y.numRows(); i++) {
			loss -= Y.get(i, 0) * Math.log(p.get(i, 0));
		}
		return loss;
	}
	
	private SimpleMatrix zeroFirstColumn(SimpleMatrix mat) {
		SimpleMatrix retMat = mat.copy();
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
		Random seed = new Random(System.nanoTime());
		Collections.shuffle(inputWindows, seed);
		Collections.shuffle(labels, seed);
		for (int i=0; i<inputWindows.size(); i++) {
			List<String> window = inputWindows.get(i);
			SimpleMatrix Y = getLabelMatrix(labels.get(i));
			SimpleMatrix X = toInputVector(window);
			// feed forward
			SimpleMatrix z1 = W.mult(X);
			SimpleMatrix a = extendVector(tanh(z1));
			SimpleMatrix p = softmax(U.mult(a));
			// loss
			double J = logLoss(Y, p);
			// calculate gradients
			SimpleMatrix delta2 = p.minus(Y);
			SimpleMatrix rU = zeroFirstColumn(U);
			SimpleMatrix dJdU = p.mult(a.transpose()).plus(rU.scale(C));
			// calculate delta1
			SimpleMatrix delta1 = U.transpose().mult(delta2);
			delta1 = delta1.extractMatrix(1, SimpleMatrix.END, 0, SimpleMatrix.END); // remove row 0 (bias part)
			delta1 = delta1.elementMult(tanhPrime(z1)); // dJdZ1
			//
			SimpleMatrix rW = zeroFirstColumn(W);
			SimpleMatrix dJdW = delta1.mult(X.transpose()).plus(rW.scale(C));;
			//
			SimpleMatrix dJdX = W.transpose().mult(delta1);
			// update weights
			U = U.minus(dJdU.scale(lr));
			W = W.minus(dJdW.scale(lr));
			X = X.minus(dJdX.scale(lr));
			updateWordVecInLookup(window, X);
		}

	}

	
	public void test(List<Datum> testData){
		// TODO
	}
	
}