package cs224n.deep;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.ops.SpecializedOps;
import org.ejml.simple.*;

import java.text.*;

public class WindowModel_Deeper implements ObjectiveFunction {

	protected SimpleMatrix L, W1, W2, U, rW1, rW2,rU;
	//
	protected int windowSize,wordSize, hiddenSize1, hiddenSize2, outputNodes;
	protected double lr;
	protected static Map<String, Integer> labelToIndex = new HashMap<String, Integer>();
	protected static Map<Integer, String> indexToLabel = new HashMap<Integer, String>();
	private double C = 0.001;
	private boolean checkGradient;
	private int Epochs;
	
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
	
	public WindowModel_Deeper(int _windowSize, int _hiddenSize1, int _hiddenSize2, double _lr, double _C, int epochs, boolean check){
		//TODO
		windowSize = _windowSize; // assuming odd window size
		hiddenSize1 = _hiddenSize1;
                hiddenSize2 = _hiddenSize2;
		wordSize = 50; // 50 dim word vector
		outputNodes = 5; // number of NER outputs
		lr = _lr;
		C = _C;
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
                W2 = initRandom(hiddenSize2, hiddenSize1 + 1);
		W1= initRandom(hiddenSize1, windowSize*wordSize + 1); // first column is bias
		U = initRandom(outputNodes, hiddenSize2+1); // first column is bias
		// change bias columns to 0
		double[] zeros = new double[W1.numRows()];
		W1.setColumn(0, 0, zeros);
                zeros = new double[W2.numRows()];
                W2.setColumn(0,0,zeros);
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
		SimpleMatrix z1 = W1.mult(X);
		SimpleMatrix a1 = extendVector(tanh(z1));
                SimpleMatrix z2 = W2.mult(a1);
                SimpleMatrix a2 = extendVector(tanh(z2));
		SimpleMatrix p = softmax(U.mult(a2));
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
		rW2 = zeroFirstColumn(W2);
                rW1 = zeroFirstColumn(W1);
		double r = SpecializedOps.elementSumSq(rW1.getMatrix());
                r += SpecializedOps.elementSumSq(rW2.getMatrix());
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

        private void printStats(List<List<String>> inputWindows, List<String>labels){
          int TP = 0, FP = 0, FN = 0, T = 0;
          double j = 0.0;   
          int numWindows = inputWindows.size();
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
                  if (!label.equals("O")) T++;
                  j += logloss(Y, p)/inputWindows.size();
                 /*
                  if (i%(numWindows/10)==0){
                      System.out.println("label = " + label);
                      System.out.println("j = " + logloss(Y,p));
                      p.print();
                  }
                 */
          }

          j +=  regularizedCost();
          j = Math.round(j*1000)/1000.0;

          //evaluate performance for each epoch
          int TN = numWindows - T - FN;

          double accuracy = Math.round((TP+TN) * 1.0/numWindows * 1000.0)/1000.0;
          double precision = Math.round(TP * 1.0/(TP+FP)*1000)/1000.0;
          double recall = Math.round(TP * 1.0/T*1000)/1000.0;
          double f1 = Math.round(2*(precision * recall)/(precision + recall)*1000)/1000.0;
          
          System.out.println("actual phrases = " + T + ", found phrases = " + (TP+FP) + ", correct phrases = " + TP);
          System.out.println("Accuracy = " +  accuracy +", Precision = " + precision +", Recall = " + recall + ", F1 = " + f1 );
          System.out.println("Average Cost = " + j);
          
        }   
	/**
	 * Simplest SGD training 
	 */
	public void train(List<List<Datum>> _trainData ) {
		List<List<String>> inputWindows = new ArrayList<List<String>>();
		List<String> labels = new ArrayList<String>();
                List<Integer> trainIndices = new ArrayList<Integer>();

		for (List<Datum> sentence: _trainData) {
			List<String> paddedSentence = pad(sentence);
			inputWindows.addAll(window(paddedSentence));
			labels.addAll(getLabels(sentence));
		}
                
                for (int i = 0; i < inputWindows.size(); i++) {
			trainIndices.add(i);
		}
		int fails = 0;
                
		for (int epoch=0; epoch<Epochs; epoch++) {
			System.out.println("Epoch = " + epoch);
			Random seed = new Random(System.nanoTime());
 			Collections.shuffle(trainIndices, seed);
			
			for (int i=0; i<trainIndices.size(); i++) {
				int ind = trainIndices.get(i);
				String label = labels.get(ind);
				List<String> window = inputWindows.get(ind);
				SimpleMatrix Y = getLabelMatrix(label);
				SimpleMatrix X = toInputVector(window);
				// feed forward
				SimpleMatrix z1 = W1.mult(X);
                                SimpleMatrix a1 = extendVector(tanh(z1));
                                SimpleMatrix z2 = W2.mult(a1);
                                SimpleMatrix a2 = extendVector(tanh(z2));
                                SimpleMatrix p = softmax(U.mult(a2));   
				//System.out.println("p = " + p);
				// loss
				// j += logloss(Y, p);
				//System.out.println("J = " + J);

				// calculate gradients
				SimpleMatrix delta3 = p.minus(Y);
				//System.out.println("delta2 = " + delta2);
				rU = zeroFirstColumn(U);
                                rW2 = zeroFirstColumn(W2); 
				rW1 = zeroFirstColumn(W1);
                                
				SimpleMatrix dJdU = delta3.mult(a2.transpose());
				dJdU = dJdU.plus(rU.scale(C));
				// calculate delta2
				SimpleMatrix delta2 = U.transpose().mult(delta3);
				delta2 = delta2.extractMatrix(1, SimpleMatrix.END, 0, SimpleMatrix.END); // remove row 0 (bias part)
				delta2 = delta2.elementMult(tanhPrime(z2)); // dJdZ1
				SimpleMatrix dJdW2 = delta2.mult(a1.transpose());
				dJdW2 = dJdW2.plus(rW2.scale(C));
                                
                                // calculate delta1
                                SimpleMatrix delta1 = W2.transpose().mult(delta2);
                                delta1 = delta1.extractMatrix(1, SimpleMatrix.END, 0, SimpleMatrix.END);    
                                delta1 = delta1.elementMult(tanhPrime(z1));
                                SimpleMatrix dJdW1 = delta1.mult(X.transpose());
                                dJdW1 = dJdW1.plus(rW1.scale(C));
				SimpleMatrix dJdX = W1.transpose().mult(delta1);
				// ***********************
				//gradient check
				if (checkGradient) {
					List<SimpleMatrix> weights = new ArrayList<SimpleMatrix>();
					weights.add(U);
					weights.add(W2);
                                        weights.add(W1);
					weights.add(X);
					List<SimpleMatrix> matrixDerivatives = new ArrayList<SimpleMatrix>();
					matrixDerivatives.add(dJdU);
					matrixDerivatives.add(dJdW2);
                                        matrixDerivatives.add(dJdW1);
					matrixDerivatives.add(dJdX);
					boolean check =
							GradientCheck.check(Y, weights, matrixDerivatives, this);
					 if (!check) {
						fails++;
						System.out.println("check failed for " + window);
					}
				}
				// ************************
				// update weights
				U = U.minus(dJdU.scale(lr));
                                W2 = W2.minus(dJdW2.scale(lr)); 
				W1 = W1.minus(dJdW1.scale(lr));
				X = X.minus(dJdX.scale(lr));
				updateWordVecInLookup(window, X);
			}
			//eval the performance
                        printStats(inputWindows,labels);

		}

	}

	
	public void test(List<List<Datum>> testData) throws IOException {
		FileWriter fw = new FileWriter("output/nn1_deeper.out");
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
				fw.write(word + "\t" + label + "\t" + output +"\n");
                           
			}
		}
                
                //eval the performance
                List<List<String>> inputWindows = new ArrayList<List<String>>();
		List<String> labels = new ArrayList<String>();
 		for (List<Datum> sentence: testData) {
			List<String> paddedSentence = pad(sentence);
			inputWindows.addAll(window(paddedSentence));
			labels.addAll(getLabels(sentence));
		}
		
                //eval performance for test data
                System.out.println("");
                System.out.println("test data evaluation");
                printStats(inputWindows,labels);
                
		fw.close();
	}

	@Override
	public double valueAt(SimpleMatrix label, SimpleMatrix input) {
		SimpleMatrix p = feedForward(input);
		return cost(label, p);
	}
        
        
        public void saveWordVectors() throws IOException {
		String outfile = "output/newVectors.txt";
		BufferedWriter bw = new BufferedWriter(new FileWriter(outfile));
		for (int col=0; col<L.numCols(); col++) {
			StringBuilder sb = new StringBuilder();
			for (int row=0; row<L.numRows(); row++) {
				sb.append(L.get(row, col));
				sb.append(" ");
			}
			sb.append("\n");
			bw.write(sb.toString());
		}
	}
	
}