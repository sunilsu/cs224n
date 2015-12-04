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

public class WindowModel_Others implements ObjectiveFunction {

	protected SimpleMatrix L, W, U, rW, rU;
	//
	protected int windowSize,wordSize, hiddenSize, outputNodes;
	protected double lr, lr_initial;
	protected static Map<String, Integer> labelToIndex = new HashMap<String, Integer>();
	protected static Map<Integer, String> indexToLabel = new HashMap<Integer, String>();
	private double C = 0.001;
	private boolean checkGradient;
	private int Epochs;
	protected int vecLen ;
        
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
	
	public WindowModel_Others(int _windowSize, int _hiddenSize, double _lr, double _C, int epochs, boolean check){
		//TODO
		windowSize = _windowSize; // assuming odd window size
		hiddenSize = _hiddenSize;
		wordSize = 50; // 50 dim word vector
		outputNodes = 5; // number of NER outputs
		lr = _lr;
		lr_initial = _lr;
		C = _C;
		Epochs = epochs;
		checkGradient = check;
                vecLen= windowSize * (wordSize+0) + 0 + 1;
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
		W = initRandom(hiddenSize, vecLen); // first column is bias
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
	public SimpleMatrix labelToOneHot(String label) {
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
		SimpleMatrix X = new SimpleMatrix(vecLen, 1); // The first row = 1, to deal with bias
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
         * Add cap index for the center word
         * @param window
         * @return 
         */
        public SimpleMatrix toInputVector_capInd(List<String> window) {
		SimpleMatrix X = new SimpleMatrix(vecLen, 1); // The first row = 1, to deal with bias
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
                String centerWord = window.get(window.size()/2+1);
                int cap = (centerWord.equals(centerWord.toLowerCase())) ? 0 : 1;
                SimpleMatrix capMat = new SimpleMatrix(1,1);
                capMat.set(0,0,cap);
                X.insertIntoThis(row, 0, capMat) ;
                row++; // increment row to point to next
                
		return X;
	}
	/**
	 * Add a capitalized indicator for each word
	 */
        
        public SimpleMatrix toInputVector_CapInd2(List<String> window) {
		SimpleMatrix X = new SimpleMatrix(vecLen, 1); // The first row = 1, to deal with bias
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
                        row += wordVec.getNumElements() ;
                        
                        int cap = (word.equals(word.toLowerCase())) ? 0 : 1;
                        SimpleMatrix capMat = new SimpleMatrix(1,1);
                        capMat.set(0,0,cap);
                        X.insertIntoThis(row, 0, capMat) ;
			row++; // increment row to point to next
												// slot to insert
		}
		return X;
	}
        /**
         * Add a missing indicator for the center word
         */
        
         public SimpleMatrix toInputVector_MissingInd(List<String> window) {
		SimpleMatrix X = new SimpleMatrix(vecLen, 1); // The first row = 1, to deal with bias
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
                String centerWord = window.get(window.size()/2+1);
                
                String key = centerWord.toLowerCase();
                int miss = 0;
			if (!FeatureFactory.wordToNum.containsKey(key)) {
				miss = 1;
			}
                SimpleMatrix missMat = new SimpleMatrix(1,1);
                missMat.set(0,0,miss);
                X.insertIntoThis(row, 0, missMat) ;
                row++; // increment row to point to next
                
		return X;
	}
         
         
         /**
         * Add both index for the center word
         * @param window
         * @return 
         */
        public SimpleMatrix toInputVector_capMissInd(List<String> window) {
		SimpleMatrix X = new SimpleMatrix(vecLen, 1); // The first row = 1, to deal with bias
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
                String centerWord = window.get(window.size()/2+1);
                //add cap indicator
                int cap = (centerWord.equals(centerWord.toLowerCase())) ? 0 : 1;
                SimpleMatrix capMat = new SimpleMatrix(1,1);
                capMat.set(0,0,cap);
                X.insertIntoThis(row, 0, capMat) ;
                row++; // increment row to point to next                
                
                //add missing indicator
                String key = centerWord.toLowerCase();
                int miss = 0;
			if (!FeatureFactory.wordToNum.containsKey(key)) {
				miss = 1;
			}
                SimpleMatrix missMat = new SimpleMatrix(1,1);
                missMat.set(0,0,miss);
                X.insertIntoThis(row, 0, missMat) ;
                row++; // increment row to point to next
                
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
			//SimpleMatrix before = L.extractVector(false, index);
			L.insertIntoThis(0, index, wordVec);
			//SimpleMatrix after = L.extractVector(false, index);
			//System.out.println(after.minus(before));
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
				tanhPrimeMat.set(i, j, 1.0 - x*x);
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
		SimpleMatrix a = tanh(z1);
		a = extendVector(a); // a = [1 a]'
		SimpleMatrix z2 = U.mult(a);
		SimpleMatrix p = softmax(z2);
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
				if (Y.get(i, k) > 0) {
					double p_ik = Math.max(p.get(i, k), 1E-7);
					loss -= Y.get(i, k) * Math.log(p_ik);
				}
			}
		}
		loss = loss / m;
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
		double lloss = logloss(Y,p);
		double rloss = regularizedCost();
		//System.out.println("lloss = " + lloss + ", rloss = " + rloss);
		return lloss + rloss;
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
                  SimpleMatrix Y = labelToOneHot(label);
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
			// shuffle inputs
			Random seed = new Random(System.nanoTime());
			Collections.shuffle(trainIndices, seed);
			
			for (int i=0; i<trainIndices.size(); i++) {
				int ind = trainIndices.get(i);
				String label = labels.get(ind);
				List<String> window = inputWindows.get(ind);
				SimpleMatrix Y = labelToOneHot(label);
				SimpleMatrix X = toInputVector(window);
				// feed forward
				SimpleMatrix z1 = W.mult(X);
				SimpleMatrix a = tanh(z1);
				a = extendVector(a); // a = [1 a]'
				SimpleMatrix z2 = U.mult(a);
				SimpleMatrix p = softmax(z2);
				//System.out.println("p = " + p);
				// loss
				//double J = cost(Y, p);
				//System.out.println("J before = " + J);

				// calculate gradients
				SimpleMatrix delta2 = p.minus(Y); // d2 = [p -Y]
				//System.out.println("delta2 = " + delta2);
				rU = zeroFirstColumn(U); // rU = [0 U] , zero bias column
				rW = zeroFirstColumn(W); // rW = [0 W]
				SimpleMatrix dJdU = delta2.mult(a.transpose()); // dU = d2 * a'
				dJdU = dJdU.plus(rU.scale(C)); // dU = dU + C * [0 U]
				// calculate delta1
				SimpleMatrix delta1 = U.transpose().mult(delta2); // d1 = U' * d2
				// d1 = d1[2:, :]
				delta1 = delta1.extractMatrix(1, SimpleMatrix.END, 0, SimpleMatrix.END); // remove row 0 (bias part)
				SimpleMatrix dadz1 = tanhPrime(z1); // dadz1 = (1 - (tanh(z1)**2)
				delta1 = delta1.elementMult(dadz1); // d1 = (U' * d2)[2:] .* dadz1
				//
				SimpleMatrix dJdW = delta1.mult(X.transpose()); // d1 * X'
				dJdW = dJdW.plus(rW.scale(C)); // dW = dW + C * [0 W]
				//
				SimpleMatrix dJdX = W.transpose().mult(delta1); // dX = W' * d1
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
					if (!check) {
						fails++;
						System.out.println("check failed for " + window);
					}
				}
				// ************************
				// update weights
				U = U.minus(dJdU.scale(lr));
				W = W.minus(dJdW.scale(lr));
				//X = X.minus(dJdX.scale(lr));
				//updateWordVecInLookup(window, X);

				// loss
				//p = feedForward(X);
				//J = J - cost(Y, p);
				//System.out.println("J reduced by  = " + J);
			}
			// check accuracy on training set
			//printStats(inputWindows, labels);
			//lr = lr*0.9;
		}
		if (checkGradient) 	System.out.println("Total checks failed: " + fails);
	}

	
	public void test(List<List<Datum>> testData) throws IOException {
		FileWriter fw = new FileWriter("output/nn_notupdateX_w" + windowSize + "_h" + hiddenSize + "_lr" + lr + "_c" + C +"_e" + Epochs +  ".out");
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
		
                System.out.println("");
                System.out.println("evaluate test data performance");
                printStats(inputWindows,labels);
                
		fw.close();
	}

	@Override
	public double valueAt(SimpleMatrix label, SimpleMatrix input) {
		SimpleMatrix p = feedForward(input);
		return cost(label, p);
	}
        
        public void saveWordVectors() throws IOException {
		String outfile = "data/newVectors.txt";
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