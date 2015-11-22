package cs224n.deep;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import org.ejml.simple.*;


public class FeatureFactory {


	private FeatureFactory() {

	}

	 
	static List<Datum> trainData;
	/** Do not modify this method **/
	public static List<Datum> readTrainData(String filename) throws IOException {
        if (trainData==null) trainData= read(filename);
        return trainData;
	}
	
	static List<Datum> testData;
	/** Do not modify this method **/
	public static List<Datum> readTestData(String filename) throws IOException {
        if (testData==null) testData= read(filename);
        return testData;
	}
	
	private static List<Datum> read(String filename)
			throws FileNotFoundException, IOException {
	    // TODO: you'd want to handle sentence boundaries
		List<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(filename));
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
				continue;
			}
			String[] bits = line.split("\\s+");
			String word = bits[0];
			String label = bits[1];

			Datum datum = new Datum(word, label);
			data.add(datum);
		}

		return data;
	}
 
 
	// Look up table matrix with all word vectors as defined in lecture with dimensionality n x |V|
	static SimpleMatrix allVecs; //access it directly in WindowModel
	public static SimpleMatrix readWordVectors(String vecFilename) throws IOException {
		if (allVecs!=null) return allVecs;
		//set allVecs from filename
		BufferedReader br = new BufferedReader(new FileReader(vecFilename));
		String line = null;
		List<String[]> vecList = new ArrayList<String[]>();
		while ((line = br.readLine()) != null) {
			if (line.trim().length() == 0) {
                continue;
			}
			String[] vectors = line.split("\\s+");
			vecList.add(vectors);
		}
		int COLS = vecList.size();
		int ROWS = vecList.get(0).length;
		allVecs = new SimpleMatrix(ROWS, COLS);
		for (int col=0; col<COLS; col++) {
			String[] vectors = vecList.get(col);
			for (int row=0; row<ROWS; row++) {
				allVecs.set(row, col, Double.valueOf(vectors[row]));
			}
			
		}
		return allVecs;
	}
	// might be useful for word to number lookups, just access them directly in WindowModel
	public static HashMap<String, Integer> wordToNum = new HashMap<String, Integer>(); 
	public static HashMap<Integer, String> numToWord = new HashMap<Integer, String>();

	public static HashMap<String, Integer> initializeVocab(String vocabFilename) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(vocabFilename));
		String line = null;
		int num = 0;
		while ((line = br.readLine()) != null) {
			wordToNum.put(line, num);
			numToWord.put(num, line);
			num++;
		}
		return wordToNum;
	}
}
