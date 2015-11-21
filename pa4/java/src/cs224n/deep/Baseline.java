package cs224n.deep;

import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Baseline {
	
	Map<String, String> neMap = new HashMap<String, String>();
	Set<String> amb = new HashSet<String>();

	public static void main(String[] args) throws IOException {
		List<Datum> trainData = FeatureFactory.readTrainData("data/train");
		List<Datum> testData = FeatureFactory.readTestData("data/dev");
		Baseline bl = new Baseline();
		bl.train(trainData);
		bl.test(testData);
	}
	
	public void train(List<Datum> trainData ){
		for (Datum datum : trainData) {
			if (neMap.containsKey(datum.word)) {
				if (!datum.label.equals(neMap.get(datum.word))) {
					neMap.remove(datum.word);
					amb.add(datum.word);
				}
			}
			else if (!amb.contains(datum.word)) {
				neMap.put(datum.word, datum.label);
			}
		}
	}

	
	public void test(List<Datum> testData) throws IOException {
		FileWriter fw = new FileWriter("baseline.out");
		for (Datum datum : testData) {
			if (neMap.containsKey(datum.word)) {
				fw.write(datum.word + "\t" + datum.label + "\t" + neMap.get(datum.word) + "\n");
			}
			else {
				fw.write(datum.word + "\t" + datum.label + "\tO" + "\n");
			}
		}
		fw.close();
	}

}
