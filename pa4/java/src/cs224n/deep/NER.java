package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NER {
    
    public static void main(String[] args) throws IOException {
	if (args.length < 2) {
	    System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev");
	    return;
	}	    

	// this reads in the train and test datasets
	//List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
	//List<Datum> testData = FeatureFactory.readTestData(args[1]);
	DataIterator iter = new DataIterator(args[0]);
	List<List<Datum>> trainData = iter.getSentences();
	iter = new DataIterator(args[1]);
	List<List<Datum>> testData = iter.getSentences();
	
	//	read the train and test data
	//TODO: Implement this function (just reads in vocab and word vectors)
	FeatureFactory.initializeVocab("data/vocab.txt");
	FeatureFactory.readWordVectors("data/wordVectors.txt");

	// initialize model 
	WindowModel model = new WindowModel(5, 100, 0.01, 0.001);
	model.initWeights();

	model.train(trainData);
	model.test(testData);
    }
}