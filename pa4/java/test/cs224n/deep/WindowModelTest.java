package cs224n.deep;

import static org.junit.Assert.*;

import java.io.IOException;
import java.util.List;

import org.ejml.simple.SimpleMatrix;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

public class WindowModelTest {

	List<List<Datum>> data = null;
	List<Datum> sentence = null;
	WindowModel model = null;

	@Before
	public void setup() {
		try {
			DataIterator iter = new DataIterator("data/train");
			data = iter.getNextSentences(3);
			sentence = data.get(1);
			model = new WindowModel(3, 1, 0.01, 0.001);
			FeatureFactory.initializeVocab("data/vocab.txt");
			FeatureFactory.readWordVectors("data/wordVectors.txt");
		} catch (IOException e) {
			fail(e.toString());
		}
	}

	@Test
	public void testPad() {
		List<String> padded = model.pad(sentence);
		System.out.println(padded);
	}

	@Test
	public void testLabels() {
		List<String> labels = model.getLabels(sentence);
		System.out.println(labels);
	}

	@Test
	public void testWindow() {
		List<String> paddedSentence = model.pad(sentence);
		List<List<String>> windows = model.window(paddedSentence);
		System.out.println(windows);

	}

	@Test
	public void testVetor() {
		List<String> paddedSentence = model.pad(sentence);
		List<List<String>> windows = model.window(paddedSentence);
		List<String> window = windows.get(0);
		System.out.println(model.toInputVector(window));
	}

}
