package cs224n.deep;

import static org.junit.Assert.*;

import java.io.IOException;

import org.junit.Before;
import org.junit.Test;

public class DataIteratorTest {

	DataIterator iter = null;
	
	@Before
	public void setup() {
		try {
			iter = new DataIterator("data/train");
		} catch (IOException e) {
			fail(e.toString());
		}
	}
	
	@Test
	public void testGetNextSentences() {
		//System.out.println(iter.getNextSentences(10));
		int total = 0;
		while (iter.hasNext())	{
			iter.getNextSentences(1);
			total++;
		}
		System.out.println("Total Sentences = " + total);
	}

}
