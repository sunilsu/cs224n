package cs224n.deep;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class DataIterator implements Iterator<List<Datum>> {

	BufferedReader br = null;
	
	public DataIterator(String filename) throws IOException {
		br = new BufferedReader(new FileReader(filename));
	}
	@Override
	public boolean hasNext() {
		boolean more = true;
		try {
			br.mark(32);
			if (br.read() == -1) {
				more = false;
			}
			br.reset();
		} catch (IOException e) {
			more = false;
		}
		return more;
	}

	@Override
	public List<Datum> next() {
		String line = null;
		List<Datum> data = new ArrayList<Datum>();
		try {
			while ((line = br.readLine()) != null) {
				if (line.trim().length() == 0) {
					break;
				}
				String[] bits = line.split("\\s+");
				String word = bits[0];
				String label = bits[1];

				Datum datum = new Datum(word, label);
				data.add(datum);
			}
		} catch (IOException e) {
			return null;
		}
		return data;
	}
	
	public List<List<Datum>> getNextSentences(int N) {
		List<List<Datum>> sentences = new ArrayList<List<Datum>>();
		int i = 0;
		while (hasNext() && i++ < N) {
			sentences.add(next());
		}
		return sentences;
	}

}
