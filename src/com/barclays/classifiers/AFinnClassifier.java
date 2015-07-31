/**
 * 
 */
package com.barclays.classifiers;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.IOUtils;

/**
 * @author G01013268
 *
 */
public class AFinnClassifier  implements IClassifier {
	private ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
	private static final String TEST_TWEETS_TXT = "tweets_test.txt";

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		AFinnClassifier classifier = new AFinnClassifier();
		List<String> lines = IOUtils.readLines(Thread.currentThread().getContextClassLoader()
				.getResourceAsStream(TEST_TWEETS_TXT));
		for(String line: lines){
			String[] words= line.split("\\t");
			int sentiment = classifier.predict(words[1]);
			System.out.println("Tweet: " + words[1] + " :OrigSentiment: " + words[0] + ":Prediction:" +  sentiment );
		}
		

	}
	
	public int predict(String tweet) {
		final String AFINN_BASE_TXT = "AFINN-111.txt";
		Map<String, Integer> afinnDict = null;
		try {
			afinnDict = readAfinnBaseFile(true, AFINN_BASE_TXT);
		} catch (IOException e) {			
			e.printStackTrace();
		}
		String[] words = tweet.toLowerCase().split("\\s+");
		List<Integer> sentiments = new ArrayList<Integer>();
		Integer sum = 0;
		for (int i = 0; i < words.length; i++) {
			// You may want to check for a non-word character before blindly
			// performing a replacement
			words[i] = words[i].replaceAll("[^\\w]", "");
			Integer afinnVal = afinnDict.get(words[i]) == null ? 0 : afinnDict
					.get(words[i]);
			sentiments.add(afinnVal);
			sum += afinnVal;
		}
		double sentiment = sum / Math.sqrt(sentiments.size());
		if (sentiment > 0) {
			return 1;
		} else if (sentiment < 0) {
			return -1;
		}
		return 0;

	}

	/**
	 * @param readExistingModel - if true, reads the trained data from the serialized file
	 * 							  if false, reads the trainingDataFile and serializes the same
	 * @return AFinn Dictionary Map
	 * @throws IOException
	 */
	private Map<String, Integer> readAfinnBaseFile(boolean readExistingModel, String trainingDataFile) throws IOException {		
		Map<String, Integer> afinnDict = new HashMap<String, Integer>();
		final String AFINN_SERIALIZED_FILE_NAME = "afinnDictModel";
		if(readExistingModel){
			return deserialize(AFINN_SERIALIZED_FILE_NAME);
		}
		List<String> lines = IOUtils.readLines(classLoader.getResourceAsStream(trainingDataFile));
		for (String line : lines) {
			afinnDict.put(line.split("\\t")[0],
					(Integer.valueOf(line.split("\\t")[1])));
		}
		serialize(afinnDict, AFINN_SERIALIZED_FILE_NAME);
		return afinnDict;
	}

	/**
	 * @param dict
	 * @param fileName
	 */
	private void serialize(Map<String, Integer> dict, String fileName) {
		try (OutputStream file = new FileOutputStream(fileName);
				OutputStream buffer = new BufferedOutputStream(file);
				ObjectOutput output = new ObjectOutputStream(buffer);) {
			output.writeObject(dict);
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

	/**
	 * @param fileName
	 * @return
	 */
	private Map<String, Integer> deserialize(String fileName) {
		Map<String, Integer> dict = null;
		try (InputStream file = classLoader.getResourceAsStream(fileName);
				InputStream buffer = new BufferedInputStream(file);
				ObjectInput input = new ObjectInputStream(buffer);) {	
			dict = (Map<String, Integer>) input.readObject();
			/*
			 * for (String key : dict.keySet()) { System.out.println("Key: " +
			 * key + ":Value: " + dict.get(key)); }
			 */		} catch (ClassNotFoundException ex) {
			ex.printStackTrace();
		} catch (IOException ex) {
			ex.printStackTrace();
		}
		return dict;
	}

}
