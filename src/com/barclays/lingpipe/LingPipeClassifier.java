package com.barclays.lingpipe;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.IOUtils;

public class LingPipeClassifier {
	private static final String AFINN_BASE_TXT = "AFINN-111.txt";

	public static void main(String[] args) throws IOException {
		SentimentClassifier sentClassifier = new SentimentClassifier();
		sentClassifier.createModelInputAndTrain();
		sentClassifier.classifyNewTweet("Have a nice day!");
		sentClassifier
				.classifyNewTweet("To everybody's surprise the critics hated it!");
		System.out.println(predictAfinnModelSentiment("Have a nice day!"));
		System.out.println(predictAfinnModelSentiment("To everybody's surprise the critics hated it!"));
	}

	private static String predictAfinnModelSentiment(String tweet) throws IOException {
		Map<String, Integer> afinnDict = readAfinnBaseFile();
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
			return "Positive";
		} else if (sentiment < 0) {
			return "Negative";
		}
		return "Neutral";

	}

	private static Map<String, Integer> readAfinnBaseFile() throws IOException {
		Map<String, Integer> afinnDict = new HashMap<String, Integer>();
		List<String> lines = IOUtils.readLines(Thread.currentThread()
				.getContextClassLoader().getResourceAsStream(AFINN_BASE_TXT));
		for (String line : lines) {
			afinnDict.put(line.split("\\t")[0],
					(Integer.valueOf(line.split("\\t")[1])));
		}
		return afinnDict;
	}

}
