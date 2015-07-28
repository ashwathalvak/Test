package com.barclays.lingpipe;

import java.io.File;
import java.io.IOException;

import com.aliasi.classify.ConditionalClassification;
import com.aliasi.classify.LMClassifier;
import com.aliasi.util.AbstractExternalizable;

public class SentimentClassifier {
	String[] categories;
	LMClassifier cls;

	public SentimentClassifier() {
		try {
			cls = (LMClassifier) AbstractExternalizable.readObject(new File(
					"classifier.txt"));
			categories = cls.categories();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public String classifyNewTweet(String text) {
		ConditionalClassification classification = cls.classify(text);
		String category = classification.bestCategory();
		System.out.println("Tweet: " + text + "\tSentiment: " + category);
		return category;
	}
}