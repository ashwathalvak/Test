package com.barclays.lingpipe;

import java.io.IOException;

public class LingPipeClassifier {

	public static void main(String[] args) throws ClassNotFoundException,
			IOException {		
		SentimentClassifier sentClassifier = new SentimentClassifier();
		sentClassifier.createModelInputAndTrain();
		sentClassifier.classifyNewTweet("Have a nice day!");
		sentClassifier
				.classifyNewTweet("To everybody's surprise the critics hated it!");
	}

}
