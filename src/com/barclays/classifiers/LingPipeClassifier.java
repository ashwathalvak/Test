package com.barclays.classifiers;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.IOUtils;

import com.aliasi.classify.Classification;
import com.aliasi.classify.Classified;
import com.aliasi.classify.ConditionalClassification;
import com.aliasi.classify.LMClassifier;
import com.aliasi.classify.NaiveBayesClassifier;
import com.aliasi.corpus.ObjectHandler;
import com.aliasi.util.AbstractExternalizable;
import com.aliasi.util.Compilable;
import com.barclays.util.NLPUtils;

public class LingPipeClassifier implements IClassifier {
	private static final String TEST_TWEETS_TXT = "tweets_test.txt";

	public static void main(String[] args) throws IOException {
		LingPipeClassifier classifier = new LingPipeClassifier();
		List<String> lines = IOUtils.readLines(Thread.currentThread()
				.getContextClassLoader().getResourceAsStream(TEST_TWEETS_TXT));
		for (String line : lines) {
			String[] words = line.split("\\t");
			int sentiment = classifier.predict(words[1]);
			System.out.println("Tweet: " + words[1] + " :OrigSentiment: "
					+ words[0] + ":Prediction:" + sentiment);
		}

	}

	private static final String NEGATIVE = "-1";
	private static final String POSITIVE = "1";
	private static final String NEUTRAL = "0";
	private static final String INPUT_TWEETS_ORIG_TXT = "tweets_orig.txt";
	// private static final TokenizerFactory NORM_TOKENIZER_FACTORY =
	// normTokenizerFactory();
	private ClassLoader classLoader = Thread.currentThread()
			.getContextClassLoader();

	/*
	 * static TokenizerFactory normTokenizerFactory() { TokenizerFactory factory
	 * = new RegExTokenizerFactory("\\S+"); return new
	 * PorterStemmerTokenizerFactory( new EnglishStopTokenizerFactory(new
	 * LowerCaseTokenizerFactory( factory))); }
	 */

	public int predict(String text) {
		LMClassifier<?, ?> model = null;
		try {
			model = createModelInputAndTrain(true, INPUT_TWEETS_ORIG_TXT);
		} catch (ClassNotFoundException | IOException e) {
			e.printStackTrace();
		}
		ConditionalClassification classification = model.classify(text);
		String category = classification.bestCategory();
		// System.out.println("Tweet: " + text + "\tSentiment: " + category);
		return Integer.parseInt(category);
	}

	/**
	 * Create the filtered base data for the model to train on
	 * 
	 * @return
	 * 
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	private LMClassifier<?, ?> createModelInputAndTrain(
			boolean readExistingModel, String trainingDataFile)
			throws IOException, ClassNotFoundException {
		final String NAIVE_BAYES_SERIALIZED_MODEL = "naiveBayesModel";
		if (readExistingModel) {
			return deserialize(NAIVE_BAYES_SERIALIZED_MODEL);
		}
		List<String> lines = IOUtils.readLines(classLoader
				.getResourceAsStream(INPUT_TWEETS_ORIG_TXT));
		List<String> linesNoStopWords = new ArrayList<String>();
		LMClassifier<?, ?> cls = NaiveBayesClassifier.createNGramProcess(
				new String[] { POSITIVE, NEGATIVE, NEUTRAL }, 7);
		Classification posClassification = new Classification(POSITIVE);
		Classification negClassification = new Classification(NEGATIVE);
		Classified<String> classified = null;
		for (String line : lines) {
			String newLine = NLPUtils.removeStopWords(line);
			if (POSITIVE.equals(newLine.split(" ")[0])) {
				classified = new Classified<String>(newLine, posClassification);
			} else {
				classified = new Classified<String>(newLine, negClassification);
			}
			linesNoStopWords.add(NLPUtils.removeStopWords(line));
			((ObjectHandler) cls).handle(classified);
		}		
		AbstractExternalizable.compileTo((Compilable) cls,  new File(NAIVE_BAYES_SERIALIZED_MODEL));
		return cls;
	}

	/**
	 * @param fileName
	 * @return
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	private LMClassifier<?, ?> deserialize(String fileName)
			throws ClassNotFoundException, IOException {
		return (LMClassifier<?, ?>) AbstractExternalizable.readObject(new File (classLoader.getResource(fileName).getFile()));

	}	

}
