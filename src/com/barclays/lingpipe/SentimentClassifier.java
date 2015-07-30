package com.barclays.lingpipe;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.en.PorterStemFilter;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.util.CharArraySet;
import org.apache.lucene.util.Version;

import com.aliasi.classify.Classification;
import com.aliasi.classify.Classified;
import com.aliasi.classify.ConditionalClassification;
import com.aliasi.classify.LMClassifier;
import com.aliasi.classify.NaiveBayesClassifier;
import com.aliasi.corpus.ObjectHandler;
import com.aliasi.tokenizer.EnglishStopTokenizerFactory;
import com.aliasi.tokenizer.LowerCaseTokenizerFactory;
import com.aliasi.tokenizer.PorterStemmerTokenizerFactory;
import com.aliasi.tokenizer.RegExTokenizerFactory;
import com.aliasi.tokenizer.TokenFeatureExtractor;
import com.aliasi.tokenizer.TokenizerFactory;
import com.aliasi.util.AbstractExternalizable;
import com.aliasi.util.Compilable;
import com.aliasi.util.FeatureExtractor;

public class SentimentClassifier {
	private LMClassifier<?, ?> cls;
	private static final String NEGATIVE = "0";
	private static final String POSITIVE = "1";
	private static final String CONF_ENGLISH_CUSTOM_STOPWORDS_TXT = "english_stopwords.txt";
	private static final String INPUT_TWEETS_ORIG_TXT = "tweets_orig.txt";
	private static final TokenizerFactory NORM_TOKENIZER_FACTORY = normTokenizerFactory();
	private ClassLoader cl = Thread.currentThread().getContextClassLoader();

	static TokenizerFactory normTokenizerFactory() {
		TokenizerFactory factory = new RegExTokenizerFactory("\\S+");
		return new PorterStemmerTokenizerFactory(
				new EnglishStopTokenizerFactory(new LowerCaseTokenizerFactory(
						factory)));
	}

	public String classifyNewTweet(String text) {
		ConditionalClassification classification = cls.classify(text);
		String category = classification.bestCategory();
		System.out.println("Tweet: " + text + "\tSentiment: " + category);
		return category;
	}

	/**
	 * Create the filtered base data for the model to train on
	 * 
	 * @throws IOException
	 */
	public void createModelInputAndTrain() throws IOException {
		List<String> lines = IOUtils.readLines(cl
				.getResourceAsStream(INPUT_TWEETS_ORIG_TXT));
		List<String> linesNoStopWords = new ArrayList<String>();
		cls = NaiveBayesClassifier.createNGramProcess(new String[] { POSITIVE,
				NEGATIVE }, 7);
		Classification posClassification = new Classification(POSITIVE);
		Classification negClassification = new Classification(NEGATIVE);
		Classified<String> classified = null;
		FeatureExtractor<CharSequence> featureExtractor = new TokenFeatureExtractor(
				NORM_TOKENIZER_FACTORY);
//		new TfIdfClassifierTrainer<CharSequence>(featureExtractor);
		for (String line : lines) {
			String newLine = removeStopWords(line);
			if (POSITIVE.equals(newLine.split(" ")[0])) {
				classified = new Classified<String>(newLine, posClassification);
			} else {
				classified = new Classified<String>(newLine, negClassification);
			}
			linesNoStopWords.add(removeStopWords(line));
			((ObjectHandler) cls).handle(classified);
		}
		File classifierTxt = new File("classifier.txt");
		AbstractExternalizable.compileTo((Compilable) cls, classifierTxt);
		System.out.println(classifierTxt.getCanonicalPath());
	}

	/**
	 * Generate the filtered base data
	 * 
	 * @param line
	 *            Base data line by line
	 * @return Filtered base data line by line
	 * @throws IOException
	 *             if there is an exception reading the data
	 */
	public String removeStopWords(String line) throws IOException {
		TokenStream tokenStream = createTokenStream(new StringReader(line));
		StringBuilder sb = new StringBuilder();
		try {
			CharTermAttribute token = tokenStream
					.getAttribute(CharTermAttribute.class);
			tokenStream.reset();
			while (tokenStream.incrementToken()) {
				if (sb.length() > 0) {
					sb.append(" ");
				}
				sb.append(token.toString());
			}
		} finally {
			tokenStream.close();
		}
		return sb.toString();
	}

	/**
	 * Create a chained filter stream to filter the base data
	 * 
	 * @param reader
	 *            Handle to base data file reader
	 * @return Handle to filtered base data file reader
	 * @throws IOException
	 *             if there is an exception reading the data
	 */
	private TokenStream createTokenStream(Reader reader) throws IOException {
		TokenStream result = new LowerCaseFilter(Version.LUCENE_48,
				new StandardFilter(Version.LUCENE_48, new StandardTokenizer(
						Version.LUCENE_48, reader)));
		result = new StopFilter(Version.LUCENE_48, result,
				readCustomStopWords());
		return new PorterStemFilter(result);
	}

	/**
	 * Read the stop word configuration file into a @CharArraySet and return the
	 * same
	 * 
	 * @return Set of stop words
	 * @throws IOException
	 *             if there was an error reading the stop-word configuration
	 *             file
	 */
	private CharArraySet readCustomStopWords() throws IOException {
		CharArraySet engStopWords = new CharArraySet(Version.LUCENE_48,
				StandardAnalyzer.STOP_WORDS_SET.size(), Boolean.TRUE);
		List<String> lines = IOUtils.readLines(cl
				.getResourceAsStream(CONF_ENGLISH_CUSTOM_STOPWORDS_TXT));
		for (String line : lines) {
			engStopWords.add(line);
		}
		return engStopWords;
	}
}