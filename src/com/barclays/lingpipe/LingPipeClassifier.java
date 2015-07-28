package com.barclays.lingpipe;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.en.PorterStemFilter;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.util.CharArraySet;
import org.apache.lucene.search.similarities.DefaultSimilarity;
import org.apache.lucene.search.similarities.TFIDFSimilarity;
import org.apache.lucene.util.Version;

import com.aliasi.classify.Classification;
import com.aliasi.classify.Classified;
import com.aliasi.classify.DynamicLMClassifier;
import com.aliasi.classify.LMClassifier;
import com.aliasi.corpus.ObjectHandler;
import com.aliasi.util.AbstractExternalizable;
import com.aliasi.util.Compilable;

public class LingPipeClassifier {
	
	private static final String NEGATIVE = "0";
	private static final String POSITIVE = "1";
	private static final String CONF_ENGLISH_CUSTOM_STOPWORDS_TXT = "conf/english_stopwords.txt";
	private static final String INPUT_TWEETS_REMOVED_STOPWORDS_TXT = "input/tweets_new.txt";
	private static final String INPUT_TWEETS_ORIG_TXT = "input/tweets_orig.txt";

	public static void main(String[] args) throws ClassNotFoundException,
			IOException {
		createModelInputAndTrain();
		SentimentClassifier sentClassifier = new SentimentClassifier();
		sentClassifier.classifyNewTweet("Have a nice day!");
		sentClassifier
				.classifyNewTweet("To everybody's surprise the critics hated it!");
	}	
	
	/**
	 * Create the filtered base data for the model to train on
	 * @throws IOException
	 */
	private static void createModelInputAndTrain() throws IOException {
		File file = new File(INPUT_TWEETS_ORIG_TXT);
		List<String> lines = FileUtils.readLines(file, "UTF-8");
		List<String> linesNoStopWords = new ArrayList<String>();		
		LMClassifier cls = DynamicLMClassifier.createNGramProcess(new String[] { POSITIVE,NEGATIVE }, 7);
		Classification posClassification = new Classification(POSITIVE);
		Classification negClassification = new Classification(NEGATIVE);
		Classified classified = null;
		for (String line : lines) {
			String newLine = removeStopWords(line);
			if(POSITIVE.equals(newLine.split(" ")[0])){
				classified = new Classified(newLine, posClassification);
			}else{
				classified = new Classified(newLine, negClassification);
			}
			linesNoStopWords.add(removeStopWords(line));
			((ObjectHandler) cls).handle(classified);
		}
		File newFile = new File(INPUT_TWEETS_REMOVED_STOPWORDS_TXT);
		FileUtils.writeLines(newFile, linesNoStopWords);
		AbstractExternalizable.compileTo((Compilable) cls, new File(
				"classifier.txt"));
	}
	
	/**
	 * Generate the filtered base data 
	 * @param line Base data line by line
	 * @return Filtered base data line by line
	 * @throws IOException if there is an exception reading the data
	 */
	public static String removeStopWords(String line) throws IOException {
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
	 * @param reader Handle to base data file reader
	 * @return Handle to filtered base data file reader
	 * @throws IOException if there is an exception reading the data
	 */
	private static TokenStream createTokenStream(Reader reader)
			throws IOException {
		TokenStream result = new LowerCaseFilter(Version.LUCENE_48,
				new StandardFilter(Version.LUCENE_48, new StandardTokenizer(
						Version.LUCENE_48, reader)));
		TFIDFSimilarity tfidf = new DefaultSimilarity();
		result = new StopFilter(Version.LUCENE_48, result, readCustomStopWords());
		return new PorterStemFilter(result);
	}
	
	/**
	 * Read the stop word configuration file into a @CharArraySet and return the
	 * same
	 * @return Set of stop words
	 * @throws IOException if there was an error reading the stop-word configuration
	 *             file
	 */
	private static CharArraySet readCustomStopWords() throws IOException {
		CharArraySet engStopWords = new CharArraySet(Version.LUCENE_48,
				StandardAnalyzer.STOP_WORDS_SET.size(), Boolean.TRUE);
		List<String> lines = FileUtils.readLines(new File(
				CONF_ENGLISH_CUSTOM_STOPWORDS_TXT));
		for (String line : lines) {
			engStopWords.add(line);
		}
		return engStopWords;
	}

}
