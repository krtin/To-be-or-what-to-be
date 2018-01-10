<h1>To be or what to be, that is the question</h1>

<h2>Running the Algorithms</h2>
<ol>
	<li>Update StanfordCoreNLP Server in config.py</li>
	<li>Run main.py in python3</li>
</ol>

<h2>Training the models</h2>
<ol>
	<li>Delete all .pkl files</li>
	<li>Run main.py in python3</li>
</ol>

<h2>Method and Intuition</h2>
<h3>General Idea</h3>
<p>The problem is to identify different forms of the base verb "be". Which essentially boils down to identifying Tense (present or past) and Person (First, Second or Third). Our aim should be to create better quality features exploiting knowledge from existing nlp libraries for pos tagging, dependency parsing and others. This will allow us to make better prediction with less data.</p>

<h3>Models Tested</h3>
<ol>
	<li><strong>Probability Model using Windowing:</strong> P(y=word|features=window) was calculated, the window size on each side was 2 and the features included the pos tags of the words. Accuracy of 30% was obatined.</li>
	<li><strong>Probability Model using Dependency Parsing and Windowing:</strong>P(y=word|features=dependency, pos) here pos tag is of the parent node of the target word and dependency is between target word and parent. During prediction we assume that dependency for any of the target word will be the same, hence we find dependency by replacing blank with any of the possible words. (This is not a very strong assumption). Further, we take the better prediction between windowing and dependency parsing by comparing the probability values, being discriminative models it is not a hard assumption that probability will be comparable. Accuracy of 36.6% was obtained.</li>
	<li><strong>Decision Tree Classifier using Windowing:</strong>After testing of dev set, a window size of 5 on each side was chosen, and the model was trained with a decision tree classifier. Accuracy of 46.67% was obtained.</li>
	<li><strong>Decision Tree Classifier using Dependency Parsing:</strong>The dependency was chosen as the feature type and pos tag was chosen as its value. POS tag of the parent and its dependency with target word was taken and additionally all children of the parent except the target node itself was also taken. Accuracy of 50% was obtained.</li>
</ol>

<h3>Further improvements</h3>
<ol>
	<li><strong>Combined Features:</strong>We could combine the features from dependency and windowing and learn a single decision tree model. This might increase accuracy but most likely not by much.</li>
	<li><strong>Better Features:</strong> Till now we have only considered features which are local to a sentence and have ignored discourse information. This leads to limitations because of ambiguities which can only be solved by incorporating discourse information. We can use co-referencing in cases where subject of the target verb is a pronoun and extract features from it. Further, in cases for Nouns as subjects, we can extract either window features or dependency features from instances of the same noun within a context of n sentences. We may have to restrict to only proper nouns, this will have to be tested.</li>
	<li><strong>Neural Model:</strong>In case we have the privilege for a larger training corpus, use deep learning model to extract both local and discourse features. We can also assist the model with dependency and pos information. One example could be using an LSTM between word occurrences of two continuous target words and using the final embedding for either direct classification like one hot vectors or indirect reduced vectors exploiting the relationship between Tense and Person.</li>
</ol>


<h2>Requirements</h2>
python 3, pandas, numpy, sklearn, pickle, collections
