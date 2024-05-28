from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.text import sent_tokenize
from nltk.tokenize import word_tokenize

import re

paragraph =  """ According to recent studies, this method is highly effective [1]. Further experiments confirmed these results [2, 3]. Researchers have found that this approach significantly improves performance across various metrics, indicating its robustness and versatility. One key aspect of its success is the integration of advanced algorithms with real-time data processing, which allows for more accurate and timely predictions.
For example, a study conducted in 2023 demonstrated that the method increased efficiency by 30 [67] machine learning models, particularly in natural language processing tasks [2]. Another study highlighted its effectiveness in reducing error rates in predictive analytics, showcasing a substantial decrease from 15% to 5% [3]. These findings underscore the method's potential in diverse applications, from healthcare diagnostics to financial forecasting.
Moreover, the adaptability of this method to different datasets and environments makes it a valuable tool in the toolkit of data scientists and engineers. Its ability to handle large volumes of data without significant degradation in performance ensures its applicability in big data scenarios.
Overall, the consistent positive outcomes reported in these studies suggest that this method is not only effective but also scalable and reliable. As more research continues to validate its benefits, it is poised to become a standard in various fields requiring precise and efficient data analysis."""

# preprocessing the data
text = re.sub(r"\[[0-9]*\]", " ", paragraph)
text = re.sub(r"\s+"," ",text)
text = re.sub(r"\d"," ",text)
# text = re.sub("^[a-zA-Z]"," ",paragraph)
text = text.lower()

pro_sent = sent_tokenize(text) # converting the story into sentences, returns a lists
words = [word_tokenize(every_sent) for every_sent in pro_sent] # converting the sentences into words

for i in range(len(words)):
    words[i] = [word for word in words[i] if word not in stopwords.words("english")]

model = Word2Vec(words,min_count=1)
sim=model.wv.most_similar("moreover") # words similar to moreover
print(sim)
