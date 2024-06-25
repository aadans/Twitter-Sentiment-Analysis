from flask import Flask, render_template, request, url_for
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle

app = Flask(__name__)

port_stem = PorterStemmer()
def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]',' ',content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)

  return stemmed_content

# Load the trained model and vectorizer
with open('rf_classifier4.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template("index1.html")

@app.route('/sentiment_analysis_prediction', methods=['POST'])
def sent_analysis_prediction():
    if request.method == 'POST':
        comment = request.form['text']
        cleaned_comment = stemming(comment)
        comment_vector = vectorizer.transform([cleaned_comment])  # Vectorize the input text
        predicted_sentiment = classifier.predict(comment_vector )  # Predict sentiment
       
        # Determine which emoji to display based on sentiment
        if predicted_sentiment == 1:
            sentiment_label = 'Positive'
            emoji_file = '93c5.gif'  # Filename for positive sentiment emoji
        elif predicted_sentiment == 0:
            sentiment_label = 'Negative'
            emoji_file = 'sad_emoji.gif'  # Filename for negative sentiment emoji
        
        # Render the template with variables
        return render_template('result1.html', text=comment, sentiment=sentiment_label, emoji=emoji_file)


if __name__ == '__main__':
    app.run(debug=True)
