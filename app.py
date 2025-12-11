from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
import re

#Discourse Markers
tagalog_discourse_markers = r"\b(?:at|kung|hanggang|hangga’t|bagama’t|nang|o|kaya|pero|dahil\ sa|dahilan\ sa|gawa\ ng|sapagka’t|upang|sakali|noon|sa\ sandali|magbuhat|magmula|bagaman|maliban|bukod|dangan|dahil|yayamang|kapag|pagka|tuwing|matapos|pagkatapos|porke|maski|imbis|sa\ lugar|sa\ halip|miyentras|para|saka|haba|samantala|bago|kundi)\b"
english_discourse_markers = r"\b(?:and|but|or|so|because|although|however|nevertheless|nonetheless|yet|still|despite\ that|in\ spite\ of\ that|even\ so|on\ the\ contrary|on\ the\ other\ hand|otherwise|instead|alternatively|in\ contrast|as\ a\ result|therefore|thus|consequently|hence|so\ that|in\ order\ that|with\ the\ result\ that|because\ of\ this|due\ to\ this|then|next|after\ that|afterwards|since\ then|eventually|finally|in\ the\ end|at\ first|in\ the\ beginning|to\ begin\ with|first\ of\ all|for\ one\ thing|for\ another\ thing|secondly|thirdly|to\ start\ with|in\ conclusion|to\ conclude|to\ sum\ up|in\ short|in\ brief|overall|on\ the\ whole|all\ in\ all|to\ summarize|in\ a\ nutshell|moreover|furthermore|what\ is\ more|in\ addition|besides|also|too|as\ well|in\ the\ same\ way|similarly|likewise|in\ other\ words|that\ is\ to\ say|this\ means\ that|for\ example|for\ instance|such\ as|namely|in\ particular|especially|more\ precisely|to\ illustrate|as\ a\ matter\ of\ fact|actually|in\ fact|indeed|clearly|surely|certainly|obviously|of\ course|naturally|apparently|evidently|no\ doubt|undoubtedly|presumably|frankly|honestly|to\ be\ honest|luckily|fortunately|unfortunately|hopefully|interestingly|surprisingly|ironically)\b"
all_discourse_markers = tagalog_discourse_markers + "|" + english_discourse_markers

#split sentence into clauses
def split_into_clauses(text):
    if not isinstance(text, str):
        return []
    #check dm in the sentence
    has_dm = re.search(all_discourse_markers, text, flags=re.IGNORECASE)
    if has_dm:
        parts = re.split(all_discourse_markers, text, flags=re.IGNORECASE)
    elif ',' in text:
        parts = text.split(',')
    else:
        return [text.strip()]
    clauses = [p.strip() for p in parts if p.strip()]
    return clauses

def extract_discourse_markers(text):
    return re.findall(all_discourse_markers, text, flags=re.IGNORECASE)

#load model
loaded = joblib.load(r"C:\Users\mynam\Downloads\Clause Level Sentiment Analysis Tool\models\taglish_sentiment_model.pkl")

if isinstance(loaded, dict):
    vectorizer = loaded.get("vectorizer")
    clf = loaded.get("model")
else:
    vectorizer = None
    clf = loaded  


app = Flask(__name__)
#temp storage
feedbacks = []

#analyze page
@app.route('/', methods=['GET', 'POST'])
def analyze():
    results = []
    overall_sentiment = None
    percentages = None
    overall_percent = None

    if request.method == 'POST':
        user_input = request.form.get('user_input', '')
        import re
        #split sentences
        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s*', user_input) if s.strip()]

        sentiment_scores = {'positive': 0, 'neutral': 0, 'negative': 0}

        for sentence in sentences:
            markers_found = extract_discourse_markers(sentence)
            clauses = split_into_clauses(sentence)

            clause_results = []
            sentiment_scores_sentence = {'positive': 0, 'neutral': 0, 'negative': 0}

            for clause in clauses:
                if vectorizer:
                    X_clause = vectorizer.transform([clause])
                else:
                    X_clause = [clause]

                prob_clause = clf.predict_proba(X_clause)[0]
                pred_clause = clf.classes_[prob_clause.argmax()]

                sentiment_scores_sentence['positive'] += prob_clause[list(clf.classes_).index('positive')]
                sentiment_scores_sentence['neutral']  += prob_clause[list(clf.classes_).index('neutral')]
                sentiment_scores_sentence['negative'] += prob_clause[list(clf.classes_).index('negative')]

                prob_dict = {
                    'positive': round(prob_clause[list(clf.classes_).index('positive')]*100, 2),
                    'neutral': round(prob_clause[list(clf.classes_).index('neutral')]*100, 2),
                    'negative': round(prob_clause[list(clf.classes_).index('negative')]*100, 2)
                }

                clause_results.append({
                    'clause': clause,
                    'sentiment': pred_clause,
                    'probabilities': prob_dict
                })

            total_sentence = sum(sentiment_scores_sentence.values())
            percentages_sentence = {
                k: round((v/total_sentence)*100, 2) if total_sentence > 0 else 0
                for k, v in sentiment_scores_sentence.items()
            }

            overall_sentence = max(percentages_sentence, key=percentages_sentence.get)
            overall_percent_sentence = percentages_sentence[overall_sentence]

            results.append({
                'sentence': sentence,
                'clauses': clause_results,
                'discourse_markers': markers_found,
                'overall': overall_sentence,
                'overall_percentage': overall_percent_sentence,
                'percentages': percentages_sentence
            })

            for k in sentiment_scores:
                sentiment_scores[k] += sentiment_scores_sentence[k]

        total = sum(sentiment_scores.values())
        percentages = {k: round((v/total)*100, 2) if total>0 else 0 for k,v in sentiment_scores.items()}
        overall_sentiment = max(percentages, key=percentages.get)
        overall_percent = percentages[overall_sentiment]
    return render_template(
        'analyze.html',
        results=results,
        overall=overall_sentiment,
        overall_percentage=overall_percent,
        percentages=percentages
    )

#feedack page
from datetime import datetime

@app.route("/leave_a_feedback", methods=["GET", "POST"])
def leave_a_feedback():
    global feedbacks
    if request.method == "POST":
        user_input = request.form.get("user_input")

        if vectorizer is not None:
            X = vectorizer.transform([user_input])
        else:
            X = [user_input]

        sentiment = clf.predict(X)[0]
        probs = clf.predict_proba(X)[0]
        prob_dict = {
            'positive': round(probs[list(clf.classes_).index('positive')] * 100, 2),
            'neutral': round(probs[list(clf.classes_).index('neutral')] * 100, 2),
            'negative': round(probs[list(clf.classes_).index('negative')] * 100, 2),
        }
        feedbacks.insert(0, {
            "text": user_input,
            "sentiment": sentiment,
            "probabilities": prob_dict,
            "timestamp": datetime.now().strftime("%B %d, %Y at %I:%M %p")
        })

        return redirect(url_for("leave_a_feedback"))

    return render_template("feedback.html", feedbacks=feedbacks)

#about page  
@app.route('/about_tool')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
