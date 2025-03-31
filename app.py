import pandas as pd
import numpy as np
import re
import string 
import joblib
from dotenv import load_dotenv
import os
import openai
from flask import Flask, render_template, request

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Global variables to store models
vectorization = None
LR = None
DT = None
GB = None
RF = None

def load_models():
    global vectorization, LR, DT, GB, RF
    
    # Only load models if they haven't been loaded yet
    if vectorization is not None:
        return True
    
    try:
        print("Current working directory:", os.getcwd())
        print("Files in directory:", os.listdir())
        if os.path.exists("trained_models"):
            print("Files in trained_models:", os.listdir("trained_models"))
        else:
            print("trained_models directory not found")
            
        print("Attempting to load vectorizer...")
        vectorization = joblib.load("trained_models/vectorizer.pkl")
        print("Vectorizer loaded successfully")
        
        print("Attempting to load LR model...")
        LR = joblib.load('trained_models/logistic_regression_model.pkl')
        print("LR model loaded successfully")
        
        print("Attempting to load DT model...")
        DT = joblib.load('trained_models/decision_tree_model.pkl')
        print("DT model loaded successfully")
        
        print("Attempting to load GB model...")
        GB = joblib.load('trained_models/gradient_boosting_model.pkl')
        print("GB model loaded successfully")
        
        print("Attempting to load RF model...")
        RF = joblib.load('trained_models/random_forest_model.pkl')
        print("RF model loaded successfully")
        
        return True
    except Exception as e:
        print(f"Detailed error loading models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Load models when application starts
@app.before_first_request
def initialize():
    load_models()

def wordopt(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text) 
    text = re.sub(r"\\W", " ", text) 
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  
    text = re.sub(r"<.*?>+", "", text)  
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)  
    text = re.sub(r"\n", "", text) 
    text = re.sub(r"\w*\d\w*", "", text)
    return text

def summarize_article(news):
    try:
        # Set up the OpenAI API key
        openai.api_key = os.getenv("API_KEY")
        
        # Define a prompt for summarization
        prompt = "Please list the key points of this article very concisely with only a few key points using - to mark the beginning of each and say nothing else."

        # Use the API method
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "the article is " + news},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the response
        generated_text = response['choices'][0]['message']['content']
        return generated_text
    except Exception as e:
        print(f"Error in summarize_article: {str(e)}")
        return "Error generating summary. Please try again."

def testing_validity(news):
    # Check if models are loaded
    if vectorization is None and not load_models():
        return "Error: Could not load models"
    
    try:
        testing_news = {"text": [news]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test["text"] = new_def_test["text"].apply(wordopt)
        new_x_test = new_def_test["text"]
        new_xv_test = vectorization.transform(new_x_test)
        
        LR_prob = LR.predict_proba(new_xv_test)[0] 
        LR_pred_class = LR.predict(new_xv_test)[0]  

        LR_confidence = LR_prob[LR_pred_class] * 100 
        RF_pred = RF.predict(new_xv_test)[0]

        if RF_pred == 1 and LR_confidence > 90:
            return "This is definitely not fake news and we are " + str(round(LR_confidence, 2)) + "% sure."
        elif RF_pred == 1 and LR_confidence > 65:
            return "This is most likely not fake news and we are " + str(round(LR_confidence, 2)) + "% sure."
        elif RF_pred == 1 and LR_confidence <= 65:
            return "This is probably not fake news but we are only " + str(round(LR_confidence, 2)) + "% sure."
        if RF_pred == 0 and LR_confidence > 90:
            return "This is definitely fake news and we are " + str(round(LR_confidence, 2)) + "% sure."
        elif RF_pred == 0 and LR_confidence > 65:
            return "This is probably fake news and we are " + str(round(LR_confidence, 2)) + "% sure."
        elif RF_pred == 0 and LR_confidence > 55:
            return "This is may be fake news but we are only " + str(round(LR_confidence, 2)) + "% sure."
        else:
            return "This is probably not fake news but we are only " + str(round(LR_confidence, 2)) + "% sure."
    except Exception as e:
        print(f"Error in testing_validity: {str(e)}")
        return "Error analyzing the news. Please try again."

@app.route("/")
def home():
    return render_template("index.html", result="")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/run-script", methods=["POST"])
def run_script():
    try:
        news = request.form.get("news")
        
        if not news:
            return render_template("index.html", result="No news provided", summary="")
            
        result = testing_validity(news)
        summary = summarize_article(news)
        summary = summary.replace("- ", "<br><br>-")
        
        return render_template("index.html", result=result, summary=summary, news=news)
    except Exception as e:
        print(f"Error in run_script: {str(e)}")
        return render_template("index.html", result="An error occurred. Please try again.", summary="", news=news if 'news' in locals() else "")

if __name__ == "__main__":
    # Load models at startup
    load_models()
    
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)  # Set debug=False in production