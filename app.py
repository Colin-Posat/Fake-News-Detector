import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string 
import joblib
from dotenv import load_dotenv
import os
import openai
import modal
load_dotenv()
from flask import Flask, render_template, request


vectorization = joblib.load("trained_models/vectorizer.pkl")

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

# Load the models
LR = joblib.load('trained_models/logistic_regression_model.pkl')
DT = joblib.load('trained_models/decision_tree_model.pkl')
GB = joblib.load('trained_models/gradient_boosting_model.pkl')
RF = joblib.load('trained_models/random_forest_model.pkl')


def summarize_article(news):
    # Set up the OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Define a prompt for summarization
    prompt = "Please list the key points of this article very consicesly with only a few key points using - to mark the beginning of each and say nothing else."

    # Use the new API method
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use gpt-4 or other models as needed
        messages=[
            {"role": "system", "content": "the article is " + news},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract and print the response
    generated_text = response['choices'][0]['message']['content']
    return(generated_text)
        

def testing_validity(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    
    LR_prob = LR.predict_proba(new_xv_test)[0] 
    LR_pred_class = LR.predict(new_xv_test)[0]  

    LR_confidence = LR_prob[LR_pred_class] * 100 

    if RF.predict(new_xv_test) == 1 and LR_confidence > 90:
        return "This is definitely not fake news and we are " + str(round(LR_confidence, 2)) + "% sure."
    elif RF.predict(new_xv_test) == 1 and LR_confidence > 65:
        return "This is most likely not fake news and we are " + str(round(LR_confidence, 2)) + "% sure."
    elif RF.predict(new_xv_test) == 1 and LR_confidence <= 65:
        return "This is probably not fake news but we are only " + str(round(LR_confidence, 2)) + "% sure."
    if RF.predict(new_xv_test) == 0 and LR_confidence > 90:
        return "This is definitely fake news and we are " + str(round(LR_confidence, 2)) + "% sure."
    elif(RF.predict(new_xv_test) == 0) and LR_confidence > 65:
        return "This is probably fake news and we are " + str(round(LR_confidence, 2)) + "% sure."
    elif(RF.predict(new_xv_test) == 0) and LR_confidence > 55:
        return "This is may be fake news but we are only " + str(round(LR_confidence, 2)) + "% sure."
    else:
        return "This is probably not fake news but we are only " + str(round(LR_confidence, 2)) + "% sure."


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html", result="")

@app.route("/about")
def about():
    return render_template("about.html")



@app.route("/run-script", methods=["POST"])
def run_script():
    news = request.form.get("news")
    
    if not news:
        return render_template("index.html", result="No news provided", summary="")
    result = testing_validity(news)
    summary = summarize_article(news)
    summary = summary.replace("- ", "<br><br>-")
    return render_template("index.html", result=result, summary=summary, news=news)

if __name__ == "__main__":
    # Ensure compatibility with Render and local environments
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)

#news = "President-elect Donald Trump on Wednesday named former acting Attorney General Matt Whitaker as his pick to be the next ambassador to NATO, a key alliance that Trump derided for years. Matt is a strong warrior and loyal Patriot, who will ensure the United States’ interests are advanced and defended. Matt will strengthen relationships with our NATO Allies, and stand firm in the face of threats to Peace and Stability — He will put AMERICA FIRST, Trump said in a statement. Whitaker, whose appointment must be confirmed by the Senate, doesn't appear to have much foreign policy experience in his professional background. Whitaker first took over the Justice Department on an acting basis during Trump's first term in November 2018, right after the midterm elections, when Trump announced on Twitter that he would succeed Jeff Sessions, whom Trump asked to resign. He served just three months in the position until William Barr was confirmed as attorney general. Before he became acting attorney general, Whitaker was chief of staff to Sessions when he was attorney general. He previously was the U.S. attorney for the Southern District of Iowa from 2004 to 2009, appointed by President George W. Bush. He has been a co-chair of the Center of Law & Justice at the nonprofit Trump-aligned think tank America First Policy Institute. Trump has long criticized NATO, accusing European allies of not contributing enough toward defense spending. He suggested at a campaign rally in February that he would allow Russia to do whatever the hell they want to countries that don't pay the bills. When he first ran for president in 2016, Trump called NATO obsolete. European Commission President Ursula von der Leyen said this year that Trump told top European officials in 2020, before he left office, that the U.S. wouldn't help Europe if it came under attack. In February, former Secretary of State Hillary Clinton warned that Trump would seek to withdraw the U.S. from NATO if he were re-elected. Vice President-elect JD Vance, who has criticized U.S. funding of Ukraine in its war with Russia, told NBC News days before this year's election that a second Trump administration would remain in NATO. Recommended Congress House Ethics panel has no agreement on releasing Matt Gaetz report after meeting From the Politics Desk After a stinging election defeat, the Democrats' next big race kicks off: From the Politics Desk Trump has repeatedly said that when it comes to Russia's war in Ukraine, he would negotiate a deal that's good for both sides as president. Whitaker, meanwhile, has made some public statements that signal he supports the alliance and Ukraine. In an interview on Fox News in 2019, Whitaker was asked how dangerous it would be if the U.S. said it would contribute only as much as other countries to NATO. We are the world's superpower, and only superpower, he said. I think we're always going to have to spend more than our fair share to make sure that democracy and freedom is defended worldwide. But at the same time, that doesn't mean that the people that we've allied with should get to, you know, sort of ride on our coattails. Speaking to Fox Business in 2022, after Russia's war in Ukraine began, Whitaker said, There is no doubt now that NATO is in the line of fire, and people like Poland are feeling the pressure as they're supplying the Ukrainian fighters. Poland is next on the list, I'm sure, for Putin. If the war doesn't end in Ukraine, it's going to be — it already is on NATO's doorstep, said Whitaker, who voiced support for the U.S.' shipping heavy weapons into Ukraine."
#testing_validity(news)
#summarize_article(news)