from datetime import datetime
import logging
import os

from flask import Flask, redirect, render_template, request

from google.cloud import storage
from google.cloud import language_v1 as language
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from google.cloud import language_v1
from google.cloud import datastore
import time
import pandas as pd
#Print all columns and all rows in a panda dataframe
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


app = Flask(__name__)


@app.route("/")
def homepage():
    # Create a Cloud Datastore client.
    datastore_client = datastore.Client()

    # # Use the Cloud Datastore client to fetch information from Datastore
    # Query looks for all documents of the 'Sentences' kind, which is how we
    # store them in upload_text()
    query = datastore_client.query(kind="gee-nlp-demo")
    text_entities = list(query.fetch())

    # # Return a Jinja2 HTML template and pass in text_entities as a parameter.
    return render_template("homepage.html", text_entities=text_entities)


@app.route("/upload", methods=["GET", "POST"])
def upload_text():
    file_name = request.form["text"]

    client = storage.Client()
    bucket = client.get_bucket('gee-nlp')
    blob = bucket.get_blob(file_name)
    downloaded_blob = blob.download_as_string()
    downloaded_blob_1  = downloaded_blob.decode('utf-8', 'ignore')
    downloaded_blob_1 = downloaded_blob_1.replace("\r\n\r\n"," ")
    text = downloaded_blob_1
    # Analyse sentiment using Sentiment API call
    #sentiment = analyze_text_sentiment(text)[0].get('sentiment score')

    results = analyze_text_sentiment(text)
    print("results: ",results)
    sentiment = results["overallResults"].get('score')
    magnitude = results["overallResults"].get('magnitude')
    print("sentiment in upload_text(): ",sentiment)
    
    sentence_sentiment = results["sentence_sentiment"]
    classify_document  = gcp_classify_text(downloaded_blob_1)
    print("classify_document: ",classify_document)

    df_sentiment = pd.DataFrame(sentence_sentiment)
    df_sentiment
    gcp_plot_sentiments(df_sentiment)

    # Assign a label based on the score
    overall_sentiment = 'unknown'
    if sentiment > 0:
        overall_sentiment = 'positive'
    if sentiment < 0:
        overall_sentiment = 'negative'
    if sentiment == 0:
        overall_sentiment = 'neutral'

    # Create a Cloud Datastore client.
    datastore_client = datastore.Client()

    # Fetch the current date / time.
    current_datetime = datetime.now()

    # The kind for the new entity. This is so all 'Sentences' can be queried.
    kind = "gee-nlp-demo"

    # Create the Cloud Datastore key for the new entity.
    key = datastore_client.key(kind, 'sample_task1')

    # Alternative to above, the following would store a history of all previous requests as no key
    # identifier is specified, only a 'kind'. Datastore automatically provisions numeric ids.
    # key = datastore_client.key(kind)

    # Construct the new entity using the key. Set dictionary values for entity
    entity = datastore.Entity(key)
    entity["file_name"] = file_name
    entity["timestamp"] = current_datetime
    entity["sentiment"] = overall_sentiment
    entity["classify_document"] = classify_document
    entity["magnitude"] = magnitude
    
    
    print("entity: ",entity)
    # Save the new entity to Datastore.
    datastore_client.put(entity)

    # Redirect to the home page.
    return redirect("/")


@app.errorhandler(500)
def server_error(e):
    logging.exception("An error occurred during a request.")
    return (
        """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(
            e
        ),
        500,
    )
def analyze_text_sentiment(text):
    client = language.LanguageServiceClient()
    document = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)

    response = client.analyze_sentiment(document=document)

    sentiment = response.document_sentiment
    overallResults = dict(
        text=text,
        score=sentiment.score,
        magnitude=sentiment.magnitude,
    )
    results ={}
    results["overallResults"] =overallResults

    for k, v in results.items():
        print(f"{k:10}: {v}")

    # Get sentiment for all sentences in the document
    sentence_sentiment = []
    for sentence in response.sentences:
        item={}
        item["text"]=sentence.text.content
        item["sentiment score"]=sentence.sentiment.score
        item["sentiment magnitude"]=sentence.sentiment.magnitude
        sentence_sentiment.append(item)

    results["sentence_sentiment"] = sentence_sentiment
    return results


def gcp_classify_text(text):
    client = language.LanguageServiceClient()
    document = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)
    response = client.classify_text(document=document)
    #response = client.classify(document=document)
    for category in response.categories:
        return category.name

def gcp_plot_sentiments(df_sentiment):
    # Plot Sentiment Scores
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import colors
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'figure.figsize':(16,8)})

    x = df_sentiment["sentiment score"]
    y =  df_sentiment["sentiment magnitude"]

    sns.scatterplot(data= df_sentiment[["sentiment score", "sentiment magnitude"]])

    n_bins=30

    #plt.hist(x, bins=n_bins)
    #plt.show()

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    # We can set the number of bins with the `bins` kwarg
    axs[0].set_xlabel("Sentiment Score")
    axs[0].set_ylabel("percentage")
    axs[0].set_title('Histogram of Sentiment Score')
    axs[1].set_xlabel("Sentiment Magnitude")
    axs[1].set_title('Histogram of Sentiment Magnitude')

    axs[0].hist(x, bins=n_bins)
    axs[1].hist(y, bins=n_bins)
    plt.show()


    fig, ax = plt.subplots(tight_layout=True)
    hist = ax.hist2d(x, y, norm=colors.LogNorm())
    plt.title("Sentiment Score and Magnitude 2-D Distribution")
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Sentiment Magnitude")
    fig.savefig('my_plot.png')

    plt.show()

if __name__ == "__main__":
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    
    app.run(host="127.0.0.1", port=8080, debug=True)


