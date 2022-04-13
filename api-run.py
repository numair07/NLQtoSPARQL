import json
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import spacy
from spacy.training import Example
nlp=spacy.load("en_core_web_sm")
ner=nlp.get_pipe('ner')
from flask import Flask, request, jsonify

nltk.download('omw-1.4')
nltk.download('wordnet')

file = open("LC-QuAD/train-data.json")
train_data = json.load(file)
file = open("LC-QuAD/test-data.json")
test_data = json.load(file)


ques = []
type_ = []
#ASK is type 3, COUNT is type 2, SELECT is type 1
for dt in train_data:
    ques.append(str(dt['corrected_question']))
    if("ASK" in dt['sparql_query']):
        type_.append(3)
    elif("COUNT" in dt['sparql_query']):
        type_.append(2)
    elif("SELECT" in dt['sparql_query']):
        type_.append(1)

train = pd.DataFrame({"question" : ques, "type" : type_})

ques = []
type_ = []
for dt in test_data:
    ques.append(str(dt['corrected_question']))
    if("ASK" in dt['sparql_query']):
        type_.append(3)
    elif("COUNT" in dt['sparql_query']):
        type_.append(2)
    elif("SELECT" in dt['sparql_query']):
        type_.append(1)

test = pd.DataFrame({"question" : ques, "type" : type_})
x_train = train['question']
y_train = train['type']
x_test = test['question']
y_test = test['type']

def clean_sentences(text):
    text = text.lower()
    text = " ".join(text.split())
    text = " ".join(lemmatizer.lemmatize(word) for word in text.split())
    return text

x_train = x_train.map(lambda a:clean_sentences(a))
x_test = x_test.map(lambda a:clean_sentences(a))

vectorizer = TfidfVectorizer(min_df=0.08, max_df=0.50,smooth_idf=False)
vectorizer.fit(x_train)
x_train = vectorizer.transform(x_train)
x_test = vectorizer.transform(x_test)

model = RandomForestClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

TRAIN_DATA = [
              ("which sector does axis bank belong to", {"entities": [(6, 12, "PROPERTY"), (18, 27, "SUBJECT")]}),
              ("company website of axis bank", {"entities": [(8, 15, "PROPERTY"), (19, 28, "SUBJECT")]}),
              ("give me the website of infosys", {"entities": [(12, 19, "PROPERTY"), (23, 30, "SUBJECT")]}),
              #stock peers
              ("give me the stock peers of icici bank", {"entities": [(12, 23, "PROPERTY"), (27, 37, "SUBJECT")]}),
              ("who are the stock peers of axis bank", {"entities": [(12, 23, "PROPERTY"), (27, 36, "SUBJECT")]}),
              ("give me the stock peers of icici bank belonging to the banking sector", {"entities": [(12, 23, "PROPERTY"), (27, 37, "SUBJECT"), (55, 62, "OBJECT"), (63, 69, "PROPERTY")]}),
              ("give me the stock peers of icici bank in the technology sector", {"entities": [(12, 23, "PROPERTY"), (27, 37, "SUBJECT"), (45, 55, "OBJECT"), (56, 62, "PROPERTY")]}),
              #symbol
              ("what is the stock symbol of coforge", {"entities": [(18, 24, "PROPERTY"), (28, 35, "SUBJECT")]}),
              ("what is the nse symbol of hdfc bank", {"entities": [(16, 22, "SUBJECT"), (26, 35, "SUBJECT")]}),
              #location
              ("where is tcs located", {"entities": [(9, 12, "SUBJECT"), (13, 20, "LOCATION")]}),
              ("which city is tcs located", {"entities": [(6, 10, "PROPERTY"), (14, 17, "SUBJECT")]}),
              ("which city is infosys based out of", {"entities": [(6, 10, "PROPERTY"), (14, 21, "SUBJECT")]}),
              ("where is infosys headquartered", {"entities": [(9, 16, "SUBJECT"), (17, 30, "LOCATION")]}),
              ("which country is hcl headquartered in", {"entities": [(6, 13, "PROPERTY"), (17, 20, "SUBJECT")]}),
              ("which state is tech mahindra headquartered in", {"entities": [(6, 11, "PROPERTY"), (15, 28, "SUBJECT")]}),
              ("which state is sbi located", {"entities": [(6, 11, "PROPERTY"), (15, 18, "SUBJECT")]}),
              ("where is the headquarter of tcs", {"entities": [(13, 24, "LOCATION"), (28, 31, "SUBJECT")]}),
              ("all companies located in bangalore", {"entities": [(14, 21, "LOCATION"), (25, 34, "OBJECT")]}),
              ("list companies located in bangalore", {"entities": [(15, 22, "LOCATION"), (26, 35, "OBJECT")]}),
              ("show companies based in pune", {"entities": [(24, 28, "OBJECT")]}),
              ("companies in pune", {"entities": [(13, 17, "OBJECT")]}),
              #executives
              ("who are the executives of sbi", {"entities": [(12, 22, "PROPERTY"), (26, 29, "SUBJECT")]}),
              ("give me all the executives of hcl", {"entities": [(16, 26, "PROPERTY"), (30, 33, "SUBJECT")]}),
              ("list all the executives of sbi", {"entities": [(13, 23, "PROPERTY"), (27, 30, "SUBJECT")]}),
              ("list the executives of sbi", {"entities": [(9, 19, "PROPERTY"), (23, 26, "SUBJECT")]}),
              ("show executives of axis bank", {"entities": [(5, 15, "PROPERTY"), (19, 28, "SUBJECT")]}),
              #institutional shareholders
              ("show me the institutional holders of sbi", {"entities": [(12, 33, "PROPERTY"), (37, 40, "SUBJECT")]}),
              ("give me the institutional holders of axis bank", {"entities": [(12, 33, "PROPERTY"), (37, 46, "SUBJECT")]}),
              ("list all institutional holders in coforge", {"entities": [(9, 30, "PROPERTY"), (34, 41, "SUBJECT")]}),
              ("show institutional holders of hcl with change greater than 500000", {"entities": [(5, 26, "PROPERTY"), (30, 33, "SUBJECT"), (39, 45, "PROPERTY"), (46, 58, "COMPARISON"), (59, 65, "NUMBER")]}),
              ("show institutional holders of hcl with shares > 500000", {"entities": [(5, 26, "PROPERTY"), (30, 33, "SUBJECT"), (39, 45, "PROPERTY"), (46, 47, "COMPARISON"), (48, 54, "NUMBER")]}),
              ("show institutional holders of hcl with change more than 500000", {"entities": [(5, 26, "PROPERTY"), (30, 33, "SUBJECT"), (39, 45, "PROPERTY"), (46, 55, "COMPARISON"), (56, 62, "NUMBER")]}),
              ("show institutional holders of hcl with shares less than 100000", {"entities": [(5, 26, "PROPERTY"), (30, 33, "SUBJECT"), (39, 45, "PROPERTY"), (46, 55, "COMPARISON"), (56, 62, "NUMBER")]}),
              ("give institutional holders of hcl with shares < 2000", {"entities": [(5, 26, "PROPERTY"), (30, 33, "SUBJECT"), (39, 45, "PROPERTY"), (46, 47, "COMPARISON"), (48, 52, "NUMBER")]}),
              ("give institutional holders of hcl holding more than 30000 shares", {"entities": [(5, 26, "PROPERTY"), (30, 33, "SUBJECT"), (42, 51, "COMPARISON"), (52, 57, "NUMBER"), (58, 64, "PROPERTY")]}),
              #historical daily stock splits
              ("show historical daily stock splits of hcl", {"entities": [(22, 34, "PROPERTY"), (38, 41, "SUBJECT")]}),
              ("give me daily stock splits of hcl", {"entities": [(14, 26, "PROPERTY"), (30, 33, "SUBJECT")]}),
              ("list the daily stock splits of hcl", {"entities": [(15, 27, "PROPERTY"), (31, 34, "SUBJECT")]}),
              ("list the stock splits of coforge", {"entities": [(9, 21, "PROPERTY"), (25, 32, "SUBJECT")]}),
              #mutual fund holders
              ("give me all the mutual fund holders of sbi", {"entities": [(16, 27, "PROPERTY"), (39, 42, "SUBJECT")]}),
              ("list mutual fund holders of sbi", {"entities": [(5, 16, "PROPERTY"), (28, 31, "SUBJECT")]}),
              ("show mutual fund holders of hcl", {"entities": [(5, 16, "PROPERTY"), (28, 31, "SUBJECT")]}),
              ("show mutual fund holders of kotak mahindra bank with shares greater than 70000", {"entities": [(5, 16, "PROPERTY"), (28, 47, "SUBJECT"), (53, 59, "PROPERTY"), (60, 72, "COMPARISON"), (73, 78, "NUMBER")]}),
              ("list mutual fund holders of infosys with shares greater than 70000", {"entities": [(5, 16, "PROPERTY"), (28, 35, "SUBJECT"), (41, 47, "PROPERTY"), (48, 60, "COMPARISON"), (61, 66, "NUMBER")]}),
              ("list mutual fund holders of infosys with change greater than 20000", {"entities": [(5, 16, "PROPERTY"), (28, 35, "SUBJECT"), (41, 47, "PROPERTY"), (48, 60, "COMPARISON"), (61, 66, "NUMBER")]}),
              ("show all mutual fund holders of infosys with change less than 35000", {"entities": [(9, 20, "PROPERTY"), (32, 39, "SUBJECT"), (45, 51, "PROPERTY"), (52, 61, "COMPARISON"), (62, 67, "NUMBER")]}),
              #stock dividends
              ("give me all the stock dividends of tcs", {"entities": [(16, 31, "PROPERTY"), (35, 38, "SUBJECT")]}),
              ("show all the stock dividends of tcs", {"entities": [(13, 28, "PROPERTY"), (32, 35, "SUBJECT")]}),
              ("list all the stock dividends of tcs", {"entities": [(13, 28, "PROPERTY"), (32, 35, "SUBJECT")]}),
              ("list all the stock dividends of tcs", {"entities": [(13, 28, "PROPERTY"), (32, 35, "SUBJECT")]}),
              ("show all the stock dividends of tcs with dividend more than 10", {"entities": [(13, 28, "PROPERTY"), (32, 35, "SUBJECT"), (41, 49, "PROPERTY"), (50, 59, "COMPARISON"), (60, 62, "NUMBER")]}),
              ("give all the stock dividends of tcs with dividend more than 10", {"entities": [(13, 28, "PROPERTY"), (32, 35, "SUBJECT"), (41, 49, "PROPERTY"), (50, 59, "COMPARISON"), (60, 62, "NUMBER")]}),
              ("show stock dividends of tcs before 25-08-2007", {"entities": [(5, 20, "PROPERTY"), (24, 27, "SUBJECT"), (28, 34, "COMPARISON"), (35, 45, "DATE")]}),
              #market cap
              ("what is the market cap of tcs", {"entities": [(12, 22, "PROPERTY"), (26, 29, "SUBJECT")]}),
              ("show market cap of infosys", {"entities": [(5, 15, "PROPERTY"), (19, 26, "SUBJECT")]}),
              ("give companies with market cap more than 200000", {"entities": [(20, 30, "PROPERTY"), (31, 40, "COMPARISON"), (41, 47, "NUMBER")]}),
              ("list all companies with market cap more than 200000", {"entities": [(24, 34, "PROPERTY"), (35, 44, "COMPARISON"), (45, 51, "NUMBER")]}),
              ("show companies with market cap less than 10000000", {"entities": [(20, 30, "PROPERTY"), (31, 40, "COMPARISON"), (41, 49, "NUMBER")]}),
              #ipo
              ("when did infosys have its ipo", {"entities": [(9, 16, "SUBJECT"), (26, 29, "PROPERTY")]}),
              ("when was tcs's ipo", {"entities": [(9, 12, "SUBJECT"), (15, 18, "PROPERTY")]}),
              ("what is the ipo date of sbi", {"entities": [(12, 15, "PROPERTY"), (24, 27, "SUBJECT")]}),
              ("show companies with ipo before 01-01-1990", {"entities": [(20, 23, "PROPERTY"), (24, 30, "COMPARISON"), (31, 41, "DATE")]}),
              #employees
              ("how many employees does tcs have", {"entities": [(9, 18, "PROPERTY"), (24, 27, "SUBJECT")]}),
              ("how many employees does icici bank have", {"entities": [(9, 18, "PROPERTY"), (24, 34, "SUBJECT")]}),
              ("list number of employees of tcs", {"entities": [(15, 24, "PROPERTY"), (28, 31, "SUBJECT")]}),
              ("give employees of tcs", {"entities": [(5, 14, "PROPERTY"), (18, 21, "SUBJECT")]}),
              ("show companies with employees more than 50000", {"entities": [(20, 29, "PROPERTY"), (30, 39, "COMPARISON"), (40, 45, "NUMBER")]}),
              ("give me the companies with employee strength less than 100000", {"entities": [(27, 36, "PROPERTY"), (45, 54, "COMPARISON"), (55, 61, "NUMBER")]}),
              #misc
              ("what is the founding date of tcs", {"entities": [(12, 25, "PROPERTY"), (29, 32, "SUBJECT")]}),
              ("give the founding date of hcl", {"entities": [(9, 22, "PROPERTY"), (26, 29, "SUBJECT")]}),
              ("what is the revenue of axis bank", {"entities": [(12, 19, "PROPERTY"), (23, 32, "SUBJECT")]}),
              ("what is the net income of coforge", {"entities": [(12, 22, "PROPERTY"), (26, 33, "SUBJECT")]}),
              ("what is the assets of infosys", {"entities": [(12, 18, "PROPERTY"), (22, 29, "SUBJECT")]}),
              ("how many assets do tcs own", {"entities": [(9, 15, "PROPERTY"), (19, 23, "SUBJECT")]}),
              ("how much assets does kotak mahindra bank own", {"entities": [(9, 15, "PROPERTY"), (21, 40, "SUBJECT")]}),
]

entities = []
for _, annotations in TRAIN_DATA:
  for ent in annotations.get("entities"):
    if ent[2] not in entities:
        entities.append(ent[2])
    ner.add_label(ent[2])

optimizer = nlp.resume_training()
move_names = list(ner.move_names)

pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

from spacy.util import minibatch, compounding
import random

with nlp.disable_pipes(*other_pipes):
    sizes = compounding(1.0, 4.0, 1.001)
    for itn in range(30):

        random.shuffle(TRAIN_DATA)

        losses = {}
        for batch in spacy.util.minibatch(TRAIN_DATA, size=sizes):
            for text, annotations in batch:
                # create Example
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                # Update the model
                nlp.update([example], losses=losses)

def question_entities(question):
    doc = nlp(question)
    return [(ent.text, ent.label_) for ent in doc.ents]

def question_type(question):
    question = clean_sentences(question)
    question = vectorizer.transform([question])
    prediction = model.predict(question)
    map_output = {1: "List Type", 2:"Count Type", 3:"Boolean Query"}
    return map_output.get(prediction[0])


app = Flask(__name__)


@app.route('/question', methods=['GET'])
def home():
    if (request.method == 'GET'):
        _question = request.headers['question']
        return jsonify([{'type': question_type(_question)}, {'entities': question_entities(_question)}])

app.run(debug = True)