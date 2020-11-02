#-- Import lib
import pandas as pd
import numpy as np
#Production
import streamlit as st

#Preprocess NLP
import spacy
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords

#Split et preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
#Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
#Evaluation
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,accuracy_score,f1_score,confusion_matrix
from sklearn.decomposition import TruncatedSVD

# Import data
@st.cache
def load_data(file_path):
    data=pd.read_csv(file_path)
    return data


def encode_label(serie_label):
    lb = LabelBinarizer()
    y_new= lb.fit_transform(serie_label)
    return y_new

# Train test split
def split (data,label_col):
    df_train,df_test = train_test_split(data,train_size=0.3,random_state=1,stratify=data[label_col])
    return df_train,df_test

# Bow binaire
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles) if t.isalpha()]


def bow(train_data,comment_col):
    parseur = CountVectorizer(binary=True,
                                tokenizer=LemmaTokenizer(),
                                strip_accents = 'unicode',
                                stop_words = 'english',
                                lowercase = True,
                                token_pattern = r'\b[a-zA-Z]{3,}\b', # keeps words of 3 or more characters
                                max_df = 0.5,
                                min_df = 10)
    X_train_parse= parseur.fit_transform(train_data[comment_col])
    return parseur,X_train_parse

# SVD
def svd(X_train_parse):
    svd=TruncatedSVD(n_components=2)
    bow_svd = svd.fit_transform(X_train_parse)
    df_svd=pd.DataFrame(data = bow_svd
             , columns = ['principal component 1', 'principal component 2'])
    return df_svd

# MODELING

# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

class Custom():
    MODELS ={'CART':{'params':
                {'class_weight': 'balanced',
                   'criterion': 'entropy',
                    'max_depth': 7,
                    # 'max_features': 31,
                    'max_leaf_nodes': 21,
                    'min_samples_leaf': 21,
                    'min_samples_split': 19},
                  'model': DecisionTreeClassifier},

            'Random Forest':{'params':
                {'bootstrap': True,
                    'class_weight': 'balanced',
                    'criterion': 'gini',
                    'max_depth': 50,
                    # 'max_features': 5,
                    # 'max_leaf_nodes': 27,
                    # 'min_samples_leaf': 15,
                    'min_samples_split': 50},
                'model': RandomForestClassifier},

            'xgb':{'params':
                    {'alpha': 2,
                    'colsample_bytree': 0.63,
                    'gamma': 8,
                    'lambda': 6,
                    'learning_rate': 0.8,
                    'max_delta_step': 21,
                    'max_depth': 18,
                    'objective': 'binary:hinge',
                    'subsample': 0.85},
                    'model': XGBClassifier},

            'SVM':{'params':
                        {'tol': 0.001,
                        'penalty': 'l2',
                        'loss': 'squared_hinge',
                        'intercept_scaling': 2,
                        'dual': False,
                        'C': 0.01,
                        'class_weight': 'balanced'},
                        'model': LinearSVC},

            'Logistic Regression':{'params':
                    {'solver': 'saga',
                        'fit_intercept': True,
                        'dual': False,
                        'class_weight': 'balanced'},
                    'model':LogisticRegression}
            }

    def __init__(self, model):
        self.model = self.MODELS[model]['model']
        self.params = self.MODELS[model]['params']

    def train_model(self, X_train_parse,train_data,label_col):

        self.pipeline = Pipeline([
            ('clf', self.model(**self.params))
        ])
        # Fit pipeline
        return self.pipeline.fit(X_train_parse, train_data[label_col])


def evaluate_model (parseur, test_data, comment_col,label_col,model):
    X_test_parse=parseur.transform(test_data[comment_col])
    y_pred=model.predict(X_test_parse)
    report= classification_report(test_data[label_col],y_pred,output_dict=True)
    report=pd.DataFrame(report).transpose().iloc[:3,[0,1,2]]
    accuracy=round(accuracy_score(test_data[label_col],y_pred),1)
    #F1-Score
    f1score=round(f1_score(test_data[label_col],y_pred,average='weighted'),1)
    #Confusion matrix
    cm=confusion_matrix(test_data[label_col], y_pred)

    return report, accuracy, f1score, cm


LANGUAGE={"af":"Afrikaans",
"ar": "Arabic",
"bg":"Bulgarian",
"bn":"Bengali",
"ca":"Catalan",
"cs": "Czech",
"cy": "Welsh",
"da":"Danish",
"de":"German",
"el":"Greek",
"en": "English",
"es":"Spanish/Castillan",
"et":"Estonian",
"fa":"Persian",
"fi":"Finnish",
"fr": "French",
"gu": "Gujarati",
"he":"Hebrew",
"hi":"Hindi",
"hr":"Croatian",
"hu":"Hungarian",
"id":"Indonesian",
"it":"Italian",
"ja":"Japanese",
"kn":"Kannada",
"ko":"Korean",
"lt":"Lithuanian",
"lv":"Latvian",
"mk":"Macedonian",
"ml":"Malayalam",
"mr":"Marathi",
"ne":"Nepali",
"nl":"Dutch",
"no":"Norwegian",
"pa":"Punjabi",
"pl":"Polish",
"pt":"Portuguese",
"ro":"Romanian",
"ru":"Russian",
"sk":"Slovak",
"sl":"Slovenian",
"so":"Somali",
"sq":"Albanian",
"sv":"Swedish",
"sw":"Swahili",
"ta":"Tamil",
"te":"Telugu",
"th":"Thai",
"tl":"Tagalog",
"tr":"Turkish",
"uk":"Ukrainian",
"ur":"Urdu",
"vi":"Vietnamese",
"zh-cn": "Chinese"}


def detecte_language(message):
    # an empty dictionary is defined
    # {language : number of common stopwords between language and message words}
    languages_shared_words = {}
    words = wordpunct_tokenize(message)
    for language in stopwords.fileids():
        # stopwords for each language
        stopwords_liste = stopwords.words(language)
        # we clean duplicates
        words = set(words)
        common_elements = words.intersection(stopwords_liste)
        # add couple to the dictionary
        languages_shared_words[language] = len(common_elements)
    #return language with max shared words
    return  max(languages_shared_words, key = languages_shared_words.get)

def detect_language_data(series_comments):
    language=series_comments.apply(lambda comment: detecte_language(comment))
    language= list(language)
    #language=[detect(comment) for comment in series_comment.values]
    language_count=[(x, language.count(x)) for x in set(language)]
    language_max=max(language_count, key = lambda x : x[1])[0]
    return language_max


def validation_comment(comment,lang_data):
    language_comment=detecte_language(comment)
    lang_comment=list(LANGUAGE.keys())[list(LANGUAGE.values()).index(language_comment.capitalize())]
    if lang_comment != lang_data:
        return 0
    else:
        return 1
