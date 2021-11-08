# :heart: Streamlit-SentimentAnalysis :broken_heart:
### First Streamlit project to train classification models and predict the feeling of a commentary.


This application allows you to train a classification model among : Logistic Regression, XGBOOST, CART, Random Forest, SVM. 
Each model has been implemented with prefixed parameters. Moreover, only Bag of word binary vectorization has been implemented here. 

In order to test the app: 
- Download the folder
- Open your terminal : 
  - Go into folder 
  - Download the dependencies : >> pip install -r requirements.txt 
  - Run the app : >> streamlit run app.py

Warning: the loaded dataset must have ";" separators and in CSV format.
