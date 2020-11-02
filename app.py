# -- Import lib

#Module
import utils
from utils import np,pd,st
import matplotlib.pyplot as plt
import SessionState

# SET TITLE
st.title("""Analyze your comments""")
st.header("Is your comment positive or negative ? ")

# Save data
session_state = SessionState.get(bar="",model_state="",train="", test="", y="", X="",vocab="", good="", hate="", xtrain_parse="",model="", pipe="",lang="")

#EXPANDER README_______________________________________________________________________________

expander1 = st.beta_expander("Read Me")
expander1.write(""" The objectives of this application are twofold:\n
:heavy_check_mark: To train and optimize your sentiment analysis model according to the parameters you choose.\n
:heavy_check_mark: Analyze new comments \n
It's very simple, here is the procedure to folow: \n
1. Train your model:
    - Unroll the side bar
    - Upload your dataset
    - Select the label and comment columns
    - Define the value of the label comment
    - Select your model

2. Press the prediction button to see the result.""")
expander1_bis = st.beta_expander("Vizualise your data")
expander2 = st.beta_expander("Train your model")
expander3 = st.beta_expander("Check a new comment with your trained model")

#SIDE BAR_______________________________________________________________________________

#DATA LOADING
uploaded_file = st.sidebar.file_uploader("Choose your dataset")
try:
    if uploaded_file is not None:
        # Create a text element and let the reader know the data is loading
        data_load_state = st.sidebar.text('Loading data...')
        data = utils.load_data(uploaded_file)
        data_load_state.text("Upload done !")
        # SELECT LABEL AND COMMENT
        session_state.X = st.sidebar.selectbox(label="Which of these columns is the comment?", options=data.columns,  key="comment")
        session_state.y = st.sidebar.selectbox(label="Which of these columns is the label?", options=data.columns,  key="label")

    pressed= st.sidebar.button('Click when selection is done')
    if pressed:
        #Select interesting columns
        data=data.loc[:,[session_state.X,session_state.y]]

        # BINARIZE LABEL
        if data[session_state.y].dtype in ["object", "str"]:
            #First value is transorfmed to 1 with Binary Labelization
            val_base=data.iloc[0,1]
            #Save other values
            val_others= [lab for lab in list(set(data[session_state.y].values)) if lab != val_base]
            assert len(data[session_state.y].value_counts())<=2, "y doit être binaire"
            data[session_state.y]=utils.encode_label(data[session_state.y])
            st.sidebar.write("Label Encoding: ")
            st.sidebar.write(f"- {val_base}: transform to 1")
            st.sidebar.write(f"- {val_others[0]}: transform to 0")
            # Improvment: Word2vec to similarity
            if str.lower(val_base) in ["good", "positive", "pos", "p", "kind", "cool"]:
                session_state.good=1
                session_state.hate=0
            else:
                session_state.good=0
                session_state.hate=1

        else:
            # Define label by selecting good and bad values for comment
            session_state.hate = st.sidebar.select_slider('Select the label for bad comment',value=0, options=list(set(data[session_state.y].values)))
            if len(data[session_state.y].value_counts())>2:
                    session_state.good=st.sidebar.select_slider('Select the label for good comment',value=2,options=list(set(data[session_state.y].values)))
            else:
                    potentiel_lab=[0,1]
                    session_state.good=[lab for lab in potentiel_lab if lab != session_state.hate][0]


        #EXPANDER VIZ_______________________________________________________________________________
        # 0. Print training dataset
        expander1_bis.subheader(":mag_right: Visualization of the training dataset")
        expander1_bis.write(data)

        # 1. Detect language
        session_state.model_state = st.sidebar.text('Model is checking the language...')
        session_state.bar=st.sidebar.progress(15)
        language= utils.detect_language_data(data.iloc[0:int(len(data)/2+1),0])
        session_state.lang=list(utils.LANGUAGE.keys())[list(utils.LANGUAGE.values()).index(language.capitalize())]
        expander1_bis.markdown(f" :grey_exclamation: **The language of your data is {utils.LANGUAGE[session_state.lang]}**")


        # 2. Print distribution of the label class
        expander1_bis.subheader(":scales: Frequency distribution of the class attribute")
        expander1_bis.write(data[session_state.y].value_counts())
        if data[session_state.y].value_counts()[session_state.hate]>data[session_state.y].value_counts()[session_state.good]:
            nb=round(data[session_state.y].value_counts()[session_state.hate]/data[session_state.y].value_counts()[session_state.good],0)
            if nb>1.5:
                expander1_bis.text(f"Imbalance between classes: Il y a {nb} fois plus de commentaires haineux que de commentaires bienveillants dans le dataset ")
            else:
                expander1_bis.text(f"Il y a {nb} fois plus de commentaires haineux que de commentaires bienveillants dans le dataset ")

        else:
            nb=round(data[session_state.y].value_counts()[session_state.good]/data[session_state.y].value_counts()[session_state.hate],0)
            if nb>1.5:
                expander1_bis.text(f"Imbalance between classes: There are {nb} times more kind than hateful comments in the dataset.")
            else:
                expander1_bis.text(f"There are {nb} times more kind than hateful comments in the dataset.")

        # 3. Get the VOCABULARY (BOW)
        session_state.model_state.text('Model is checking the vocabulary...')
        session_state.bar.progress(20)

        #Split data into train an test dataset
        session_state.train,session_state.test= utils.split(data,session_state.y)

        # Get BOW from train dataset
        expander1_bis.subheader(":notebook: Most frequent words in the vocabulary")
        session_state.vocab,session_state.xtrain_parse=utils.bow(session_state.train,session_state.X)

        # Vocab done, progress bar
        session_state.bar.progress(50)


        #Check most famous words in BOW
        freq_mots = np.sum(session_state.xtrain_parse.toarray(),axis=0)
        index = np.argsort(freq_mots)[::-1]
        imp = {'words':np.asarray(session_state.vocab.get_feature_names())[index],'freq':freq_mots[index]}
        expander1_bis.write(pd.DataFrame(imp))

        # Vocab done, progress bar
        session_state.model_state.text("Now you can train your model")
        session_state.bar.progress(100)


        # Size Reduction : Truncated svd  -
        expander1_bis.subheader(":bar_chart: Reduction of vocabulary dimensionality")
        df_svd=utils.svd(session_state.xtrain_parse)
        df_svd[session_state.y]=session_state.train[session_state.y].values
        expander1_bis.write(df_svd)

        fig = plt.figure(figsize = (4,4))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Principal Component 1', fontsize = 8)
        ax.set_ylabel('Principal Component 2', fontsize = 8)
        ax.set_title('2 Components Truncated-SVD', fontsize = 8)
        targets = list(set(df_svd[session_state.y].values))
        colors_max = ['red', 'yellow', 'blue','green']
        colors=[]
        for i in range(len(targets)):
            colors.append(colors_max[i])
        for target, color in zip(targets,colors):
            indicesToKeep = df_svd[session_state.y] == target
            ax.scatter(df_svd.loc[indicesToKeep, 'principal component 1']
                       , df_svd.loc[indicesToKeep, 'principal component 2']
                       , c = color
                       , s = 50, alpha=0.5,edgecolors='None')
        ax.legend(targets)
        ax.grid()
        expander1_bis.pyplot(fig)


    #EXPANDER TRAIN_______________________________________________________________________________
    #2. Train the model and get the metrics
    choice_model=expander2.selectbox("Which model do you want to train? ",list(utils.Custom.MODELS.keys()))
    button=expander2.button('Train')
    if button:
        session_state.pipe=utils.Custom(choice_model)
        session_state.model=session_state.pipe.train_model(session_state.xtrain_parse.toarray(),session_state.train, session_state.y)
        metrics, accuracy, f1, cm = utils.evaluate_model(session_state.vocab, session_state.test,session_state.X,session_state.y, session_state.model)
        expander2.subheader(":brain: Evaluation du modele")
        expander2.write(metrics)
        expander2.markdown("**Score :**")
        left_column,middle1,middle2,right_column = expander2.beta_columns(4)
        left_column.write(f"The accuracy is : {accuracy}")
        middle1.write(f"The f1 score is : {f1}")
        expander2.markdown("**Matrix of confusion :**")
        expander2.write(cm)

        #session_state.model_state.text('Model is done !')
        #session_state.bar.progress(100)
except Exception:
    st.markdown(":exclamation: **RELOAD YOUR DATA OR CHECK IF YOU HAVE SELECTED THE RIGHT COLUMNS**")


    #_______________________________________________________________________________
#3. Predict
new_comment=expander3.text_input("Write your comment")
if expander3.button('Predict') and new_comment!= "":
# Checl if langauge is ok
    detection=utils.validation_comment(new_comment, session_state.lang)
    if detection ==0 :
        expander3.markdown(f" :exclamation: **Write your comment in the same language of your data : {utils.LANGUAGE[session_state.lang]}**")
    else:
        expander3.markdown(":grey_exclamation: **Language used is ok... Labelization is loading...**")

        # set iterable to predict function
        new_comment_list=[new_comment]
        comment_parse=session_state.vocab.transform(new_comment_list)
        prediction = session_state.model.predict(comment_parse)
        if prediction[0]==session_state.good:
            expander3.markdown(":heart: **Your comment is great**")
        else:
            expander3.markdown(":broken_heart: **Your comment is hateful**")
        expander3.balloons()
