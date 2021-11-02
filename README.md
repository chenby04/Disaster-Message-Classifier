# disaster_message_classifier

<summary><h2 style="display: inline-block">Table of Contents</h2></summary>
<ol>
<li>
    <a href="#about-the-project">About The Project</a>
</li>
<li>
    <a href="#getting-started">Getting Started</a>
</li>
<li><a href="#methods-and-results">Methods and Results</a></li>
<li><a href="#limitations-and-future-directions">Limitations and Future Directions</a></li>
<li><a href="#acknowledgements">Acknowledgements</a></li>
</ol>

## About The Project
When diaster happens, people rely on social media and text messages to request for help. If these messages can be processed in a timely and precise manner, there is a higher chance that people with needs can be connected to the appropriate disaster relief agency.

Inspired by this need, this project trains a ***supervised machine learning model*** to automatically process diaster messages and classify people's needs into multiple categories, such as "clothing", "food", or "shelter". The training dataset is a set of real messages collected following diasters, where each message has been labeled with zero or more classes of needs. This project also builds a ***web application*** under the Flask framework, which allows an emergency worker to input a message and get classification results instantly. 

 
## Getting Started
- Prerequisites
    ```
    Python==3.8.8
    Flask==1.1.2
    SQLAlchemy==1.4.7
    plotly==5.3.1
    pandas==1.2.4
    dill==0.3.4
    nltk==3.6.1
    scikit_learn==1.0
    ```
- Clone the repo
   ```
   git clone https://github.com/chenby04/disaster_message_classifier.git
   ```
- File descriptionsd
  
    This project has three main folders. The `app` folder holds the Flask-based web application. The `data` folder holds the raw dataset, the script for an ETL pipeline, and the cleaned data. The `models` folder holds the script for an ML pipeline and the trained model.
   ```
   disaster_message_classifier/
    - app/
    | - templates/
    | |- master.html  # main page of web app
    | |- go.html  # classification result page of web app
    |- run.py  # Flask file that runs app

    - data/
    |- disaster_categories.csv  # raw data to process 
    |- disaster_messages.csv  # raw data to process
    |- process_data.py  # ETL pipeline - merge and clean raw data
    |- InsertDatabaseName.db   # cleaned data saved as sqlite database

    - models/
    |- train_classifier.py # ML pipeline - supervised classification model
    |- classifier.pkl  # model saved as a pickle file

    - requirements.txt # python packages are required to run the project
    - overview.png # snapshot of web app
    - results.png # snapshot of web app
    ```

- Usage
    
    Run the ETL pipeline from `data` folder to clean the raw data and save cleaned data as an sqlite database:
    ```
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
    ```

    Run the ML pipeline from `models` folder to train a supervised classification model using the cleaned data: 
    ```
    python train_classifier.py ../data/DisasterResponse.db classifier.pd
    ```

    Run the web app script from `app` folder and see the app in browser at http://0.0.0.0:3001/
    ```
    python run.py
    ```


## Methods and Results
- Data inspection and cleaning

    The raw dataset consists of 26384 disaster messages, each labeled with zero or more of the 36 pre-defined classes. Upon inspection, 206 messages were duplicated and removed. The inspection also shows a highly skewed class distribution, where major classes like `aid_related` was tagged to over 10k messages while minor classes like `clothing` was tagged to only 404 messages. More surprisingly, the class `child_alone` was tagged to none of the messages and therefore becoming untrainable. This skewed class distribution is known as **class imbalance**, which needs to be taken into consideration during training, as discussed in the following section.

- Machine learning model  

    To prepare for the training, each message was tokenized and transformed. Steps of tokenization included removing punctuations, normalizing case, spliting words by space, removing stop words, lemmatizing, and stemming. Steps of transformation included vectorization and TF-IDF transformation. 
 
    A multi-label classification model was trained. Because accuracy is not a good metric for imbalanced dataset, f1-score was used instead. Note that f1-score has different flavors : `macro f1-score` weighs each class equally whereas `micro f1-score` weighs each sample equally. If the goal of the training is to get as many samples correct as possible not favoring any class, micro is preferred, but if the goal of the training is to get minor classes correct as well, macro is preferred. In this project, macro was employed because I consider minor classes such as `clothing` equally or sometimes more important than major classes such as `aid_related`.

    Five algorithms from the scikit-learn library were evaluated: naive Bayes, logistic regression, random forest, SVM, and a hard voting ensamble of the previous four. To combat the class imbalance, penalty was used in SVM. 

    Grid search was employed to tune the hyperparameters for each algorithm, and results are summarized below.

    

    | Algorithm           	|   Train   	|    Test    	| Train-Test 	|
    |---------------------	|:---------:	|:----------:	|:----------:	|
    | Naive Bayes         	|    0.64   	|    0.60    	|    0.04    	|
    | Logistic Regression 	|    0.74   	|    0.67    	|    0.07    	|
    | Random Forest       	|    0.72   	|    0.64    	|    0.08    	|
    | SVM               	|    0.73   	|    0.68    	|    0.05    	|
    | Voting              	|    0.71   	|    0.64    	|    0.07    	|

    Among the five algorithms, SVM shows the highest score and the second least overfitting. Class-specific scores using SVM were also calculated.

- Web application
    
    The web application serves two purposes. First, it provides a graphical overview of the training data. Second, it implements the trained model to classify user input messages into pre-defined classes. 
    ![Overview](overview.png)
    ![Results](results.png)


## Limitations and Future Directions
The training dataset is limited in size (26384 data entries), which restricts the performance of the model. Future improvements might exploit pretrained word embeddings such as [Word2Vec](https://arxiv.org/pdf/1310.4546.pdf). The word embeddings were trained on large datasets, whose knowledge could potentially be transferred to small datasets to help understand their context. In addition to word embeddings, neural network (deep learning) might also improve the classification performance by extracting higher level features.


## Acknowledgements
- This project is part of the training provided by Udacity's [Data Scientiest Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

- The pre-labeled disaster message dataset was provided by [Figure Eight](https://en.wikipedia.org/wiki/Figure_Eight_Inc.), a human-in-the-loop machine learning and artificial intelligence company.



