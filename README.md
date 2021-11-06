# Disaster-Message-Classifier

<summary><h2 style="display: inline-block">Table of Contents</h2></summary>
<ol>
<li>
    <a href="#about-the-project">About The Project</a>
</li>
<li>
    <a href="#getting-started">Getting Started</a>
</li>
<li><a href="#methods-and-results">Methods and Results</a></li>
<li><a href="#discussion">Discussion</a></li>
<li><a href="#acknowledgements">Acknowledgements</a></li>
</ol>

## About The Project
When diaster happens, people rely on social media and text messages to request for help. If these messages can be processed in a timely and precise manner, there is a higher chance that people with needs can be connected to the appropriate disaster relief agency.

Inspired by this need, this project trains a ***supervised machine learning model*** to automatically process diaster messages and classify people's needs into multiple classes, such as "clothing", "food", and "shelter". This project also builds a Flask-based ***web application***, which allows an emergency worker to input a message and instantly get classification results. 

 
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
    scikit_learn==1.0.1
    ```
- Clone the repo
   ```
   git clone https://github.com/chenby04/Disaster-Message-Classifier.git
   ```
- File descriptions
  
    This project has three main folders. The `app` folder holds the Flask-based web application. The `data` folder holds the raw dataset (messages collected following diasters with class labels), the script for an ETL (Extract, Transform, Load) pipeline, and the cleaned data. The `models` folder holds the script for an ML (Machine Learning) pipeline and the trained model.
   ```
   Disaster-Message-Classifier/
    - app/
    | - templates/
    | |- master.html  # main page of web app
    | |- go.html  # classification result page of web app
    |- run.py  # Flask file that runs app

    - data/
    |- disaster_categories.csv  # raw data to process 
    |- disaster_messages.csv  # raw data to process
    |- process_data.py  # ETL pipeline - merge and clean raw data
    |- DiasterResponse.db   # cleaned data saved as sqlite database

    - models/
    |- train_classifier.py # ML pipeline - supervised classification model
    |- classifier.pkl  # model saved as a pickle file

    - requirements.txt # python packages are required to run the project
    - Overview.png # snapshot of web app
    - Results.png # snapshot of web app
    ```

- Usage
    
    Run the ETL pipeline from `data` folder to clean the raw data and save cleaned data as an sqlite database:
    ```
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
    ```

    Run the ML pipeline from `models` folder to train a supervised classification model using the cleaned data: 
    ```
    python train_classifier.py ../data/DisasterResponse.db classifier.pkl
    ```

    Run the web app script from `app` folder and see the app in browser at localhost:3001/
    ```
    python run.py
    ```


## Methods and Results
- Data inspection and cleaning

    The raw training dataset consists of 26384 messages collected following diasters, where each message has been labeled with zero or more of the 36 pre-defined classes. Each class is a type of help needed, such as "clothing" or "food". 
    
    Upon inspection, 206 duplicated messages were removed. The inspection also shows a highly skewed class distribution, where major classes like `aid_related` was tagged to over 10k messages while minor classes like `clothing` was tagged to only 404 messages. More surprisingly, the class `child_alone` was tagged to none of the messages and therefore becoming untrainable. This skewed class distribution is known as ***class imbalance***, which needs to be taken into consideration during training, as discussed in the following section.

- Machine learning model  

    To prepare for the training, each message was tokenized and transformed. Steps of tokenization included removing punctuations, normalizing case, spliting words by space, removing stop words, lemmatizing, and stemming. Steps of transformation included vectorization and TF-IDF transformation. 
 
    A multi-label classification model was trained. Because accuracy is not a good metric for imbalanced dataset, f1-score was used instead. Note that f1-score has different flavors: `macro f1-score` weighs each class equally whereas `micro f1-score` weighs each sample equally. If the goal of the training is to get as many samples correct as possible not favoring any class, micro is preferred, but if the goal of the training is to get minor classes correct as well, macro is preferred. ***In this project, macro was employed because I consider minor classes such as `clothing` equally or sometimes more important than major classes such as `aid_related`***.

    Five estimators from the scikit-learn library were evaluated: naive Bayes, logistic regression, random forest, SVM, and a hard voting ensamble of the previous four. Balanced class-weights were employed when applicable to help combat imbalaned classes. Grid search was used to tune the hyperparameters for each algorithm, and the optimized macro f1-scores are summarized below.
    
    | Estimator           	|   Train   	|    Test    	| Train-Test Diff	|
    |---------------------	|:---------:	|:----------:	|:---------------:	|
    | Naive Bayes         	|    0.35   	|    0.29    	|    0.06    	    |
    | Logistic Regression 	|    0.59   	|    0.43    	|    0.16       	|
    | Random Forest       	|    0.73   	|    0.47    	|    0.26       	|
    | SVM               	|    0.83   	|    0.48    	|    0.35       	|
    | Voting              	|    0.76   	|    0.48    	|    0.28       	|

    Among the five estimators, Voting and SVM have the highest score. Voting further has less overfitting (less train-test score difference), so it is considered the best estimator. The class-specific scores of the voting estimator are displayed below. Note that although several methods were used to combat class imbalance, the estimator still underperformed when classifying minority classes. 

    | Class                    	| Precision 	| Recall 	| F1-score 	| Count 	|
    |------------------------	|:---------:	|:------:	|:--------:	|:-----:	|
    | related                	|    0.93   	|  0.79  	|   0.86   	|  3987 	|
    | request                	|    0.61   	|  0.73  	|   0.67   	|  925  	|
    | offer                  	|    0.25   	|  0.14  	|   0.18   	|   28  	|
    | aid_related            	|    0.75   	|  0.70  	|   0.72   	|  2207 	|
    | medical_help           	|    0.48   	|  0.62  	|   0.54   	|  428  	|
    | medical_products       	|    0.40   	|  0.65  	|   0.49   	|  261  	|
    | search_and_rescue      	|    0.37   	|  0.35  	|   0.36   	|  148  	|
    | security               	|    0.16   	|  0.13  	|   0.14   	|  103  	|
    | military               	|    0.47   	|  0.67  	|   0.56   	|  192  	|
    | water                  	|    0.65   	|  0.82  	|   0.73   	|  356  	|
    | food                   	|    0.74   	|  0.83  	|   0.78   	|  583  	|
    | shelter                	|    0.60   	|  0.80  	|   0.69   	|  466  	|
    | clothing               	|    0.46   	|  0.54  	|   0.50   	|   74  	|
    | money                  	|    0.40   	|  0.59  	|   0.48   	|  122  	|
    | missing_people         	|    0.34   	|  0.24  	|   0.28   	|   51  	|
    | refugees               	|    0.41   	|  0.51  	|   0.46   	|  196  	|
    | death                  	|    0.55   	|  0.65  	|   0.60   	|  243  	|
    | other_aid              	|    0.36   	|  0.54  	|   0.43   	|  692  	|
    | infrastructure_related 	|    0.28   	|  0.44  	|   0.34   	|  371  	|
    | transport              	|    0.39   	|  0.41  	|   0.40   	|  239  	|
    | buildings              	|    0.46   	|  0.67  	|   0.54   	|  283  	|
    | electricity            	|    0.41   	|  0.58  	|   0.48   	|  103  	|
    | tools                  	|    0.00   	|  0.00  	|   0.00   	|   27  	|
    | hospitals              	|    0.30   	|  0.23  	|   0.26   	|   61  	|
    | shops                  	|    0.00   	|  0.00  	|   0.00   	|   35  	|
    | aid_centers            	|    0.19   	|  0.17  	|   0.18   	|   72  	|
    | other_infrastructure   	|    0.22   	|  0.36  	|   0.27   	|  243  	|
    | weather_related        	|    0.80   	|  0.80  	|   0.80   	|  1387 	|
    | floods                 	|    0.65   	|  0.68  	|   0.66   	|  392  	|
    | storm                  	|    0.67   	|  0.83  	|   0.74   	|  489  	|
    | fire                   	|    0.47   	|  0.40  	|   0.43   	|   58  	|
    | earthquake             	|    0.80   	|  0.84  	|   0.82   	|  456  	|
    | cold                   	|    0.52   	|  0.59  	|   0.56   	|  113  	|
    | other_weather          	|    0.28   	|  0.48  	|   0.36   	|  243  	|
    | direct_report          	|    0.56   	|  0.69  	|   0.62   	|  1053 	|
    |                        	|           	|        	|          	|       	|
    | **macro avg**             |    **0.46**   |  **0.53** |   **0.48**| **16687** |

- Web application
    
    A Flask-based web application was developed. The web application serves two purposes. First, it provides a graphical overview of the training data. Second, it implements the trained model to classify user input messages into pre-defined classes. 
    ![Overview](Overview.png)
    ![Results](Results.png)


## Discussion
Despite trying multiple models and using grid search to search for optimized parameters, the performance of the model was mediocre. Part of this can be attributed to the training dataset's limited size and imbalanced classes. However, ***what limited the performance most may be the poor labeling quality of the training data***. For example, one training data contains the message "How we can find food and water? we have people in many differents needs, and also for medicine at Fontamara 43 cite Tinante.", but it was labeled as negative for classes such as "medical_products", "water", and "food". In another example, the training message is "People from Dal blocked since Wednesday in Carrefour, we having water shortage, food and medical assistance.", but it was labeled as true for "earthquake". 

No model can make up for bad data - especially in supervised learning. To truly improve the classification performance, the labels of the training data must be accurate and consistent. 


## Acknowledgements
- This project is part of the training provided by Udacity's [Data Scientiest Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025).



