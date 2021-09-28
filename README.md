# disaster_message_classifier

<summary><h2 style="display: inline-block">Table of Contents</h2></summary>
<ol>
<li>
    <a href="#about-the-project">About The Project</a>
</li>
<li>
    <a href="#getting-started">Getting Started</a>
    <ul>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#installation">Installation</a></li>
    </ul>
</li>
<li><a href="#results">Results</a></li>
<li><a href="#limitations-and-future-directions">Limitations and Future Directions</a></li>
<li><a href="#acknowledgements">Acknowledgements</a></li>
</ol>

## About The Project
When diaster happens, people rely on social media and text messages to request for help. If these messages can be processed in a timely and precise manner, there is a higher chance that people with needs can be connected to the appropriate disaster relief agency.

Inspired by this need, this project trains a **supervised machine learning model** to automatically process diaster messages and classify people's needs into 36 categories, such as "clothing", "child_alone", and "shelter". The training dataset is a set of real messages collected following diasters, where each message has been labeled with one or more classes of needs. This project also builds a **web application** under the Flask framework, which allows an emergency worker to input a message and get classification results instantly. 

 
## Getting Started
### Prerequisites

### Installation
- Clone the repo
   ```
   git clone https://github.com/chenby04/disaster_message_classifier.git

   ```
- File descriptions
  
    This project has three main folders. The `app` folder holds the Flask-based web application. The `data` folder holds the raw dataset, the script for an ETL pipeline, and the cleaned data. The `models` folder holds the script for a ML pipeline and the trained model.
   ```
   disaster_message_classifier/
    - app/
    | - template/
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
    ```

- Usage
    
    Run the ETL pipeline from `data` folder to clean the raw data and save processed data in an sqlite database:
    ```
    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
    ```

    Run the ML pipeline from `models` folder to train a supervised classification model: 
    ```
    python train_classifier.py ../data/DisasterResponse.db classifier.pkl
    ```

    Run the web app from `app` folder to 


## Results
- Machine learning model
    Upon inspection of the raw data, the class `child_alone` had all zero entries and was therefore excluded from the training.  206 rows contained duplicated infomation and were also excluded from the training.

    Five classification algorithms were evaluated: naive Bayes, logistic regression, random forest, SVM, and a hard voting ensamble of the previous four. After hyperparameter fine-tuning with grid search, micro-f1 score was calculated for each algorithm and summarized below.

    | Algorithm           	|   Train   	|    Test    	| Train-Test 	|
    |---------------------	|:---------:	|:----------:	|:----------:	|
    | Naive Bayes         	|    0.64   	|    0.60    	|    0.04    	|
    | Logistic Regression 	|    0.74   	|    0.67    	|    0.07    	|
    | Random Forest       	|    0.72   	|    0.64    	|    0.08    	|
    | **SVM**            	|  **0.73**   	|  **0.68**    	|  **0.05**    	|
    | Voting              	|    0.71   	|    0.64    	|    0.07    	|

    Among the five algorithms, SVM shows the highest score and the second least overfitting. Class-specific scores using SVM were also calculated.


- Web application


## Limitations and Future Directions
The training dataset is imbalanced: it has 36 classes with highly unequal class distribution. One extreme case is the class `child_alone`, which is negative throughout the dataset and therefore being untrainable. Besides, the limited size of the dataset (26384 instances) also restricts the performance of the model.

To combat the imbalanced classes as well as the limited size of the dataset, future improvements might take advantage of pretrained word embeddings such as [Word2Vec](https://arxiv.org/pdf/1310.4546.pdf). The word embeddings was trained on large datasets, whose knowledge could potentially be transferred to small datasets to help understand their context. In addition to word embeddings, neural network (deep learning) can extract higher level features, which could also improve the classification performance.


## Acknowledgements
- This project is part of the training provided by Udacity's [Data Scientiest Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

- The pre-labeled disaster message dataset was provided by [Figure Eight](https://en.wikipedia.org/wiki/Figure_Eight_Inc.), a human-in-the-loop machine learning and artificial intelligence company.



