# Project: Disaster Response


# Description
This project aims to implement a quick identification and response to a text messages that could be sent in a disaster situation. The project tackle 3 parts which is part of Data scientists role.

- **Extracting, transforming and loading the data.** 
This follow ETL pipeline which in out application the we will extract the data from two .csv files. And merge the files together and apply some transformation, and finally loading these data into sqlite database to be ready to use in the next phase.
- **Building a machine learning model and optimize it.**
This pipeline will firstly read the database and will split the data into training set and testing set. after that the model will be build using a ML pipeline then fitting the model with training data. 
we will save the model to compressed file with extension .lzma. The evaluation function will produce a report that shows the confusion matrix and the accuracy for each category 
- **Placing the machine learning model into a web app for external user to easily interact with the application.**
Finally, the model will be loaded and unzipped, then for each message passed from the front end application, it will be predicted using the model and return a table for which this message it likely to be. 

There are 3 folders in this project which contains the files necessary to execute this project:

```
.
+-- _app
|   +-- templates
|		+-- go.html
|		+-- master.html
|   +-- run.py
+-- _data
|   +-- disaster_categories.csv
|   +-- disaster_messages.csv
|   +-- DisasterDB.db
|   +-- process_data.py
+-- _models
|   +-- classifer_c.lzma
|   +-- train_classifier.py
+-- _README.dm

```
- the `data folder` has the `process_data.py` script which read the two .csv files and return the Disaster.db database.
- the `model folder` has the `train_classifier.py` script which output a **zipped** classifier model. this zipped object can be read by _joblib.load_ function
- the `app folder` contains the `run.py` script that uses flask for the back-end app and internal folder `templates` for the front-end part of the application

# Installation
In order for this application to work correctly all the files and folder should be place in their correct place as shown above. 
follow the steps below to corectly run the application:

 1. Run the script `process_data.py` by typing the command below

    

    `python process_data.py disaster_messages.csv disaster_categories.csv DisasterDB.db`

 2. Run the script `train_classifier.py` by typing the command below:

    `python train_classifier.py ../data/DisasterDB.db classifer_c.lzma`

3. Finally running the script `run.py`

    `python run.py`

this should give you a message in the terminal as bellow:

 ![enter image description here](https://github.com/NasserSubait/p5-disaster-response-project/blob/master/command_line_run_app.PNG?raw=true)   

Once this command line shows that the server is active, you can go to the web browser and type in the URL bar 
`http://localhost:3001`

the front page should appear as the image below:
![enter image description here](https://github.com/NasserSubait/p5-disaster-response-project/blob/master/front_page.PNG?raw=true)

Now you can place the message and find out its categories,
