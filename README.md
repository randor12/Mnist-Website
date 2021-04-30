# Mnist-Website

This is a flask website for running and testing machine learning models and deep learning models for the Mnist dataset in live time.

# About

- Currently for this program, there is a CNN model created with TensorFlow, a SVM (One vs One) model created from scratch, and a Naive Bayes Model created from scratch. The training code is not currently on here. Only the minimum code and the saved models for prediction. 
- The `app.py` file contains the Flask Website backend. To start the site locally, clone the repo and either run `python app.py` or `flask run`. Makes sure the required packages are installed with either pip or conda.
- The `templates` folder contains the HTML used for the website. 
- The `static` folder contains the CSS, JS and bootstrap template used for this project. 
- In order to test the models, draw a number between 0 through 9 (inclusive) on the canvas, and sumbit for prediction. The machine learning model can be selected through the drop down menu for testing. If you accidentally write something you didn't mean to, don't worry! Just press the clear button to clear the canvas. 

# Created By:
- Ryan
- Jeremy
- TK
