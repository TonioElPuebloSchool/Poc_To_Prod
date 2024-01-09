<img style="float: left; padding-right: 10px; width: 200px" src="https://upload.wikimedia.org/wikipedia/fr/b/b1/Logo_EPF.png?raw=true"> 

## Poc To Prod
**P2024** MMMDE4IN19-22-antoine-courbi

# Developping a web application

### Created by Antoine Courbi

-----
### [*Moodle course*](https://moodle.epf.fr/mod/folder/view.php?id=344972)
### [*Github depository*](https://github.com/TonioElPuebloSchool/Poc_To_Prod)
-----

This **README** is meant to explain **how to run** the application and what **features** are implemented.

The application is a **simple application** using **flask** enabling a user to make a **prediction** on a **trained ML model**. The dataset is a **stackoverflow topics posts dataset**.

# **Requirements**

The `requirements` can be installed as follow using **command line**:
```bash
pip install -r requirements.txt
```
The two following **commands** can then be used in the `predict\predict` folder to **run the application**:
```bash
set FLASK_APP=app.py
flask run
```
After which the **application** can be accessed at http://127.0.0.1:5000/.

# **Overview**

This project is a `Flask-based` **web application** designed to perform **text predictions**. It uses a **trained machine learning model** to **predict the label** of a given text input. The application provides a simple interface where users can enter their text and receive a prediction.

# **Architecture**

The application is structured as follows:

- `app.py`: The main application file. It initializes the Flask application and defines the routes.
- `predict/predict.py`: Contains the `TextPredictionModel` class used for making predictions.
- `artefacts_path`: A directory containing the trained model and other necessary artefacts.

# **Details**

The application provides a **single route** (`/`) that accepts both `GET` and `POST` requests. On a `GET request`, it simply renders the **prediction form**. On a `POST request`, it takes the user's text from the form, makes a prediction using the model, and then **displays the prediction on the page**.

<img style="float: center; padding-right: 10px; width: 250px" src="data_readme\webpage.png">


# **Testing**

This project includes a suite of **unit tests** to ensure the functionality of the `TextPredictionModel` class in the `run.py` file. These tests are implemented using the **unittest framework** and are located in the `test_predict.py` file.

**The tests cover the following areas:**

- Loading **model**, **parameters**, and **labels** from **artefacts** 
>The `test_from_artefacts_loads_model_params_and_labels` test verifies that the `from_artefacts` class method of `TextPredictionModel` correctly loads the **model**, **parameters**, and **labels** from the specified **artefacts** directory. The test uses **mocking** to avoid actual file reading and model loading.


- Calling **embedding** and **model prediction** 
>The `test_predict_calls_embedding_and_model_predict` test checks the `predict` method of `TextPredictionModel`. It verifies that the method correctly calls the **embed function** and the **predict method** of the model. The test uses a **mock model** and the **embed function** is also mocked.

- **Format** and **length** of predictions
>The `test_predict_with_mock_model` test checks the predict method of `TextPredictionModel` and verifies the **format** and **length** of the **predictions** returned by the predict method. It uses a **mock model** and the **embed function** is also mocked.

The `setUp` method is run before each test method, and it sets up a **temporary directory** for testing. The `tearDown` method is run after each test method, and it removes the **temporary directory**.

To run these tests, navigate to the project directory and use the following command in your terminal:
```bash
python -m unittest test_predict.py
```
This will **run all the tests** and display the **results** in the **terminal**.


------
Thank you for reading this README. I hope you enjoyed it. If you have any remarks or questions, feel free to contact me on [LinkedIn](https://www.linkedin.com/in/antoine-courbi/).

<p align="center">&mdash; ⭐️ &mdash;</p>
<p align="center"><i>This README was created during the Poc-To-Prod course</i></p>
<p align="center"><i>Created by Antoine Courbi</i></p>