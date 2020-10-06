# Making a Dashboard with Heroku

To make interactive python applications, we here use Flask through Dash. We make interactive plots using plotly and present the finished applications on Heroku.
Heroku can be used for free. However, note that the free version comes with a few limitations: you are only given 1 duno, meaning that your app only can respond to
one user request at a time. If each request only takes ms, a couple of users will, however, not feel any drag. You are also limited to 300 MB in total per application.
To see how the app looks on Heroku, click [here](https://apokasc2.herokuapp.com/). The present example is very minimalistic.

To create the example, we use publically available data from the [APOKASC-II sample](https://arxiv.org/abs/1804.09983).

## app.py

This python script is the core of the application. It contains an app-layout, in which you can set up your homepage, with an interactive callback (that changes as a result of the user input).
This python script could be run on your own computer through the terminal. All the remaining files are only there to enable heroku to run the script as well.

## Procfile

This file tells Heroku how to launch your application. We do this through gunicorn.

## runtime.txt

This file specifies how to run app.py. Specify the python version.

## requirements.txt

This file tells Heroku which packages to load. Since we use gunicorn, we must load this. In addition, we must specify all the python packages involved. Note that plotly needs pandas.

