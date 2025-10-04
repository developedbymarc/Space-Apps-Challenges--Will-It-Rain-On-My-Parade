# Cloud 67 Team Members

- Marc Gulgulian (developedbymarc)
- Boghos Hamalian (Boghos)
- Joe Mozian (IsolatedLun)
- Levon Jamgossian (Ghostycodes2077)

# Space Apps Challenges: Will It Rain On My Parade?

## Project overview

Our project is a web-based probabilistic weather prediction application which provides the user with a dashboard to compute the probability of their desired weather conditions for a given day and suggests an alternative date if the probability of the event is below a certain threshold.

## ✍️ Project description

The "Will It Rain On My Parade?" application is a web-based tool designed to provide users with probabilistic weather predictions. The application will consist of several key components:

* Data Acquisition: We download and store meteorological data from NASA's MERRA-2, specifically the M2T1NXSLV and M2T1NXFLX collections. The variables we selected will serve as the foundation for our model.
* Data Processing and Cleaning: Implement data cleaning techniques using pandas to remove any garbage values, handle missing datapoints, and ensure the dataset is structured for analysis.
* Feature Engineering: We create features from the cleaned and discretized data, such as temperature, humidity, precipitation categories.
* Model Development: We use bnlearn to fit the data to a Bayesian Network using Hill Climbing and BIC scoring. This model will be trained on the processed data to compute the probability of desired weather conditions.
* User Interface Development: We utilize Streamlit to create a frontend dashboard where users can input a target date and location alongside their desired weather conditions.
* Probability Computation: We calculate the likelihood of the specified conditions for the given date and location using the Bayesian model and present the results in an easy-to-understand format.
* Alternative Date Suggestions: If the probability is below a specified threshold, the application will analyze nearby dates and suggest alternatives that are closer to the desired weather criteria.
* Multi-Day Event Support: We allow users to select date ranges for events and provide suggestions for these periods, enhancing functionality for more complex planning scenarios.

## 🎯 Goals

1. Process, clean and store NASA meteorological data
2. Build and fit a probabilistic model (Bayesian Network) with the cleaned data
3. Provide a user-friendly interface where the user can input their desired weather conditions for a given day of the year and return its computed probability
4. Suggest days closer to the desired weather conditions in a time window around the initial desired date
5. Allow the user to select date ranges for multi-day events
6. Suggest a date range closer to the desired weather conditions in a time window around the initial desired date range

## 👥 Team

| Role | Name |
|------|------|
| Project Manager, Tech & Research Lead | Marc Gulgulian |
| Backend & Data Engineer | Boghos Hamalian |
| Frontend Engineer | Joe Mozian |
| Frontend Engineer | Levon Jamgossian |


### Team Background

Marc Gulgulian, Levon Jamgossian and Joe Mozian are junior Computer Science (undergraduate) at Haigazian University. Boghos Hamalian has earned his BSc. in Computer Science from Haigazian University in 2024-2025. 

## ✅ Task tracker

We use this task tracker to keep track of team tasks:
[Task Tracker](https://app.slack.com/lists/T09FYL0G6AD/F09F520711V)

🔑 Key resources

1. https://disc.gsfc.nasa.gov/datasets/M2T1NXSLV_5.12.4/summary?keywords=M2T1NXSLV_5.12.4
2. https://disc.gsfc.nasa.gov/datasets/M2T1NXFLX_5.12.4/summary?keywords=M2T1NXFLX_5.12.4
3. https://www.researchgate.net/publication/251920405_Bayesian_network_probability_model_for_weather_prediction
4. https://www.researchgate.net/publication/220836968_Bayesian_Networks_for_Probabilistic_Weather_Prediction

