# How to run the app

## Prerequisites

- Python 3.12
- Git
- At least 4GB RAM available

## Installation

### 1. Clone the Repository

```
git clone https://github.com/YOUR_USERNAME/space-apps-challenges--will-it-rain-on-my-parade.git;

cd space-apps-challenges--will-it-rain-on-my-parade;
```

### 2. Create Virtual Environment

*Windows:*

```
python -m venv venv;

venv\Scripts\activate;
```

*macOS/Linux:*

```
python3 -m venv venv;

source venv/bin/activate;
```

### 3. Install Dependencies

```
pip install --upgrade pip

pip install -r requirements.txt
```

If you don't have a `requirements.txt`, create one with:

```
streamlit
pandas
numpy
bnlearn
plotly
streamlit-folium
folium
google-generativeai
```

### 4. Set Up Google Gemini API Key (Optional - for AI summaries)

1. Create a folder called `.streamlit`

2. Create a `secrets.toml` file in the `.streamlit` folder

2. Add the following API key to .streamlit/secrets.toml:
   
   ```
   GEMINI_API_KEY = "GENERATE_AND_ENTER_YOUR_OWN_KEY"
   ```   

### 5. Verify Database File

Make sure `large_merra2_data.db` is in your project root directory. The app will crash if this file is missing.

## Running the App

### Start the Streamlit Server

```
python -m streamlit run app.py
```

The app will automatically open in your browser.

If it doesn't open automatically, navigate to the URL shown in your terminal (usually http://localhost:8501).

### First Run

- *First startup will take 1-2 minutes* while the Bayesian Network trains on the data
- Subsequent runs will be faster due to caching

## Usage

1. *Select Location*: Choose from available locations in the database
2. *Select Date*: Pick any date (predictions based on seasonal patterns)
3. *Set Preferences*: Choose your preferred weather conditions
4. *Adjust Search Range*: Set how many days to search for best matches
5. *Click "Predict Weather"*: Generate predictions and view results

## Features

- Single-day weather predictions with probability distributions
- Best days finder for your preferred conditions
- Interactive visualizations and charts
- AI-generated weather summaries (requires API key)
- CSV export of predictions

## Troubleshooting

### "Unable to make predictions" Error
- The selected location may not be in the training data
- Try selecting a different location from the dropdown

### App is Slow
- First run trains the model and takes time
- Large date ranges in multi-day mode take longer to compute
- Try reducing the search range slider

### Missing Database Error
- Ensure large_merra2_data.db is in the project root
- File size should be around 40MB

### Import Errors
- Make sure your virtual environment is activated
- Reinstall dependencies: pip install -r requirements.txt

### API Key Not Working
- Check that .streamlit/secrets.toml exists and contains your key
- Verify the key is valid at Google AI Studio
- The app will work without the API key (AI summary feature won't work)

## Project Structure

```
project/
├── app.py                      # Main application
├── large_merra2_data.db        # Weather data (40MB)
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── secrets.toml           # API keys (gitignored)
├── .gitignore
└── README.md
```

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
* AI-powered comprehensive summary of the computed probabilities

## 🎯 Goals

1. Process, clean and store NASA meteorological data
2. Build and fit a probabilistic model (Bayesian Network) with the cleaned data
3. Provide a user-friendly interface where the user can input their desired weather conditions for a given day of the year and return its computed probability
4. Suggest days closer to the desired weather conditions in a time window around the initial desired date
5. AI-powered comprehensive summary of the computed probabilities

### ✨ Uniqueness

We suggest days closer to the desired weather conditions in a time window around the initial desired date if it happens that the desired conditions weren't met on the initial date. Additionally, we give comprehensive and user-friendly AI summaries of the computed probabilities.

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

### Strengths:

The project effectively integrates a Bayesian Network with historical NASA MERRA-2 data, providing probabilistic weather predictions. It accounts for location (latitude/longitude) and time (day of year with leap-year normalization), which improves accuracy and consistency. The categorical discretization of weather variables (temperature, wind, precipitation, snow, humidity) simplifies predictions and aligns well with human-understandable categories. The interactive Streamlit interface enhances usability, with map-based location selection, date inputs, and preference-based “best days” analysis. Visualization tools, including Plotly charts and probability progress bars, help users interpret results intuitively. Integration of AI-generated natural language summaries adds a friendly, accessible layer for non-technical users. Finally, caching mechanisms reduce repeated computations, improving responsiveness.

### Weaknesses:

The model relies solely on historical data, limiting its responsiveness to real-time or rapidly changing weather phenomena. The discretization, while user-friendly, reduces granularity, potentially oversimplifying subtle variations. Bayesian Network inference can fail when a location or day combination isn’t present in the training data, leading to gaps in predictions. The AI summary relies on an external API key, which may be unavailable for all users.

### Limitations:

Predictions are constrained to locations in the training dataset; remote or unobserved coordinates cannot be modeled. Extreme weather events outside historical patterns may be inaccurately predicted. Future forecasts assume stationarity in climate patterns, which may not hold under climate change. Computationally, large datasets increase training time and memory usage.

### Areas of Improvement:

Incorporating real-time satellite data or numerical weather models could enhance accuracy. Hybrid approaches combining Bayesian Networks with machine learning models might also improve accuracy. User experience could improve with automated alerts, or mobile-friendly design. Additionally, extending the model to seasonal or climate trend predictions could broaden its utility for planning and research. Finally, the project could also benefit from being incorporated in a larger event planning application, boosting its reach in the field of business.
