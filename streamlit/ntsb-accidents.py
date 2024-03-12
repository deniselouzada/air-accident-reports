# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Imports
import streamlit as st
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime as dt
import warnings

warnings.filterwarnings("ignore")
# -

df = pd.read_csv("AviationData2.csv")

df


# ## Data cleaning and preprocessing
#


df.drop_duplicates(inplace=True)


# Let's remove columns we are not interested in

# Let's drop irrelevant columns
df.drop(
    columns=[
        "Mkey",
        "ReportNo",
        "N",  # N is the ID of the plane
        "HasSafetyRec",
        "ReportType",
        "EventID",
        "Latitude",
        "Longitude",
        "Unnamed: 37",
        "DocketPublishDate",
        "DocketUrl",
        "AirportID",
        "AirportName",
        "ReportStatus",
        "Operator",
        "AmateurBuilt",
        "NtsbNo",
        "OriginalPublishDate",
        "FAR",
        "RepGenFlag",
        "Scheduled",
        "EventType",
    ],
    inplace=True,
)


df.isna().sum().sort_values(ascending=False)

# Probable cause would be a valuable metric, however that information is missing from a really large number of entries. Therefore, the column will be dropped.
df.drop("ProbableCause", axis=1, inplace=True)

# State is not super important, so we will drop it as well, along with PurposeOfFlight, Make and Model
df.drop(columns=["PurposeOfFlight", "State", "Make", "Model"], axis=1, inplace=True)

df.isna().sum().sort_values(ascending=False)

# HighestInjuryLevel can be extracted from the injury counts, which don't have any missing values, so we will remove this column as well and extract the information later
df.drop("HighestInjuryLevel", axis=1, inplace=True)

df.isna().sum().sort_values(ascending=False)

# Now we are left with all the information we want and reduced the number of missing values significantly

# Given the shear amount of data, it's acceptable to simply drop the null values
df.dropna(inplace=True)

df.isna().sum().sort_values(ascending=False)

# Now let's change the types of the numerical data that are mistakenly described as objects
df.EventDate = pd.to_datetime(df.EventDate)


# We are only interested in accidents involving airplanes, so we can remove some of this data
itemList = ["HELI", "UNK", "GLI", "ULTR", "GYRO", "PPAR"]
for string in itemList:
    df.drop(df[df.AirCraftCategory.str.contains(string)].index, inplace=True)


df.drop(df[df.AirCraftCategory == "AIR,"].index, inplace=True)


# ## Exploratory Analysis and Feature Engineering
#

# Let's extract the date and the time of the accidents
df.EventDate = pd.to_datetime(df.EventDate)
df.EventDate = df.EventDate.dt.tz_convert("US/Eastern")

# Let's first extract some relevant information from the data and create new columns
df["EventTime"] = df.EventDate.dt.timetz


df.EventTime = df.EventTime.astype(str).str.replace(":", "")


df.EventTime = df.EventTime.astype(int)


df.EventDate = pd.to_datetime(df.EventDate.dt.strftime("%Y-%m-%d"))



# Let's make the date the index of the data frame
df.set_index(df.EventDate, inplace=True)
df.drop(columns="EventDate", inplace=True)

# Let's extract the year from the date. That can be useful to determine how accidents have changed through the years
df["Year"] = df.index.year


# Let's create a new column to store the accident severity based on the amount and type of injury
df["AccidentSeverity"] = pd.Series()

df.loc[df.FatalInjuryCount > 0, "AccidentSeverity"] = "Fatal"
df.loc[(df.SeriousInjuryCount > 0) & (df.FatalInjuryCount == 0), "AccidentSeverity"] = (
    "Serious"
)
df.loc[
    (df.MinorInjuryCount > 0)
    & (df.SeriousInjuryCount == 0)
    & (df.FatalInjuryCount == 0),
    "AccidentSeverity",
] = "Minor"
df.loc[
    (df.MinorInjuryCount == 0)
    & (df.SeriousInjuryCount == 0)
    & (df.FatalInjuryCount == 0),
    "AccidentSeverity",
] = "No injuries"

# Let's rename these columns for brevity
df.rename(
    columns={
        "FatalInjuryCount": "FatalInjuries",
        "SeriousInjuryCount": "SeriousInjuries",
        "MinorInjuryCount": "MinorInjuries",
    },
    inplace=True,
)


# Let's transform the category of the aircraft into the number of airplanes involved in the accident, since they all have the type "Airplane"

df["NumberOfAircraftInvolved"] = df.AirCraftCategory.str.count("AIR")

df.NumberOfAircraftInvolved.value_counts()


# Now we can remove the category of aircraft because we only have airplanes
df.drop(columns=["AirCraftCategory"], inplace=True)

df.WeatherCondition.value_counts()

# Let's consolidate the unknown values
df.loc[df.WeatherCondition == "Unknown", "WeatherCondition"] = "UNK"

df.WeatherCondition.value_counts()


df.Country.value_counts()

df.Country.value_counts().head(10)


df.AirCraftDamage.value_counts()

# +
# To help our model, let's use the most severe damage to give this column only one description
df.loc[df.AirCraftDamage.str.contains("Destroyed"), "AirCraftDamage"] = "Destroyed"
df.loc[df.AirCraftDamage.str.contains("Substantial"), "AirCraftDamage"] = "Substantial"
df.loc[df.AirCraftDamage.str.contains("Minor"), "AirCraftDamage"] = "Minor"
df.loc[df.AirCraftDamage.str.contains("Unknown"), "AirCraftDamage"] = "Unknown"

df.AirCraftDamage.value_counts()
# -


# We can see that the EventTime median is 0, equivalent to 00:00 EST, which tells us this is probably a missing value, in fact.
#
fig, ax = plt.subplots()
ax = sns.histplot(df.EventTime, kde=True)
st.pyplot(fig)

df.loc[df.EventTime == 0, "Year"].value_counts()

df.loc[df.EventTime == 0, "Year"].value_counts().sum()

df.loc[df.EventTime == 10000, "Year"].value_counts().sum()

df.loc[df.EventTime == 230000, "Year"].value_counts().sum()

# The fact that most of the 0 values are concentrated in accidents which occured more than 40 years ago is one more clue that these are actually missing values. In that case, we will, instead of simply removing the values, replace them with the median. However, to find the real median, we need to remove the 0 values first. Let's bear in mind that the missing values are not only 0, but concentrated around it.
#

df.EventTime.loc[
    (df.EventTime == 0) | (df.EventTime == 10000) | (df.EventTime == 230000)
] = None


fig, ax = plt.subplots()
ax = sns.histplot(df.EventTime, kde=True)
st.pyplot(fig)


df_time = df.dropna()

# *** About half of the values are missing

# ## Extracting insights with Multivariate Analysis
#
# Now that our data is cleaned and organized, let's analyse the data and answer some questions
#

# ### What's the distribution of the variables?
#

# +
# hist = pd.melt(df, value_vars = df)
# hist = sns.FacetGrid (hist, col = 'variable', col_wrap = 3, sharex = False, sharey = False)
# hist.map(sns.histplot, 'value')
# -


# Let's see the outlier distribution
df_box = df.drop(
    columns=[
        "City",
        "Country",
        "NumberOfEngines",
        "AirCraftDamage",
        "WeatherCondition",
        "AccidentSeverity",
    ]
)


df_box.plot(
    subplots=True, layout=(3, 3), kind="box", figsize=(12, 14), patch_artist=True
)
plt.subplots_adjust(wspace=0.5)

df_injuries = df_box.drop(columns=["EventTime", "Year", "NumberOfAircraftInvolved"])

# Since the data is heavily skewed to high valued outliers, we'll plot these variables in logarithmic scale
hist = pd.melt(df_injuries, value_vars=df_injuries)
hist = sns.FacetGrid(hist, col="variable", col_wrap=3, sharex=False)
hist.map(sns.histplot, "value").set(yscale="log")

df_hist = df_box.drop(columns=["FatalInjuries", "SeriousInjuries", "MinorInjuries"])

hist = pd.melt(df_hist, value_vars=df_hist)
hist = sns.FacetGrid(hist, col="variable", col_wrap=3, sharex=False, sharey=False)
hist.map(sns.histplot, "value").set(yscale="log")

sns.histplot(df.Year, kde=True)

# We can see that our numerical data are mostly not normal in distribution. Moreover, we know these outliers are not errors. The fact is that most airplane accidents involve small aircraft with one or two engines, but accidents involving big commercial airplanes do happen and skew the data upwards when it comes to injuries and fatalities. We shall confirm that with our model later.
#

# ### What's the proportion of accidents that happened on Visual vs Instrument conditions?
#

import plotly.express as px

# Let's create a data frame with the weather condition but with both the codes and wether the condition is visual or intrument
df_weather = pd.DataFrame(data=df.WeatherCondition)

df_weather["Condition"] = pd.Series()

df_weather.rename(columns={"WeatherCondition": "Code"}, inplace=True)
df_weather

df_weather.loc[df_weather.Code.str.contains("V"), "Condition"] = "Visual"
df_weather.loc[df_weather.Code.str.contains("I"), "Condition"] = "Instrument"
df_weather.loc[df_weather.Code.str.contains("UNK"), "Condition"] = "Unknown"

data = df_weather.value_counts()

data = pd.DataFrame(data)
data.reset_index(inplace=True)
data

data2 = data.groupby("Condition").sum().sort_values(by="count", ascending=False)

explode = (0.2, 0, 0.1)
palette = sns.color_palette("Set2", n_colors=3, desat=0.9)
fig, ax = plt.subplots()
ax.pie(
    data2["count"],
    explode=explode,
    labels=data2.index.to_list(),
    autopct="%0.0f%%",
    shadow={"ox": -0.04, "edgecolor": "none", "shade": 0.9},
    startangle=40,
    colors=palette,
)
ax.set_title("Percent of accidents by weather condition")
plt.show()

# ### What's the distribution of severity of accidents?
#

df_severity = df.AccidentSeverity.value_counts().reset_index()
df_severity

explode = (0.1, 0, 0.1, 0)
palette = sns.color_palette("Set2", n_colors=4, desat=0.9)
fig, ax = plt.subplots()
ax.pie(
    df_severity["count"],
    explode=explode,
    labels=df_severity.AccidentSeverity.to_list(),
    autopct="%0.0f%%",
    shadow={"ox": -0.04, "edgecolor": "none", "shade": 0.9},
    startangle=90,
    colors=palette,
)
ax.set_title("Percent of accidents by severity")
plt.show()

# ### How have fatalities fared through the years?
#

plt.plot(df.FatalInjuries)
plt.show()

avg_fatalities = df.groupby("Year")["FatalInjuries"].mean()
avg_fatalities = pd.DataFrame(avg_fatalities)
avg_fatalities

plt.plot(avg_fatalities)

median_fatalities = df.groupby("Year")["FatalInjuries"].median()
median_fatalities = pd.DataFrame(median_fatalities)
median_fatalities

plt.plot(median_fatalities)

# let's find the most fatal accidents
df.sort_values(by="FatalInjuries", ascending=False).head()

# ### What's the correlation between the numeric variables?
#

numeric_data = df_box.reset_index()

numeric_data.drop(columns=["EventDate"], inplace=True)

numeric_data.head()

# +
corr = numeric_data.corr()

# Generate a mask for the upper triangle
mask = np.triu(corr)
col = corr.columns.tolist()
row = corr.index.tolist()
mask = np.array(mask)
mask = pd.DataFrame(mask, columns=col, index=row)
mask = mask.mask(mask == 0)
mask = mask == corr

corr = corr.mask(corr == 1, 0)

# Draw Heat Map

# Set up the matplotlib figure
plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(10, 8))


g = sns.heatmap(
    corr,
    vmax=0.32,
    vmin=-0.16,
    mask=mask,
    fmt=".2f",
    annot=True,
    center=0,
    cbar_kws={"shrink": 0.5},
    linewidths=0.6,
    cmap="flare",
)
# -

# ### What's the percentage of accidents per type of damage?
#

damage = df.AirCraftDamage.value_counts().reset_index()
damage

damage.drop(index=3, inplace=True)
damage

explode = (0.1, 0, 0.1)
palette = sns.color_palette("Set2", n_colors=4, desat=0.9)
fig, ax = plt.subplots()
ax.pie(
    damage["count"],
    explode=explode,
    labels=damage.AirCraftDamage.to_list(),
    autopct="%0.1f%%",
    shadow={"ox": -0.04, "edgecolor": "none", "shade": 0.9},
    startangle=45,
    colors=palette,
)
ax.set_title("Percent of accidents by aircraft damage")
plt.show()

# ### What's the percent of accidents per number of aircraft involved?
#

num_aircraft = df.NumberOfAircraftInvolved.value_counts().reset_index()
num_aircraft

# +
ax = sns.barplot(
    num_aircraft,
    x=num_aircraft.NumberOfAircraftInvolved,
    y=num_aircraft["count"],
    palette="muted"
    
)
ax.set_yscale('log')
for p in ax.patches:
    ax.annotate(
        f'{p.get_height():.0f}',
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="bottom",
    )
# -

# ### Out of the accidents involving one aircraft, what's the percentage of accidents per number of engines?
#

df.sample(5)

engines = (
    df.loc[df.NumberOfAircraftInvolved == 1, "NumberOfEngines"]
    .value_counts()
    .reset_index()
)
engines

engines.drop(index=4, inplace=True)

engines = engines.sort_values(by="NumberOfEngines")

# +
ax = sns.barplot(
    engines, x=engines.NumberOfEngines, 
    y=engines["count"], 
    palette="muted"
)
ax.set_yscale('log')

for p in ax.patches:
    ax.annotate(
        f'{p.get_height():.0f}',
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="bottom",
    )
# -

df.sample(5)

# ### How important is each feature to the fatality of accidents?
#

df_imp = df.copy()

df_imp.WeatherCondition = df.WeatherCondition.map(
    {
        "VMC": "Visual",
        "VFR": "Visual",
        "IMC": "Instrument",
        "IFR": "Instrument",
        "UNK": "Unknown",
    }
)
df_imp

# Convert categorical variables to numeric using one hot encoding
df_imp = pd.get_dummies(df_imp, columns=["AirCraftDamage", "WeatherCondition"])

X = df_imp.drop(
    columns=[
        "City",
        "Country",
        "FatalInjuries",
        "MinorInjuries",
        "SeriousInjuries",
        "AccidentSeverity",
        "NumberOfEngines",
        "EventTime",
    ]
)

y = df_imp["AccidentSeverity"].apply(lambda x: 1 if x == "Fatal" else 0)

rf_model = RandomForestClassifier(
    class_weight={0: 0.84, 1: 0.16}, max_features=None)

model = rf_model.fit(X, y)

importances = model.feature_importances_

indices = np.argsort(importances)

std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

# Plot
plt.figure(1, figsize=(11, 10))
plt.title("Importance of features to the fatality of accidents", fontsize=20)
plt.barh(
    range(X.shape[1]),
    importances[indices],
    color="steelblue",
    xerr=std[indices],
    align="center",
)
plt.yticks(range(X.shape[1]), X.columns[indices], fontsize=16)
plt.ylim([-1, X.shape[1]])
plt.xticks(fontsize=16)
plt.show()

# ### What's the relationship of fatalities and time of day?
#


