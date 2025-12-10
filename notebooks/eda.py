import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


CLEANED_DATA = '/Users/Marcy_Student/Desktop/marcy/DA2025_Lectures_Kevin/Mod6/Mod6_Project/Data/cleaned_crash_data.csv'
pd.set_option('display.max_columns', None)

df = pd.read_csv(CLEANED_DATA)

df["CRASH TIME"] = pd.to_datetime(df["CRASH TIME"], errors="coerce")
df["hour"] = df["CRASH TIME"].dt.hour

df["CRASH DATE"] = pd.to_datetime(df["CRASH DATE"], errors="coerce")
df["weekday"] = df["CRASH DATE"].dt.day_name()

weekday_predestriansInjured = df.groupby('weekday')['NUMBER OF PEDESTRIANS INJURED'].sum().reset_index().sort_values(by='NUMBER OF PEDESTRIANS INJURED',ascending=False)

#print(weekday_predestriansInjured)

inj_by_hour = df.groupby(['hour','weekday'])['NUMBER OF PEDESTRIANS INJURED'].sum().reset_index().sort_values('NUMBER OF PEDESTRIANS INJURED',ascending=False)
#print(inj_by_hour.head())

#print(df.keys())
vehByHour = df.groupby(['veh_group', 'hour'])['NUMBER OF PEDESTRIANS INJURED'].sum().reset_index().sort_values('NUMBER OF PEDESTRIANS INJURED',ascending=False)
#print(vehByHour.head())

# Create Baseline Model - 

X = df[['hour']]
y = df['NUMBER OF PEDESTRIANS INJURED']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Combine X_train (DataFrame) and y_train (Series) correctly
train_temp = X_train.copy()
train_temp['NUMBER OF PEDESTRIANS INJURED'] = y_train

# Compute mean injuries by hour
hour_means = train_temp.groupby('hour')['NUMBER OF PEDESTRIANS INJURED'].mean()

# Map hourly mean to X_test
baseline_hour_avg = X_test['hour'].map(hour_means)

# In case some hours are missing in training, fallback to global mean
baseline_hour_avg = baseline_hour_avg.fillna(y_train.mean())

mae_hour = mean_absolute_error(y_test, baseline_hour_avg)
mse_hour = mean_squared_error(y_test, baseline_hour_avg)

print("Baseline (Hourly Avg) MAE:", mae_hour)
print("Baseline (Hourly Avg) MSE:", mse_hour)



