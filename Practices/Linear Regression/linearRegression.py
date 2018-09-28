import pandas as pd
from sklearn.linear_model import LinearRegression

bmi_life_data = pd.read_csv('dataset.csv')
bmi_life_model = LinearRegression()
x_values = bmi_life_data[['BMI']]
y_values = bmi_life_data[['Life expectancy']]
bmi_life_model.fit(x_values,y_values)
print(bmi_life_model.predict(21.07931))