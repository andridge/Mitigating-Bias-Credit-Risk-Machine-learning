import numpy as np
import pandas as pd

# Generate synthetic data
np.random.seed(0)
num_samples = 1000

# Features
age = np.random.randint(18, 70, num_samples)
gender = np.random.choice(['Male', 'Female'], num_samples)
income = np.random.randint(20000, 120000, num_samples)
credit_score = np.random.randint(300, 850, num_samples)

# Label: 1 for approved, 0 for not approved
approved = np.random.choice([0, 1], num_samples)

# Create a DataFrame
data = pd.DataFrame({'Age': age, 'Gender': gender, 'Income': income, 'CreditScore': credit_score, 'Approved': approved})

# Save the dataset as a CSV file
data.to_csv('credit_risk_dataset.csv', index=False)
