import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# Load the dataset
df = pd.read_csv('dat5.csv')

# Check initial class distribution
print("Initial class distribution:")
print(df['nfti_positive'].value_counts())

# Separate input features and target variable
X = df.drop('nfti_positive', axis=1)  # Assuming 'nfti_positive' is the target
y = df['nfti_positive']

# Initialize the Random Under Sampler
rus = RandomUnderSampler(random_state=42)

# Resample the dataset
X_resampled, y_resampled = rus.fit_resample(X, y)

# Combine the resampled data back into a DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['nfti_positive'] = y_resampled

# Check new balance
print("New class distribution after undersampling:")
print(df_resampled['nfti_positive'].value_counts())

# Save the balanced dataset
df_resampled.to_csv('dat5_balanced.csv', index=False)
print('Balanced dataset saved as dat5_balanced.csv')
