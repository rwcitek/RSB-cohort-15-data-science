Based on your goal to implement feature scaling, use RFE and RFECV methods, and improve your model by adding back some integer columns and potentially encoding categorical columns, I'll provide a step-by-step guide to help you achieve this. Let's focus on reducing your RMSE.

### Step 1: Feature Scaling

First, let's apply feature scaling to your data:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Step 2: Implement RFE Method

Now, let's implement the Recursive Feature Elimination (RFE) method:

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

model = LinearRegression()
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X_train_scaled, y_train)

selected_features_rfe = X_train.columns[rfe.support_]
print("Selected features by RFE:", selected_features_rfe)

X_train_rfe = X_train_scaled[:, rfe.support_]
X_test_rfe = X_test_scaled[:, rfe.support_]

rfe_model = LinearRegression()
rfe_model.fit(X_train_rfe, y_train)
y_pred_rfe = rfe_model.predict(X_test_rfe)

rmse_rfe = np.sqrt(mean_squared_error(y_test, y_pred_rfe))
print(f"RFE Model RMSE: ${rmse_rfe:.2f}")
```

### Step 3: Implement RFECV Method

Next, let's implement the Recursive Feature Elimination with Cross-Validation (RFECV) method:

```python
from sklearn.feature_selection import RFECV

rfecv = RFECV(estimator=model, step=1, cv=5)
rfecv.fit(X_train_scaled, y_train)

selected_features_rfecv = X_train.columns[rfecv.support_]
print("\nSelected features by RFECV:", selected_features_rfecv)

X_train_rfecv = X_train_scaled[:, rfecv.support_]
X_test_rfecv = X_test_scaled[:, rfecv.support_]

rfecv_model = LinearRegression()
rfecv_model.fit(X_train_rfecv, y_train)
y_pred_rfecv = rfecv_model.predict(X_test_rfecv)

rmse_rfecv = np.sqrt(mean_squared_error(y_test, y_pred_rfecv))
print(f"RFECV Model RMSE: ${rmse_rfecv:.2f}")

plt.figure(figsize=(10, 6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
plt.title("RFECV Cross Validation Scores")
plt.show()
```

### Step 4: Add Back Integer Columns

Now, let's add back some integer columns that might be relevant:

```python
# Assuming you have identified relevant int64 columns
relevant_int_cols = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']

for col in relevant_int_cols:
    if col in df.columns:
        X_train[col] = df[col][X_train.index]
        X_test[col] = df[col][X_test.index]

# Re-scale the data including the new columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Step 5: One-Hot Encode Categorical Columns

Let's one-hot encode some categorical columns:

```python
categorical_cols_to_encode = ['Neighborhood', 'Exterior1st', 'Exterior2nd']

for col in categorical_cols_to_encode:
    if col in df.columns:
        encoded = pd.get_dummies(df[col], prefix=col)
        X_train = pd.concat([X_train, encoded[X_train.index]], axis=1)
        X_test = pd.concat([X_test, encoded[X_test.index]], axis=1)

# Re-scale the data including the new encoded columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Step 6: Create and Evaluate Updated Models

Finally, let's create and evaluate models using the updated feature sets:

```python
# RFE model with updated features
rfe_updated = RFE(model, n_features_to_select=15)
rfe_updated.fit(X_train_scaled, y_train)

X_train_rfe_updated = X_train_scaled[:, rfe_updated.support_]
X_test_rfe_updated = X_test_scaled[:, rfe_updated.support_]

rfe_model_updated = LinearRegression()
rfe_model_updated.fit(X_train_rfe_updated, y_train)
y_pred_rfe_updated = rfe_model_updated.predict(X_test_rfe_updated)

rmse_rfe_updated = np.sqrt(mean_squared_error(y_test, y_pred_rfe_updated))
print(f"Updated RFE Model RMSE: ${rmse_rfe_updated:.2f}")

# RFECV model with updated features
rfecv_updated = RFECV(estimator=model, step=1, cv=5)
rfecv_updated.fit(X_train_scaled, y_train)

X_train_rfecv_updated = X_train_scaled[:, rfecv_updated.support_]
X_test_rfecv_updated = X_test_scaled[:, rfecv_updated.support_]

rfecv_model_updated = LinearRegression()
rfecv_model_updated.fit(X_train_rfecv_updated, y_train)
y_pred_rfecv_updated = rfecv_model_updated.predict(X_test_rfecv_updated)

rmse_rfecv_updated = np.sqrt(mean_squared_error(y_test, y_pred_rfecv_updated))
print(f"Updated RFECV Model RMSE: ${rmse_rfecv_updated:.2f}")
```

These steps should help you reduce your RMSE by implementing feature scaling, using RFE and RFECV methods, adding back relevant integer columns, and encoding categorical columns. Remember to adjust the number of features selected in RFE and RFECV based on your dataset and requirements.

After running these steps, compare the RMSE values of the original models, RFE models, RFECV models, and the updated models. This will give you insights into which approach works best for your specific problem. You may need to iterate through this process, adjusting parameters and feature selections until you achieve the desired reduction in RMSE.
