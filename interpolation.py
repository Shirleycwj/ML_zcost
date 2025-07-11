import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xarray as xr  # For gridded data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
import plotly.io as pio
import os

# 1. Load point-scale CSV data
df = pd.read_csv(r"C:\Users\shirl\OneDrive - Imperial College London\Postdoc_240501\ICCS_summer_school\site_full.csv")

# 2. Prepare predictors and target
predictor_cols = [col for col in df.columns if col not in ['z_cost', 'SiteLatitude', 'SiteLongitude']]
# print('Predictor columns:', predictor_cols) 
X = df[predictor_cols]
y = df['z_cost']

# Log-transform all predictors and target
print('Applying log1p transform to all predictors and target (z_cost)')
X = np.log1p(X)
y = np.log1p(y)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 5. Evaluate on test set
y_pred = rf.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))

# 6. Visualise
# 1. Scatter plot: Predicted vs Actual
plt.figure(figsize=(6,6))
plt.scatter(np.array(y_test), np.array(y_pred), alpha=0.6)
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'r--', lw=2)
plt.xlabel('Actual z_cost')
plt.ylabel('Predicted z_cost')
plt.title('Random Forest Regression: Actual vs Predicted')
plt.grid(True)
# Add metrics as text box
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
plt.text(0.05, 0.95, f'R2 = {r2:.2f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}',
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
plt.tight_layout()
plt.show()

# 2. Residual plot
residuals = np.array(y_test) - np.array(y_pred)
plt.figure(figsize=(6,4))
plt.scatter(np.array(y_pred), residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted z_cost')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.grid(True)
# Add metrics as text box
plt.text(0.05, 0.95, f'R2 = {r2:.2f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}',
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
plt.tight_layout()
plt.show()

# 7. Feature Importance Analysis
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 6))
plt.title('Feature Importances')
plt.bar(range(len(predictor_cols)), importances[indices], align='center')
plt.xticks(range(len(predictor_cols)), list(np.array(predictor_cols)[indices]), rotation=90)
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# 8. 3D scatter plot of top 3 predictors
top3_cols = [predictor_cols[i] for i in indices[:3]]
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_test[top3_cols[0]], X_test[top3_cols[1]], X_test[top3_cols[2]], c=y_test, cmap='viridis', alpha=0.7)
ax.set(xlabel=top3_cols[0], ylabel=top3_cols[1], zlabel=top3_cols[2])
cb = plt.colorbar(sc, ax=ax, pad=0.1)
cb.set_label('Actual z_cost')
plt.title('3D Scatter: Top 3 Predictors vs Actual z_cost')
plt.tight_layout()
plt.show()

# 8. Interactive 3D scatter plot of top 3 predictors using plotly


# Prepare data for plotly
x = X_test[top3_cols[0]]
y_ = X_test[top3_cols[1]]
z = X_test[top3_cols[2]]
c = y_test

trace = go.Scatter3d(
    x=x,
    y=y_,
    z=z,
    mode='markers',
    marker=dict(
        size=5,
        color=c,
        colorscale='Viridis',
        colorbar=dict(title='Actual z_cost'),
        opacity=0.8
    ),
    text=[f'Actual z_cost: {val:.2f}' for val in c]
)

layout = go.Layout(
    title='Interactive 3D Scatter: Top 3 Predictors vs Actual z_cost',
    scene=dict(
        xaxis_title=top3_cols[0],
        yaxis_title=top3_cols[1],
        zaxis_title=top3_cols[2]
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig = go.Figure(data=[trace], layout=layout)
fig.show()
fig.write_html(r"C:\Users\shirl\OneDrive - Imperial College London\Postdoc_240501\ICCS_summer_school\interactive_3d_scatter.html")

# # 6. Interpolate to site-scale/global level
# # Assume you have gridded predictor data as NetCDF, with same variable names as in predictor_cols
# ds = xr.open_dataset('gridded_predictors.nc')

# # For missing root traits, fill with mean from site data
# root_trait_cols = [col for col in predictor_cols if 'root' in col]
# for col in root_trait_cols:
#     if col not in ds:
#         ds[col] = (('lat', 'lon'), np.full(ds[predictor_cols[0]].shape, X[col].mean()))

# # Flatten grid for prediction
# grid_shape = ds[predictor_cols[0]].shape
# grid_df = pd.DataFrame({var: ds[var].values.flatten() for var in predictor_cols})

# # Predict on grid
# grid_pred = rf.predict(grid_df)
# grid_pred_reshaped = grid_pred.reshape(grid_shape)

# # 7. Evaluate with gridded z (if available)
# if 'z' in ds:
#     z_true = ds['z'].values
#     mask = ~np.isnan(z_true)
#     print("Grid R2 Score:", r2_score(z_true[mask].flatten(), grid_pred_reshaped[mask].flatten()))
#     print("Grid RMSE:", mean_squared_error(z_true[mask].flatten(), grid_pred_reshaped[mask].flatten(), squared=False))
#     print("Grid MAE:", mean_absolute_error(z_true[mask].flatten(), grid_pred_reshaped[mask].flatten()))

# # 8. Save results (optional)
# ds_out = ds.copy()
# ds_out['z_pred'] = (ds[predictor_cols[0]].dims, grid_pred_reshaped)
# ds_out.to_netcdf('global_z_prediction.nc')


## use the only first 3 predictors

# 1. Use only the top 3 predictors
top3_predictors = ['ph', 'cn', 'olp']  # Replace with your actual top 3 names if different

# Subset your data
X_top3 = df[top3_predictors]
y = df['z_cost']

X_top3 = np.log1p(X_top3)
y = np.log1p(y)

# Train/test split (optional, for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X_top3, y, test_size=0.2, random_state=42)

# Train Random Forest on top 3 predictors
rf_top3 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_top3.fit(X_train, y_train)

# Predict on test set (optional, for evaluation)
y_pred = rf_top3.predict(X_test)
print("Top 3 predictors - R2 Score:", r2_score(y_test, y_pred))
print("Top 3 predictors - RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Top 3 predictors - MAE:", mean_absolute_error(y_test, y_pred))

# Scatter plot: Predicted vs Actual for top 3 predictors
plt.figure(figsize=(6,6))
plt.scatter(np.array(y_test), np.array(y_pred), alpha=0.6)
plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'r--', lw=2)
plt.xlabel('Actual z_cost')
plt.ylabel('Predicted z_cost')
plt.title('Top 3 Predictors: Actual vs Predicted')
plt.grid(True)
# Add metrics as text box
r2_top3 = r2_score(y_test, y_pred)
rmse_top3 = np.sqrt(mean_squared_error(y_test, y_pred))
mae_top3 = mean_absolute_error(y_test, y_pred)
plt.text(0.05, 0.95, f'R2 = {r2_top3:.2f}\nRMSE = {rmse_top3:.2f}\nMAE = {mae_top3:.2f}',
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
plt.tight_layout()
plt.show()

os.chdir("C:/Users/shirl/OneDrive - Imperial College London/Postdoc_240501/C allocation framework/")

# 1. Open each NetCDF file
ds_ph = xr.open_dataset('ph_map.nc')
ds_cn = xr.open_dataset('cn_map.nc')
ds_olp = xr.open_dataset('./root_exudation_collective/olp_raster.nc')

# 2. Choose a target grid (e.g., ph.nc)
target_lat = ds_ph['latitude']
target_lon = ds_ph['longitude']

# 3. Interpolate sn and olp to ph grid
# No need to interpolate sn if it's already on the same grid as ph
ph_grid = ds_ph['layer']  # or whatever the variable name is
ph_grid = ph_grid.squeeze()
cn_grid = ds_cn['layer']  # or whatever the variable name is
cn_grid = cn_grid.squeeze()
olp_on_ph_grid = ds_olp['Band_1'].interp(latitude=target_lat, longitude=target_lon)

# 4. Stack predictors into DataFrame for prediction
ph_flat = ph_grid.values.flatten()
cn_flat = cn_grid.values.flatten()
olp_flat = olp_on_ph_grid.values.flatten()


grid_df = pd.DataFrame({
    'ph': ph_flat,
    'cn': cn_flat,
    'olp': olp_flat
})

# 5. Predict using your trained model (rf_top3)
grid_pred = rf_top3.predict(grid_df)
grid_pred_reshaped = grid_pred.reshape(cn_grid.shape)

# 6. Save results
ds_out = ds_ph.copy()
ds_out['z_pred_top3'] = (cn_grid.dims, grid_pred_reshaped)
# ds_out.to_netcdf('global_z_prediction_top3.nc')

# Assume observed z is in ds_ph['z_obs'] (adjust if in another file/variable)
z_obs = xr.open_dataset('./apprent_z.nc')['layer']
z_obs = z_obs.squeeze()

# Mask invalid values
mask = ~np.isnan(z_obs) & ~np.isnan(grid_pred_reshaped)

# Calculate differences
abs_diff = np.where(mask, np.abs(grid_pred_reshaped - z_obs), np.nan)
rel_diff = np.where(mask, abs_diff / np.abs(z_obs), np.nan)

# Print summary statistics
print("Absolute difference: mean =", np.nanmean(abs_diff), ", median =", np.nanmedian(abs_diff))
print("Relative difference: mean =", np.nanmean(rel_diff), ", median =", np.nanmedian(rel_diff))



# Plot absolute difference
plt.figure(figsize=(8,4))
plt.imshow(abs_diff, cmap='viridis')
plt.colorbar(label='Absolute Difference')
plt.title('Absolute Difference: Predicted vs Observed z')
plt.tight_layout()
plt.show()

# Plot relative difference
plt.figure(figsize=(8,4))
plt.imshow(rel_diff, cmap='viridis', vmin=0, vmax=float(np.nanpercentile(rel_diff, 99)))  # clip outliers for better color scaling
plt.colorbar(label='Relative Difference')
plt.title('Relative Difference: Predicted vs Observed z')
plt.tight_layout()
plt.show()

# Plot observed z and predicted z side by side with color scale limited to 0-40
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
im1 = plt.imshow(z_obs, cmap='viridis', vmin=0, vmax=40)
plt.colorbar(im1, label='Observed z')
plt.title('Observed z')
plt.subplot(1,2,2)
im2 = plt.imshow(grid_pred_reshaped, cmap='viridis', vmin=0, vmax=40)
plt.colorbar(im2, label='Predicted z')
plt.title('Predicted z')
plt.tight_layout()
plt.show()


