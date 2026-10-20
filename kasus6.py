import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_alerts = 1500

drivers = ['Large_Agri', 'Small_Agri', 'Mining', 'Selective_Logging', 'Wildfire']
driver_dist = [0.50, 0.25, 0.10, 0.10, 0.05]
labels = np.random.choice(drivers, size=n_alerts, p=driver_dist)

ndvi_drop = []
sar_vv_backscatter = []
sar_vh_backscatter = []
distance_to_road_km = []

for l in labels:
  if l == 'Large_Agri':
    ndvi_drop.append(np.random.normal(0.60, 0.08))
    sar_vv_backscatter.append(np.random.normal(-12.0, 1.5))
    sar_vh_backscatter.append(np.random.normal(-18.0, 1.8))
    distance_to_road_km.append(np.random.exponential(5.0))
  elif l == 'Small_Agri':
    ndvi_drop.append(np.random.normal(0.40, 0.10))
    sar_vv_backscatter.append(np.random.normal(-10.5, 1.8))
    sar_vh_backscatter.append(np.random.normal(-16.0, 2.0))
    distance_to_road_km.append(np.random.exponential(3.0))
  elif l == 'Mining':
    ndvi_drop.append(np.random.normal(0.70, 0.05))
    sar_vv_backscatter.append(np.random.normal(-7.0, 1.2))
    sar_vh_backscatter.append(np.random.normal(-13.0, 1.5))
    distance_to_road_km.append(np.random.exponential(8.0))
  elif l == 'Selective_Logging':
    ndvi_drop.append(np.random.normal(0.25, 0.07))
    sar_vv_backscatter.append(np.random.normal(-9.0, 1.4))
    sar_vh_backscatter.append(np.random.normal(-14.5, 1.6))
    distance_to_road_km.append(np.random.exponential(1.5))
  else:
    ndvi_drop.append(np.random.normal(0.65, 0.12))
    sar_vv_backscatter.append(np.random.normal(-14.0, 2.0))
    sar_vh_backscatter.append(np.random.normal(-21.0, 2.2))
    distance_to_road_km.append(np.random.exponential(12.0))

df = pd.DataFrame({
    'ndvi_loss': np.clip(ndvi_drop, 0.0, 1.0),
    'sar_vv_db': sar_vv_backscatter,
    'sar_vh_db': sar_vh_backscatter,
    'dist_road_km': distance_to_road_km,
    'driver': labels,
})

X = df.drop(columns=['driver'])
y = df['driver']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=120, max_depth=7, class_weight='balanced', random_state=42
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print('=== DISTRIBUSI KELAS DATA UJI (IMBALANCE CHECK) ===')
print(y_test.value_counts())

print('\n=== LAPORAN EVALUASI MODEL (FOKUS MACRO F1) ===')
print(classification_report(y_test, y_pred))

print('=== CONFUSION MATRIX ===')
labels_order = sorted(df['driver'].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels_order)
print(pd.DataFrame(cm, index=labels_order, columns=labels_order))