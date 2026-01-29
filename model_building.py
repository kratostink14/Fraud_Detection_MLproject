# Импорт библиотек
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Загрузка данных
df = pd.read_csv('Updated_Paysim_data.csv')
print(df.head(5))

# Разделение данных на признаки и целевую переменную
x = df.drop(columns=['isFraud' ])
y = df['isFraud']

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


# Построение моделей и их оценка
from sklearn.metrics import classification_report, recall_score, roc_auc_score, confusion_matrix

models = {
    # Модель Логистической регрессии
    # 'Logistic Regression': LogisticRegression(class_weight='balanced',
    #                                           random_state=42,
    #                                           solver='liblinear'),
    # # Модель алгоритма случайного леса
    # 'RandomForestClassifier': RandomForestClassifier(class_weight='balanced',
    #                                                  n_estimators=100,
    #                                                  max_depth=10,
    #                                                  min_samples_split=2,
    #                                                  random_state=42),
    # Модель градиентного бустинга
    'XGBoost': XGBClassifier(scale_pos_weight=100,
                             learning_rate=0.1,
                             random_state=42,
                             objective='binary:logistic',
                             ),
}

# Обучение и оценка модели
for name, model in models.items():
    print(f" Обучение модели: {name} ")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Recall модели XGBoost (способность находить мошенников): {recall_score(y_test, y_pred):.4f}")

# Преды на тестовой выборке
y_pred_proba = models['XGBoost'].predict_proba(X_test)[:, 1]
custom_threshold = 0.7
y_pred_custom = (y_pred_proba >= custom_threshold).astype(int)

print(f"Отчет для порога {custom_threshold}")
print(classification_report(y_test, y_pred_custom))
# Оцениваю модели метриками для классификации

# Матрица ошибок (Confusion matrix), чтобы понять как сильно модель промахивается
conf_matrix = confusion_matrix(y_test, y_pred_custom)
print(conf_matrix)
# Визуал матрицы ошибок
plt.figure(figsize=[10,8])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
plt.title('Confusion matrix')
plt.ylabel('Реальные значения')
plt.xlabel('Предсказанные значения ')
plt.savefig('Confusion Matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ROC AUC (оценивают баланс между tpr и fpr
print(y_pred_proba)
print(f'ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}')
from sklearn.metrics import precision_recall_curve

# PR кривая чтобы рассчитать нужный порог для модели
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# График
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], 'b--', label='Precision (Точность)', lw=2)
plt.plot(thresholds, recall[:-1], 'g-', label='Recall (Полнота)', lw=2)

plt.xlabel('Порог (Threshold)')
plt.ylabel('Значение метрики')
plt.title('Выбор порога: Precision против Recall')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('threshold.png', dpi=300, bbox_inches='tight')
plt.show()

custom_threshold = 0.7

# Метрика shap
# Влияние каждого признака на фрод
import shap
explainer = shap.TreeExplainer(models['XGBoost'])
X_shap = X_test.sample(5000, random_state=42)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Сохранение модели
import joblib
# Создаем словарь "Всё в одном"
model_package = {
    'model': models['XGBoost'],
    'features': x.columns.tolist(),
    'threshold': 0.7,
    'metrics': {
        'auc': roc_auc_score(y_test, y_pred_proba),
        'recall': recall_score(y_test, y_pred_custom)
    }
}

joblib.dump(model_package, 'full_fraud_package.pkl')

print("Все, что нужно, упаковано в full_fraud_package.pkl")

