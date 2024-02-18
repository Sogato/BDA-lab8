import graphviz
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz


# Функция для преобразования данных
def transform_data(df):
    """Преобразование данных датафрейма."""
    df["Gender"] = df["Gender"].apply(lambda toLabel: 0 if toLabel == 'male' else 1)
    df["Age"] = df["Age"].fillna(df["Age"].mean())


# Загрузка и первоначальный анализ обучающего набора данных
training = pd.read_csv("titanic-train.csv")
print("Информация об обучающем наборе данных:")
training.info()
print("\nПервые пять строк обучающего набора данных:")
print(training.head())

# Преобразование обучающего набора данных
transform_data(training)

print("\nИнформация об обучающем наборе данных после преобразования:")
training.info()
print("\nПервые пять строк обучающего набора данных после преобразования:")
print(training.head())

# Подготовка данных для модели
y_target = training["Survived"].values
columns = ["Fare", "Pclass", "Gender", "Age", "SibSp"]
X_input = training[columns].values

# Обучение модели
clf_train = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf_train.fit(X_input, y_target)

# Вывод оценки точности модели
print(f"\nТочность модели на обучающих данных: {clf_train.score(X_input, y_target)}")

# Генерация визуализации дерева решений
dot_data = export_graphviz(clf_train, out_file=None, feature_names=columns,
                           class_names=["Not Survived", "Survived"], filled=True)
graph = graphviz.Source(dot_data)
graph.render(filename='tree_visualization', format='png', cleanup=True)

# Загрузка и преобразование тестового набора данных
testing = pd.read_csv("titanic-test.csv")
print("\nИнформация о тестовом наборе данных до преобразования:")
testing.info()
print("\nПервые пять строк тестового набора данных:")
print(testing.head())

transform_data(testing)

print("\nИнформация о тестовом наборе данных после преобразования:")
testing.info()
print("\nПервые пять строк тестового набора данных после преобразования:")
print(testing.head())

print(f"\nКоличество записей в тестовом наборе данных: {testing.shape[0]}")
print("\nКоличество отсутствующих значений по переменным:")
print(testing.isnull().sum())

# Применение модели к тестовым данным
X_input_test = testing[columns].values
predicted_survival = clf_train.predict(X_input_test)
target_labels = pd.DataFrame({'Est_Survival': predicted_survival, 'Name': testing['Name']})

print("\nПервые пять предсказаний выживания:")
print(target_labels.head())

# Сравнение с фактическими данными
all_data = pd.read_csv("titanic_all.csv")
testing_results = pd.merge(target_labels, all_data[['Name', 'Survived']], on='Name')
acc = np.sum(testing_results['Est_Survival'] == testing_results['Survived']) / float(len(testing_results))
print(f"\nТочность модели на тестовых данных: {acc}")

# Дополнительный анализ с использованием полного набора данных
all_data_transformed = all_data.copy()
transform_data(all_data_transformed)
X = all_data_transformed[columns].values
y = all_data_transformed["Survived"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
clf_train.fit(X_train, y_train)

train_score = clf_train.score(X_train, y_train)
test_score = clf_train.score(X_test, y_test)
print(f'\nTraining score = {train_score}')
print(f'Testing score = {test_score}')
