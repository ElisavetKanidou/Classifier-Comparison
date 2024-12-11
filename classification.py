# Κάνουμε το drive του λογαριασμού προσβάσιμο στο notepad του google colab
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
# Διαβάζουμε τα δεδομένα από το Excel που βρίσκεται στον γονικό φάκελο του drive του λογαριασμού ο οποίος τρέχει τον κώδικα
df = pd.read_excel('/content/drive/MyDrive/Dataset2Use_Assignment1.xlsx')

# Τα απαραίτητα import
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import data_table

# Επιλέγουμε τις στήλες που θέλουμε να κανονικοποιήσουμε και να ελέγξουμε για ελλείπεις εγγραφές
# Αυτο το list θα χρησιμοποιηθει για πολλές χρήσεις
columns_to_normalize = ['365* ( Β.Υ / Κοστ.Πωλ )', 'Λειτ.Αποτ/Συν.Ενεργ. (ROA)', 'ΧΡΗΜ.ΔΑΠΑΝΕΣ / ΠΩΛΗΣΕΙΣ', ' ΠΡΑΓΜΑΤΙΚΗ ΡΕΥΣΤΟΤΗΤΑ :  (ΚΕ-ΑΠΟΘΕΜΑΤΑ) / Β.Υ', '(ΑΠΑΙΤ.*365) / ΠΩΛ.', 'Συν.Υποχρ/Συν.Ενεργ', 'Διάρκεια Παραμονής Αποθεμάτων', 'Λογαριθμος Προσωπικού']

# Η στήλη που θέλουμε να προβλέψουμε είναι η 'Κατάσταση Εταιρείας'
target_column = 'ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)'

df_healthy = df[df[target_column] == 1]
df_bankrupt = df[df[target_column] == 2]

# Ομαδοποίηση κατά έτος
healthy_counts = df_healthy.groupby('ΕΤΟΣ')[target_column].count()
bankrupt_counts = df_bankrupt.groupby('ΕΤΟΣ')[target_column].count()

# Σχεδίαση του γραφήματος
fig, ax = plt.subplots()
healthy_counts.plot(kind='bar', width=0.4, position=0, color='green', label='Υγιείς')
bankrupt_counts.plot(kind='bar', width=0.4, position=1, color='red', label='Χρεωκοπημένες')

# Προσαρμογή του γραφήματος
ax.set_title('Αριθμός Υγιών και Χρεωκοπημένων Επιχειρήσεων για κάθε Έτος')
ax.set_xlabel('Έτος')
ax.set_ylabel('Αριθμός Επιχειρήσεων')
ax.legend()
plt.show()

# ΕΠΙΛΕΞΑΜΕ ΝΑ ΚΑΝΟΥΜΕ ΤΗΝ ΚΑΝΟΝΙΚΟΠΟΙΗΣΗ ΚΑΙ ΤΟΝ ΕΛΕΓΧΟ ΕΛΛΕΙΠΩΝ ΣΤΗΛΩΝ ΠΡΙΝ ΤΟΝ ΣΧΕΔΙΑΣΜΟ ΤΟΥ FIGURE 2 ΕΠΕΙΔΗ ΣΕ ΑΛΛΗ ΠΕΡΙΠΤΩΣΗ ΔΕΝ ΒΓΑΙΝΕΙ ΣΩΣΤΑ

# Επιλογή των γραμμών που έχουν τουλάχιστον ένα 0 στις επιλεγμένες στήλες
rows_with_zeros = df[(df[columns_to_normalize] == 0).any(axis=1)] #Η λίστα columns_to_normalize συμπίπτει με τις στήλες που θέλουμε να ελέγξουμε για ελλείπεις εγγραφές

# Εκτύπωση των γραμμών που πληρούν τη συνθήκη
data_table.DataTable(rows_with_zeros)

# Φτιάχνουμε ένα υπο-DataFrame μόνο με τις επιλεγμένες στήλες
df_selected = df[columns_to_normalize]

# Δημιουργούμε ένα αντικείμενο MinMaxScaler
scaler = MinMaxScaler()

# Εφαρμόζουμε την κανονικοποίηση στο υπο-DataFrame
df_normalized = pd.DataFrame(scaler.fit_transform(df_selected), columns=df_selected.columns)

# Συνδυάζουμε τα κανονικοποιημένα δεδομένα με τα υπόλοιπα του DataFrame
df[columns_to_normalize] = df_normalized

# Επιλέγουμε μόνο τις στήλες Α έως Η και την L
selected_columns = ['365* ( Β.Υ / Κοστ.Πωλ )', 'Λειτ.Αποτ/Συν.Ενεργ. (ROA)', 'ΧΡΗΜ.ΔΑΠΑΝΕΣ / ΠΩΛΗΣΕΙΣ', ' ΠΡΑΓΜΑΤΙΚΗ ΡΕΥΣΤΟΤΗΤΑ :  (ΚΕ-ΑΠΟΘΕΜΑΤΑ) / Β.Υ', '(ΑΠΑΙΤ.*365) / ΠΩΛ.', 'Συν.Υποχρ/Συν.Ενεργ', 'Διάρκεια Παραμονής Αποθεμάτων', 'Λογαριθμος Προσωπικού','ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)']
df_selected = df[selected_columns]

# Ομαδοποίηση κατά κατάσταση εταιρείας
grouped_by_status = df_selected.groupby(target_column) # target_column = 'ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)'

# Δημιουργία Figure με δύο subfigures
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 12))

# Υγιείς Εταιρείες
healthy_companies = grouped_by_status.get_group(1)
healthy_companies = healthy_companies.drop(columns=target_column)
healthy_stats = healthy_companies.describe().loc[['min', 'max', 'mean']]
healthy_stats.plot(kind='bar', ax=ax1, title='Υγιείς Εταιρείες')
ax1.set_ylabel('Τιμές')

# Πτωχευμένες Εταιρείες
bankrupt_companies = grouped_by_status.get_group(2)
bankrupt_companies = bankrupt_companies.drop(columns=target_column)
bankrupt_stats = bankrupt_companies.describe().loc[['min', 'max', 'mean']]
bankrupt_stats.plot(kind='bar', ax=ax2, title='Πτωχευμένες Εταιρείες')
ax2.set_ylabel('Τιμές')

# Σχεδίαση
plt.tight_layout()
plt.show()

# Δημιουργούμε ένα αντικείμενο StratifiedKFold με 4 folds
n_splits = 4
skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# counter για τα folds
counter = 0

# Δημιουργούμε DataFrame για να αποθηκεύσουμε τα αποτελέσματα
results_df_columns = ['Classifier Name', 'Training or Test Set', 'Balanced or Unbalanced Train Set',
                      'Number of Training Samples', 'Number of Non-Healthy Companies in Training Sample',
                      'TP', 'TN', 'FP', 'FN', 'ROC-AUC']
results_df = pd.DataFrame(columns=results_df_columns)

# Ορίζουμε τους classifiers
classifiers = {
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(),
    'Decision Trees': DecisionTreeClassifier(),
    'Random Forests': RandomForestClassifier(),
    'k-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machines': SVC(),
    'Gradient Boosting': GradientBoostingClassifier()  # Additional model
}

# Επαναληπτική εκτέλεση των classifiers
for classifier_name, classifier in classifiers.items():
    print(''.center(60,'-'))
    print((f' Training and Evaluating {classifier_name} ').center(60, "-"))
    print(''.center(60,'-'))

    # Επαναφορά counter για τα folds
    counter = 0

    # Αρχικοποίηση is_balnanced
    is_balanced = False

    # Επαναληπτική εκτέλεση για κάθε fold
    for train_index, test_index in skf.split(df, df[target_column]):
        train_set, test_set = df.iloc[train_index].copy(), df.iloc[test_index].copy()

        # Υπολογισμός των υγιών και χρεωκοπημένων στα train και test sets
        healthies_train = train_set[train_set[target_column] == 1]
        bankrupts_train = train_set[train_set[target_column] == 2]
        healthies_test = test_set[test_set[target_column] == 1]
        bankrupts_test = test_set[test_set[target_column] == 2]

        # Εκτύπωση των αποτελεσμάτων για κάθε fold
        print('\n',(f"\n--- Fold {counter + 1} ").ljust(45,'-'))
        print(f"Train set - Υγιείς: {len(healthies_train)}, Χρεωκοπημένες: {len(bankrupts_train)}")
        print(f"Test set - Υγιείς: {len(healthies_test)}, Χρεωκοπημένες: {len(bankrupts_test)}")


        # Εάν η αναλογία είναι πάνω από 3 υγιείς / 1 χρεωκοπημένη
        if len(healthies_train) > 3 * len(bankrupts_train):
            # Διαλέγουμε τυχαία όσες υγιείς εταιρείες χρειάζεται
            random_healthies = healthies_train.sample(n=3 * len(bankrupts_train), random_state=42)
            # Αφαιρούμε τις επιλεγμένες εταιρείες από το train set
            train_set = train_set.drop(random_healthies.index)
            train_set = train_set.drop(bankrupts_train.index)

            # Προσθέτουμε τις επιλεγμένες εταιρείες στο test set
            test_set = pd.concat([test_set, train_set])

            # Ενημερώνουμε το index των υγιών εταιριών
            healthies_train = train_set[train_set[target_column] == 1]
            train_set = train_set.drop(healthies_train.index) # άδειασμα του train_set

            # Επανυπολογισμός του train_set
            train_set = pd.concat([bankrupts_train, random_healthies])

            # Ενημέρωση index για train set
            healthies_train = train_set[train_set[target_column] == 1]
            bankrupts_train = train_set[train_set[target_column] == 2]

            # Ενημέρωση index για test set
            healthies_test = test_set[test_set[target_column] == 1]
            bankrupts_test = test_set[test_set[target_column] == 2]
            is_balanced = True # Ο πίνακας train_set είναι πλέον balanced 3 προς 1

        # Εκτύπωση νέου Balanced Training Set
        print('\n',(f"\n--- Fold {counter + 1}: Νέο Balanced Training Set ").ljust(45,'-'))
        print(f"Train set - Υγιείς: {len(healthies_train)}, Χρεωκοπημένες: {len(bankrupts_train)}")
        print(f"Test set - Υγιείς: {len(healthies_test)}, Χρεωκοπημένες: {len(bankrupts_test)}")

        # Ενημέρωση του counter
        counter += 1

        # Features and target για το training
        features_train = train_set.drop(columns=[target_column])
        target_train = train_set[target_column]

        # Features and target για το testing
        features_test = test_set.drop(columns=[target_column])
        target_test = test_set[target_column]

        # Εκπαιδεύουμε το μοντέλο
        classifier.fit(features_train, target_train)

        # Μέθοδος για εκτίμηση μοντέλου
        def evaluate_model(classifier, features, target, set_type):
            predictions = classifier.predict(features)
            accuracy = accuracy_score(target, predictions)
            precision = precision_score(target, predictions)
            recall = recall_score(target, predictions)
            f1 = f1_score(target, predictions)
            roc_auc = roc_auc_score(target, predictions)

            # Εκτύπωση μετρικών
            print(f"\n{set_type} Set Evaluation:")
            print(f"Accuracy:  {accuracy:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall:    {recall:.2f}")
            print(f"F1 Score:  {f1:.2f}")
            print(f"ROC-AUC:   {roc_auc:.2f}")

            # Εκτύπωση confusion matrix
            conf_matrix = confusion_matrix(target, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Confusion Matrix - {set_type} Set')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()

            # Επιστροφή μετρικών/μεταβλητών
            return accuracy, precision, recall, f1, roc_auc

        # Αξιολόγηση του training set
        evaluate_model(classifier, features_train, target_train, 'Training')

        # Αξιολόγηση του test set
        accuracy_test, precision_test, recall_test, f1_test, roc_auc_test = evaluate_model(classifier, features_test, target_test, 'Test')

        # Αποθήκευση αποτελεσμάτων στο DataFrame
        results_df = pd.concat([results_df, pd.DataFrame.from_records([{
            'Classifier Name': classifier_name,
            'Training or Test Set': 'Training',
            'Balanced or Unbalanced Train Set': 'Balanced' if is_balanced else 'Unbalanced',
            'Number of Training Samples': len(features_train),
            'Number of Non-Healthy Companies in Training Sample': len(target_train[target_train == 2]),
            'TP': confusion_matrix(target_train, classifier.predict(features_train))[1, 1],
            'TN': confusion_matrix(target_train, classifier.predict(features_train))[0, 0],
            'FP': confusion_matrix(target_train, classifier.predict(features_train))[0, 1],
            'FN': confusion_matrix(target_train, classifier.predict(features_train))[1, 0],
            'ROC-AUC': roc_auc_score(target_train, classifier.predict(features_train))
        }])], ignore_index=True, axis=0)

        results_df = pd.concat([results_df, pd.DataFrame.from_records([{
            'Classifier Name': classifier_name,
            'Training or Test Set': 'Test',
            'Balanced or Unbalanced Train Set': 'Balanced' if is_balanced else 'Unbalanced',
            'Number of Training Samples': len(features_test),
            'Number of Non-Healthy Companies in Training Sample': len(target_train[target_train == 2]),
            'TP': confusion_matrix(target_test, classifier.predict(features_test))[1, 1],
            'TN': confusion_matrix(target_test, classifier.predict(features_test))[0, 0],
            'FP': confusion_matrix(target_test, classifier.predict(features_test))[0, 1],
            'FN': confusion_matrix(target_test, classifier.predict(features_test))[1, 0],
            'ROC-AUC': roc_auc_score(target_test, classifier.predict(features_test))
        }])], ignore_index=True, axis=0)


# Αποθήκευση αποτελεσμάτων σε CSV αρχείο
results_df.to_csv('/content/drive/My Drive/balancedDataOutcomesTEST.csv', index=True)