from utils._utils import load_data_from_csv
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

if __name__ == '__main__':

    # Load training and test data
    all_padded_resampled_beats, y_train, all_resampled_pre_rr, all_resampled_post_rr, all_resampled_avg_rr = load_data_from_csv(os.path.join('data', 'ecg_training.csv'))
    padded_beats_test, y_test, all_pre_rr_test, all_post_rr_test, all_avg_rr_test = load_data_from_csv(os.path.join('data', 'ecg_test.csv'))

    # Combine the ECG beats with the RR features
    X_combined_train = np.hstack((all_padded_resampled_beats,
                            all_resampled_pre_rr.reshape(-1, 1),
                            all_resampled_post_rr.reshape(-1, 1),
                            all_resampled_avg_rr.reshape(-1, 1)))
    
    # Combine the ECG beats with the RR features
    X_combined_test = np.hstack((padded_beats_test,
                            all_pre_rr_test.reshape(-1, 1),
                            all_post_rr_test.reshape(-1, 1),
                            all_avg_rr_test.reshape(-1, 1)))

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_combined_train, y_train)

    # Initialize the Gradient Boosting Classifier
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    # Fit the model
    gb_clf.fit(X_train_resampled, y_train_resampled)

    # Predict and evaluate
    y_pred = gb_clf.predict(X_combined_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    