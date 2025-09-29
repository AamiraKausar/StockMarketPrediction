#Evaluate the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
print("Accuracy for Logistic regression:", accuracy_score(y_test, y_pred))
print("Precision for Logistic regression:", precision_score(y_test, y_pred))
print("Recall for Logistic regression:", recall_score(y_test, y_pred))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

# Evaluate
print("Accuracy for Random Forest:", accuracy_score(y_test, y_pred1))
print("Precision for Random Forest:", precision_score(y_test, y_pred1))
print("Recall for Random Forest:", recall_score(y_test, y_pred1))

cm = confusion_matrix(y_test, y_pred1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for RandomForest')
plt.show()


#Visualize predictions vs actual outcomes in a plot
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(y_pred, label='Predicted', marker='x')
plt.title("Actual vs Predicted Stock Movement")
plt.xlabel("Days")
plt.ylabel("Label (0 = Down, 1 = Up)")
plt.legend()
plt.show()