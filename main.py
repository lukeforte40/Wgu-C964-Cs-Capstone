import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot
import seaborn as sns

# Import data sets
df = pd.read_csv("data/test.csv",encoding='unicode_escape')

# Prep data
df.drop(columns = ["textID",'Time of Tweet', 'Age of User',
       'Country', 'Population -2020', 'Land Area (Km²)', 'Density (P/Km²)' ], inplace=True)

df.dropna()

df_labels = df["sentiment"].astype(str)
df_data = df["text"].astype(str)

vectorizer = CountVectorizer(
    analyzer = 'word',
    lowercase = False,
)

vectorizer = vectorizer.fit_transform(
    df_data
)

vectorizer_nd = vectorizer.toarray()

# Split data into dependent and independent
X_train, X_test, y_train, y_test  = train_test_split(
        vectorizer_nd,
        df_labels,
        train_size=0.70,
        random_state=1234)

# Set up Linear regression model
model = linear_model.LogisticRegression()

model.fit(X=X_train,y=y_train)

y_prediction = model.predict(X_test)

accuracy = metrics.accuracy_score(y_test,y_prediction)

# Input loop to start program
run = 1
while run:

    input_text = input("Select an item from the list to begin:\n"
    "1) Input a custom sentence to analyze.\n"
    "2) View a chart of ratios of each sentiment for training data.\n"
    "3) View a chart of ratios of each sentiment from results.\n"
    "4) View a confusion matrix on the correctness of the model for the sample data.\n"
    "5) View accuracy of prediction.\n"
    "6) "
    "3) Exit.\n"
    "Input: ")
    try:
        match int(input_text):
            case 1: # Input a custom sentence to analyze
                sentiment_input = input("Write a sentence to analyze (Type 'exit' to close):")
                sentiment = model.predict(sentiment_input)
                print(f"Sentiment: {sentiment}")
            case 2: # View a chart of ratios of each sentiment for training data.
                pyplot.pie(y_train.value_counts(), labels=["positive", "negative", "neutral", "other"])
                pyplot.title("Ratios of each sentiment for training data")
                pyplot.show()
            case 3: # View a chart of ratios of each sentiment from results.
                pyplot.pie(pd.Series(y_prediction).value_counts(), labels=["positive", "negative", "neutral", "other"])
                pyplot.title("Ratios of each sentiment from results")
                pyplot.show()
            case 4: # View a confusion matrix on the correctness of the model for the sample data.
                cm = metrics.confusion_matrix(y_test, y_prediction)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                pyplot.xlabel('Predicted')
                pyplot.ylabel('Actual')
                pyplot.title('Confusion Matrix')
                pyplot.show()
            case 5: # View accuracy of prediction.
                print("Accuracy: ")
                print(accuracy)
            case 6: # Exit
                run = 0
            case _:
                print("Error! Command not recognized.")
                pass
    except:
        print("Error! Please try again.")