import pandas as pd
from matplotlib import pyplot

df = pd.read_csv("data/test.csv")

print(df)

"""# Read the positive and negative text files into lists with the correct encoding
with open('data/positive.txt', 'r', encoding='ISO-8859-1') as file:
    positive_words = file.read().splitlines()

with open('data/negative.txt', 'r', encoding='ISO-8859-1') as file:
    negative_words = file.read().splitlines()

# Create dataframe and add words
df_positive = pd.DataFrame(positive_words, columns=['Numbers'])
df_negative = pd.DataFrame(negative_words, columns=['Numbers'])

df_positive.hist()"""

# Function to perform sentiment analysis
def analyze_sentiment(text):
    words = text.lower().split()
    
    #Initialize a sentiment score to keep track of the sentiment of the text.
    sentiment_score = 0

    # Count positive and negative words #Loop through the words in the preprocessed text and adjust the sentiment score based on the presence of words in positive and negative lists.
    for word in words:
        if word in positive_words:
            sentiment_score += 1
        elif word in negative_words:
            sentiment_score -= 1

    # Classify sentiment based on score
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"
    
# Input loop to start program
run = 1
while run:
    input_text = input("Write a sentence to analyze (Type 'exit' to close):")
    if input_text != "exit":
        sentiment = analyze_sentiment(input_text)
        print(f"Sentiment: {sentiment}")
    else:
        run = 0
