Project Overview
Develop and present a machine learning application solving a proposed problem.This project will act as an AI model that detects sentence sentiment in 3 categories positive, negative or neutral. There are also functions to analyze the data in various charts and graphs.

Project Tasks

A.  Create a letter of transmittal and a project proposal to convince senior, nontechnical managers and executives to implement your data product approved in Task 1. The proposal should include each of the following:

•   a summary of the problem

•   a description of how the data product benefits the customer and supports the decision-making process

•   an outline of the data product

•   a description of the data that will be used to construct the data product

•   the objectives and hypotheses of the project

•   an outline of the project methodology

•   funding requirements

•   the impact of the solution on stakeholders

•   ethical and legal considerations and precautions that will be used when working with and communicating about sensitive data

•   your expertise relevant to the solution you propose

 
    Note: Expertise described here could be real or hypothetical to fit the project topic you have created.

 

B.  Write an executive summary directed to IT professionals that addresses each of the following requirements:

•   the decision support problem or opportunity you are solving for

•   a description of the customers and why this product will fulfill their needs

•   existing gaps in the data products you are replacing or modifying (if applicable)

•   the data available or the data that needs to be collected to support the data product lifecycle

•   the methodology you use to guide and support the data product design and development

•   deliverables associated with the design and development of the data product

•   the plan for implementation of your data product, including the anticipated outcomes from this development

•   the methods for validating and verifying that the developed data product meets the requirements and, subsequently, the needs of the customers

•   the programming environments and any related costs, as well as the human resources that are necessary to execute each phase in the development of the data product

•   a projected timeline, including milestones, start and end dates, duration for each milestone, dependencies, and resources assigned to each task

 

C.  Design and develop your fully functional data product that addresses your identified business problem or organizational need from part A. Include each of the following attributes, as they are the minimum required elements for the product:

•   one descriptive method and one non-descriptive (predictive or prescriptive) method

        # descriptive
        pyplot.pie(y_train.value_counts(), labels=["postive","negative","neutral","other"])
        pyplot.title("Ratios of each sentiment for training data")
        pyplot.show()
        
        # non-descriptive
        pyplot.pie(pd.Series(prediction).value_counts(), labels=["postive","negative","neutral","other"])
        pyplot.title("Ratios of each sentiment from results")
        pyplot.show()

•   collected or available datasets

        https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset/data?select=train.csv

•   decision support functionality

        Model-Driven

•   ability to support featurizing, parsing, cleaning, and wrangling datasets

        # Prep data
        df.drop(columns = ["textID",'Time of Tweet', 'Age of User',
               'Country', 'Population -2020', 'Land Area (Km²)', 'Density (P/Km²)' ], inplace=True)
        
        df.dropna()
        
        df_labels = df["sentiment"].astype(str)
        df_data = df["text"].astype(str)

•   methods and algorithms supporting data exploration and preparation

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

•   data visualization functionalities for data exploration and inspection

        # descriptive
        pyplot.pie(y_train.value_counts(), labels=["postive","negative","neutral","other"])
        pyplot.title("Ratios of each sentiment for training data")
        pyplot.show()
        
        # non-descriptive
        pyplot.pie(pd.Series(prediction).value_counts(), labels=["postive","negative","neutral","other"])
        pyplot.title("Ratios of each sentiment from results")
        pyplot.show()

•   implementation of interactive queries

        case 1: # Input a custom sentence to analyze
        sentiment_input = input("Write a sentence to analyze (Type 'exit' to close):")
        sentiment = model.predict(sentiment_input)
        print(f"Sentiment: {sentiment}")

•   implementation of machine-learning methods and algorithms

        # Set up Linear regression model
        model = linear_model.LogisticRegression()
        
        model.fit(X=X_train,y=y_train)


•   functionalities to evaluate the accuracy of the data product

        accuracy = metrics.accuracy_score(y_test,y_prediction)

        #confusion matrix
        cm = metrics.confusion_matrix(y_test, y_prediction)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        pyplot.xlabel('Predicted')
        pyplot.ylabel('Actual')
        pyplot.title('Confusion Matrix')
        pyplot.show()

•   industry-appropriate security features
        
        See code

•   tools to monitor and maintain the product

        See code

•   a user-friendly, functional dashboard that includes three visualization types

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

        # plots and graphs
        # descriptive pie
        pyplot.pie(y_train.value_counts(), labels=["positive","negative","neutral","other"])
        pyplot.title("Ratios of each sentiment for training data")
        pyplot.show()
        
        # non-descriptive
        pyplot.pie(pd.Series(y_prediction).value_counts(), labels=["positive","negative","neutral","other"])
        pyplot.title("Ratios of each sentiment from results")
        pyplot.show()
        
        #confusion matrix
        cm = metrics.confusion_matrix(y_test, y_prediction)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        pyplot.xlabel('Predicted')
        pyplot.ylabel('Actual')
        pyplot.title('Confusion Matrix')
        pyplot.show()

D.  Create each of the following forms of documentation for the product you have developed:

•   a business vision or business requirements document

•   raw and cleaned datasets with the code and executable files used to scrape and clean data (if applicable)

•   code used to perform the analysis of the data and construct a descriptive, predictive, or prescriptive data product

•   assessment of the hypotheses for acceptance or rejection

•   visualizations and elements of effective storytelling supporting the data exploration and preparation, data analysis, and data summary, including the phenomenon and its detection

•   assessment of the product’s accuracy 

•   the results from the data product testing, revisions, and optimization based on the provided plans, including screenshots

•   source code and executable file(s)

•   a quick-start guide summarizing the steps necessary to install and use the product

 

E.  Acknowledge sources, using in-text citations and references, for content that is quoted, paraphrased, or summarized.

 

F.  Demonstrate professional communication in the content and presentation of your submission.