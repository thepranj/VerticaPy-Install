from verticapy import *
from verticapy.connect import *
import sys
import getopt
from verticapy.learn.linear_model import LogisticRegression
from verticapy.learn.model_selection import cross_validate

def connect():
    conn_info = {"host": "34.237.154.116", 
                     "port": "5433", 
                     "database": "testdrive", 
                     "password": "password", 
                     "user": "dbadmin"}
    new_auto_connection(conn_info, name = 'my_cluster')
    change_auto_connection('my_cluster')
    #print(read_auto_connect())           
    print("Connected to the Vertica Database.\n")

def main():
    connect()
    #print("Hello Vertica!")
    churn = vDataFrame('public.churn')
    #print(churn.head())
    # Data Prep
    print("Running one hot encoding on new data...")
    churn = prep(churn)
    print("Success!")
    #print("""Saving to database as 'demo.churn_clean'\n""")
    # Model Train
    print("Begin training")
    churn = train(churn)
    # Predict
    print("Begin predicting")
    results(churn)
    #print("""Predictions saved to 'Desktop/churn_results.csv'\n""")
    print('Workflow complete... Exiting')

# Run Main 2 if you want to walk through the workflow step by step
# def main2():
#     connect()
#     churn = vDataFrame('public.churn')
#     val1 = ''
#     intro = '''Please Enter a command: \n1) data prep\n2) train\n3) predict\n4) quit\n\n'''
#     while val1 != '4':
#         val1 = input(intro)
#         if val1 == '1':
#             print("Begin data prep")
#             print("Running one hot encoding on new data...")
#             churn = prep(churn)
#             print("Success!")
#             print("""Saving to database as 'demo.churn_clean'\n""")
#         elif val1 == '2':
#             pass
#         elif val1 == '3':
#             print("Begin predicting")
#             print("""Predictions saved to 'Desktop/pred.csv'\n""")
#             #model = LogReg  
#             #predict(model)  
#         elif val1 == '4':
#             print("quitting\n")
    
def prep(churn):
    for column in ["DeviceProtection", 
                "MultipleLines",
                "PaperlessBilling",
                "Churn",
                "TechSupport",
                "Partner",
                "StreamingTV",
                "OnlineBackup",
                "Dependents",
                "OnlineSecurity",
                "PhoneService",
                "StreamingMovies"]:
        churn[column].decode("Yes",1,0)
    churn.get_dummies().drop(["gender", 
                              "Contract", 
                              "PaymentMethod", 
                              "InternetService"])

    return(churn)                              

def train(churn):
    drop(name = "public.churn_model")
    model = LogisticRegression("churn_model", 
                           penalty = 'L2', 
                           tol = 1e-6, 
                           max_iter = 1000, 
                           solver = "BFGS")
    # print("Running cross_validate function\n")                       
    # cross_validate(model, churn, churn.get_columns(exclude_columns = ["churn"]), 'churn')
    print("Fitting logistic regression model...")
    model.fit(churn, 
          churn.get_columns(exclude_columns = ["churn", "customerID"]), 
          'churn')      
    print("Success! demo.churn_model created")
    print("Model AUC: " + str(model.score(method="auc")) + '\n')
    # Begin Predict
    model.predict(churn,
                  X = churn.get_columns(exclude_columns = ["churn", "customerID"]),
                  name = 'pred_probs')
    churn.sort({"pred_probs":"desc"})       
    churn['pred_probs'].dropna()
    return(churn)

def results(churn):
    # Saves results with predictions to DB as a new table
    churn.to_db(name = '"public"."churn_results"', relation_type = 'table')
    print("New table created in database: churn_results")
    # Save csv with predictions
    #churn.to_csv(name = 'churn_results', path = "C:/Users/PSingh5/Desktop/python/")       

def predict(model):
    # Moved to train function
    pass
    
if __name__ == "__main__":
    main()