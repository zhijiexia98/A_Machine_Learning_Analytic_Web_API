

import numpy as np
import pandas as pd
import csv
import io
import matplotlib.pyplot as plt
import seaborn as sns
import mpld3
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.utils import shuffle

def run_algo1(file_path):

    diabetes = pd.read_csv('diabetes.csv')
    diabetes.head()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=66)

    #KNN
    from sklearn.neighbors import KNeighborsClassifier
    training_accuracy = []
    test_accuracy = []
    # try n_neighbors from 1 to 10
    neighbors_settings = range(1, 11)
    for n_neighbors in neighbors_settings:
    # build the model
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)


    #SVM
    from sklearn.svm import SVC
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    svc = SVC(gamma=0.001, C=1000.)
    svc.fit(X_train_scaled, y_train)
    
    #DecisionTreeClassifier
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(max_depth=3, random_state=1000)
    clf = clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)

    df=pd.read_csv(file_path, low_memory=False, skipinitialspace=True, na_filter=False)
    df["Outcome"]=" "
    features=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    Xnew=df[features].values 
    y_predict=knn.predict(Xnew)
    # print(y_prednew)
    # df = pd.DataFrame(y_prednew)
    # df.to_csv("score.csv", index=False, header=["Outcome"])
    
    #get the figure plot
    fig = plt.figure()
    pos = plt.scatter(Xnew[:,1],Xnew[:,2], y_predict == 1)
    neg = plt.scatter(Xnew[:,1],Xnew[:,2], y_predict == 0)
    plt.xlabel('Glucose')
    plt.ylabel('Blood Pressure')
    plt.legend((pos,neg),('Positive', 'Negative'))
    plt.title('Classification Results on Diabetes with KNN')
    return mpld3.fig_to_html(fig)


# run_algo1('unknowns.csv')
# plt.show()

#def run_algo2(file_path):


#     def load_data(filename='white_wine.csv'):
#         with open(filename,'r') as file:
#             reader = csv.reader(file)
#             columnNames = next(reader)
#             rows = np.array(list(reader), dtype = float)
#             return columnNames, rows
        # df = pd.read_csv(filename)
        # # Now seperate the dataset as response variable and feature variabes
        # X = df.drop(['volatile acidity', 'total sulfur dioxide', 'chlorides', 'density','quality'], axis = 1)
        # y = df['quality']
        # columnNames = df.columns

        # from sklearn.preprocessing import StandardScaler
        # from sklearn.model_selection import train_test_split
        # # Train and Test splitting of data
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=50)
        # # Applying Standard scaling to get optimized result
        # sc = StandardScaler()    
        # X_train = sc.fit_transform(X_train)
        # X_test = sc.fit_transform(X_test)
        
        # return columnNames,rows,X_test,y_test

    # def seperate_labels(columnNames, rows):
    #         labelColumnIndex = columnNames.index('quality')
    #         ys = rows[:,labelColumnIndex]
    #         xs = np.delete(rows,[1, 4, 6, 7, 11],axis=1)
    #         del columnNames[labelColumnIndex]
    #         return columnNames, xs,ys
            
    # def preposess(columnNames, rows):
    #         xs = np.delete(rows,[1,4,6,7],axis=1)
    
    
    #         return xs

    # from sklearn.model_selection import train_test_split
    # from sklearn.ensemble import RandomForestClassifier
    # import pandas as pd


    

    # def rtf(X_train,X_test,y_train):
        
    #     gnb = RandomForestClassifier(n_estimators = 580)
    #     gnb.fit(X_train,y_train)
    #     y_pred=gnb.predict(X_test)
    #     return y_pred


    # columnNames,rows =load_data(file_path)

    # xtest=preposess(columnNames, rows)

    # columnNames,data =load_data('white_wine.csv')
    # columnNames, xs, ys = seperate_labels(columnNames, data)
    # y_pred = rtf(xs,xtest,ys)
    
    # fig = plt.figure()
    # plt.scatter(xs[:,0],xs[:,1],10,ys != y_pred, cmap = 'jet',alpha=0.3)
    # plt.xlabel('fixed acidity')
    # plt.ylabel('citric acid')
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

def run_algo2(file_path):
    wine = pd.read_csv('white_wine.csv')
    wine.head()
    wine= shuffle(wine)
    features=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol']
    y = wine['quality']
    X = wine[features]
    X = preprocessing.scale(X)
    X = StandardScaler().fit_transform(X)
    test_data_path = file_path
    test_data = pd.read_csv(test_data_path)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state=1)
    rf.fit(X_train,y_train)
    y_predrf = rf.predict(X_test)
    
    #test
    test_data_path = file_path
    test_data = pd.read_csv(test_data_path)
    test_X = test_data[features]

    test_X = preprocessing.scale(test_X)
    test_preds = rf.predict(test_X)
    #print(test_preds)

    fig = plt.figure()
    plt.scatter(test_X[:,10],test_X[:,8],test_preds, cmap = 'BuPu' )
    plt.xlabel('Alcohol')
    plt.ylabel('pH')
    return mpld3.fig_to_html(fig)

# run_algo2('unknowns2.csv')
# plt.show()


# from sklearn.metrics import confusion_matrix as cm
# from sklearn.metrics import classification_report as cr
# from mpl_toolkits.mplot3d import Axes3D

# outdir = ' ./OneClassSVM'
# if not os.path.exists(outdir):
#     os.mkdir(outdir)

#def run_algo3(file_path):


#     from sklearn import preprocessing

#     features = ['PTS','TRB','AST']
#     data = pd.read_csv(file_path)
#     data = data.fillna(0)
#     # features = ['PTS','TRB','AST']
#     X = data[features]
#     from sklearn.ensemble import IsolationForest
#     A = IsolationForest(contamination = 0.05, random_state = 42).fit(X)
#     pred = A.predict(X)
#     data = data.assign(prediction = pred)
#     #get sample decision boundary and sorted it
#     decision = A.decision_function(X)
#     data  = data.assign(decision = decision)

#     Ax = []
#     for a, b in zip(data.iloc[np.where(pred == -1)[0],1], decision[np.where(pred  == -1)[0]]):
#         Ax.append([a,round(b,2)])
#     Ax = sorted(Ax, key = lambda x: x[1])
#     # for i in Ax:
#     #     print(i)

#     print(pred.shape)
#     print('{} outliers and  {} inliers' .format(sum(pred == -1),sum(pred == 1)))
#     columna = data[['Player','decision']].to_numpy()
#     sorted_column = sorted(columna, key = lambda x: x[1])
#     pd.DataFrame(sorted_column,columns = ['Player','Scores']).to_csv( 'Isolation_Forrest_Scores.csv',index = False)

#     # for the top 3
#     Top3 = Ax[:3]
#     #print(Top3)
#     plot_num = 1
#     #plt.figure(figsize =(32,16))

#     for name, _ in Top3:
#         for i in features:
#             f_val = data.loc[data['Player'] == name, [i]].to_numpy()[0,0]
#             #plt.subplot(3,3,plot_num)
#             plot_num += 1
#             counts, bin_edges = np.histogram(data[i], bins = 100)
#             counts = counts/np.sum(counts)
#             cdf = np.cumsum(counts)
#             #plt.plot(bin_edges[1:],cdf)
#             #plt.vlines(f_val,0,1,colors = ['r'])
#             #plt.xlabel(i)
#             #plt.ylabel('Percentage of Players')
#             #plt.title(name.split('\\')[0]+' on CDF of '+ i)

#     #plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
#     #plt.savefig('Top3_3.png')
#     #plt.close()


#     #plot scattor
#     fig = plt.figure()
#     bx = fig.add_subplot(111, projection = '3d')

#     f1 = []
#     f2 = []
#     f3 = []
#     for name, _ in Top3:
#         f1.append(data.loc[data['Player'] == name, [features[0]]].to_numpy()[0,0])
#         f2.append(data.loc[data['Player'] == name, [features[1]]].to_numpy()[0,0])
#         f3.append(data.loc[data['Player'] == name, [features[2]]].to_numpy()[0,0])
#     bx.scatter(f1,f2,f3, marker = '^', color = 'r',depthshade = False)

#     for name, _ in Top3:
#         data.drop(data.loc[data['Player']==name].index,inplace=True)

#     f1 = data.loc[data['prediction']==-1,[features[0]]].to_numpy()[:,0]
#     f2 = data.loc[data['prediction']==-1,[features[1]]].to_numpy()[:,0]
#     f3 = data.loc[data['prediction']==-1,[features[2]]].to_numpy()[:,0]
#     bx.scatter(f1,f2,f3, depthshade=False)

#     f1 = data.loc[data['prediction']==1,[features[0]]].to_numpy()[:,0]
#     f2 = data.loc[data['prediction']==1,[features[1]]].to_numpy()[:,0]
#     f3 = data.loc[data['prediction']==1,[features[2]]].to_numpy()[:,0]
#     bx.scatter(f1,f2,f3, depthshade=False)

#     bx.set_xlabel(features[0])
#     bx.set_ylabel(features[1])
#     bx.set_zlabel(features[2])
#     plt.title('inliers and outliers in 3D')
#     return mpld3.fig_to_html(fig)


# run_algo3('nba_players_stats_19_20_per_game.csv')
# plt.show()

def run_algo3(file_path):

    import csv
    import numpy as np
    import pandas as pd
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split

    data = pd.read_csv(file_path)
    df = data.drop(['Player','Pos','Tm'],axis=1) # drop columns with strings
    df = df.dropna(axis=1) #drop col with missing values
    df.head()


    selected_features=['TRB','AST','PTS']
    def load_clean_normed_data():
        df = pd.read_csv('nba_players_stats_19_20_per_game.csv')[['Player']+selected_features]
        for stat in selected_features:
            df[stat] = df[stat]/df[stat].max() #Normalize
        return df

    df = load_clean_normed_data()

    def train_one_class_svm(data):
        from sklearn.svm import OneClassSVM
        return OneClassSVM(kernel = 'rbf').fit(data[selected_features])

    def train_elliptic_envelope(data):
        from sklearn.covariance import EllipticEnvelope
        return EllipticEnvelope(contamination = 0.05,random_state = 42).fit(data[selected_features])

    def train_isolation_forest(data):
        from sklearn.ensemble import IsolationForest
        return IsolationForest(contamination = 0.05,random_state = 42).fit(data[selected_features])
    clf_ee = train_elliptic_envelope(df)
    scores = clf_ee.decision_function(df[selected_features])
    topthreeIndices = np.argsort(scores)[:3] #gives indices of top3 , sorted list of scores
    top3 = df.iloc[topthreeIndices] # iloc: gives me samples of these indices

    outliers = scores < 0 #find outliers
    inliers = scores >= 0 #find inliers
    outliersSansTopThree = outliers.copy()
    outliersSansTopThree[topthreeIndices] = False

    fig = plt.figure()

    ax = fig.add_subplot(111,projection = '3d')
    ax.scatter(df.iloc[~outliers][selected_features[0]],
            df.iloc[~outliers][selected_features[1]],
            df.iloc[~outliers][selected_features[2]],label = 'Inliers')
    ax.scatter(df.iloc[outliersSansTopThree][selected_features[0]],
            df.iloc[outliersSansTopThree][selected_features[1]],
            df.iloc[outliersSansTopThree][selected_features[2]],label = 'Outliers')
    ax.scatter(top3[selected_features[0]],
            top3[selected_features[1]],
            top3[selected_features[2]], label='Top three outliers')

    ax.legend()
    ax.set_xlabel('TRB')
    ax.set_ylabel('AST')
    ax.set_zlabel('PTS')
    #plt.legend((inliers,outliers, top3),('Inliers', 'Outliers', 'Top three outliers'))
    plt.title('Outliers Detection on NBA Data with Elliptic Envelop')

    # with io.StringIO() as stringbuffer:
    #     fig.savefig(stringbuffer, format = 'svg')
    #     svgstring = stringbuffer.getvalue()
    # return svgstring

    with io.StringIO() as stringbuffer:
        fig.savefig(stringbuffer,format='svg')
        svgstring = stringbuffer.getvalue()
    return svgstring

run_algo3('nba_players_stats_19_20_per_game.csv')
plt.show()
