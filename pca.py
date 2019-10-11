#import everythong here las
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pca():
    print("hellow i am here las ")
    data=pd.read_csv("Wine.csv")
    print(data.head())
    print(data.describe())

    #make the x and the y las
    #here we had to make the whether the wine is of type one or two las
    x=data.iloc[:,0:13].values
    y=data.iloc[:,13].values

    #spliting it into the training and the testing set las

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=0)

    #performing the standard scaler las
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    #performing the logistic regression here

    from sklearn.linear_model import LogisticRegression
    clas=LogisticRegression(random_state=0,solver='liblinear')
    clas.fit(x_train,y_train)

    #predicting the vlaues las
    ypred11=clas.predict(x_test)

    #making the confucion matrices las

    from sklearn.metrics import confusion_matrix
    cn2=confusion_matrix(y_test,ypred11)
    print("Befor the performition of the pca las")
    print(cn2)

    #perform the pca by taking the compnents of the
    from sklearn.decomposition import PCA
    pca=PCA(n_components=2)
    x_train=pca.fit_transform(x_train)
    x_test=pca.transform(x_test)
    print(pca.explained_variance_)

    #performing the logistic regression la
    from sklearn.linear_model import LogisticRegression
    clasf=LogisticRegression(random_state=0,solver='liblinear')
    clasf.fit(x_train,y_train)

    #predicting the logistic regression la
    ypred=clasf.predict(x_test)

    #making the confusion matrix las

    from sklearn.metrics import confusion_matrix
    cn=confusion_matrix(y_test,ypred)
    print("After the applycation of the pca laa")
    print(cn)

    #visualising the datas las
    from matplotlib.colors import ListedColormap
    xset,yset=x_train,y_train
    x1,x2=np.meshgrid(np.arange(start=xset[:,0].min()-1,stop=xset[:,0].max()+1,step=0.01),
                      np.arange(start=xset[:,1].min()-1,stop=xset[:,1].max()+1,step=0.01))
    # print(x1,x2)
    print(x1.shape)
    print("Hey i am her of the ")
    plt.contourf(x1,x2,clasf.predict(np.array([x1.ravel(),x2.ravel()]).T)
                .reshape(x1.shape),
                 alpha=0.75,cmap=ListedColormap(('red','green','blue')))

    plt.xlim(x1.min(),x1.max())
    plt.ylim(x2.min(),x2.max())
    print("Hwy ia am ther you amm ther wrqwe")
    for i,j in enumerate(np.unique(yset)):
        plt.scatter(xset[yset==j,0],xset[yset==j,1],
                    c=ListedColormap(('red','green','blue'))(i),label=j)
    #adding the title las
    plt.title("Plotting of the logistic regression ")
    plt.xlabel("pc1")
    plt.ylabel("pc2")
    plt.legend()
    plt.show()


    pass

#main function las

def main():
    pca()
#calling the main function las

if __name__ =="__main__":
    main()