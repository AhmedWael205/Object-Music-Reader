from commonfunctions import *


def Train(file1='Classifier1.npy',file2='Classifier2.npy'):

    x_train = []
    y_train = []
    shapes1 = ['Note','Others']

    for i in range(len(shapes1)):
        LoadImage('OurDataset/'+shapes1[i]+'/*.png',i,x_train = x_train,y_train=y_train)

    x_train1 = np.asarray(x_train)
    y_train1 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train1.shape)
    print ("The size of the labels of the training set is: ", y_train1.shape)


    training_features1 = np.zeros((x_train1.shape[0],7))
    for i in range(training_features1.shape[0]):
        training_features1[i] = extract_features(x_train1[i])

    knn1 = KNeighborsClassifier(n_neighbors=3).fit(training_features1, y_train1)


    ##############################################################################################


    x_train = []
    y_train = []
    shapes6 = ['Single','Double']

    for i in range(len(shapes6)):
        LoadImage('OurDataset/'+shapes6[i]+'/*.png',i,x_train = x_train,y_train=y_train)

    x_train6 = np.asarray(x_train)
    y_train6 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train6.shape)
    print ("The size of the labels of the training set is: ", y_train6.shape)


    training_features6 = np.zeros((x_train6.shape[0],7))
    for i in range(training_features6.shape[0]):
        training_features6[i] = extract_features(x_train6[i])

    knn6 = KNeighborsClassifier(n_neighbors=3).fit(training_features6, y_train6)


    ##############################################################################################

    x_train = []
    y_train = []
    shapes2 = [ 'Basic','Secondary']

    for i in range(len(shapes2)):
        LoadImage('OurDataset/AllOther/'+shapes2[i]+'/*.png',i,removeLines=False,width=200,height=200,x_train = x_train,y_train=y_train)

    x_train2 = np.asarray(x_train)
    y_train2 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train2.shape)
    print ("The size of the labels of the training set is: ", y_train2.shape)

    training_features2 = np.zeros((x_train2.shape[0],7))
    for i in range(training_features2.shape[0]):
        training_features2[i] = extract_features(x_train2[i])

    knn2 = KNeighborsClassifier(n_neighbors=3).fit(training_features2, y_train2) 


    ##############################################################################################


    x_train = []
    y_train = []
    shapes3 = ['Flagged','UnFlagged']

    for i in range(len(shapes3)):
        LoadImage('OurDataset/AllSingle/'+shapes3[i]+'/*.png',y=i,removeLines=False,x_train = x_train,y_train=y_train)

    x_train3 = np.asarray(x_train)
    y_train3 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train3.shape)
    print ("The size of the labels of the training set is: ", y_train3.shape)


    training_features3 = np.zeros((x_train3.shape[0],7))
    for i in range(training_features3.shape[0]):
        training_features3[i] = extract_features(x_train3[i])

    knn3 = KNeighborsClassifier(n_neighbors=3).fit(training_features3, y_train3) 

    ##############################################################################################


    x_train = []
    y_train = []
    shapes7 = ['Saturated','UnSaturated']

    for i in range(len(shapes7)):
        LoadImage('OurDataset/AllSingle/'+shapes7[i]+'/*.png',y=i,removeLines=False,x_train = x_train,y_train=y_train)

    x_train7 = np.asarray(x_train)
    y_train7 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train7.shape)
    print ("The size of the labels of the training set is: ", y_train7.shape)


    training_features7 = np.zeros((x_train7.shape[0],7))
    for i in range(training_features7.shape[0]):
        training_features7[i] = extract_features(x_train7[i])

    knn7 = KNeighborsClassifier(n_neighbors=3).fit(training_features7, y_train7) 

    ##############################################################################################

    x_train = []
    y_train = []
    shapes4 = ['Hollow','Whole']

    for i in range(len(shapes4)):
        LoadImage('OurDataset/AllSingle/'+shapes4[i]+'/*.png',y=i,removeLines=False,x_train = x_train,y_train=y_train)

    x_train4 = np.asarray(x_train)
    y_train4 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train4.shape)
    print ("The size of the labels of the training set is: ", y_train4.shape)

    training_features4 = np.zeros((x_train4.shape[0],7))
    for i in range(training_features4.shape[0]):
        training_features4[i] = extract_features(x_train4[i])

    knn4 = KNeighborsClassifier(n_neighbors=3).fit(training_features4, y_train4) 

    ##############################################################################################
    x_train = []
    y_train = []
    shapes8 = ['Chord','Filled']

    for i in range(len(shapes8)):
        LoadImage('OurDataset/AllSingle/'+shapes8[i]+'/*.png',y=i,removeLines=False,x_train = x_train,y_train=y_train)

    x_train8 = np.asarray(x_train)
    y_train8 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train8.shape)
    print ("The size of the labels of the training set is: ", y_train8.shape)

    training_features8 = np.zeros((x_train8.shape[0],7))
    for i in range(training_features8.shape[0]):
        training_features8[i] = extract_features(x_train8[i])

    knn8 = KNeighborsClassifier(n_neighbors=3).fit(training_features8, y_train8) 

    ##############################################################################################


    x_train = []
    y_train = []
    shapes5 = ['2FLAGS','3FLAGS']

    for i in range(len(shapes5)):
        LoadImage('OurDataset/AllSingle/'+shapes5[i]+'/*.png',i,x_train = x_train,y_train=y_train)

    x_train5 = np.asarray(x_train)
    y_train5 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train5.shape)
    print ("The size of the labels of the training set is: ", y_train5.shape)

    training_features5 = np.zeros((x_train5.shape[0],7))
    for i in range(training_features5.shape[0]):
        training_features5[i] = extract_features(x_train5[i])

    knn5 = KNeighborsClassifier(n_neighbors=3).fit(training_features5, y_train5)


    ##############################################################################################

    x_train = []
    y_train = []
    shapes9 = [ 'Must','Optional']

    for i in range(len(shapes9)):
        LoadImage('OurDataset/AllOther/'+shapes9[i]+'/*.png',i,removeLines=False,width=200,height=200,x_train = x_train,y_train=y_train)

    x_train9 = np.asarray(x_train)
    y_train9 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train9.shape)
    print ("The size of the labels of the training set is: ", y_train9.shape)

    training_features9 = np.zeros((x_train9.shape[0],7))
    for i in range(training_features9.shape[0]):
        training_features9[i] = extract_features(x_train9[i])

    knn9 = KNeighborsClassifier(n_neighbors=3).fit(training_features9, y_train9) 

    ##############################################################################################

    x_train = []
    y_train = []
    shapes10 = [ 'KeyS','End']

    for i in range(len(shapes10)):
        LoadImage('OurDataset/AllOther/'+shapes10[i]+'/*.png',i,removeLines=False,width=200,height=200,x_train = x_train,y_train=y_train)

    x_train10 = np.asarray(x_train)
    y_train10 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train10.shape)
    print ("The size of the labels of the training set is: ", y_train10.shape)

    training_features10 = np.zeros((x_train10.shape[0],7))
    for i in range(training_features10.shape[0]):
        training_features10[i] = extract_features(x_train10[i])

    knn10 = KNeighborsClassifier(n_neighbors=3).fit(training_features10, y_train10) 

    ##############################################################################################

    x_train = []
    y_train = []
    shapes11 = [ 'Sharp','Natural']

    for i in range(len(shapes11)):
        LoadImage('OurDataset/AllOther/'+shapes11[i]+'/*.png',i,removeLines=False,width=200,height=200,x_train = x_train,y_train=y_train)

    x_train11 = np.asarray(x_train)
    y_train11 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train11.shape)
    print ("The size of the labels of the training set is: ", y_train11.shape)

    training_features11 = np.zeros((x_train11.shape[0],7))
    for i in range(training_features11.shape[0]):
        training_features11[i] = extract_features(x_train11[i])

    knn11 = KNeighborsClassifier(n_neighbors=3).fit(training_features11, y_train11) 

    ##############################################################################################


    x_train = []
    y_train = []
    shapes12 = [ 'Large','Small']

    for i in range(len(shapes12)):
        LoadImage('OurDataset/AllOther/'+shapes12[i]+'/*.png',i,removeLines=False,width=200,height=200,x_train = x_train,y_train=y_train)

    x_train12 = np.asarray(x_train)
    y_train12 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train12.shape)
    print ("The size of the labels of the training set is: ", y_train12.shape)

    training_features12 = np.zeros((x_train12.shape[0],7))
    for i in range(training_features12.shape[0]):
        training_features12[i] = extract_features(x_train12[i])

    knn12 = KNeighborsClassifier(n_neighbors=3).fit(training_features12, y_train12) 

    ##############################################################################################


    x_train = []
    y_train = []
    shapes13 = [ 'Numbers','Ands']

    for i in range(len(shapes13)):
        LoadImage('OurDataset/AllOther/'+shapes13[i]+'/*.png',i,removeLines=False,width=200,height=200,x_train = x_train,y_train=y_train)

    x_train13 = np.asarray(x_train)
    y_train13 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train13.shape)
    print ("The size of the labels of the training set is: ", y_train13.shape)

    training_features13 = np.zeros((x_train13.shape[0],7))
    for i in range(training_features13.shape[0]):
        training_features13[i] = extract_features(x_train13[i])

    knn13 = KNeighborsClassifier(n_neighbors=3).fit(training_features13, y_train13) 

    ##############################################################################################

    x_train = []
    y_train = []
    shapes14 = [ 'And','DAnd']

    for i in range(len(shapes14)):
        LoadImage('OurDataset/AllOther/'+shapes14[i]+'/*.png',i,removeLines=False,width=200,height=200,x_train = x_train,y_train=y_train)

    x_train14 = np.asarray(x_train)
    y_train14 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train14.shape)
    print ("The size of the labels of the training set is: ", y_train14.shape)

    training_features14 = np.zeros((x_train14.shape[0],7))
    for i in range(training_features14.shape[0]):
        training_features14[i] = extract_features(x_train14[i])

    knn14 = KNeighborsClassifier(n_neighbors=3).fit(training_features14, y_train14) 

    ##############################################################################################

    x_train = []
    y_train = []
    shapes15 = [ '1Number','2Numbers']

    for i in range(len(shapes15)):
        LoadImage('OurDataset/AllOther/'+shapes15[i]+'/*.png',i,removeLines=False,width=200,height=200,x_train = x_train,y_train=y_train)

    x_train15 = np.asarray(x_train)
    y_train15 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train14.shape)
    print ("The size of the labels of the training set is: ", y_train14.shape)

    training_features15 = np.zeros((x_train15.shape[0],7))
    for i in range(training_features15.shape[0]):
        training_features15[i] = extract_features(x_train15[i])

    knn15 = KNeighborsClassifier(n_neighbors=3).fit(training_features15, y_train15) 

    ##############################################################################################

    x_train = []
    y_train = []
    shapes16 = [ 'SameNumber','DifferentNumber']

    for i in range(len(shapes16)):
        LoadImage('OurDataset/AllOther/'+shapes16[i]+'/*.png',i,removeLines=False,width=200,height=200,x_train = x_train,y_train=y_train)

    x_train16 = np.asarray(x_train)
    y_train16 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train16.shape)
    print ("The size of the labels of the training set is: ", y_train16.shape)

    training_features16 = np.zeros((x_train16.shape[0],7))
    for i in range(training_features16.shape[0]):
        training_features16[i] = extract_features(x_train16[i])

    knn16 = KNeighborsClassifier(n_neighbors=3).fit(training_features16, y_train16) 

    ##############################################################################################
    x_train = []
    y_train = []
    shapes17 = [ '22','44']

    for i in range(len(shapes17)):
        LoadImage('OurDataset/AllOther/'+shapes17[i]+'/*.png',i,removeLines=False,width=200,height=200,x_train = x_train,y_train=y_train)

    x_train17 = np.asarray(x_train)
    y_train17 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train14.shape)
    print ("The size of the labels of the training set is: ", y_train14.shape)

    training_features17 = np.zeros((x_train17.shape[0],7))
    for i in range(training_features17.shape[0]):
        training_features17[i] = extract_features(x_train17[i])

    knn17 = KNeighborsClassifier(n_neighbors=3).fit(training_features17, y_train17) 

    ##############################################################################################
    x_train = []
    y_train = []
    shapes18 = [ '24','42']

    for i in range(len(shapes18)):
        LoadImage('OurDataset/AllOther/'+shapes18[i]+'/*.png',i,removeLines=False,width=200,height=200,x_train = x_train,y_train=y_train)

    x_train18 = np.asarray(x_train)
    y_train18 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train18.shape)
    print ("The size of the labels of the training set is: ", y_train18.shape)

    training_features18 = np.zeros((x_train18.shape[0],7))
    for i in range(training_features18.shape[0]):
        training_features18[i] = extract_features(x_train18[i])

    knn18 = KNeighborsClassifier(n_neighbors=3).fit(training_features18, y_train18) 

    ##############################################################################################
    x_train = []
    y_train = []
    shapes19 = [ '2','4']

    for i in range(len(shapes19)):
        LoadImage('OurDataset/AllOther/'+shapes19[i]+'/*.png',i,removeLines=False,width=200,height=200,x_train = x_train,y_train=y_train)

    x_train19 = np.asarray(x_train)
    y_train19 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train19.shape)
    print ("The size of the labels of the training set is: ", y_train19.shape)

    training_features19 = np.zeros((x_train19.shape[0],7))
    for i in range(training_features19.shape[0]):
        training_features19[i] = extract_features(x_train19[i])

    knn19 = KNeighborsClassifier(n_neighbors=3).fit(training_features19, y_train19) 

    ##############################################################################################
    x_train = []
    y_train = []
    shapes20 = [ '2Beams','4Beams']

    for i in range(len(shapes20)):
        LoadImage('OurDataset/'+shapes20[i]+'/*.png',i,removeLines=False,width=200,height=200,x_train = x_train,y_train=y_train)

    x_train20 = np.asarray(x_train)
    y_train20 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train20.shape)
    print ("The size of the labels of the training set is: ", y_train20.shape)

    training_features20 = np.zeros((x_train20.shape[0],7))
    for i in range(training_features20.shape[0]):
        training_features20[i] = extract_features(x_train20[i])

    knn20 = KNeighborsClassifier(n_neighbors=3).fit(training_features20, y_train20) 

    ##############################################################################################

    x_train = []
    y_train = []
    shapes21 = [ 'l2','ge2']

    for i in range(len(shapes21)):
        LoadImage('OurDataset/AllSingle/'+shapes21[i]+'/*.png',i,removeLines=False,width=200,height=200,x_train = x_train,y_train=y_train)

    x_train21 = np.asarray(x_train)
    y_train21 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train21.shape)
    print ("The size of the labels of the training set is: ", y_train21.shape)

    training_features21 = np.zeros((x_train21.shape[0],7))
    for i in range(training_features21.shape[0]):
        training_features21[i] = extract_features(x_train21[i])

    knn21 = KNeighborsClassifier(n_neighbors=3).fit(training_features21, y_train21) 

    ##############################################################################################


    x_train = []
    y_train = []
    shapes22 = [ '2Chord','3Chord']

    for i in range(len(shapes21)):
        LoadImage('OurDataset/AllSingle/'+shapes22[i]+'/*.png',i,removeLines=False,width=200,height=200,x_train = x_train,y_train=y_train)

    x_train22 = np.asarray(x_train)
    y_train22 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train22.shape)
    print ("The size of the labels of the training set is: ", y_train22.shape)

    training_features22 = np.zeros((x_train22.shape[0],7))
    for i in range(training_features22.shape[0]):
        training_features22[i] = extract_features(x_train22[i])

    knn22 = KNeighborsClassifier(n_neighbors=3).fit(training_features22, y_train22) 

    ##############################################################################################


    x_train = []
    y_train = []
    shapes23 = [ '1D','2D']

    for i in range(len(shapes23)):
        LoadImage('OurDataset/AllSingle/'+shapes23[i]+'/*.png',i,removeLines=False,width=200,height=200,x_train = x_train,y_train=y_train)

    x_train23 = np.asarray(x_train)
    y_train23 = np.asarray(y_train)

    print ("The size of the training set is: ", x_train23.shape)
    print ("The size of the labels of the training set is: ", y_train23.shape)

    training_features23 = np.zeros((x_train23.shape[0],7))
    for i in range(training_features23.shape[0]):
        training_features23[i] = extract_features(x_train23[i])

    knn23 = KNeighborsClassifier(n_neighbors=3).fit(training_features23, y_train23) 

    ##############################################################################################


    x = []
    x.append(training_features1)
    x.append(training_features2)
    x.append(training_features3)
    x.append(training_features4)
    x.append(training_features5)
    x.append(training_features6)
    x.append(training_features7)
    x.append(training_features8)
    x.append(training_features9)
    x.append(training_features10)
    x.append(training_features11)
    x.append(training_features12)
    x.append(training_features13)
    x.append(training_features14)
    x.append(training_features15)
    x.append(training_features16)
    x.append(training_features17)
    x.append(training_features18)
    x.append(training_features19)
    x.append(training_features20)
    x.append(training_features21)
    x.append(training_features22)
    x.append(training_features23)

    x = np.asarray(x)

    print("Data1 save to file:"+file1)
    np.save(file1, x)


    y = []
    y.append(y_train1)
    y.append(y_train2)
    y.append(y_train3)
    y.append(y_train4)
    y.append(y_train5)
    y.append(y_train6)
    y.append(y_train7)
    y.append(y_train8)
    y.append(y_train9)
    y.append(y_train10)
    y.append(y_train11)
    y.append(y_train12)
    y.append(y_train13)
    y.append(y_train14)
    y.append(y_train15)
    y.append(y_train16)
    y.append(y_train17)
    y.append(y_train18)
    y.append(y_train19)
    y.append(y_train20)
    y.append(y_train21)
    y.append(y_train22)
    y.append(y_train23)

    y = np.asarray(y)

    print("Data2 save to file:"+file2)
    np.save(file2, y)

