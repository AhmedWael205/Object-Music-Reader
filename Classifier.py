from commonfunctions import *

def getLetters(locations,Staffs,StaffThickness,StaffHeight,dots=1,j=0):
    try:
        x,y,radius = locations[j]
        n = np.argsort(np.abs(y - np.asarray(Staffs)))
        if n[0] == 4:
            #c or d or e or f
            letter = {0:'f1',1:'e1',2:'d1',3:'c1'}
        elif n[0] == 3:
            # f or g or a
            letter = {0:'a1',1:'g1',2:'f1',3:'f1'}
        elif n[0] == 2:
            # a or b or c2
            letter = {0:'c2',1:'b1',2:'a1',3:'a1'}
        elif n[0] == 1:
            # c2 or d2 or e2
            letter = {0:'e2',1:'d2',2:'c2',3:'c2'}
        else:
            # e2 or f2 or g2 or a2 or b2
            letter = {0:'b2',1:'a2',2:'g2',3:'f2',4:'e2'}
    except:
        return 'e1'

    return letter[getNumber(Staffs[n[0]],y,StaffHeight)]
        
def getNumber(n,y,StaffHeight):
    x = n-y
    if x < 0:
        # Dot is Under the line
        if -1*x > StaffHeight*1.4:
            return 3
        elif -1*x > (StaffHeight/2)*1.4:
            return 2
        else:
            return 1
    else:
        # Dot is above the line
        if x < (StaffHeight/2)*0.8:
            return 0
        else:
            return 1
    
def Classifier(SegmentedNotes,NotesPerOctave,locations,Staffs,StaffThickness,StaffHeight,out = "out.txt",debug=False,k=5):
    i = -1

    features = np.load('Classifier1.npy')
    y = np.load('Classifier2.npy')

    shapes1 = ['Note','Others']
    shapes2 = [ 'Basic','Secondary']
    shapes3 = ['Flagged','UnFlagged']
    shapes4 = ['Hollow','Whole']
    shapes5 = ['2FLAGS','3FLAGS']
    shapes6 = ['Single','Double']
    shapes7 = ['Saturated','UnSaturated']
    shapes8 = ['Chord','Filled']
    shapes9 = [ 'Must','Optional']
    shapes10 = [ 'KeyS','End']
    shapes11 = [ 'Sharp','Natural']
    shapes12 = [ 'Large','Small']
    shapes13 = [ 'Numbers','Ands']
    shapes14 = [ 'And','DAnd']
    shapes15 = [ '1Number','2Numbers']
    shapes16 = [ 'SameNumber','DifferentNumber']
    shapes17 = [ '22','44']
    shapes18 = [ '24','42']
    shapes19 = [ '2','4']
    shapes20 = [ '2Beams','4Beams']
    shapes21 = [ 'l2','ge2']
    shapes22 = [ '2Chord','3Chord']
    shapes23 = [ '1D','2D']


    f = open(out, "w")


    m = 0
    n = 0
    if len(NotesPerOctave) != 1:
        f.write("{\n")

        
    f.write("[")
    if len(NotesPerOctave) != 1:
        f.write(' \\meter<"4/4"> ')

    for Note in SegmentedNotes:
        i = i + 1
        x = preprocessing(Note,removeLines=False,saveImage=False)
        if debug:
            show_images([x])
        test_point_h= extract_features(x)
        
        knn_prediction1 = KNN(test_point_h,features[0],k,y[0])
        if debug:
            print(shapes1[knn_prediction1])
        
        if knn_prediction1 == 0:
        ## Note
        
            knn_prediction6 = KNN(test_point_h,features[5],k,y[5])
            if debug:
                print(shapes6[knn_prediction6])
            
            if knn_prediction6 == 0:
                #Single
                knn_prediction3 = KNN(test_point_h,features[2],k,y[2])
                if debug:
                    print(shapes3[knn_prediction3])
                
                if knn_prediction3 == 1:
                    #UnFlagged
                    knn_prediction7 = KNN(test_point_h,features[6],k,y[6])
                    if debug:
                        print(shapes7[knn_prediction7])
                    
                    if knn_prediction7 == 0:
                        #Saturated
                        knn_prediction8 = KNN(test_point_h,features[7],k,y[7])
                        if debug:
                            print(shapes8[knn_prediction8])
                        if knn_prediction8 == 1:
                            #Filled
                            # place < 100 then Unflipped
                            # else Flipped
                            ttt = {0:'UnFlipped',1:'Flipped'}
                            place = np.argmax(np.sum(1-rgb2gray(x),axis=0)) < 100
                            if debug:
                                print(ttt[place])
                            f.write((getLetters(locations[i],Staffs[n],StaffThickness,StaffHeight)+'/4 '))
                        else:
                            #Chord
                            knn_prediction22 = KNN(test_point_h,features[21],k,y[21])
                            if debug:
                                print(shapes22[knn_prediction22])
                            if knn_prediction22 == 0:
                                #2Chords
                                z = {0:'c1',1:'d1',2:'e1',3:'f1',4:'g1',5:'a1',6:'b1'}
                                m1 = (getLetters(locations[i],Staffs[n],StaffThickness,StaffHeight))
                                ff = list(z.keys())[list(z.values()).index(m1)]
                                m2 = z[(ff+2)%6]     
                                f.write('{'+m1+'/4,'+m2+'/4}')
                            else:
                                #3 Chords
                                knn_prediction23 = KNN(test_point_h,features[22],k,y[22])
                                if debug:
                                    print(shapes23[knn_prediction23])
                                if knn_prediction23 == 0:
                                    #1D
                                    z = {0:'c1',1:'d1',2:'e1',3:'f1',4:'g1',5:'a1',6:'b1'}
                                    m1 = (getLetters(locations[i],Staffs[n],StaffThickness,StaffHeight))
                                    ff = list(z.keys())[list(z.values()).index(m1)]
                                    m2 = z[(ff+2)%6]
                                    m3 = z[(ff+4)%6]
                                    f.write('{'+m1+'/4,'+m2+'/4,'+m3+'/4}')
                                else:
                                    #2D
                                    z = {0:'c1',1:'d1',2:'e1',3:'f1',4:'g1',5:'a1',6:'b1'}
                                    m1 = (getLetters(locations[i],Staffs[n],StaffThickness,StaffHeight))
                                    ff = list(z.keys())[list(z.values()).index(m1)]
                                    m2 = z[(ff+1)%6]
                                    m3 = z[(ff+2)%6]
                                    f.write('{'+m1+'/4,'+m2+'/4,'+m3+'/4}')
                    else:
                        #Unsaturated
                        knn_prediction4 = KNN(test_point_h,features[3],k,y[3])
                        if debug:
                            print(shapes4[knn_prediction4])
                        if knn_prediction4 == 0:
                            #Hollow
                            # place < 100 then Unflipped
                            # else Flipped
                            ttt = {0:'UnFlipped',1:'Flipped'}
                            place = np.argmax(np.sum(1-rgb2gray(x),axis=0)) < 100
                            if debug:
                                print(ttt[place])
                            f.write((getLetters(locations[i],Staffs[n],StaffThickness,StaffHeight)+'/2 '))
                        else:
                            f.write((getLetters(locations[i],Staffs[n],StaffThickness,StaffHeight)+'/1 '))
                            
                    
                else:
                    # Flagged
                    knn_prediction21 = KNN(test_point_h,features[20],k,y[20])
                    if debug:
                        print(shapes21[knn_prediction21])
                    
                    if knn_prediction21 == 1:
                    # 2 or more flagse
                        knn_prediction5 = KNN(test_point_h,features[4],k,y[4])
                        if debug:
                            print(shapes5[knn_prediction5])
                        if knn_prediction5 == 0:
                            #2 flags
                            f.write((getLetters(locations[i],Staffs[n],StaffThickness,StaffHeight)+'/16 '))
                        else:
                            #3 flags
                            f.write((getLetters(locations[i],Staffs[n],StaffThickness,StaffHeight)+'/32 '))
                    else:
                        # 1 Flag
                        f.write((getLetters(locations[i],Staffs[n],StaffThickness,StaffHeight)+'/8 '))
            else:
                # Double
                knn_prediction20 = KNN(test_point_h,features[19],k,y[19])
                if debug:
                    print(shapes20[knn_prediction20])

                if knn_prediction20 == 0:
                    #2Beams
                    for j in range(2):
                        f.write((getLetters(locations[i],Staffs[n],StaffThickness,StaffHeight,j=j)+'/8 '))
                else:
                    #4Beams
                    for j in range(4):
                        f.write((getLetters(locations[i],Staffs[n],StaffThickness,StaffHeight,j=j)+'/16 '))
                
        elif knn_prediction1 == 1:
            # Others
            knn_prediction2 = KNN(test_point_h,features[1],k,y[1])
            if debug:
                print(shapes2[knn_prediction2])
            
            if knn_prediction2 == 0:
                #Basic
                knn_prediction9 = KNN(test_point_h,features[8],k,y[8])
                if debug:
                    print(shapes9[knn_prediction9])
                if knn_prediction9 == 0:
                    #Must
                    knn_prediction10 = KNN(test_point_h,features[9],k,y[9])
                    if debug:
                        print(shapes10[knn_prediction10])
                else:
                    #Optional
                    knn_prediction11 = KNN(test_point_h,features[10],k,y[10])
                    if debug:
                        print(shapes11[knn_prediction11])
            else:
                #Secondary
                knn_prediction12 = KNN(test_point_h,features[11],k,y[11])
                if debug:
                    print(shapes12[knn_prediction12])
                if knn_prediction12 == 0:
                    #Large
                    knn_prediction13 = KNN(test_point_h,features[12],k,y[12])
                    if debug:
                        print(shapes13[knn_prediction13])
                    if knn_prediction13 == 0:
                        #Numbers
                        knn_prediction15 = KNN(test_point_h,features[14],k,y[14])
                        if debug:
                            print(shapes15[knn_prediction15])
                        if knn_prediction15 == 0:
                            #1Number
                            knn_prediction19 = KNN(test_point_h,features[18],k,y[18])
                            if debug:
                                print(shapes19[knn_prediction19])

                        else:
                            #2Numbers
                            knn_prediction16 = KNN(test_point_h,features[15],k,y[15])
                            if debug:
                                print(shapes16[knn_prediction16])
                            
                            if knn_prediction16 == 0:
                                #Same
                                knn_prediction17 = KNN(test_point_h,features[16],k,y[16])
                                if debug:
                                    print(shapes17[knn_prediction17])

                            else:
                                #Different
                                knn_prediction18 = KNN(test_point_h,features[17],k,y[17])
                                if debug:
                                    print(shapes18[knn_prediction18])

                    else:
                        #Ands
                        knn_prediction14 = KNN(test_point_h,features[13],k,y[13])
                        if debug:
                            print(shapes14[knn_prediction14])
                    
        m = m+1
        if len(NotesPerOctave) != 1 and m == NotesPerOctave[n] and n != len(NotesPerOctave)-1:
            n = n + 1
            m = 0
            f.write(']\n[ \\meter<"4/4"> ')
                    
    f.write("]")

    if len(NotesPerOctave) != 1:
        f.write("\n}")

    f.close()
                    