from PreProcessing import *
from Segmentation import *
from Classifier import *
from TrainClassifier import *
import argparse
import os
import datetime
# Initialize parser
parser = argparse.ArgumentParser()

parser.add_argument("inputfolder", help = "Input File")
parser.add_argument("outputfolder", help = "Output File")

args = parser.parse_args()

np_load_old1 = np.load

# modify the default parameters of np.load
try:
    np.load = lambda *a,**k: np_load_old1(*a, allow_pickle=True, **k)
except:
    pass


#  To Train First Uncomment The Following Line
# Train()

# InputFolder = "PublicTestCases/Input"
# OutputFolder = "PublicTestCases/Output1"
InputFolder = args.inputfolder
OutputFolder = args.outputfolder



for filename in sorted(glob.glob(InputFolder+"/*")):
    try:
        outfilename = OutputFolder+filename.rsplit('.', 1)[0].replace(InputFolder,"")+".txt"

        image = io.imread(filename)

        x = PreProcessings(image)

        SegmentedNotes,NotesPerOctave,locations,Staffs,StaffThickness,StaffHeight = Segmentation(x)

        Classifier(SegmentedNotes,NotesPerOctave,locations,Staffs,StaffThickness,StaffHeight,out = outfilename,k=5)
    except:
        f = open(outfilename, "w")
        f.write("[]")
        f.close()