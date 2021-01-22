
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv
from skimage.util.shape import view_as_windows

# Convolution:
from scipy.signal import convolve2d
import glob
from mpl_toolkits.mplot3d import Axes3D
import scipy.misc
from scipy import fftpack
from scipy import ndimage
import math

from skimage.transform import hough_line, hough_line_peaks,rotate,resize,rescale
from skimage.util import random_noise
from skimage.filters import median,sobel,apply_hysteresis_threshold,threshold_otsu,threshold_local,threshold_sauvola
from skimage.morphology import rectangle,disk,square,binary_erosion,binary_dilation,binary_closing,binary_opening,skeletonize,convex_hull_image
from skimage.feature import canny
from skimage.measure import label
from skimage.color import label2rgb

#Segmentation
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
from skimage.color import label2rgb


# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

# Show the figures / plots inside the notebook
def show_images(images,titles=None,saveImage=False,name="untitled",axis=False):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    if not axis:
        plt.axis("off")
    plt.show()
    if saveImage:
        show_images.counter +=1
        path = "temp/"
        x = path + name + "_" + str(show_images.counter) + ".png"
        fig.savefig(x, bbox_inches='tight',dpi=240)
show_images.counter = 0


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')


def feng_threshold(img, w_size1=15, w_size2=30, k1=0.15, k2=0.01, alpha1=0.1):
    """ Runs the Feng's thresholding algorithm.
    Reference:
    Algorithm proposed in: Meng-Ling Feng and Yap-Peng Tan, “Contrast adaptive
    thresholding of low quality document images”, IEICE Electron. Express,
    Vol. 1, No. 16, pp.501-506, (2004).
    Modifications: Using integral images to compute the local mean and the
    standard deviation
    @param img: The input image. Must be a gray scale image
    @type img: ndarray
    @param w_size1: The size of the primary local window to compute
        each pixel threshold. Should be an odd window
    @type w_size1: int
    @param w_size2: The size of the secondary local window to compute
        the dynamic range standard deviation. Should be an odd window
    @type w_size2: int
    @param k1: Parameter value that lies in the interval [0.15, 0.25].
    @type k1: float
    @param k2: Parameter value that lies in the interval [0.01, 0.05].
    @type k2: float
    @param alpha1: Parameter value that lies in the interval [0.15, 0.25].
    @type alpha1: float
    @return: The estimated local threshold for each pixel
    @rtype: ndarray
    """
    # Obtaining rows and cols
    rows, cols = img.shape
    i_rows, i_cols = rows + 1, cols + 1

    # Computing integral images
    # Leaving first row and column in zero for convenience
    integ = np.zeros((i_rows, i_cols), np.float)
    sqr_integral = np.zeros((i_rows, i_cols), np.float)

    integ[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.float), axis=0), axis=1)
    sqr_img = np.square(img.astype(np.float))
    sqr_integral[1:, 1:] = np.cumsum(np.cumsum(sqr_img, axis=0), axis=1)

    # Defining grid
    x, y = np.meshgrid(np.arange(1, i_cols), np.arange(1, i_rows))

    # Obtaining local coordinates
    hw_size = w_size1 // 2
    x1 = (x - hw_size).clip(1, cols)
    x2 = (x + hw_size).clip(1, cols)
    y1 = (y - hw_size).clip(1, rows)
    y2 = (y + hw_size).clip(1, rows)

    # Obtaining local areas size
    l_size = (y2 - y1 + 1) * (x2 - x1 + 1)

    # Computing sums
    sums = (integ[y2, x2] - integ[y2, x1 - 1] -
            integ[y1 - 1, x2] + integ[y1 - 1, x1 - 1])
    sqr_sums = (sqr_integral[y2, x2] - sqr_integral[y2, x1 - 1] -
                sqr_integral[y1 - 1, x2] + sqr_integral[y1 - 1, x1 - 1])

    # Computing local means
    means = sums / l_size

    # Computing local standard deviation
    stds = np.sqrt(sqr_sums / l_size - np.square(means))

    # Obtaining windows
    padded_img = np.ones((rows + w_size1 - 1, cols + w_size1 - 1)) * np.nan
    padded_img[hw_size: -hw_size, hw_size: -hw_size] = img

    winds = view_as_windows(padded_img, (w_size1, w_size1))

    # Obtaining maximums and minimums
    mins = np.nanmin(winds, axis=(2, 3))

    # Obtaining local coordinates for std range calculations
    hw_size = w_size2 // 2
    x1 = (x - hw_size).clip(1, cols)
    x2 = (x + hw_size).clip(1, cols)
    y1 = (y - hw_size).clip(1, rows)
    y2 = (y + hw_size).clip(1, rows)

    # Obtaining local areas size
    l_size = (y2 - y1 + 2) * (x2 - x1 + 2)

    # Computing sums
    sums = (integ[y2, x2] - integ[y2, x1 - 1] -
            integ[y1 - 1, x2] + integ[y1 - 1, x1 - 1])
    sqr_sums = (sqr_integral[y2, x2] - sqr_integral[y2, x1 - 1] -
                sqr_integral[y1 - 1, x2] + sqr_integral[y1 - 1, x1 - 1])

    # Computing local means2
    means2 = sums / l_size

    # Computing standard deviation range
    std_ranges = np.sqrt(sqr_sums / l_size - np.square(means2))

    # Computing normalized standard deviations and extra alpha parameters
    n_stds = stds / std_ranges
    n_sqr_std = np.square(n_stds)
    alpha2 = k1 * n_sqr_std
    alpha3 = k2 * n_sqr_std

    thresholds = ((1 - alpha1) * means + alpha2 * n_stds
                    * (means - mins) + alpha3 * mins)

    return thresholds

    
def getPoints(points):
    x = (points[points[:,0].argsort()])

    if x[0,1] > x[1,1]:
        x[[0, 1]] = x[[1, 0]]
    if x[2,1] > x[3,1]:
        x[[2, 3]] = x[[3, 2]]

    expand = 30
        
    if x[0,0] - expand > 0:
        x[0,0] -= expand
    else:
        x[0,0] = 0
        
    if x[0,1] - expand > 0:
        x[0,1] -= expand
    else:
        x[0,1] = 0

    if x[1,0] - expand > 0:
        x[1,0] -= expand
    else:
        x[1,0] = 0

    if x[1,1] + expand < 9*128:
        x[1,1] += expand
    else:
        x[1,1] = 9*128

    if x[2,0] + expand < 16*128:
        x[2,0] += expand
    else:
        x[2,0] = 16*128

    if x[2,1] - expand> 0:
        x[2,1] -= expand
    else:
        x[2,1] = 0

    if x[3,1] + expand < 16*128:
        x[3,1] += expand
    else:
        x[3,1] = 16*128

    if x[3,1] + expand < 9*128:
        x[3,1] += expand
    else:
        x[3,1] = 9*128
        
    return x


################################################################################################################################

# from commonfunctions import *

## TODO :: Add to commonfunctions.py when approved

from sklearn.neighbors import KNeighborsClassifier
import mahotas 
# import numpy
def BDratio(img):
    blackPixels=len(img[img==255])
    whitePixels=len(img[img==0])
    ratio=blackPixels/whitePixels
    #print(ratio)
    return ratio
def WHratio(img):
    width=sum((img < 255).any(axis=0))
    height=sum((img < 255).any(axis=1))
    ratio=height/width
    return ratio

    
def zernikeMoments(img):
    value =[]
    if len(img.shape) > 2:
        value=mahotas.features.zernike_moments(img.max(2), 20) 
    else:
        value=mahotas.features.zernike_moments(img, 20)
    return value

def calculateDistance(x1, x2):
    distance = np.linalg.norm(x1-x2)
    return distance

def KNN(test_point, training_features, k,y_train):
    x = np.argsort([calculateDistance(training_features[i,:],test_point) for i in range(training_features.shape[0])])
    unique_elements, counts_elements = np.unique(y_train[x[0:k]], return_counts=True)
    classification = unique_elements[np.argmax(counts_elements)]
    return classification


def extract_features(img,mode="Hu"):
    if mode != "Hu":
        BDratio = BDratio(img)
        WHratio = WHratio(img)
    Hu=fd_hu_moments(img)
    Hu=np.asarray(Hu)
    
    features=[]
    if mode != "Hu":
        features.append(BDratio)
        features.append(WHratio)
    features.extend(Hu)
    
    return features

def fd_hu_moments(image):
    if len(image.shape)==2:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    image=np.max(image)-image
    if len(image.shape)==3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #show_images([image])
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature



def LoadImage(path,y=0,width=200,height=200,many=True,test=False,removeLines=False,saveImage=False,x_train = [],y_train=[]):
    if many:
        x = sorted(glob.glob(path))
    else:
        x = [path]
        
    for filename in x:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
        ksize = (3, 3) 
        blur = cv2.blur(gray, ksize) 
        result=[]
        if removeLines:
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
            detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            for c in cnts:
                cv2.drawContours(img, [c], -1, (255,255,255), 2)

            # Repair image
            repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,7))
            result = 255 - cv2.morphologyEx(255 - img, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
        else:
            result = cv2.threshold(blur,100 , 255, cv2.THRESH_OTSU)[1]
            
        resized = cv2.resize(result, (width,height), interpolation = cv2.INTER_AREA)
#         show_images([resized])
        if not test:
            x_train.append(resized)
            y_train.append(y)
        if saveImage:
            show_images([resized],[""],saveImage=saveImage)
#         show_images([resized])
    return resized



def preprocessing(Image,y=0,width=200,height=200,removeLines=False,saveImage=False):
    
    img = np.copy((Image*255).astype(np.uint8))
    if removeLines:
        gray = np.copy(img)
        ksize = (3, 3) 
        blur = cv2.blur(gray, ksize) 
        thresh = cv2.threshold(blur.astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(img, [c], -1, (255,255,255), 2)

        # Repair image
        repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,19))

        result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    else:
        # result = img
        gray = np.copy(img)
        ksize = (3, 3) 
        blur = cv2.blur(gray, ksize) 
        result = cv2.threshold(blur,0 , 255, cv2.THRESH_OTSU)[1]
    resized = cv2.resize(result, (width,height), interpolation = cv2.INTER_AREA)

    if saveImage:
            show_images([resized],[""],saveImage=saveImage)
    return resized