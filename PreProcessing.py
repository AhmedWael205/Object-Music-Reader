from commonfunctions import *

def PreProcessings(image,debug=False,saveImage=False,sample="Sample"):


    if debug:
        print("image shape: ",image.shape)
        show_images([image],["in"])
    
    step0_out = [image]
    step1_in = step0_out[0]

        
    image_gray = (rgb2gray(image)*255).astype("int")
    
    if debug:
        show_images([image_gray],["Orignal Image"])

    step1_out = [image_gray]
    step2_in = step1_out[0]

    thresh = threshold_otsu(step2_in)


    thresh_feng = feng_threshold(image_gray)
    
    if np.isnan(np.sum(thresh_feng)):
        thresh_feng = np.where(np.isnan(thresh_feng),9999,thresh_feng)
    binary_feng = image_gray>thresh_feng


    binary_ostu = image_gray>thresh
    binary= binary_ostu * binary_feng
    if debug:
        show_images([binary],[sample + "Thresholding"],saveImage=saveImage)

    high_thresh, thresh_im = cv2.threshold((step2_in).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    low_thresh = 0.5*high_thresh
    if debug:
        print("high_thresh: ",high_thresh)
        print("low_thresh: ",low_thresh)

    step2_1_out = [binary]
    step2_2_in = step2_1_out[0]

    image_binary_erosion = binary_erosion(step2_2_in)
    
    if debug:
        show_images([image_binary_erosion],[sample + "erosion"],saveImage=saveImage)

    step2_out = [binary,image_binary_erosion]
    step3_in = step2_out[1]

    image_edges = cv2.Canny((step2_in).astype(np.uint8), low_thresh, high_thresh, apertureSize=3)

    if debug:
        show_images([image_edges],[sample + " Image Edges"])


    step4_out = [image_edges]
    step5_in = step4_out[0]

    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(step5_in, theta=tested_angles)

    angles = []
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        angles.append(angle)

    rotation_angle = (np.median(angle) * 180)/np.pi
    if debug:
        print("rotation_angle: ",rotation_angle)

    image_rotate = step3_in
    if abs(rotation_angle) < 85:
        image_rotate = rotate(step3_in,90+rotation_angle,resize=True,clip=True,cval=1)
        
    if debug:
        show_images([image_rotate],["Rotated Image"])


    step5_1_out = [image_rotate]
    step5_2_in = step5_1_out[0]

    image_resized = resize(step5_2_in, (9*128, 16*128), anti_aliasing=True)
    if debug:
        show_images([image_resized],[sample])


    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats((255-image_resized*255).astype(np.uint8), connectivity=8)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1
    min_size = 20000  
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 1
            
    if debug:
        show_images([1-img2])            
    
    hull1 = convex_hull_image(img2)
    if debug:
        show_images([hull1])
    contours,hierarchy = cv2.findContours((hull1).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    points = []
    for cnt in contours : 
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 
        n = approx.ravel()  
        i = 0
        for j in n : 
            if(i % 2 == 0): 
                points.append([n[i] ,n[i + 1]])
            i = i + 1


    points = np.asarray(points)
    if points.shape[0] != 4:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        points = np.int0(box)

    pts1 = np.float32(getPoints(points))
    pts2 = np.float32([[0, 0], [0, 9*128], [16*128, 0], [16*128, 9*128]]) 


    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    result = cv2.warpPerspective(image_resized, matrix, (16*128, 9*128))
    if debug:
        show_images([result],["Prespective"],saveImage=saveImage)

    outputs = [image_resized,result]
    Wael_out = outputs[1]
    
    if debug:
        show_images([Wael_out],["out"],name="out",saveImage=saveImage)
    
    return Wael_out
