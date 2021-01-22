from commonfunctions import *


def Segmentation(image, debug=False, saveImage=False):
    img = np.copy(image)

    # Remove Staff Brace
    coor = np.where(img == 0)
    img2 = (np.round(np.copy(img[:, coor[0][0] + 100:img.shape[1]])) * 255).astype(np.uint8)

    if debug:
        print("Cropped Image")
        show_images([img2])

    label_image, num = label(img2, background=255, return_num=True, connectivity=1)
    if debug:
        print("Label Image")
        show_images([label_image])
    # print(num)

    # image_label_overlay = label2rgb(label_image, image=img1, bg_label=0)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image_label_overlay)

    SegmentedImgs = []

    for region in regionprops(label_image):
        # take regions with large enough areas
        minr, minc, maxr, maxc = region.bbox
        width = maxc - minc
        height = maxr - minr
        if height >= 50 and width >= img.shape[0] / 2:
            # draw rectangle around segmented coins
            #         rect = mpatches.Rectangle((minc, minr), width, height,
            #                                   fill=False, edgecolor='red', linewidth=2)
            #         ax.add_patch(rect)
            #         print(rect)
            # Append Segment to list of segmented images
            SegmentedImgs.append(img[minr:minr + height, minc: img.shape[1]])

    ResizedImgs = []
    for imgs in SegmentedImgs:
        img1 = resize(imgs, (9 * 128, 16 * 128), anti_aliasing=True)
        #         src = np.copy(imgs)
        #         #percent by which the image is resized
        #         scale_percent = 150

        #         #calculate the 50 percent of original dimensions
        #         width = int(src.shape[1] * scale_percent / 100)
        #         height = int(src.shape[0] * scale_percent / 100)

        #         # dsize
        #         dsize = (width, height)

        #         # resize image
        #         outputR = cv2.resize(src, dsize)
        ResizedImgs.append(img1)
    if debug:
        print("Resized Images")
        show_images(ResizedImgs, saveImage=saveImage)

    StaffThicknesses = []
    for imgr in ResizedImgs:
        BlackRuns = []
        for i in range(imgr.shape[1]):
            c = 0
            col = imgr[:, i]
            for row in col:
                if (row == 0):
                    c += 1
                else:
                    if (c != 0):
                        BlackRuns.append(c)
                    c = 0
        BlackRuns = np.asarray(BlackRuns)
        StaffThicknesses.append(np.bincount(BlackRuns).argmax())

    masks = []
    Vmasks = []
    index = 0
    for img in ResizedImgs:
        er = np.ones((StaffThicknesses[index] + 15, 1))
        img = cv2.bitwise_not(img)
        th2 = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

        horizontal = th2
        vertical = th2
        rows, cols = horizontal.shape

        # inverse the image, so that lines are black for masking
        horizontal_inv = cv2.bitwise_not(horizontal)
        # perform bitwise_and to mask the lines with provided mask
        masked_img = cv2.bitwise_and(img, img, mask=horizontal_inv)
        # reverse the image back to normal
        masked_img_inv = cv2.bitwise_not(masked_img)

        horizontalsize = int(cols / 30)
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
        horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
        horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))

        verticalsize = int(rows / 30)
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
        vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))

        Vmasks.append(vertical)

        # step1
        edges = cv2.adaptiveThreshold(horizontal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)

        # step2
        kernel = np.ones((2, 2), dtype="uint8")
        dilated = cv2.dilate(edges, kernel)

        # step3
        smooth = horizontal.copy()

        # step 4
        smooth = cv2.blur(smooth, (4, 4))

        # step 5
        (rows, cols) = np.where(img == 0)
        horizontal[rows, cols] = smooth[rows, cols]

        binhorizontal = horizontal / 255
        invBin = 1 - binhorizontal
        openHor = binary_erosion(invBin, er)
        masks.append(openHor)
        index += 1

    if debug:
        print("Horiztonal Masks")
        show_images(masks, saveImage=saveImage)
        print("Vertical Masks")
        show_images(Vmasks, saveImage=saveImage)

    maskedImgs = []
    VmaskedImgs = []
    i = 0
    for mask in masks:
        img = np.copy(ResizedImgs[i])
        img[mask == 0] = 1
        maskedImgs.append(img)

        Vimg = np.copy(ResizedImgs[i])
        Vimg = (Vimg) * (Vmasks[i])
        VmaskedImgs.append(1 - Vimg)
        i += 1

    if debug:
        print("Horizontal Masked Images")
        show_images(maskedImgs, saveImage=saveImage)
        print("Vertical Masked Images")
        show_images(VmaskedImgs, saveImage=saveImage)

    SegmentedNotes = []
    SegmentedNotesCenter = []
    NotesPerOctave = []
    D = []
    imgNum = 0
    for maskImg in maskedImgs:
        Accumlator = 0
        if debug:
            print("Image Num ", imgNum)

        #     show_images([ResizedImgs[2]],[sample + " Image Edges"])

        image_edges = cv2.Canny((np.round(ResizedImgs[imgNum])).astype(np.uint8), 0, 100 / 255, apertureSize=3)

        #     show_images([image_edges],[sample + " Image Edges"])

        step4_out = [image_edges]
        step5_in = step4_out[0]

        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
        h, theta, d = hough_line(step5_in, theta=tested_angles)

        angles = []
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            angles.append(angle)

        rotation_angle = (np.median(angles) * 180) / np.pi
        if debug:
            print("rotation_angle: ", rotation_angle)

        angle = np.abs(rotation_angle)
        if (angle >= 85 and angle <= 95) or (angle >= 0 and angle <= 2) or 0:
            Notes = []
            if debug:
                print("Perfectly Horizontal Staff Lines")
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
                (255 - maskImg * 255).astype(np.uint8), connectivity=8)
            sizes = stats[1:, -1];
            nb_components = nb_components - 1
            min_size = 4000

            for i in range(0, nb_components):
                img = np.zeros((output.shape))
                if sizes[i] >= min_size:
                    #                 print(sizes[i])
                    img[output == i + 1] = 1
                    Notes.append(img)
            #                 show_images([1-img])

            # Sort Notes
            StartList = []
            EndList = []
            SortedNotes = []
            for Note in Notes:
                VerticalMax = np.max(Note, axis=0)
                VerticalDiff = np.diff(VerticalMax, axis=0)
                start = np.argwhere(VerticalDiff == 1)
                end = np.argwhere(VerticalDiff == -1) + 1
                if start.size != 0:
                    StartList.append(start[0][0])
                if end.size != 0:
                    EndList.append(end[0][0])

            zipped_lists = zip(EndList, Notes)

            sorted_zipped_lists = sorted(zipped_lists)

            SortedNotes = [element for _, element in sorted_zipped_lists]

            ResizedIndex = 0
            for SN in SortedNotes:
                #             show_images([1-SN])
                img2 = np.copy(maskedImgs[ResizedIndex])
                for s in range(len(StartList)):
                    img = np.copy(1 - SN)
                    out = img[:, StartList[s] - 10:EndList[s] + 10]
                    out2 = img2[:, StartList[s] - 10:EndList[s] + 10]
                    if (out.size != 0 and np.sum(1 - out)):
                        SegmentedNotes.append(out)
                        SegmentedNotesCenter.append(out2)
                        D.append(np.sum(1 - out))
                        Accumlator += 1
            ResizedIndex += 1
        else:
            if debug:
                print("Sekwed Staff Line")
            img = np.copy(maskedImgs[imgNum])
            imgC = np.copy(SegmentedImgs[imgNum])
            vmask = np.copy(Vmasks[imgNum])
            VerticalMax = np.max(vmask, axis=0)
            VerticalDiff = np.diff(VerticalMax, axis=0)
            start = np.argwhere(VerticalDiff == 255)
            end = np.argwhere(VerticalDiff == -255) + 1

            appPos = np.sort(np.append(start, end))
            diff = np.diff(appPos)

            for i in range(0, len(appPos) - 1, 2):
                if (np.sum(vmask[:, appPos[i]:appPos[i + 1]]) > 150000):
                    #                 print(np.sum(vmask[:,appPos[i]:appPos[i+1]]))
                    #                 io.imshow(img[:,appPos[i]-10:appPos[i+1]+100])
                    #                 io.show()
                    SegmentedNotes.append(img[:, appPos[i] - start[0][0]:appPos[i + 1] + start[0][0]])
                    SegmentedNotesCenter.append(imgC[:, appPos[i] - start[0][0]:appPos[i + 1] + start[0][0]])
                    D.append(np.sum(vmask[:, appPos[i]:appPos[i + 1]]))
                    Accumlator += 1
        imgNum += 1
        NotesPerOctave.append(Accumlator)

    Notess = []

    if len(NotesPerOctave) == 1:
        for Note in SegmentedNotes:
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
                (255 - Note * 255).astype(np.uint8), connectivity=8)
            sizes = stats[1:, -1];
            nb_components = nb_components - 1
            min_size = 4000

            for i in range(0, nb_components):
                img = np.zeros((output.shape))
                if sizes[i] >= min_size:
                    img[output == i + 1] = 1
                    Notess.append(np.max(img) - img)

        SegmentedNotes = Notess

    if debug:
        for Note in SegmentedNotes:
            show_images([Note])

    # Staff Line Specs Detection

    # Average Staff Gaps
    StaffHeights = []
    for i in range(len(ResizedImgs)):
        imgr = np.round(ResizedImgs[i])
        WhiteRuns = []
        for i in range(imgr.shape[1]):
            c = 0
            col = imgr[:, i]
            for row in col:
                if (row == 1):
                    c += 1
                else:
                    if (c != 0):
                        WhiteRuns.append(c)
                        c = 0
        WhiteRuns = np.asarray(WhiteRuns)
        StaffHeights.append(np.bincount(WhiteRuns).argmax())

    StaffHeight = np.mean(StaffHeights, dtype='int64')
    if debug:
        print(StaffHeight)

    # Average Staff Thickness
    StaffThickness = np.mean(StaffThicknesses, dtype='int64')
    if debug:
        print(StaffThickness)

    # Staff Lines Beginning
    FirstRows = []
    Staffs = []
    for i in range(len(ResizedImgs)):
        # for i in range(1):
        imgr = 1 - masks[i]
        FirstRow = np.where(imgr == 1, )[0][0]
        Staffs.append([FirstRow + j * (StaffHeights[i] + StaffThicknesses[i]) for j in range(5)])
    if debug:
        print(Staffs)

    # Find Where 3enaba is
    Nearest_Line = []
    Nearest_Distance = []
    locations = []
    SE = np.ones((15, 3))
    ER = np.ones((StaffThickness, 1))
    for i in range(len(SegmentedNotes)):
        # for i in range(1,2):
        cimg = np.copy(SegmentedNotes[i])
        Density = np.sum(1 - cimg)
        n = 1
        if Density > 30000:
            n = 2
        cimg = binary_closing(cimg, SE.T)
        #     cimg = binary_closing(cimg, SE)
        #     cimg = binary_dilation(cimg, SE.T)
        cimg = binary_dilation(cimg, ER)
        if debug:
            show_images([cimg])

        #     HorizontalProj = np.sum(1-cimg, axis=1).argsort()[-n:-(StaffThickness+StaffHeight)*n:-(StaffThickness+StaffHeight)]
        #     MaxRow = HorizontalProj
        #     VerticalProj = np.sum(1-cimg, axis=0).argsort()[-n:-(StaffThickness+StaffHeight)*n:-(StaffThickness+StaffHeight)]
        #     MaxCol = VerticalProj
        #     print(MaxRow, MaxCol)

        gray = (255 - cimg * 255).astype(np.uint8)
        ## threshold
        # th, threshed = cv2.threshold(gray, 100, 255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        if debug:
            show_images([gray])
        ## findcontours
        cnts = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # cv2.circle(img,center,radius,(0,255,0),2)
        # print ('Circle' + str(i) + ': Center =' + str(center) + 'Radius =' + str(radius))
        if debug:
            print('countors:', len(cnts))
        ## filter by area
        s1 = 1000
        s2 = 20000
        Dots = 0
        dots_1 = []
        for cnt in cnts:
            if s1 < cv2.contourArea(cnt) < s2:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                #             print(gray)
                dots_1.append([x, y, radius])
                if StaffHeight // 2 < radius < StaffHeight // 2 + 100:
                    Dots += 1
        locations.append(dots_1)
        if debug:
            print("Dots number: ", Dots)

    return SegmentedNotes, NotesPerOctave, locations, Staffs, StaffThickness, StaffHeight