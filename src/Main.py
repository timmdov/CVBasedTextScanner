import cv2
import numpy as np
import utils
import easyocr
import os
webCamFeed = False
pathImage = "TestDataPack/datasrc1.jpg"
heightImg = 640
widthImg = 480
utils.initializeTrackbars()
count = 0

if not os.path.exists('Scanned'):
    os.makedirs('Scanned')

if not os.path.exists('Transcript'):
    os.makedirs('Transcript')

if not os.path.exists('Result'):
    os.makedirs('Result')


reader = easyocr.Reader(['en','tr'])


def apply_ocr_to_saved_images(count):

    image_types = ["Original", "Gray", "Threshold", "Contours", "BiggestContour", "WarpPerspective", "WarpGray",
                   "AdaptiveThreshold"]

    best_confidence = 0
    best_transcript = ""
    best_image_type = ""
    for img_type in image_types:
        img_path = f"Scanned/myImage_{count}_{img_type}.jpg"
        img = cv2.imread(img_path)

        if img is None:
            print(f"Error: Image {img_path} could not be loaded.")
            continue

        print(f"Applying OCR to {img_path}")

        # Apply OCR
        text_results = reader.readtext(img)
        extracted_text = []
        total_confidence = 0

        for bbox, text, score in text_results:
            total_confidence += score
            if score > 0.25:
                bbox = np.array(bbox).astype(int)
                cv2.rectangle(img, tuple(bbox[0]), tuple(bbox[2]), (0, 255, 0), 2)
                cv2.putText(img, text, (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
                extracted_text.append(f"{text} (confidence: {score:.2f})")

        if not extracted_text:
            print(f"No text detected in {img_path}")

        annotated_img_path = f"Scanned/myImage_{count}_{img_type}_OCR.jpg"
        cv2.imwrite(annotated_img_path, img)
        transcript_path = f"Transcript/myImage_{count}_{img_type}.txt"
        with open(transcript_path, 'w') as file:
            for line in extracted_text:
                file.write(line + "\n")

        print(f"OCR applied and results saved for {img_path}")
        if total_confidence > best_confidence:
            best_confidence = total_confidence
            best_transcript = extracted_text
            best_image_type = img_type
    if best_transcript:
        final_transcript_path = f"Result/Final_Transcript.txt"
        with open(final_transcript_path, 'w') as file:
            file.write(f"Best OCR result from {best_image_type} filter:\n\n")
            for line in best_transcript:
                file.write(line + "\n")
        print(f"Best OCR result from {best_image_type} saved as Final_Transcript.txt")



if not webCamFeed:
    img = cv2.imread(pathImage)
    if img is None:
        print(f"Error: Image {pathImage} could not be loaded.")
        exit()
    img = cv2.resize(img, (widthImg, heightImg))
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    thres = utils.valTrackbars()
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)


    imgContours = img.copy()
    imgBigContour = img.copy()
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)


    imageArray = ([img, imgGray, imgThreshold, imgContours],
                  [imgBlank, imgBlank, imgBlank, imgBlank])


    biggest, maxArea = utils.biggestContour(contours)
    if biggest.size != 0:
        biggest = utils.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
        imgBigContour = utils.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))


        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))


        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)


        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])


        cv2.imwrite(f"Scanned/myImage_{count}_Original.jpg", img)
        cv2.imwrite(f"Scanned/myImage_{count}_Gray.jpg", imgGray)
        cv2.imwrite(f"Scanned/myImage_{count}_Threshold.jpg", imgThreshold)
        cv2.imwrite(f"Scanned/myImage_{count}_Contours.jpg", imgContours)
        cv2.imwrite(f"Scanned/myImage_{count}_BiggestContour.jpg", imgBigContour)
        cv2.imwrite(f"Scanned/myImage_{count}_WarpPerspective.jpg", imgWarpColored)
        cv2.imwrite(f"Scanned/myImage_{count}_WarpGray.jpg", imgWarpGray)
        cv2.imwrite(f"Scanned/myImage_{count}_AdaptiveThreshold.jpg", imgAdaptiveThre)


        apply_ocr_to_saved_images(count)


        count += 1


    labels = [["Original", "Gray", "Threshold", "Contours"],
              ["Biggest Contour", "Warp Perspective", "Warp Gray", "Adaptive Threshold"]]

    stackedImage = utils.stackImages(imageArray, 0.75, labels)
    cv2.imshow("Result", stackedImage)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
