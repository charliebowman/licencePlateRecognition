import cv2

# count = 1 
# while count < 10:

img = cv2.imread('cars/Car3.jpg')
cv2.imshow('Original', img)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', gray_img)

smooth_img = cv2.bilateralFilter(gray_img, 5, 55, 55)
cv2.imshow('Bilateral Filtering', smooth_img)

canny_edge = cv2.Canny(smooth_img, 100, 300)
cv2.imshow('Canny Edge', canny_edge)


# Determine number plate from contours 
contours, hierachy = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

# Find the contour rectangular corners and isolate the number plate 
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    plate_approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

    if len(plate_approx) == 4:
        x, y, w, h = cv2.boundingRect(contour)
        break 

# Display the number plate bordered with a rectangle overlaying original image
img = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 5)

cv2.imshow("Number Plate", img)
cv2.waitKey(0)

#    count += 1

