import streamlit as st
import cv2
from PIL import Image
import numpy as np
import imutils


def train():
        left = st.file_uploader("Upload The Left Side Of The Image", type=['jpg', 'png', 'jpeg'])
        right = st.file_uploader("Upload The Right Side Of The Image", type=['jpg', 'png', 'jpeg'])
        if not left or not right:
               return None

        original_image_left = Image.open(left)
        original_image_left = np.array(original_image_left)

        original_image_right = Image.open(right)
        original_image_right = np.array(original_image_right)

        st.image(original_image_left, caption="★ Original Left Image★")
        st.image(original_image_right, caption="★ Original Right Image★")

        
        

        st.text("______________________________________________________________________________________________")

                # read images and transform them to grayscale

        trainImg_gray = cv2.cvtColor(original_image_left, cv2.COLOR_RGB2GRAY)


        # Opencv defines the color channel in the order BGR. 
        # Transform it to RGB to be compatible to matplotlib
        queryImg_gray = cv2.cvtColor(original_image_right, cv2.COLOR_RGB2GRAY)

        
        label = "✵Query image✵" 
        st.image(original_image_right, caption=label)

        label = "✵Train image (Image to be transformed)✵" 
        st.image(original_image_left, caption=label)

        feature_extractor = 'orb' # one of 'sift', 'surf', 'brisk', 'orb'
        feature_matching = 'bf'
        kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor)
        kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor)


        st.text("______________________________________________________________________________________________")
        st.text("★display the keypoints and features detected on both images★")
        # display the keypoints and features detected on both images
        label = "✵Query image✵" 
        st.image(trainImg_gray, caption=label)

        label = "✵Train image (Image to be transformed)✵" 
        st.image(queryImg_gray, caption=label)
      

        if feature_matching == 'bf':
            matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
            img3 = cv2.drawMatches(original_image_left,kpsA,original_image_right,kpsB,matches[:100],
                                   None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        elif feature_matching == 'knn':
            matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
            img3 = cv2.drawMatches(original_image_left,kpsA,original_image_right,kpsB,np.random.choice(matches,100),
                                   None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            

        label = "✵After Feature Matching✵" 
        st.image(img3, caption=label)
        #plt.show()
        M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
        if M is None:
            print("Error!")
        (matches, H, status) = M
        print(H)



        # Apply panorama correction
        width = original_image_left.shape[1] + original_image_right.shape[1]
        height = original_image_left.shape[0] + original_image_right.shape[0]

        result = cv2.warpPerspective(original_image_left, H, (width, height))
        result[0:original_image_right.shape[0], 0:original_image_right.shape[1]] = original_image_right

        label = "✵Apply panorama correction✵" 
        st.image(result, caption=label)



        # transform the panorama image to grayscale and threshold it 
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        # Finds contours from the binary image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # get the maximum contour area
        c = max(cnts, key=cv2.contourArea)

        # get a bbox from the contour area
        (x, y, w, h) = cv2.boundingRect(c)

        # crop the image to the bbox coordinates
        result = result[y:y + h, x:x + w]

 

        label = "✵Final Image✵" 
        st.image(result, caption=label)





def detectAndDescribe(image, method=None):
    """
    Compute key points and feature descriptors using an specific method
    """
    
    
    
    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        
    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)







def createMatcher(method,crossCheck):
    "Create and return a Matcher Object"
    
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf



def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)
        
    # Match descriptors.
    best_matches = bf.match(featuresA,featuresB)
    
    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches



def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m,n in rawMatches:
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches





def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        
        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            reprojThresh)

        return (matches, H, status)
    else:
        return None





def main_opration():
    st.title("Panorama Stiching")
   

    train()
    st.text("_____________________________________________________________________________________________________________")

if __name__ == "__main__":
    main_opration()



