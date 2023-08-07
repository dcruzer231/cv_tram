#Divide and Conque strategy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import transform as tf
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
import cv2
import os
from time import time


saveCounter = 0
def getPoints(img1, img2,imgshape):
    #convert to integer if not already
    if img1.dtype != 'uint8':
        img1 = img_as_ubyte(img1)
    if img2.dtype != 'uint8':
        img2 = img_as_ubyte(img2)
    orb = cv2.ORB_create()

    #create mask to mask half the image
    mask2 = np.zeros(img2.shape[:-1],np.uint8)
    mask2[:,-3*imgshape[1]//4:] = 255

    mask1 = np.zeros(img1.shape[:-1],np.uint8)
    mask1[:,:3*imgshape[1]//4] = 255

    keypoints1, descriptors1 = orb.detectAndCompute(img1,mask=None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2,mask=None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1,descriptors2)
    # Extract data from orb objects and matcher
    dist = np.array([m.distance for m in matches])
    ind1 = np.array([m.queryIdx for m in matches])
    ind2 = np.array([m.trainIdx for m in matches])
    keypoints1 = np.array([p.pt for p in keypoints1])
    keypoints2 = np.array([p.pt for p in keypoints2])
    keypoints1 = keypoints1[ind1]
    keypoints2 = keypoints2[ind2]
    keypoints1, keypoints2 = select_matches_ransac(keypoints1, keypoints2)

    display_mask1 = np.maximum(img1, (np.reshape(mask1,(img1.shape[0],img1.shape[1],1))//255*130))
    display_mask2 = np.maximum(img2,(np.reshape(mask2,(img2.shape[0],img2.shape[1],1))//255*130))

    display_control_lines(img1,img2,keypoints1,keypoints2)
    return keypoints1, keypoints2

#warps images best on the src and dest points and stitches img1 to img2.
def warpAndStitchImagesAvg(src,dest,img1, img2):
    global saveCounter
    global savePath

    #get the warp to check coordinates

    avgDest = (src + dest)/2
    H0 = tf.ProjectiveTransform()
    H1 = tf.ProjectiveTransform()
    H0.estimate(src, avgDest)
    
    #img1 indexed to avoid the colour index
    coords = tf.warp_coords(H0,img1.shape[:2])
    rowCorners1 = coords[0,[0,0,-1,-1],[0,-1,-1,0]]
    colCorners1 = coords[1,[0,0,-1,-1],[0,-1,-1,0]]

    #round the coordinates and convert to int
    rowCorners1 = np.around(rowCorners1).astype(int)
    colCorners1 = np.around(colCorners1).astype(int)

    #Get the most negative coordinates or 0.
    xdisplacementLeft1 = max(-np.min(colCorners1),0)
    ydisplacementLeft1 = max(-np.min(rowCorners1),0) 
    
    #Get any necessary displament to pad the right
    #it will either return the padding needed or 0.
    xdisplacementRight1 = max(np.max(colCorners1)-img2.shape[1],0)
    ydisplacementRight1 = max(np.max(rowCorners1)-img2.shape[0],0)


    H0.estimate(dest, avgDest)
    #img2 indexed to avoid the colour index
    coords = tf.warp_coords(H0,img2.shape[:2])
    rowCorners2 = coords[0,[0,0,-1,-1],[0,-1,-1,0]]
    colCorners2 = coords[1,[0,0,-1,-1],[0,-1,-1,0]]

    #round the coordinates and convert to int
    rowCorners2 = np.around(rowCorners2).astype(int)
    colCorners2 = np.around(colCorners2).astype(int)

    #Get the most negative coordinates or 0.
    xdisplacementLeft2 = max(-np.min(colCorners2),0)
    ydisplacementLeft2 = max(-np.min(rowCorners2),0) 
    
    #Get any necessary displament to pad the right
    #it will either return the padding needed or 0.
    xdisplacementRight2 = max(np.max(colCorners2)-img1.shape[1],0)
    ydisplacementRight2 = max(np.max(rowCorners2)-img1.shape[0],0)

    #Find the maximum needed displacement from either images
    xdisplacementLeft = max(xdisplacementLeft1,xdisplacementLeft2)
    ydisplacementLeft = max(ydisplacementLeft1,ydisplacementLeft2)

    xdisplacementRight = max(xdisplacementRight1,xdisplacementRight2)
    ydisplacementRight = max(ydisplacementRight1,ydisplacementRight2)
    

    # Add displacement to the shift the warped image and recompute homeography
    newDest = np.copy(avgDest)
    newDest[:,0] = avgDest[:,0]+xdisplacementLeft
    newDest[:,1] = avgDest[:,1]+ydisplacementLeft
    

    H0.estimate(src, newDest)

    #new output shape needs to make room for all the shifting.
    ysize = ydisplacementLeft + max((ydisplacementRight1 + img2.shape[0]), (ydisplacementRight2 + img1.shape[0]))
    xsize = xdisplacementLeft + max(np.max(colCorners2),np.max(colCorners1))#max((xdisplacementRight1 + img2.shape[1]), (xdisplacementRight2 + img1.shape[1]))
    output_shape = (ysize, xsize)



    H1.estimate(dest,newDest)

    thresholds = (2,2)
    warped1 = tf.warp(img1, H0.inverse, output_shape=output_shape)
    warped2 = tf.warp(img2, H1.inverse,output_shape=output_shape)
    
    cond0,det0 = cond_num_and_det(H0)
    cond1,det1 = cond_num_and_det(H1)

    if (abs(cond0),abs(det0)) < thresholds and (abs(cond1),abs(det1)) < thresholds:
        fig2, ax2 = plt.subplots(figsize=(12,10))
        ax2.imshow(np.maximum(warped1, warped2), cmap=plt.cm.gray)
        fig2.suptitle('warp at ' + str(saveCounter-1), fontsize=14)
        fig2.savefig(savePath+'warp at '+str(saveCounter-1),dpi=dpiValue)        
        return np.maximum(warped1, warped2)

    #If there is a bad warp, just place images side by side
    else:
        print("could not stich at ",saveCounter-1)
        print("cond_num_and_det H1",cond_num_and_det(H1))
        print("cond_num_and_det H0",cond_num_and_det(H0))

        #save the offending warp
        fig5, ax5 = plt.subplots(figsize=(12,10))
        ax5.imshow(np.maximum(warped1, warped2), cmap=plt.cm.gray)
        fig5.suptitle('Bad warp at ' + str(saveCounter-1), fontsize=14)
        fig5.savefig(savePath+'bad warp at '+str(saveCounter-1),dpi=dpiValue)

        canvas_shape = (max(img1.shape[0],img2.shape[0]),img1.shape[1]+img2.shape[1],3)
        canvas = np.zeros(canvas_shape,dtype=type(img1[0,0,0]))
        canvas[:img1.shape[0],:img1.shape[1]] = img1
        canvas[:img2.shape[0],img1.shape[1]:]= img2
        canvas[:,img1.shape[1]:img1.shape[1]+20,0] = 1

        fig2, ax2 = plt.subplots(figsize=(12,10))
        ax2.imshow(canvas, cmap=plt.cm.gray)
        fig2.suptitle('canvas at ' + str(saveCounter-1), fontsize=14)
        fig2.savefig(savePath+'canvas at '+str(saveCounter-1),dpi=dpiValue)        
        return canvas

def display_control_lines(im0,im1,pts0=np.array([[0,0]]),pts1=np.array([[0,0]]),clr_str = 'rgbycmwk',tag=""):
    global saveCounter
    canvas_shape = (max(im0.shape[0],im1.shape[0]),im0.shape[1]+im1.shape[1],3)
    canvas = np.zeros(canvas_shape,dtype=type(im0[0,0,0]))
    canvas[:im0.shape[0],:im0.shape[1]] = im0
    canvas[:im1.shape[0],im0.shape[1]:canvas.shape[1]]= im1
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(canvas)
    ax.axis('off')
    pts2 = pts1+np.array([im0.shape[1],0])
    for i in range(pts0.shape[0]):
        ax.plot([pts0[i,0],pts2[i,0]],[pts0[i,1],pts2[i,1]],color=clr_str[i%len(clr_str)],linewidth=1.0)
    fig.suptitle('Point correpondences', fontsize=16)
    fig.savefig(savePath+str(saveCounter)+"_Point_correpondences"+tag,dpi=dpiValue)
    

def cond_num_and_det(H):
    # Very large condition numbers usually indicate a bad homography
    # Negative determinants and those with low absolute values usually indicate a bad homography
    # Large determinant also usually indicate a bad homography
    w,v = np.linalg.eig(np.array(H.params))
    w = np.sort(np.abs(w))
    cn = w[2]/w[0]
    d = np.linalg.det(H.params)
    print('condition number {:7.3f}, determinant {:7.3f}'.format(cn,d))
    return cn, d

def homography_error(H,pts1,pts2):
    pts1 =  np.hstack((pts1,np.ones((pts1.shape[0],1))))
    pts2 =  np.hstack((pts2,np.ones((pts2.shape[0],1))))
    proj = np.matmul(H,pts1.T).T
    proj = proj/proj[:,2].reshape(-1,1)
    err = proj[:,:2] - pts2[:,:2]
    return np.mean(np.sqrt(np.sum(err**2,axis=1)))

def select_matches_ransac(pts0, pts1):
    H, mask = cv2.findHomography(pts0.reshape(-1,1,2), pts1.reshape(-1,1,2), cv2.RANSAC,5)
    choice = np.where(mask.reshape(-1) ==1)[0]
    return pts0[choice], pts1[choice]

def read_images(image_dir):
    image_files = sorted(os.listdir(image_dir))
    image_list = []
    for im in image_files:
        img = mpimg.imread(image_dir+im)#[:,:,:3]
        # img=img/np.amax(img)
        image_list.append(img)
    return image_list  


def merge_images(image_list,shape):

    image1, image2 = None, None
    #recusive case
    if len(image_list) > 2:
        half = len(image_list)//2
        image2 = merge_images(image_list[half:],shape)
        image1 = merge_images(image_list[:half],shape)
        
        plt.close('all')

    else:
        #only one image, just return
        if len(image_list) < 2:
            return image_list[0]
        #two images, stitch and return
        else:
            image1 = image_list[0]
            image2 = image_list[1]

    keypoints1,keypoints2 = getPoints(image1,image2,shape)

    dest,src = keypoints1,keypoints2

    combined = warpAndStitchImagesAvg(src,dest,image2,image1)
    return combined



def similarityTransform(image1, image2):
    global saveCounter
    H1 = tf.SimilarityTransform()
    H2 = tf.SimilarityTransform()
    shape = image1.shape
    src,dest = getPoints(image1,image2,shape)
    avgDest = (src + dest)/2
    H1.estimate(src,avgDest)
    H2.estimate(dest,avgDest)

    saveCounter += 1
    return H1, H2


def doWarp(image1, image2, H1,H2):
    global saveCounter

    warp1 = tf.warp(image1,H1.inverse)
    warp2 = tf.warp(image2,H2.inverse)

    display_control_lines(warp1,warp2,tag="output")
 
    # a blend of the images to check for ultimate aligntment
    blend = 0.5*warp1 + 0.5*warp2
    fig5, ax5 = plt.subplots(figsize=(12,10)) #plt.subplots(figsize=(12,10))
    ax5.imshow(blend, cmap=plt.cm.gray)
    fig5.suptitle('Blended output', fontsize=14)
    fig5.savefig(savePath+'blended_'+str(saveCounter),dpi=dpiValue)

    halfRow = warp1.shape[0]//2
    rowSlice = np.zeros_like(warp1)
    rowSlice[:halfRow] += warp1[:halfRow]
    rowSlice[halfRow:] += warp2[halfRow:]

    halfCol = warp1.shape[1]//2
    colSlice = np.zeros_like(warp1)
    colSlice[:,:halfCol] += warp1[:,:halfCol]
    colSlice[:,halfCol:] += warp2[:,halfCol:]



    fig5, ax5 = plt.subplots(figsize=(12,10)) #plt.subplots(figsize=(12,10))
    ax5.imshow(blend, cmap=plt.cm.gray)
    fig5.suptitle('Blended output', fontsize=14)
    fig5.savefig(savePath+'blended_'+str(saveCounter),dpi=dpiValue)


    fig6, ax6 = plt.subplots(figsize=(12,10)) #plt.subplots(figsize=(12,10))
    ax6.imshow(rowSlice, cmap=plt.cm.gray)
    fig6.suptitle('row cut output', fontsize=14)
    fig6.savefig(savePath+'rowCut_'+str(saveCounter),dpi=dpiValue)


    fig7, ax7 = plt.subplots(figsize=(12,10)) #plt.subplots(figsize=(12,10))
    ax7.imshow(colSlice, cmap=plt.cm.gray)
    fig7.suptitle('column cut output', fontsize=14)
    fig7.savefig(savePath+'colCut_'+str(saveCounter),dpi=dpiValue)
    saveCounter += 1




dpiValue=300

if __name__ == '__main__':


    start = time()
    debug = 1

    plt.close('all')

    savePath = "/home/dan/linux_share/"

    path = "C:\\Users\\dcruz\\Documents\\System_ecology-lab\\TramData\\20191206\\"
    path1 = "/home/dan/Documents/TramData/20191220/"
    path2 = "/home/dan/Documents/TramData/20191206/"
    path3 = "/home/dan/Documents/TramData/20191115/" #This is the very bad dataset


    image_list1 = read_images(path1)
    image_list2 = read_images(path2)
    image_list3 = read_images(path3)

    cutoff = len(image_list1)
    image_list_p1 = image_list1[:cutoff]
    image_list_p1 = [img_as_float(x) for x in image_list_p1]

    image_list_p2 = image_list2[:cutoff]
    image_list_p2 = [img_as_float(x) for x in image_list_p2]

    image_list_p3 = image_list3[:cutoff]
    image_list_p3 = [img_as_float(x) for x in image_list_p3]

    H1,H2 = similarityTransform(image_list_p3[0],image_list_p2[0])
    avgCount = 1
    threshold = 0.2
    for i in range(1,len(image_list_p1)):
        print("Homography in ",saveCounter)
        newH1,newH2 = similarityTransform(image_list_p3[i],image_list_p2[i])
        plt.close('all')

        cond0,det0 = cond_num_and_det(newH1)
        cond1,det1 = cond_num_and_det(newH2)

        if(abs(1-cond0) < threshold and abs(1-cond1) < threshold and abs(1-det0) < threshold and abs(1-det1) < threshold):
            H1.params += newH1.params
            H2.params += newH2.params
            avgCount += 1
        else:
            print("bad warp!")

    
    saveCounter = 0
    H1.params /= avgCount 
    H2.params /= avgCount 

    for i in range(0,len(image_list_p1)):
        doWarp(image_list_p3[i],image_list_p2[i],H1,H2)
        plt.close('all')
        
    print(time()-start)