
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import transform as tf
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
import cv2
import os
from time import time


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

    keypoints1, descriptors1 = orb.detectAndCompute(img1,mask=mask1)
    keypoints2, descriptors2 = orb.detectAndCompute(img2,mask=mask2)
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

    # display_control_lines(display_mask2,display_mask1,keypoints2,keypoints1)
    return keypoints1, keypoints2

avgCounter = 0


#warps images best on the src and dest points and stitches img1 to img2.
def warpAndStitchImagesAvg(corresLeft,corresRight):
    global saveCounter
    global savePath

    #get the warp to check coordinates
    src = corresLeft[2]
    dest = corresRight[0]

    img1 = corresLeft[1]
    img2 = corresRight[1]

    avgDest = (src + dest)/2
    H1 = tf.ProjectiveTransform()
    H2 = tf.ProjectiveTransform()
    H1.estimate(src, avgDest)
    H2.estimate(dest, avgDest)
    
    thresholds = (2,2)

    
    cond0,det0 = cond_num_and_det(H1)
    cond1,det1 = cond_num_and_det(H2)
    print("H1")
    print(H1)
    print("H2")
    print(H2)
    print()

    print("warp and stitch at", str(saveCounter))

    print("cond_num_and_det H2",cond_num_and_det(H2))
    print("cond_num_and_det H1",cond_num_and_det(H1))

    #img1 indexed to avoid the colour index
    coords = tf.warp_coords(H1,img1.shape[:2])
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


    #img2 indexed to avoid the colour index
    coords = tf.warp_coords(H2,img2.shape[:2])
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
    

    H1.estimate(src, newDest)

    #new output shape needs to make room for all the shifting.
    ysize = ydisplacementLeft + max((ydisplacementRight1 + img2.shape[0]), (ydisplacementRight2 + img1.shape[0]))
    xsize = xdisplacementLeft + max(np.max(colCorners2),np.max(colCorners1))#max((xdisplacementRight1 + img2.shape[1]), (xdisplacementRight2 + img1.shape[1]))
    output_shape = (ysize, xsize)



    H2.estimate(dest,newDest)

    oldCoordinates1 = corresLeft[0]
    oldCoordinates2 = corresRight[2]
    
    if (abs(1-cond0),abs(1-det0)) < thresholds and (abs(1-cond1),abs(1-det1)) < thresholds:
        warped1 = tf.warp(img1, H1.inverse, output_shape=output_shape)
        warped2 = tf.warp(img2, H2.inverse,output_shape=output_shape)
        
        coordsGrid1 = tf.warp_coords(H1,shape=output_shape)
        coordsGrid2 = tf.warp_coords(H2,shape=output_shape)


        if type(oldCoordinates1) == np.ndarray:
            oldCoordinates1 = oldCoordinates1.astype(np.int)
            newCoords1 = coordsGrid1[:,oldCoordinates1[:,1],oldCoordinates1[:,0]]
            newCoords1 = np.transpose(newCoords1)
            newCoords1 = newCoords1[:,::-1]
        else:
            newCoords1 = None

        if type(oldCoordinates2) == np.ndarray:
            oldCoordinates2 = oldCoordinates2.astype(np.int)
            newCoords2 = coordsGrid2[:,oldCoordinates2[:,1],oldCoordinates2[:,0]]
            newCoords2 = np.transpose(newCoords2)
            newCoords2 = newCoords2[:,::-1]

        else:
            newCoords2 = None
        fig2, ax2 = plt.subplots(figsize=(12,10))
        ax2.imshow(np.maximum(warped1, warped2), cmap=plt.cm.gray)
        fig2.suptitle('warp at ' + str(saveCounter), fontsize=14)
        fig2.savefig(savePath+'warp at '+str(saveCounter),dpi=dpiValue)
        
        return [newCoords1,np.maximum(warped1, warped2), newCoords2]

    #If there is a bad warp, just place images side by side
    else:
        print("could not stich at ",saveCounter-1)


        canvas_shape = (max(img1.shape[0],img2.shape[0]),img1.shape[1]+img2.shape[1],3)
        canvas = np.zeros(canvas_shape,dtype=type(img1[0,0,0]))
        canvas[:img2.shape[0],:img2.shape[1]] = img2
        canvas[:img1.shape[0],img2.shape[1]:]= img1
        canvas[:,img2.shape[1]:img2.shape[1]+20,0] = 1

        fig2, ax2 = plt.subplots(figsize=(12,10))
        ax2.imshow(canvas, cmap=plt.cm.gray)
        fig2.suptitle('bad warp at ' + str(saveCounter-1), fontsize=14)
        fig2.savefig(savePath+'warp at '+str(saveCounter-1),dpi=dpiValue)

        newCoords1 = np.copy(oldCoordinates1)
        if  type(oldCoordinates1) == np.ndarray and len(oldCoordinates1.shape)>1:
            print(newCoords1.shape)
            newCoords1[:,0] = newCoords1[:,0]+img2.shape[1] # This is new coordinates adjusted

        return [newCoords1,canvas, oldCoordinates2]


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
    return cn, d
    print('condition number {:7.3f}, determinant {:7.3f}'.format(cn,d))

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


def merge_images(corres,shape,left,right):
    global saveCounter

    image1, image2 = None, None
    plt.close('all')
    #recusive case
    if len(corres) > 2:
        half = len(corres)//2
        corresRight = merge_images(corres[half:],shape,half,right)
        corresLeft = merge_images(corres[:half],shape,left,half-1)
        display_control_lines(corresRight[1],corresLeft[1],corresRight[0],corresLeft[2],tag="_current_Correspondence")


        combined = warpAndStitchImagesAvg(corresLeft,corresRight)
        saveCounter += 1
        return combined        

    else:
        #only one image, just return
        if len(corres) < 2:
            return corres[0]
        #two images, stitch and return
        else:
            image1 = corres[0][1]
            image2 = corres[1][1]

        combined = warpAndStitchImagesAvg(corres[0],corres[1])
        saveCounter += 1
        return combined

#returns a 2D array [i][j] where i is the image index, j=0 is left points
#j=1 is the image, and j=2 is the right points
def getCorrespondence(image_list):

    correspondenceList = [[None]*3 for i in range(len(image_list))]
    correspondenceList[0][1] = image_list[0]

    for i in range(len(image_list)-1):
        j = i+1
        keypoints1,keypoints2 = getPoints(image_list[i],image_list[j],shape)

        #keypoints1 is how to connect image i to j
        #keypoints2 is how to connect j to i.
        
        correspondenceList[i][2] = keypoints1
        correspondenceList[j][0] = keypoints2
        correspondenceList[j][1] = image_list[j]
    return correspondenceList




def displayAllKeyPoints(corres,tag="",clr_str = 'rgbycmwk'):
    img=corres[1]
    lPoints = corres[0]
    rPoints = corres[2]

    fig, ax = plt.subplots(figsize=(12,10)) #plt.subplots(figsize=(12,10))
    ax.imshow(img, cmap=plt.cm.gray)
    fig.suptitle('Points_'+str(tag), fontsize=14)
    fig.savefig(savePath+'Keypoints_'+str(tag),dpi=dpiValue)

    if type(lPoints) == np.ndarray:
        for i,point in enumerate(lPoints):
            ax.plot(point[0],point[1],marker="v",color=clr_str[i%len(clr_str)])
    if type(rPoints) == np.ndarray:    
        for i,point in enumerate(rPoints):
            ax.plot(point[0],point[1],marker="*",color=clr_str[i%len(clr_str)])


dpiValue=300

if __name__ == '__main__':


    start = time()
    debug = 1

    plt.close('all')

    savePath = "/home/dan/linux_share/"

    path = "/home/dan/Documents/TramData/20191206/"


    image_list = read_images(path)
    image_list_p = image_list[:30]
    image_list_p = [img_as_float(x) for x in image_list_p]

    shape = image_list_p[0].shape

    corres = getCorrespondence(image_list_p)

    saveCounter = 0

    combined = merge_images(corres,shape,0,len(image_list_p)-1)[1]

    # there should only be one image left, this is the output image
    
    fig5, ax5 = plt.subplots(figsize=(12,10)) #plt.subplots(figsize=(12,10))
    ax5.imshow(combined, cmap=plt.cm.gray)
    fig5.suptitle('Fully Stitched Image', fontsize=14)
    fig5.savefig(savePath+'Fully Stitched Image',dpi=dpiValue)
    print(time()-start)