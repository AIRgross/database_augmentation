import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import os
def rotation_im(image, angle):
	return imutils.rotate_bound(image, angle)


def blur_img(image, coefx,coefy):
        return cv2.blur(image, (coefx,coefy))


def resize(image, scale_x,scale_y):
	return cv2.resize(image,(0,0),fx=scale_x,fy=scale_y)

def inser_obj(thresh,img_obj,img_back):
	mat=np.where(thresh==255)
	img_back[mat,:]=img_obj[mat,:] 
	
def extract(img,thresh):
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	dilated = cv2.dilate(thresh,kernel,iterations = 13) # dilate
	im, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
 	contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
 
	#print('contours size',coutours.shape)
#	return 1,1,img.shape[0]-1,img.shape[1]-1
 	for contour in contours:
 
	    # get rectangle bounding contour
 
		[x,y,w,h] = cv2.boundingRect(contour)
#		print("extract exp",y,x,h,w)
 
   		return y,x,h,w
    # discard areas that are too large
#	   if h>img.shape[1]-2 and w>img.shape[0]-2:
 
 #	       continue
	    # discard areas that are too small
 
 		if h<5 or w<5:
 
 	        	continue
 
   	return x,y,w,h

def colorful(img):
	filter=np.random.randint(-20,20,size=img.shape,dtype=int)

	filter[:,:,0]=filter[:,:,0]+np.random.randint(-40,40)
	filter[:,:,1]=filter[:,:,1]+np.random.randint(-40,40)
	filter[:,:,2]=filter[:,:,2]+np.random.randint(-40,40)

	filter=img+filter
 	filter=np.clip(filter,0,255) 
	filter=np.uint8(filter/2)

 	filter = cv2.GaussianBlur(filter,(3,3),0)+img/2

	filter = np.uint8(filter)
#	cv2.imshow("img Random",filter)
#	cv2.waitKey(0)
#	cv2.imshow("img ",img)
#	cv2.waitKey(0)
	#filter = cv2.medianBlur(filter,5)
#	cv2.waitKey(0)


	return filter


def overlay_image_alpha(img, img_overlay, pos,xi,yi,h,w, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """
    x0=xi
    y0=yi
 
    x, y = pos

    # Image ranges
    x1, x2 = max(0, x), min(img.shape[0], x + x0+h)
    y1, y2 = max(0, y), min(img.shape[1], y + y0+w)

    # Overlay ranges
    x1o, x2o = max(x0, -x), min(x0+h, img.shape[0] - x)
    y1o, y2o = max(y0, -y), min(y0+w, img.shape[1] - y)

    # Exit if nothing to do
 #   if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
 #       return

    channels = img.shape[2]
    x1=x
    y1=y
    x2=x+h
    y2=y+w
    x1o=xi
    y1o=yi
    x2o=xi+h
    y2o=yi+w  
    delta_x=0
    delta_y=0

    if(x+h>img.shape[0]):
	delta_x=x2-img.shape[0]

    if(y+w>img.shape[1]):
	delta_y=y2-img.shape[1]

    alpha = alpha_mask[x1o:x2o-delta_x, y1o:y2o-delta_y]
    alpha_inv = 1.0 - alpha
 #   print('test length',x1o-x2o,x1-x2-delta_x)
 #   print('test length',y1o-y2o,y1-y2-delta_y)
    for c in range(channels):
        img[x1:x2-delta_x, y1:y2-delta_y, c] = (alpha * img_overlay[x1o:x2o-delta_x, y1o:y2o-delta_y, c] + alpha_inv * img[x1:x2-delta_x, y1:y2-delta_y, c])
#    print("box position :",x1, y1, x2-delta_x,"<=",img.shape[0], y2-delta_y,"<=",img.shape[1])
    return x1, y1, x2-delta_x, y2-delta_y


 

def create_pic_label(path_back,list_over,nbr_add,path_image,path_label,label):
	image_back = cv2.imread(path_back)	
	f= open(path_label,"w+")
	list_x=list(range(nbr_add))
	list_y=list(range(nbr_add))
	for i in range(nbr_add):
		index=np.random.randint(0,len(list_over))
		image_ref  = cv2.imread(list_over[index])

		angle=np.random.randint(0,360)
		scale_x=np.random.uniform(0.10,1.2)
		scale_y=scale_x*np.random.uniform(0.92,1.07)
#	print("angle",angle)
#	print("scale x y",scale_x,scale_y)

#	print("i:",i,"over",nbr_add,image_back.shape)
 		image = rotation_im(image_ref,angle)
  		while( image.shape[0]>100 or image.shape[0]>100 ):
			image= resize(image, 0.90,0.90)
 
#	cv2.imshow("test",image)
#	cv2.waitKey(0)
 		image = resize(image, scale_x,scale_y)
 
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
		im,thresh = cv2.threshold(gray,50,1.,cv2.THRESH_BINARY) 
	#plt.imshow(thresh)
	#plt.show()
		x_0,y_0,h,w=extract(image,thresh)
#	print("extract ",x_0,y_0,h,w)
		x=image_back.shape[0]-h/2
		y=image_back.shape[1]-w/2

		pos=(np.random.randint(0,x/nbr_add) , np.random.randint(0,y/nbr_add))
		indice_x=np.random.randint(0,nbr_add-i)
		indice_y=np.random.randint(0,nbr_add-i)

		pos=(pos[0]+x/nbr_add*list_x[indice_x],pos[1]+y/nbr_add*list_y[indice_y])
		list_x.remove(list_x[indice_x])
		list_y.remove(list_y[indice_y])	

#	print("pos",pos,"box",h,w,"img",image_back.shape)
		if(np.random.randint(0,2)>0):
			image=colorful(image)
		x1,y1,x2,y2=overlay_image_alpha(image_back, image, pos,x_0,y_0,h,w, thresh)

	 
		#cv2.rectangle(image_back,(y1,x1),(y2,x2),(255,255,0),2)
		max_x=float(image_back.shape[0])
		max_y=float(image_back.shape[1])
		#print(max_x,max_y)
     		f.write("%d %.6f %.6f %.6f %.6f\n" %(label,(y1/max_y+y2/max_y)/2.,(x1/max_x+x2/max_x)/2,y2/max_y-y1/max_y,x2/max_x-x1/max_x))
		#print(y1,x1,y2,x2)
	#cv2.imshow("test",image_back)
	#cv2.waitKey(0) 

	scale_x=np.random.randint(1,4)
	scale_y=np.random.randint(1,4)	
 	image_back=blur_img(image_back, scale_x,scale_y)
	cv2.imwrite(path_image, image_back) 
	f.close()




path_back="/home/eamslab/drivers/database_creator/background"
path_over="/home/eamslab/drivers/database_creator/object"

nbr_add=np.random.randint(1,6) #number of object to add in each picture
label=0  #numero of label

path_label="/home/eamslab/drivers/database_creator/labels/drone_" #name for labels
path_image="/home/eamslab/drivers/database_creator/images/drone_" #name for image

iter=0 
nbr=5000 #number of picture to create

ftrain= open("/home/eamslab/drivers/database_creator/train.txt","w+") 
flabel= open("/home/eamslab/drivers/database_creator/labels.txt","w+")


list_name_o=[]
list_name_back=[]

for (direpath,dirnames,filenames) in os.walk(path_over):
    	for f in filenames:
		files=direpath+'/'+f
       		if os.path.isfile(files):
			list_name_o.append(files)

for (direpath,dirnames,filenames) in os.walk(path_back):
    	for f in filenames:
		files=direpath+'/'+f
       		if os.path.isfile(files):
			list_name_back.append(files)

for i in range(nbr):
  #  for f in filenames:
	   iter=iter+1
	   path_label_name=path_label+str(iter)+".txt"
	   path_image_name=path_image+str(iter)+".jpg"

     	   ftrain.write("data/images/drone_%s.jpg\n" %(str(iter)))
     	   flabel.write("data/labels/drone_%s.txt\n" %(str(iter)))
   
	   index=np.random.randint(0,len(list_name_back))
	   create_pic_label(list_name_back[index],list_name_o,nbr_add,path_image_name,path_label_name,label)

#cv2.imshow("test",image_back)
#cv2.waitKey(0)
#cv2.imshow("test",image_ref)
#cv2.waitKey(0)
ftrain.close()
flabel.close()






