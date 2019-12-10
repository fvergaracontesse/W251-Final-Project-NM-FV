import cv2 as cv2
import os.path
import  numpy as np
from os.path import isfile, join
from tqdm import tqdm
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import json

class InputImage:

    def __init__(self, filename=None):
        if filename is None:
            self.filename = "edge_file.png"
        else:
            self.filename = filename
        self.img = None
        self.thresh = None
        self.dilate = None
        self.sub_images = []
        self.np_sub_images = []

    def readImage(self):
        try:
            if(os.path.exists(self.filename)):
              img = cv2.imread(self.filename)
              print("File read as image successfully")
              self.img = img
            else:
              print("File not found")
        except:
            print("Something went wrong reading image")

    def pilToCV(self, pilImg):
        img = cv2.cvtColor(np.array(pilImg), cv2.COLOR_RGB2BGR)
        img_filename = "edgeImage.png"
        cv2.imwrite(img_filename,img)
        #scale_percent = 800 # percent of original size
        #width = int(img.shape[1] * scale_percent / 100)
        #height = int(img.shape[0] * scale_percent / 100)
        #dim = (width, height)
        # resize image
        #resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        self.img  = img

    def modifyImage(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilate = cv2.dilate(thresh, kernel , iterations=4)
        self.thresh = thresh
        self.dilate = dilate

    def getSubImages(self):
        # Find contours in the image
        cnts = cv2.findContours(self.dilate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts=  sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])
        contours = []
        sub_images = []
        threshold_min_area = 400
        threshold_max_area = 6000
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if area > threshold_min_area and area < threshold_max_area:
                contours.append(c)
                sub_images.append(self.img[y : y + h, x : x + w])
                #cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),1)
        self.sub_images = sub_images

    def saveSubImages(self):
        if len(self.filename.split("/"))>1:
            image_to_symbol_directory = 'images_to_symbol'+"/"+self.filename.split("/")[1].split(".")[0]
        else:
            image_to_symbol_directory = 'images_to_symbol'+"/"+self.filename.split(".")[0]
        if not os.path.exists(image_to_symbol_directory):
            os.makedirs(image_to_symbol_directory)
        for i,image in enumerate(self.sub_images):
            img_filename = image_to_symbol_directory+"/"+str(i)+".png"
            cv2.imwrite(img_filename,image)
        print("Subimages save successfully")

    def absoluteFilePaths(self,directory):
        paths=[]
        for dirpath,_,filenames in os.walk(directory):
            for f in filenames:
                #print (os.path.abspath(os.path.join(dirpath, f)))
                if isfile(os.path.abspath(os.path.join(dirpath, f))):
                    paths.append(os.path.abspath(os.path.join(dirpath, f)))
                else:
                    continue
        return paths

    def getLabelfromPath(self,path):
        return path.split('/')[-2]

    def convertToArray(self, image_loc):
        img = Image.open(image_loc).convert('L')
        img = img.resize((100,100),Image.ANTIALIAS)
        array = np.asarray(img)
        return array

    def prepareImagesForClassifications(self,labelPath):
        imgArray=[]
        labels = []
        if len(self.filename.split("/"))>1:
            image_to_symbol_directory = 'images_to_symbol'+"/"+self.filename.split("/")[1].split(".")[0]
        else:
            image_to_symbol_directory = 'images_to_symbol'+"/"+self.filename.split(".")[0]
        filePaths=self.absoluteFilePaths(image_to_symbol_directory)
        labelPaths=self.absoluteFilePaths(labelPath)
        for f in tqdm(filePaths):
            imgArray.append(self.convertToArray(f))
        for s in tqdm(labelPaths):
            labels.append(self.getLabelfromPath(s))
        data=np.array([imgArray]).T
        #b=np.array([x.ravel() for x in data.ravel()])
        labels_final = list(set(labels))
        labels_final.sort()
        np_image_array=data.reshape(len(imgArray),100,100,1)
        return np_image_array, labels_final

    def crop_image(self, imgarray):
        height, width = imgarray.shape
        xmin = None
        for i,l in enumerate(imgarray.T):
            #print (i)
            if np.average(l) < 255:
                xmin = max(i-1,0)
                break
        ymin = None
        for i,l in enumerate(imgarray):
            #print (i)
            if np.average(l) < 255:
                ymin = max(i-1,0)
                break
        ymax = None
        for i,l in enumerate(imgarray[::-1]):
            #print (i, sum(l))
            if np.average(l) < 255:
                ymax = min(height-i+1,height)
                break
        xmax = None
        for i,l in enumerate(imgarray.T[::-1]):
            #print (i)
            if np.average(l) < 255:
                xmax = min(width-i+1,width)
                break
        #print ("%d,%d : %d,%d"%(xmin,xmax,ymin,ymax))
        script_type= None
        if ymax - ymin < 0.5 * height and ymin > 0.25*height:
            script_type = '_'
        elif ymax - ymin < 0.5 * height and ymax < 0.75*height:
            script_type = '^'
        #print (script_type)
        return (cv2.resize(imgarray[ymin:ymax,xmin:xmax], dsize=(64, 64), interpolation=cv2.INTER_CUBIC),script_type)

    def predict(self, model, imgarray):
        classes={0: '(', 1: ')', 2: '+', 3: '-', 4: '1', 5: '2', 6: '3', 7: '4', 8: '=', 9: 'x', 10: 'y'}
        y_prob = model.predict(imgarray.reshape(1,64,64,1))
        y_classes = y_prob.argmax(axis=-1)
        return classes[y_classes[0]]
