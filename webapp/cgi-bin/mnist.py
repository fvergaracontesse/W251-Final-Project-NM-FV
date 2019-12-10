#!/usr/bin/env pipenv-shebang
"""
CGI script that accepts image urls and feeds them into a ML classifier. Results
are returned in JSON format.
"""

import io
import json
import sys
import os
import re
import base64
import numpy as np
from PIL import Image
#from model import model
from InputImage import InputImage
from BaselineModel import BaselineModel
from keras.models import model_from_json
import cv2 as cv2


# Default output
res = {"result": 0,
       "data": [],
       "error": ''}

# Import modules for CGI handling
import cgi, cgitb

# Create instance of FieldStorage
form = cgi.FieldStorage()

try:
    # Get post data
    if os.environ["REQUEST_METHOD"] == "POST":
        #data = sys.stdin.read(int(os.environ["CONTENT_LENGTH"]))
        #Input images tools
        input_image = InputImage(None)
        #Load model
        model_json_filename = "cgi-bin/models/baseline1.json"
        model_weights_filename = "cgi-bin/models/baseline1.h5"
        # load json and create model
        json_file = open(model_json_filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(model_weights_filename)

        # Get data from fields
        url = form.getvalue('url')
        options  = form.getvalue('option')

        # Convert data url to numpy array
        img_str = re.search(r'base64,(.*)', url ).group(1)
        image_bytes = io.BytesIO(base64.b64decode(img_str))
        image=np.asarray(Image.open(image_bytes).convert('L'))

        height, width = image.shape
        for i,l in enumerate(image.T):
            #print (sum(l))
            if np.average(l) == 255:
                #print (i, sum(l))
                cv2.line(image, (i, 0), (i, height), (0,0,0))
        symbols=[]
        prev_sum = 0
        height, width = image.shape
        for i,l in enumerate(image.T):
            # start of boundary
            if sum(l) != 0 and prev_sum == 0:
                #print (sum(l),prev_sum)
                boundary_start = i
            # end of boundary
            if sum(l) == 0 and prev_sum != 0:
                boundary_end=i
                #print (boundary_start , boundary_end)
                if (boundary_end - boundary_start) > 1:
                    #draw_image(image[0 : 64, boundary_start : boundary_end])
                    symbols.append(image[0 : height, boundary_start : boundary_end])
            prev_sum = sum(l)

        operators=['=','-','+']
        equation=[]
        ## Get Prediction
        for k in symbols:
            img,script_type = input_image.crop_image(k)
            prediction = input_image.predict(model, img)
            if script_type != None and prediction not in operators:
                prediction=script_type+prediction
            equation.append(prediction)

        # Return label data
        res['result'] = 1
        res['labels'] = "labels"
        res['height'] = height
        res['width'] = width
        res['symbols'] = len(symbols)
        res['equation'] = " ".join(equation)
        #res['prediction'] = " ".join(prediction)

except Exception as e:
    # Return error data
    res['error'] = str(e)



# Print JSON response
print("Content-type: application/json")
print("")
print(json.dumps(res))
