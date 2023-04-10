import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import model_from_json
from pytesseract import image_to_string
import pytesseract


pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'


# Load the saved model
# json_file = open('models/model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
#loaded_model = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights("models/model.hdf5")
#print("Loaded saved model from disk.")
 
# evaluate loaded model on test data
#def identify_number(image):
   # image=image/255.0
    #print(image)
    #image_resize = cv2.resize(image, (28,28))    # For plt.imshow
    #cv2.imshow(mat = image)
    #image_resize_2 = image_resize.reshape(1,28,28,1)    # For input to model.predict_classes
#    cv2.imshow('number', image_test_1)
    # loaded_model_pred = loaded_model.predict(image_resize_2 , verbose = 0)
    #loaded_model_pred=loaded_model.predict(image_resize_2.reshape(1,28,28))
#    print('Prediction of loaded_model: {}'.format(loaded_model_pred[0]))
    # loaded_model_pred = loaded_model_pred.reshape((10,))
    # maxProb = float('-inf')
    # probIdx = -1
    # for idx, prob in enumerate(loaded_model_pred):
    #     if(prob > maxProb):
    #         maxProb = prob
    #         probIdx = idx
    #return loaded_model_pred[0]
    #return np.argmax(loaded_model_pred,axis=0)
    #return loaded_model_pred

def extract_number(sudoku):
    sudoku = cv2.resize(sudoku, (450,450))
#    cv2.imshow('sudoku', sudoku)

    # split sudoku

    grid = np.zeros([9,9])
    for i in range(9):
        for j in range(9):
#            image = sudoku[i*50+3:(i+1)*50-3,j*50+3:(j+1)*50-3]
            image = sudoku[i*50:(i+1)*50,j*50:(j+1)*50]
            # print(image)
            #filename = "images/sudoku/file_%d_%d.jpg"%(i, j)
            imgcrop = image[2:48,2:48]
            #cv2.imwrite(filename, image)
            #if image.sum() > 25000:
            t=pytesseract.image_to_string(imgcrop,lang="eng",config='--psm 6 --oem 3')
            t=t.strip()
            if t.isnumeric() :
             grid[i][j]=int(t)
            #grid[i][j] = n.recognize(image) # recognise the image
            else:
             grid[i][j] = 0
             #print('0') 
            #grid=[[8,0,0,9,6,0,1,0,0],[7,0,1,8,0,4,0,0,0],[6,3,0,0,1,0,8,0,0],[5,2,0,0,0,0,0,0,0],[0,0,0,2,0,6,0,0,0],[0,0,0,0,0,0,0,9,2],[0,0,5,0,2,0,0,1,6],[0,0,0,3,0,7,2,0,5],[0,0,4,0,5,1,0,0,8]]
    #print(grid)
    #return grid.astype(int)
    #grid=int(grid)
    return grid.astype(int)




