import zipfile
from PIL import Image
import pytesseract as tess
import cv2 as cv
import numpy as np

# loading the face detection classifier
face_cascade = cv.CascadeClassifier('readonly/haarcascade_frontalface_default.xml')

#open the .zip file with the images to be processed:
imgs_file = zipfile.ZipFile('readonly/images.zip')
imgs_name_list = imgs_file.namelist() #list of image file names

imgs_data_file = [] #list with a dict for each image

#add file names and images to each dictionary:
for name in imgs_name_list:
    imgs_data_file.append({'name':name, 'image':Image.open(imgs_file.open(name))})

num_images = len(imgs_data_file) # no. of images to be processed
print('There are {} images:'.format(num_images))
print(imgs_name_list)


#add text to each image dictionary
for i in range(num_images):
    print('Processing ',imgs_data_file[i]['name'],' ...')
    txt = tess.image_to_string(imgs_data_file[i]['image'])
    imgs_data_file[i]['text'] = txt.replace('-\n','') #remove hyphenated words at EOL
print('Done!')


#add face bounding boxes to each image dictionary
for i in range(num_images):
    print('Processing ',imgs_data_file[i]['name'],' ...')
    img = np.array(imgs_data_file[i]['image']) #get the image array for CV processing
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #use grayscale for finding faces
    imgs_data_file[i]['faces'] = face_cascade.detectMultiScale(gray,1.3) #find faces and store in dict
print('Done!')


#Contact sheet printer for faces found in each image
def print_faces(i):
    face_size = 100
    image = imgs_data_file[i]['image']
    faces = imgs_data_file[i]['faces']
    # print them 5 across
    contact_sht = Image.new(imgs_data_file[i]['image'].mode,(face_size*5,face_size*(len(faces)//5+1)))
    
    cx,cy = 0,0  #(cx, cy) = location of face on the contact sheet
    
    for x,y,w,h in faces:
        face = image.crop((x,y,x+w,y+h))
        if face.width > face_size:
            face.thumbnail((face_size,face_size))
        contact_sht.paste(face,(cx,cy))
        if cx+face_size == contact_sht.width:
            cx = 0
            cy += face_size
        else:
            cx += face_size
    
    return contact_sht

# Search for specified text in each image.
# If the text is found, print the contact sheet of faces for that image
search_text = 'Mark'

for i in range(num_images):
    if search_text in imgs_data_file[i]['text']:
        print("Results found in file {}".format(imgs_data_file[i]['name']))
        if len(imgs_data_file[i]['faces']) == 0:
            print('But there were no faces in that file!')
        else:
            display(print_faces(i))

