#cv2 handles video capture, image processing and displays the results
import cv2
#to perform multiple functions at the same time, real time updating the face verification
import threading
#does the actual face recognition tasks
from deepface import DeepFace
#To handle system operations
import os

#This function was created to check if the necessary model files for face verifications are downloaded and available

def ensure_model_files():

    #this sets the variable weights_dir to the path were DeepFace model weights should be stored.
    #Furthermore, os.path.expanduser(), expandes the ~ to the user's current home directory
    weights_dir = os.path.expanduser("~/.deepface/weights/")
    #checks if the weights_dir directory exists, if not creates it
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    #this is a list of models that are required for face verification
    model_files = ['VGG-Face', 'Facenet', 'OpenFace']
    for file in model_files:
        #creates a file path to where of the models should be stored
        #the .join combines the path of weights_dir and file, by adding file in the end
        file_path = os.path.join(weights_dir, file)
        #To check if this path already exists there, don't want to re download each time
        if not os.path.exists(file_path):
            try:
                print(f"Downloading {file}...")
                #The build_model method is called to download and built the models
                #simply have it as file because the model names that are provided are already in the right format
                DeepFace.build_model(file) 
            #This is to catch any exceptions and print what the error was
            except Exception as e:
                print(f"Error downloading {file}: {e}")

# Ensure model files are available before starting the main loop
ensure_model_files()

#This line opens a connection to the webcam, and allows operations to capture frames from it.
#the 0 indicates to the first available camera, and CAP_DSHOW is DirectShow backend used on Windows operating systems.
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

#Sets the corresponding width and height of video capture frame to 640 and 480 respectively
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

#A counter to keep track on how many frames
counter = 0

matched_user_id = None

def preprocess_image(image_path):
    # Read image in BGR format
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Failed to load image from path: {image_path}")
    
    # Resize image to 160x160 pixels
    image_resize = cv2.resize(image,(160,160))

    return image

def preprocess_frame(frame):
    #Resizes the input image to 160x160 pixels
    frame_resize = cv2.resize(frame,(160,160))
    #Converts the BGR to YUV color format
    yuv = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2YUV)
    #Applies histogram equalization, because increases contrast by distributing brightness evenly
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    #Converts the YUV back to BGR format, since face recognition models usually expects BGR format.
    frame_equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return frame_equalized

#Acts as the database for matching
dic_ref = {"Alen": [preprocess_image("C:\\Users\\AlenC\\OneDrive\\Skrivbord\\Learning\\MachineLearning\\Python\\FaceRecognition\\AlenPicture.jpg"),
                    preprocess_image("C:\\Users\\AlenC\\OneDrive\\Skrivbord\\Learning\\MachineLearning\\Python\\FaceRecognition\\AlenPicture2.jpg")],

            "Emil": [preprocess_image("C:\\Users\\AlenC\\OneDrive\\Skrivbord\\Learning\\MachineLearning\\Python\\FaceRecognition\\emilPicture1.jpg")],

            "Amel": [preprocess_image("C:\\Users\\AlenC\\OneDrive\\Skrivbord\\Learning\\MachineLearning\\Python\\FaceRecognition\\baboPicture1.jpg"),
                     preprocess_image("C:\\Users\\AlenC\\OneDrive\\Skrivbord\\Learning\\MachineLearning\\Python\\FaceRecognition\\baboPicture2.jpg"),
                     preprocess_image("C:\\Users\\AlenC\\OneDrive\\Skrivbord\\Learning\\MachineLearning\\Python\\FaceRecognition\\baboPicture3.jpg"),
                     preprocess_image("C:\\Users\\AlenC\\OneDrive\\Skrivbord\\Learning\\MachineLearning\\Python\\FaceRecognition\\baboPicture4.jpg")],
                     
            "Mevlida": [preprocess_image("C:\\Users\\AlenC\\OneDrive\\Skrivbord\\Learning\\MachineLearning\\Python\\FaceRecognition\\mamaPicture1.jpg"),
                     preprocess_image("C:\\Users\\AlenC\\OneDrive\\Skrivbord\\Learning\\MachineLearning\\Python\\FaceRecognition\\mamaPicture2.jpg"),
                     preprocess_image("C:\\Users\\AlenC\\OneDrive\\Skrivbord\\Learning\\MachineLearning\\Python\\FaceRecognition\\mamaPicture3.jpg"),
                     preprocess_image("C:\\Users\\AlenC\\OneDrive\\Skrivbord\\Learning\\MachineLearning\\Python\\FaceRecognition\\mamaPicture4.jpg"),
                     preprocess_image("C:\\Users\\AlenC\\OneDrive\\Skrivbord\\Learning\\MachineLearning\\Python\\FaceRecognition\\mamaPicture5.jpg"),
                     preprocess_image("C:\\Users\\AlenC\\OneDrive\\Skrivbord\\Learning\\MachineLearning\\Python\\FaceRecognition\\mamaPicture6.jpg"),
                     preprocess_image("C:\\Users\\AlenC\\OneDrive\\Skrivbord\\Learning\\MachineLearning\\Python\\FaceRecognition\\mamaPicture7.jpg"),
                     preprocess_image("C:\\Users\\AlenC\\OneDrive\\Skrivbord\\Learning\\MachineLearning\\Python\\FaceRecognition\\mamaPicture8.jpg")]} 




#only one thread can be executed at a time
lock = threading.Lock()

#function to check the given face
def check_face(frame):
    #make face_match global so it can be affected outside the function to
    global matched_user_id
    try:
        
        print(f"Start face verification")
        #only one thread at a time can execute the following code
        with lock:
            #Iterates through each user and their reference images
            for user_id,ref_imgs in dic_ref.items():
                print(f"Checks: {user_id}")
                for ref_img in ref_imgs:
                    #Compares the input image, against each reference image
                    print(f"Comparing with reference image for {user_id}")
                    result = DeepFace.verify(img1_path=frame,img2_path=ref_img,model_name="VGG-Face",distance_metric="cosine")
                    print(result)
                    #If a match is found sets the matched_user_id to the corresponding user_id   
                    if result['verified']:
                        matched_user_id = user_id
                        return
            #If no match is found, the matched_user_id is reseted
            print(f"No match found")
            matched_user_id = None
    #Catches and logs any error during the verification process
    except Exception as e:
        #If something is wrong it sets matched_user_id to None by default
        print(f"The error in face verification is: {user_id} and : {e}")
        matched_user_id = None

#infinite loop to continously do face verifications        
while True:
    #the cap.read function reutrns 2 values, first if it read the frame succesfully or not, and the actual frame data 
    ret, frame = cap.read()

    #only if the frame was succesfully read
    if ret:
        #Every 30th frame
        if counter % 30 == 0:
            #try and except incase it doesn't recognizes a face
            try:
                # the .Thread is used to create another thing that runs simultaneously to the video capture
                #and we chose the face verification to be run simultaneously
                #arguments needs to be a tuple and therefore we add another , after the copy. We also use the copy to not affect the original image.
                #Then simply start the thread
                threading.Thread(target =check_face,args = (frame.copy(),)).start()
            #We will get ValueErrors because it won't recongnize every face
            except ValueError:
                pass
        #helping to count the frames
        counter +=1

        #an if-else statement if face_match is true or false
        if matched_user_id:
            #this adds a text to the displayed video
            #img,text,org(where the text will be),font,fontscale,color,thickness
            cv2.putText(frame, f"User: {matched_user_id}", (20,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame, f"No User Found", (20,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
        #displays the current frame in a window for the user to see
        #video is the name of the window, the second argument is the image that will be displayed
        cv2.imshow("video",frame)
    #waits for an input from the keyboard, the 1 specifies the delay in miliseconds
    key = cv2.waitKey(1)
    #checks if q was pressed on the keyboard and breaks the loop if that's the case
    if key == ord("q"):
        break

#This is used to close all the windows from OpenCV effectively ending the code
cv2.destroyAllWindows()
