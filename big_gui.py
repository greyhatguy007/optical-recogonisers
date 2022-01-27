import tkinter as tk
import cv2
from cv2 import cv2


def facedetect():
    import cv2
    from cv2 import cv2

    face_Cascade = cv2.CascadeClassifier("/home/ritvik/.workspace/opencv-master/data/haarcascades_cuda/haarcascade_frontalface_default.xml")
    #eye_Cascade = cv2.CascadeClassifier("/home/ritvik/.workspace/opencv-master/data/haarcascades_cuda/haarcascade_eye.xml")
    #body_Cascade = cv2.CascadeClassifier("/home/ritvik/.workspace/opencv-master/data/haarcascades_cuda/haarcascade_lowerbody.xml")
    #smile_Cascade = cv2.CascadeClassifier("/home/ritvik/.workspace/opencv-master/data/haarcascades_cuda/haarcascade_smile.xml")
    cap = cv2.VideoCapture(0)
    cap.set(10, 100)
    while cap.isOpened():
        _ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_Cascade.detectMultiScale(gray, 1.1, 4)

        for (x,y,w,h) in faces:

            #roiface_gray = gray[y:y+h, x:x+h]

            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2) 
            cv2.putText(frame, "FACE", (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2, 3)
        cv2.imshow("face detect", frame)
        if cv2.waitKey(40) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def vehicledetect():
    import cv2
    from cv2 import cv2
    import numpy as np
    from time import sleep

    min1=80 
    min2=80 

    offset=6 

    position1=550 

    delay= 60 

    detection1 = []
    var_bla= 0

    
    def function_to_find_pos(x, y, w, h):
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        return cx,cy

    cap = cv2.VideoCapture('/home/ritvik/.workspace/proj/project_final/video.mp4')
    subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()

    while True:
        _ret , frame1 = cap.read()
        tempo = float(1/delay)
        sleep(tempo) 
        grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey,(3,3),5)
        img_sub = subtracao.apply(blur)
        dilat = cv2.dilate(img_sub,np.ones((5,5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilatada = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
        dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)
        contorno,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        cv2.line(frame1, (25, position1), (1200, position1), (255,127,0), 3) 
        for(_i,c) in enumerate(contorno):
            (x,y,w,h) = cv2.boundingRect(c)
            validar_contorno = (w >= min1) and (h >= min2)
            if not validar_contorno:
                continue

            cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
            centro = function_to_find_pos(x, y, w, h)
            detection1.append(centro)
            cv2.circle(frame1, centro, 4, (0, 0,255), -1)

            for (x,y) in detection1:
                if y<(position1+offset) and y>(position1-offset):
                    var_bla+=1
                    cv2.line(frame1, (25, position1), (1200, position1), (0,127,255), 3)  
                    detection1.remove((x,y))     

        cv2.putText(frame1, "VEHICLE COUNT : "+str(var_bla), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
        cv2.imshow("Video Original" , frame1)

        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
    cap.release()

def show_this_code():
    text="""
     import tkinter as tk


    def facedetect():
        import cv2
        from cv2 import cv2

        face_Cascade = cv2.CascadeClassifier("/home/ritvik/.workspace/opencv-master/data/haarcascades_cuda/haarcascade_frontalface_default.xml")
        #eye_Cascade = cv2.CascadeClassifier("/home/ritvik/.workspace/opencv-master/data/haarcascades_cuda/haarcascade_eye.xml")
        #body_Cascade = cv2.CascadeClassifier("/home/ritvik/.workspace/opencv-master/data/haarcascades_cuda/haarcascade_lowerbody.xml")
        #smile_Cascade = cv2.CascadeClassifier("/home/ritvik/.workspace/opencv-master/data/haarcascades_cuda/haarcascade_smile.xml")
        cap = cv2.VideoCapture(0)
        cap.set(10, 100)
        while cap.isOpened():
            _ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_Cascade.detectMultiScale(gray, 1.1, 4)

            for (x,y,w,h) in faces:

                #roiface_gray = gray[y:y+h, x:x+h]

                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2) 
                cv2.putText(frame, "FACE", (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2, 3)
            cv2.imshow("face detect", frame)
            if cv2.waitKey(40) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def vehicledetect():
        import cv2
        from cv2 import cv2
        import numpy as np
        from time import sleep

        min1=80 
        min2=80 

        offset=6 

        position1=550 

        delay= 60 

        detection1 = []
        var_bla= 0


        def function_to_find_pos(x, y, w, h):
            x1 = int(w / 2)
            y1 = int(h / 2)
            cx = x + x1
            cy = y + y1
            return cx,cy

        cap = cv2.VideoCapture('/home/ritvik/.workspace/proj/project_final/video.mp4')
        subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()

        while True:
            _ret , frame1 = cap.read()
            tempo = float(1/delay)
            sleep(tempo) 
            grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(grey,(3,3),5)
            img_sub = subtracao.apply(blur)
            dilat = cv2.dilate(img_sub,np.ones((5,5)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilatada = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
            dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)
            contorno,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            cv2.line(frame1, (25, position1), (1200, position1), (255,127,0), 3) 
            for(_i,c) in enumerate(contorno):
                (x,y,w,h) = cv2.boundingRect(c)
                validar_contorno = (w >= min1) and (h >= min2)
                if not validar_contorno:
                    continue

                cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
                centro = function_to_find_pos(x, y, w, h)
                detection1.append(centro)
                cv2.circle(frame1, centro, 4, (0, 0,255), -1)

                for (x,y) in detection1:
                    if y<(position1+offset) and y>(position1-offset):
                        var_bla+=1
                        cv2.line(frame1, (25, position1), (1200, position1), (0,127,255), 3)  
                        detection1.remove((x,y))     

            cv2.putText(frame1, "VEHICLE COUNT : "+str(var_bla), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
            cv2.imshow("Video Original" , frame1)

            if cv2.waitKey(1) == 27:
                break
            
        cv2.destroyAllWindows()
        cap.release()

    def show_this_code():
        text=""

    
        root = tk.Tk()
        T = tk.Text(root, height=100, width=150)
        T.pack()
        T.insert(tk.END, text)
        tk.mainloop()


    def cctv_capture():
        import cv2
        from cv2 import cv2
        import numpy as np

        cam = "/home/ritvik/.workspace/opencv-master/samples/data/vtest.avi"
        cap = cv2.VideoCapture(cam)
        cap.set(10, 20)

        _ret, frame1 = cap.read()
        _ret, frame2 = cap.read()
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter("/home/ritvik/workspace/detect.avi", fourcc, 60.0, (720,720)) 
        while cap.isOpened():
            #ret, frame = cap.read()

            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _,thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                #(x1 ,y1, w1, h1) = cv2.boundingRect(contour-1)
                (x ,y, w, h) = cv2.boundingRect(contour)
                #area = str(cv2.contourArea(contour))\
                #print(area)
                if cv2.contourArea(contour) < 500:
                    continue
                #if abs(x-x1) < 20000:
                #    cv2.putText(frame1, "violation", (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255),1)
                cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255.0), 2)
                cv2.putText(frame1, "Status: {}".format("Detected Moment !!!"), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.putText(frame1, "HUMAN DETECTED", (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)
                #if cv2.contourArea(contour) > 250 and cv2.contourArea(contour) < 600:
                #    cv2.putText(frame1,"VEHICLE",(x,y) ,cv2.FONT_HERSHEY_SIMPLEX ,0.5, (0,255,0), 2)
            #cv2.drawContours(frame1, contours, -1, (0,255,0), 2)
                out.write(frame2)

            cv2.imshow("feed", frame1)
            frame1 = frame2
            _ret, frame2 = cap.read()
            if cv2.waitKey(40) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    import time
    tim = time.strftime("%H:%M:%S %p")

    def mask_detect(cam):
        import cv2
        from cv2 import cv2
        import numpy as np
        from keras.models import load_model
        model=load_model("/home/ritvik/.workspace/proj/project_final/mask.model")
        #model=load_model("/home/ritvik/.workspace/model2-003.model")

        results={0:'without mask',1:'mask'}
        GR_dict={0:(0,0,255),1:(0,255,0)}

        rect_size = 4
        cap = cv2.VideoCapture(cam)


        haarcascade = cv2.CascadeClassifier("/home/ritvik/.workspace/opencv-master/data/haarcascades_cuda/haarcascade_frontalface_default.xml")

        while True:
            (_ret, im) = cap.read()
            rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
            faces = haarcascade.detectMultiScale(rerect_size)
            for f in faces:
                (x, y, w, h) = [v * rect_size for v in f] 

                face_img = im[y:y+h, x:x+w]
                rerect_sized=cv2.resize(face_img,(150,150))
                normalized=rerect_sized/255.0
                reshaped=np.reshape(normalized,(1,150,150,3))
                reshaped = np.vstack([reshaped])
                result=model.predict(reshaped)


                label=np.argmax(result,axis=1)[0]

                cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
                cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
                cv2.putText(im, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                cv2.putText(im, "PLEASE WEAR BLUE MASK FOR BEST RESULT", (4,73), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)


            #cv2.imshow('LIVE1', blur)
            cv2.imshow('LIVE', im)
            key = cv2.waitKey(10)

            if key == 27: 
                break

        cap.release()
        cv2.destroyAllWindows()

    def time1():
        import time
        global tim
        tim = time.strftime("%H:%M:%S %p")
        tim = str(tim)
        return tim

    def nothing():
        pass

    def showcode_model():
        text=""
        from keras.optimizers import RMSprop
        from keras.preprocessing.image import ImageDataGenerator
        import cv2
        from keras.models import Sequential
        from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
        from keras.models import Model, load_model
        from keras.callbacks import TensorBoard, ModelCheckpoint
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score
        from sklearn.utils import shuffle
        import tensorflow as tf
        import imutils
        import numpy as np

        model = Sequential([
            Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D(2,2),
            Conv2D(100, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dropout(0.5),
            Dense(50, activation='relu'),
            Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        TRAINING_DIR = "/home/ritvik/.workspace/proj/project_final/face-mask-dataset/Dataset/train"
        train_datagen = ImageDataGenerator(rescale=1.0/255,
                                           rotation_range=40,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           fill_mode='nearest')
        train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
                                                            batch_size=10, 
                                                            target_size=(150, 150))
        VALIDATION_DIR = "/home/ritvik/.workspace/proj/project_final/face-mask-dataset/Dataset/test"
        validation_datagen = ImageDataGenerator(rescale=1.0/255)
        validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
                                                                 batch_size=10, 
                                                                 target_size=(150, 150))
        checkpoint = ModelCheckpoint('model2-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
        history = model.fit_generator(train_generator,
                                      epochs=10,
                                      validation_data=validation_generator,
                                      callbacks=[checkpoint])
        ""
        root = tk.Tk()
        T = tk.Text(root, height=100, width=150)
        T.pack()
        T.insert(tk.END, text)

        #my_button5 = tk.Button(my_label, text="Exit",command=root.quit, bg="blue",font=("Algerian" ,15), padx=40 , pady=10 , borderwidth=7)
        #my_button5.grid(row=0, column=3, padx=10 , pady=250 )
        #my_button5.place(x=880,y=900)

        tk.mainloop()


    root = tk.Tk()
    root.title("PYTHON PROJECT")
    root.geometry("1920x1080")
    bg = tk.PhotoImage(file="/home/ritvik/Pictures/wp1913251.png")
    my_label = tk.Label(root, image=bg)
    my_label.place(x=0, y=0, relwidth=1, relheight=1)

    my_text = tk.Label(root, text="😎Optical Recognisers😎", font=("Jocker", 50), fg="blue",bg="BLACK")
    my_text.pack(pady=50)

    my_button2 = tk.Button(my_label, text="Face Detector" ,command=facedetect , bg="blue", font=("Algerian" ,15) , padx=40 , pady=10 , borderwidth=7)
    my_button2.grid(row=0, column=1, padx=10 , pady=250)
    my_button2.place(x=580,y=170)

    my_button3 = tk.Button(my_label, text="Human Detection for Monitoring" ,command=cctv_capture ,bg="blue", font=("Algerian" ,15), padx=40 , pady=10 , borderwidth=7)
    my_button3.grid(row=0,column=5,  padx=10 , pady=250)
    my_button3.place(x=50,y=170)

    my_button4 = tk.Button(my_label, text="Vehicle Detect" ,command=vehicledetect ,bg="blue", font=("Algerian" ,15), padx=40 , pady=10 , borderwidth=7)
    my_button4.grid(row=0,column=5,  padx=10 , pady=250)
    my_button4.place(x=1400,y=170)

    my_button4 = tk.Button(my_label, text="Mask Detector" ,command=nothing ,bg="blue", font=("Algerian" ,15), padx=40 , pady=10 , borderwidth=7)
    my_button4.grid(row=0,column=5,  padx=10 , pady=250)
    my_button4.place(x=1100,y=170)

    my_button5 = tk.Button(my_label, text="Exit",command=root.quit, bg="blue",font=("Algerian" ,15), padx=40 , pady=10 , borderwidth=7)
    my_button5.grid(row=0, column=3, padx=10 , pady=250 )
    my_button5.place(x=880,y=900)

    my_button6 = tk.Button(my_label, text="Show Trainer Code",command=showcode_model, bg="blue",font=("Algerian" ,15), padx=40 , pady=10 , borderwidth=7)
    my_button6.grid(row=0, column=3, padx=10 , pady=250 )
    my_button6.place(x=300,y=500)

    my_button7 = tk.Button(my_label, text="Show This Code",command=showcode_model, bg="blue",font=("Algerian" ,15), padx=40 , pady=10 , borderwidth=7)
    my_button7.grid(row=0, column=3, padx=10 , pady=250 )
    my_button7.place(x=1300,y=500)

    my_text2 = tk.Label(root, text='👽', font=("Jocker", 50), fg="blue")
    my_text2.place(x=928, y=500)

    #my_text1 = tk.Label(root, text='HELLO WORLD', font=("Jocker", 50), fg="blue")
    #my_text1.place(x=600, y=600)

    root.mainloop()
    """
    root = tk.Tk()
    T = tk.Text(root, height=100, width=150)
    T.pack()
    T.insert(tk.END, text)
    tk.mainloop()
    exit()
        
    


def cctv_capture():
    import cv2
    from cv2 import cv2
    import numpy as np

    cam = "/home/ritvik/.workspace/opencv-master/samples/data/vtest.avi"
    cap = cv2.VideoCapture(cam)
    cap.set(10, 20)

    _ret, frame1 = cap.read()
    _ret, frame2 = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("/home/ritvik/workspace/detect.avi", fourcc, 60.0, (720,720)) 
    while cap.isOpened():
        #ret, frame = cap.read()

        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _,thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        for contour in contours:
            #(x1 ,y1, w1, h1) = cv2.boundingRect(contour-1)
            (x ,y, w, h) = cv2.boundingRect(contour)
            #area = str(cv2.contourArea(contour))\
            #print(area)
            if cv2.contourArea(contour) < 500:
                continue
            #if abs(x-x1) < 20000:
            #    cv2.putText(frame1, "violation", (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255),1)
            cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255.0), 2)
            cv2.putText(frame1, "Status: {}".format("Detected Moment !!!"), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame1, "HUMAN DETECTED", (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)
            #if cv2.contourArea(contour) > 250 and cv2.contourArea(contour) < 600:
            #    cv2.putText(frame1,"VEHICLE",(x,y) ,cv2.FONT_HERSHEY_SIMPLEX ,0.5, (0,255,0), 2)
        #cv2.drawContours(frame1, contours, -1, (0,255,0), 2)
            out.write(frame2)

        cv2.imshow("feed", frame1)
        frame1 = frame2
        _ret, frame2 = cap.read()
        if cv2.waitKey(40) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

import time
tim = time.strftime("%H:%M:%S %p")

def mask_detect(cam):
    import cv2
    from cv2 import cv2
    import numpy as np
    from keras.models import load_model
    model=load_model("/home/ritvik/.workspace/proj/project_final/mask.model")
    #model=load_model("/home/ritvik/.workspace/model2-003.model")

    results={0:'without mask',1:'mask'}
    GR_dict={0:(0,0,255),1:(0,255,0)}

    rect_size = 4
    cap = cv2.VideoCapture(cam)


    haarcascade = cv2.CascadeClassifier("/home/ritvik/.workspace/opencv-master/data/haarcascades_cuda/haarcascade_frontalface_default.xml")

    while True:
        (_ret, im) = cap.read()
        rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
        faces = haarcascade.detectMultiScale(rerect_size)
        for f in faces:
            (x, y, w, h) = [v * rect_size for v in f] 

            face_img = im[y:y+h, x:x+w]
            rerect_sized=cv2.resize(face_img,(150,150))
            normalized=rerect_sized/255.0
            reshaped=np.reshape(normalized,(1,150,150,3))
            reshaped = np.vstack([reshaped])
            result=model.predict(reshaped)


            label=np.argmax(result,axis=1)[0]

            cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
            cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
            cv2.putText(im, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            cv2.putText(im, "PLEASE WEAR BLUE MASK FOR BEST RESULT", (4,73), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)


        #cv2.imshow('LIVE1', blur)
        cv2.imshow('LIVE', im)
        key = cv2.waitKey(10)

        if key == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

def time1():
    import time
    global tim
    tim = time.strftime("%H:%M:%S %p")
    tim = str(tim)
    return tim

def nothing():
    pass

def showcode_model():
    text="""
    from keras.optimizers import RMSprop
    from keras.preprocessing.image import ImageDataGenerator
    import cv2
    from keras.models import Sequential
    from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
    from keras.models import Model, load_model
    from keras.callbacks import TensorBoard, ModelCheckpoint
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from sklearn.utils import shuffle
    import tensorflow as tf
    import imutils
    import numpy as np

    model = Sequential([
        Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2,2),
        Conv2D(100, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dropout(0.5),
        Dense(50, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    TRAINING_DIR = "/home/ritvik/.workspace/proj/project_final/face-mask-dataset/Dataset/train"
    train_datagen = ImageDataGenerator(rescale=1.0/255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
                                                        batch_size=10, 
                                                        target_size=(150, 150))
    VALIDATION_DIR = "/home/ritvik/.workspace/proj/project_final/face-mask-dataset/Dataset/test"
    validation_datagen = ImageDataGenerator(rescale=1.0/255)
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
                                                             batch_size=10, 
                                                             target_size=(150, 150))
    checkpoint = ModelCheckpoint('model2-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
    history = model.fit_generator(train_generator,
                                  epochs=10,
                                  validation_data=validation_generator,
                                  callbacks=[checkpoint])
    """
    root = tk.Tk()
    T = tk.Text(root, height=100, width=150)
    T.pack()
    T.insert(tk.END, text)
    
    #my_button5 = tk.Button(my_label, text="Exit",command=root.quit, bg="blue",font=("Algerian" ,15), padx=40 , pady=10 , borderwidth=7)
    #my_button5.grid(row=0, column=3, padx=10 , pady=250 )
    #my_button5.place(x=880,y=900)
    
    tk.mainloop()
    exit()
    

root = tk.Tk()
root.title("PYTHON PROJECT")
root.geometry("1920x1080")
bg = tk.PhotoImage(file="/home/ritvik/Pictures/wp1913251.png")
my_label = tk.Label(root, image=bg)
my_label.place(x=0, y=0, relwidth=1, relheight=1)

my_text = tk.Label(root, text="😎Optical Recognisers😎", font=("Jocker", 50), fg="blue",bg="BLACK")
my_text.pack(pady=50)

my_button2 = tk.Button(my_label, text="Face Detector" ,command=facedetect , bg="blue", font=("Algerian" ,15) , padx=40 , pady=10 , borderwidth=7)
my_button2.grid(row=0, column=1, padx=10 , pady=250)
my_button2.place(x=580,y=170)

my_button3 = tk.Button(my_label, text="Human Detection for Monitoring" ,command=cctv_capture ,bg="blue", font=("Algerian" ,15), padx=40 , pady=10 , borderwidth=7)
my_button3.grid(row=0,column=5,  padx=10 , pady=250)
my_button3.place(x=50,y=170)

my_button4 = tk.Button(my_label, text="Vehicle Detect" ,command=vehicledetect ,bg="blue", font=("Algerian" ,15), padx=40 , pady=10 , borderwidth=7)
my_button4.grid(row=0,column=5,  padx=10 , pady=250)
my_button4.place(x=1400,y=170)

my_button4 = tk.Button(my_label, text="Mask Detector" ,command=nothing ,bg="blue", font=("Algerian" ,15), padx=40 , pady=10 , borderwidth=7)
my_button4.grid(row=0,column=5,  padx=10 , pady=250)
my_button4.place(x=1100,y=170)

my_button5 = tk.Button(my_label, text="Exit",command=root.quit, bg="blue",font=("Algerian" ,15), padx=40 , pady=10 , borderwidth=7)
my_button5.grid(row=0, column=3, padx=10 , pady=250 )
my_button5.place(x=880,y=900)

my_button6 = tk.Button(my_label, text="Show Trainer Code",command=showcode_model, bg="blue",font=("Algerian" ,15), padx=40 , pady=10 , borderwidth=7)
my_button6.grid(row=0, column=3, padx=10 , pady=250 )
my_button6.place(x=300,y=500)

my_button7 = tk.Button(my_label, text="Show This Code",command=show_this_code, bg="blue",font=("Algerian" ,15), padx=40 , pady=10 , borderwidth=7)
my_button7.grid(row=0, column=3, padx=10 , pady=250 )
my_button7.place(x=1300,y=500)

my_text2 = tk.Label(root, text='👽', font=("Jocker", 50), fg="blue")
my_text2.place(x=928, y=500)

#my_text1 = tk.Label(root, text='HELLO WORLD', font=("Jocker", 50), fg="blue")
#my_text1.place(x=600, y=600)

root.mainloop()
exit()