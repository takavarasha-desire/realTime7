import cv2
import time
from keras.models import model_from_json


emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
cascPath = "./haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

# load json and create model arch
json_file = open('./models/fer2013_Model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('./models/fer2013_Model_weights.keras')


def predict_emotion(face_image_gray):  # a single cropped face
    resized_img = cv2.resize(face_image_gray, (48, 48), interpolation=cv2.INTER_AREA)
    image = resized_img.reshape(1, 48, 48)
    list_of_list = model.predict(image, batch_size=1, verbose=1)
    anger, disgust, fear, joy, neutral, sadness, surprise = [prob for lst in list_of_list for prob in lst]
    return [anger, disgust, fear, joy, neutral, sadness, surprise]


video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, 1)

    faces = faceCascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        face_image_gray = img_gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Predict emotion probabilities
        emotion_probs = predict_emotion(face_image_gray)
        predicted_emotion = emotion_labels[emotion_probs.index(max(emotion_probs))]

        # Embed predicted emotion label on the frame
        label_text = f'Emotion: {predicted_emotion}'
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        with open('emotion.txt', 'a') as f:
            f.write('{},{},{},{},{},{},{}\n'.format(time.time(), *emotion_probs))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
