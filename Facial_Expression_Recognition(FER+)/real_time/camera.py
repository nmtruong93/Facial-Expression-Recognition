import cv2
from real_time.predictor import FacialExpressionModel
from mtcnn.mtcnn import MTCNN
from real_time.image_utils import compute_norm_mat, preproc_img

rgb = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX


def start_app(cnn):
    ix = 0
    detector = MTCNN(min_face_size=50)
    while True:
        ix += 1

        _, fr = rgb.read()
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = detector.detect_faces(fr)
        for face in faces:
            box = face["box"]
            x,y,w,h = box
            fc = gray[y:y + h, x:x + w]
            try:
                A, A_pinv = compute_norm_mat(w, h)
                image = preproc_img(fc, A, A_pinv)
                roi = cv2.resize(image, (64, 64))
                roi = roi.reshape((1, 64, 64, 1))
                pred = cnn.predict_emotion(roi)
            except:
                continue
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('Filter', fr)
        print(fr)
    rgb.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = FacialExpressionModel("face_model.json", "face_model.h5")
    start_app(model)
