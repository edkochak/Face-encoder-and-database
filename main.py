import PIL
import torch.nn as nn
import torch
import cv2
import numpy as np
from PIL import Image
import models
import os
import torchvision.transforms as T
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random
import argparse
from torch.utils.tensorboard import SummaryWriter
import scipy.spatial.distance as distance
import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
def rand(): return random.randint(100, 255)


def get_net(path):
    model = models.Net_50()
    model.pretrain(embad=50, new_fc=2)
    model.load_state_dict(torch.load(path))
    return model


def main(path_to_net, filename_capture, record=False, use_tensorboard=0):
    transform = T.Compose([
        T.Resize([244, 244]),
        T.ToTensor()
    ])
    if use_tensorboard:
        tb = SummaryWriter()
    last = []
    lastimage = []
    threshold = 5.5

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml')

    d = 0.25
    N = 10

    if filename_capture == 'camera':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(filename_capture)

    if record:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(
            *'mp4v'), 25.0, (frame_width, frame_height))

    model = get_net(path_to_net)

    model.eval()
    count = 0
    meta = []
    vec1 = []

    while True:
        ret, original_image = cap.read()
        if not ret:
            break
        count += 1
        h1, w1, _ = original_image.shape

        grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        detected_faces = list(face_cascade.detectMultiScale(grayscale_image))

        imgs = torch.Tensor()
        if len(detected_faces) > 0:
            img_faces = []
            # Проходимся по всем лицам
            for faces in detected_faces:
                (column, row, width, height) = faces

                # Отступы
                dx = int(d*width)
                dy = int(d*height)
                # Обрезка
                face_image = original_image[max(row-dy, 0):min(row +
                                                               height+dy, h1), max(column-dx, 0):min(column+width+dx, w1), ::-1]
                img = Image.fromarray(face_image)
                img = transform(img)
                img_faces.append(img)
                batch = img.view(img.size(0), -1)
                img = T.Normalize(batch.mean(1), batch.var(1))(img)

                if 0 in imgs.shape:
                    # если первая фотка, то создаем пачку
                    imgs = img.unsqueeze(0)
                else:
                    # Добавляем в пачку
                    imgs = torch.cat((imgs, img.unsqueeze(0)), dim=-1)

            if 0 in imgs.shape:
                continue
            predict = model(imgs)
            dx_image = 0

            # Лица на фотке
            for n, face in enumerate(predict):
                # Тензор в numpy и добавление в Tensorboard
                face = face.detach().numpy()
                vec1.append((face, img_faces[n].numpy()))

                # Поиск ближайших соседей
                if last == []:
                    sum_res = []
                else:
                    d1 = distance.cdist([i[1] for i in last], [
                                        face],  'minkowski', p=2.)
                    sum_res = [(last[n][0], i[0])
                               for n, i in enumerate(d1)]
                    sum_res = sorted(sum_res, key=lambda x: x[1])
                    sum_res = sum_res[:N]

                # Голосование
                voices = {}
                for id, w, *_ in sum_res:
                    if id not in voices:
                        voices[id] = 0
                    voices[id] += np.e**(-((w*100)**2))

                # Если порог не пройден, то это новый человек
                if sum_res == [] or (sum_res != [] and sum_res[0][1] > threshold):
                    color = (rand(), rand(), rand())
                    last.append((len(lastimage), face))
                    (column, row, width, height) = detected_faces[n]
                    dx = int(d*width)
                    dy = int(d*height)
                    face_image = original_image[max(row-dy, 0):min(row +
                                                                   height+dy, h1), max(column-dx, 0):min(column+width+dx, w1)]
                    maxkey = len(lastimage)
                    lastimage.append((face_image, color))
                else:
                    (column, row, width, height) = detected_faces[n]
                    maxkey = max(voices, key=lambda x: voices[x])
                    face_image, color = lastimage[maxkey]
                    last.append((maxkey, face))

                meta.append(maxkey)

                # Обрезка и добавление фотки в левый вверхний угол, а также рамка
                face_image = cv2.resize(
                    face_image, (100, 100), interpolation=cv2.INTER_AREA)
                img_height, img_width, _ = face_image.shape
                original_image[0:img_height,
                               dx_image:dx_image+img_width] = face_image
                dx_image += img_width
                # расстояние до ближайщего лица
                if sum_res:
                    original_image = cv2.putText(original_image, str(round(sum_res[0][1], 5)),
                                                 (dx_image, 30),
                                                 cv2.FONT_HERSHEY_COMPLEX,
                                                 1,
                                                 (255, 255, 255),
                                                 1,
                                                 2)

                start_point, end_point = (max(
                    column-dx, 0), max(row-dy, 0)), (min(column+width+dx, w1), min(row + height+dy, h1))
                original_image = cv2.rectangle(
                    original_image, start_point, end_point, color, thickness=6)

            # Преобразуем данные, удобные Tensorboard
            if len(vec1) > 400 and use_tensorboard:
                imgs1 = [i[1] for i in vec1]
                vec1 = [i[0] for i in vec1]
                vec1 = torch.from_numpy(np.array(vec1))
                imgs1 = torch.from_numpy(np.array(imgs1))
                tb.add_embedding(vec1, label_img=imgs1,
                                 metadata=meta, global_step=count)
                meta = []
                vec1 = []
        if record:
            out.write(original_image)
        cv2.imshow("camera", original_image)
        if cv2.waitKey(1) == 27:
            break

    imgs1 = [i[1] for i in vec1]
    vec1 = [i[0] for i in vec1]
    vec1 = torch.from_numpy(np.array(vec1))
    imgs1 = torch.from_numpy(np.array(imgs1))
    if use_tensorboard:
        tb.add_embedding(vec1, label_img=imgs1,
                         metadata=meta, global_step=count)
        tb.close()

    cap.release()
    if record:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN кодировщик в прямом эфире')
    parser.add_argument("--filename", default='videoplayback_Trim.mp4',
                        help="Путь до видео")
    parser.add_argument("--tensorboard", default=0,
                        help="Добавлять ли векторы в Tensorboard. 1/0")
    args = parser.parse_args()
    main('32_35.model', args.filename, use_tensorboard=int(args.tensorboard))
