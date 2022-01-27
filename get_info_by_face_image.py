import asyncio
import scipy.spatial.distance as distance
from PIL import Image
import aiosqlite
import models
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import io
import torchvision.transforms as T
import numpy as np
import tensorflow as tf
import cv2
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
transform = T.Compose([
    T.Resize([244, 244]),
    T.ToTensor()
])


async def prepare_db(filename):
    db = await aiosqlite.connect(filename)
    await db.execute('CREATE TABLE IF NOT EXISTS images (owner_id INTEGER, date INTEGER, post_id INTEGER, vector BLOB, url TEXT);')
    await db.commit()
    return db


def get_net(path):
    model = models.Net_50()
    model.pretrain(embad=50, new_fc=2)
    model.load_state_dict(torch.load(path))
    return model


async def get_data(db):
    cursor = await db.execute('SELECT * FROM images')
    _data = await cursor.fetchall()
    data = []
    for d in _data:
        d = list(d)
        d[3] = np.load(io.BytesIO(d[3]))['arr_0']
        data.append(d)
    return data


def vizualize(data):
    writer = SummaryWriter()
    meta = [d[0] for d in data]
    vectors = np.array([d[3] for d in data])
    writer.add_embedding(vectors, metadata=meta)
    writer.close()


def compare(filename, data):
    model = get_net(r"32_35.model")
    model.eval()

    d = 0.25
    img = cv2.imread(filename)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h1, w1, _ = img.shape
    detected_faces = list(face_cascade.detectMultiScale(grayscale_image))

    if not detected_faces:
        print('Лиц на данной фотографии не обнаружено.')
        return None

    (column, row, width, height) = detected_faces[0]

    # Отступы
    dx = int(d*width)
    dy = int(d*height)
    # Обрезка
    face_image = img[max(row-dy, 0):min(row + height+dy, h1),
                     max(column-dx, 0):min(column+width+dx, w1), ::-1]
    img = Image.fromarray(face_image)

    img = transform(img)
    batch = img.view(img.size(0), -1)
    img = T.Normalize(batch.mean(1), batch.var(1))(img)
    face = model(img.unsqueeze(0))
    d1 = distance.cdist([i[3] for i in data], face.detach().numpy(), "cosine")
    sum_res = [(data[n], i[0]) for n, i in enumerate(d1)]
    sum_res = sorted(sum_res, key=lambda x: x[1])
    account_id = sum_res[0][0][0]
    photo_url = sum_res[0][0][4]
    photo_id = sum_res[0][0][2]
    cos = sum_res[0][1]
    text = f'''
    Индификатор Аккаунта - {account_id}
    Индификатор Поста - {photo_id}
    Косинусная мера - {cos}
    Ссылка на изображение - {photo_url}
    '''

    print(text)


async def main(filename):
    db = await prepare_db('data.db')
    data = await get_data(db)
    vizualize(data)
    compare(filename, data)
    await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN кодировщик')
    parser.add_argument("--filename", default='test.webp',
                        help="Путь до изображения")
    args = parser.parse_args()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args.filename))
