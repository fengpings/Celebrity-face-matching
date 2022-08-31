import cv2
import numpy as np
import insightface
import pickle
from collections import OrderedDict
import time
from functools import reduce
import matplotlib.pyplot as plt
from dbface_detect_align_module import dbface_detect as dbface
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
import sys

def performance(f):
    def fn(*args, **kw):
        t1 = time.time()
        r = f(*args, **kw)
        t2 = time.time()
        print('call {} in {}s'.format(f.__name__,(t2-t1)))
        return r
    return fn


def compute_cos_sim(emb1, emb2):
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return sim


@performance
def topk_sim(img_emb, mean_emb_dict, k):
    sims = []
    for celeb in mean_emb_dict:
        sims.append((celeb, compute_cos_sim(img_emb, mean_emb_dict[celeb])))
    
    sorted_sims = sorted(sims, key=lambda x:x[1], reverse=True)
    return sorted_sims[:k]


def softmax(res):
    def f1(x):
        return np.exp(x[1])

    def f2(x, y):
        return x + y

    total = reduce(f2, map(f1, res))
    sims = []

    for item in res:
        sims.append((item[0], f1(item)/total))

    return sims

def knn_sim(res):
    total = 0
    for item in res:
        total += item[1]
    for i in range(len(res)):
        res[i] = (res[i][0], res[i][1]/total)
        
        
if __name__ == '__main__':
    pins = './research_data/105_classes_pins_dataset/'
    with open('./embeddings/w600k_r50.onnx_aligned_img_dict.pkl', 'rb') as f:
        image_dict = pickle.load(f)
    
    print('='*20 + 'Pins Face Recognition images loaded' + '='*20)
    
    with open('./embeddings/w600k_r50.onnx_aligned_emb_dict.pkl', 'rb') as f:
        emb_dict = pickle.load(f)
    
    print('='*20 + 'Pins Face Recognition embeddings loaded' + '='*20)
    
    model = 'w600k_r50.onnx'

    handler = insightface.model_zoo.get_model(model,
                                              providers=['CPUExecutionProvider'])
    handler.prepare(ctx_id=0)
    
    print('='*20 + 'ArcFace-ResNet50 loaded' + '='*20)
    
    dbface_detect = dbface(net_type='dbface',device='cpu', align=True)
    print('='*20 + 'DBFace loaded' + '='*20)

    X = []
    ori_y = []
    y = []

    for celeb in emb_dict:
        embs = emb_dict[celeb]
        for emb in embs:
            X.append(emb)
            ori_y.append(celeb)

    le = LabelEncoder()
    y = le.fit_transform(ori_y)

    X = normalize(X)
    knn = KNN(n_neighbors=100,algorithm='brute')
    knn.fit(X,y)
    print('='*20 + 'KNN-Cosine Distance fitted' + '='*20)
    while True:
        test_img = input('Please enter the path of the image you want to test: ')
        image = cv2.imread(test_img)
        if image is None:
            print('image does not exist!')
            continue

        print('='*20 + 'Test image loded' + '='*20)

        print('='*20 + 'Matching starts' + '='*20)

        st = time.time()
        _, aligned_face = dbface_detect.detect(image)

        if len(aligned_face) == 0:
            sys.exit('faces are not detected in the input image!')
        else:
            aligned = aligned_face[0]

        face_embedding = handler.get(aligned)

        face_embedding = normalize(face_embedding.reshape((1,face_embedding.shape[0])))
        preds = knn.predict_proba(face_embedding)

        predictions = []
        count = 0
        for pred in preds[0]:
            predictions.append((le.classes_[count],pred))
            count += 1
        sorted_preds = sorted(predictions, key=lambda x:x[1], reverse=True)

        sims = sorted_preds[:3]
        knn_sim(sims)
        et = time.time()

        print('='*20 + 'Matching ends' + '='*20)
        print(sims)
        print('time used: {}s'.format(et-st))

        img_paths = OrderedDict()

        print('='*20 + 'Display most similar celebrities' + '='*20)
        for celeb in sims:
            max_sim = -1
            max_img = None

            for img, emb in image_dict[celeb[0]]:
                emb = normalize(emb.reshape((1,-1)))
                sim = compute_cos_sim(emb.reshape((-1,)), face_embedding.reshape((-1,)))
                if sim > max_sim:
                    max_sim = sim
                    max_img = img

            img_paths[celeb[0]] = pins + celeb[0] + '/' + max_img

        imgs = list(img_paths.values())
        celebs = list(img_paths.keys())

        plt.rcParams['figure.figsize'] = (18, 18)

        plt.subplot(232)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        plt.imshow(image)
        plt.xlabel('You are: ')
        plt.text(0, 0, 'You', size=30, c='red')
        plt.tight_layout(h_pad=0, w_pad=0)
        plt.axis('off')

        plt.subplot(234)
        img1 = cv2.imread(imgs[0])
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        plt.imshow(img1)
        # plt.xlabel('{}% like {}'.format(round(sims[0][1]*100), sims[0][0].split('_')[1]))
        plt.tight_layout(h_pad=0, w_pad=1)
        plt.text(0, 0, '{}% {}'.format(round(sims[0][1]*100, 2), sims[0][0].split('_')[1]), size=30, c='red')
        plt.axis('off')

        plt.subplot(235)
        img2 = cv2.imread(imgs[1])
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        plt.imshow(img2)
        # plt.xlabel('{}% like {}'.format(round(sims[1][1]*100), sims[1][0].split('_')[1]))
        # plt.tight_layout(h_pad=0, w_pad=1)
        plt.text(0, 0, '{}% {}'.format(round(sims[1][1]*100, 2), sims[1][0].split('_')[1]), size=30, c='red')
        plt.axis('off')

        plt.subplot(236)
        img3 = cv2.imread(imgs[2])
        img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
        plt.imshow(img3)
        # plt.xlabel('{}% like {}'.format(round(sims[2][1]*100), sims[2][0].split('_')[1]))
        plt.tight_layout(h_pad=0, w_pad=1)
        plt.text(0, 0, '{}% {}'.format(round(sims[2][1]*100, 2), sims[2][0].split('_')[1]), size=30, c='red')
        plt.axis('off')

        plt.show()
        
        another = input('Test another image? (yes/no):')
        if another == 'no':
            print('Bye!')
            break