import cv2
import numpy as np

# reference facial points, a list of coordinates (x,y)
REFERENCE_FACIAL_POINTS = [
    [30.29459953, 30.29459953],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156]
]

# def face_alignment(im, kpts_locations, kpts_locations_ref, FACE_SIZE_REF=(200,200)):
#     M = cv2.estimateAffine2D(kpts_locations, kpts_locations_ref)
#     face_im_aligned = cv2.warpAffine(im, M[0], FACE_SIZE_REF)
#     return face_im_aligned

def align_process(img, bbox, landmark, image_size):
    """
    crop and align face
    Parameters:
    ----------
        img: numpy array, bgr order of shape (1, 3, n, m)
            input image
        points: numpy array, n x 10 (x1, x2 ... x5, y1, y2 ..y5)
        desired_size: default 256
        padding: default 0
    Retures:
    -------
    crop_imgs: list, n
        cropped and aligned faces
    """
    M = None
    if landmark is not None:
        assert len(image_size) == 2
        # 这个基准是112*96的面部特征点的坐标
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8
        src[:, 1] -= 8

        if image_size[0] == image_size[1] and image_size[0] != 112:
            src = src / 112 * image_size[0]

        dst = landmark.astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(dst.reshape(1,5,2), src.reshape(1,5,2))

    if M is None:
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox

        margin = 44
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:  # do align using landmark
        assert len(image_size) == 2

        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)

        return warped

# def transformation_from_points(points1, points2):
#     points1 = points1.astype(numpy.float64)
#     points2 = points2.astype(numpy.float64)
#     c1 = numpy.mean(points1, axis=0)
#     c2 = numpy.mean(points2, axis=0)
#     points1 -= c1
#     points2 -= c2
#     s1 = numpy.std(points1)
#     s2 = numpy.std(points2)
#     points1 /= s1
#     points2 /= s2
#     U, S, Vt = numpy.linalg.svd(points1.T * points2)
#     R = (U * Vt).T
#     return numpy.vstack([numpy.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),numpy.matrix([0., 0., 1.])])

# def warp_im(img_im, orgi_landmarks,tar_landmarks):
#     pts1 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in orgi_landmarks]))
#     pts2 = numpy.float64(numpy.matrix([[point[0], point[1]] for point in tar_landmarks]))
#     M = transformation_from_points(pts1, pts2)
#     dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
#     return dst

# def align_process(img, bbox, landmark, image_size):
#     # img_im = cv2.imread(pic_path)
#     # cv2.imshow('affine_img_im', img_im)
#     dst = warp_im(img, landmark, REFERENCE_FACIAL_POINTS)
#     # cv2.imshow('affine', dst)
#     crop_im = dst[0:image_size[0], 0:image_size[1]]
#     return crop_im
#     # cv2.imshow('affine_crop_im', crop_im)
