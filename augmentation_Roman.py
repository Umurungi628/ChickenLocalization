import cv2, random, json, os
import numpy as np
from shapely.geometry import Polygon

"""------------------------------------input-img-annotation-dirs-----------------------------------"""
in_im_dir = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\resultimage2'     #path to the input images
out_im_dir = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\augmentation_Images'    #output path


dir = r'C:\Users\User\PycharmProjects\chickenLocalization\chickenLocalization-main\resultimage2'   #path to the the *.json file
jfile = "annotations.json"
jfile_out = "annotations.json"
imread_type = 'cv_read'     #'cv_read' / 'np_read'

"""------------------------------------------------------------------------------------------------"""
"""-------------------------------------variables--------------------------------------------------"""
"""type_aug:
- type - max count of output images after augmentation, n_i - number of input images
------------
- 'small' - 19n_i (12geom_aug + 7im_enhance_aug)
- 'medium' - 64n_i (4geom_aug*9geom_aug + 4geom_aug*7im_enhance_aug)
- 'large' - 672n_i (84geom_aug * (7im_enhance_aug+1basis))-->84geom_aug + 84geom_aug * 7im_enhance_aug
- 'xl' - 2048n_i (256geom_aug * (7im_enhance_aug+1basis))-->256geom_aug + 256geom_aug * 7im_enhance_aug
------------
aug
------------
- 0 - make 3 crop from image
- 1 - resize image to same size as crop
- 2 - make 3 rotation
- 3 - vertical flip
- 4 - horizontal flip
- 5 - make 3 perspective distortion
- 6 - blur
- 7 - histogram enhance
- 8 - contrast
- 9 - brightness
- 10 - add noise
- 11 - color shift
0 - 5 --> geom_aug
6 - 11 --> im_enhance_aug
"""

aug = [0,1,4,5,6,8,9,10,11]     #steps of the augmentation
tar_size = [1280, 720]
res_size = tar_size
rot_crop = 'no'     #yes/no
ang = 70
bl = 7
type_blur = 'gauss'
type_h = 'eq_hist'
con = 1.2
s = 0.7
pr = 20
n = 0
k = 0
im_e = []
anot_e = []
nonfull_cat = False
nfc = [1, 2]

"""--------------------------------------------------------"""
"""----------augmentation function-------------------------"""
if not os.path.isdir(out_im_dir):
    os.makedirs(out_im_dir)

def imread(name, read_type):
    if read_type == 'cv_read':
        img = cv2.imread(name)
    elif read_type == 'np_read':
        img = cv2.imdecode(np.fromfile(name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return img


def get_poly_coords(polygon):
    pol_area = []
    if polygon.geom_type == 'Polygon':
        out = np.array(polygon.exterior.coords, dtype=float).transpose()
    elif polygon.geom_type == 'MultiPolygon':
        pol_i = list(polygon.geoms)
        for i in range(0, len(pol_i)):
            pol_area.append(pol_i[i].area)
        inter1 = pol_i[pol_area.index(max(pol_area))]
        out = np.array(inter1.exterior.coords, dtype=float).transpose()
    elif polygon.geom_type == 'GeometryCollection':
        for pol_c in polygon.geoms:
            if pol_c.geom_type == 'Polygon':
                out = np.array(pol_c.exterior.coords, dtype=float).transpose()
    return out


def area_seg_bbox(x, y):
    ar_points = np.array([x, y], dtype=int).transpose()
    coords = np.array([x, y], dtype=int).transpose().reshape(1, 2 * len(x)).tolist()
    bbox = np.array([(min(x)), (min(y)), (max(x) - min(x)), (max(y) - min(y))], dtype=int).tolist()
    area = cv2.contourArea(ar_points)
    return area, coords, bbox


def imcrop(img, img_size, tar_size, res_size, n):
    n0 = n + 1
    h, w = [img_size[0], img_size[1]]
    hw_min = np.amin([img_size[0], img_size[1]])
    if tar_size[0] > img_size[1]:
        #w1 = (img_size[1] // 4) * 3
        w1 = res_size[0]
        h1 = w1 * tar_size[1] // tar_size[0]
    elif tar_size[1] > img_size[0]:
        #h1 = (img_size[0] // 4) * 3
        h1 = res_size[1]
        w1 = h1 * tar_size[0] // tar_size[1]
        #h1, w1 = [hw_min, hw_min]
    else:
        h1, w1 = [tar_size[1], tar_size[0]]  # rozlišení výřezu
    left = random.randint(0, w - w1)  # vygenerování náhodné hodnoty z rozpětí 0 až polovina šířky původního obrázku
    upper = random.randint(0,h - h1)  # random.randint(1, (vyska // (2*50)))*20  # vygenerování náhodné hodnoty z rozpětí 1 až polovina výšky původního obrázku
    right = left + w1
    lower = upper + h1
    im0 = img[upper:lower, left:right]
    l0, u0 = 0, 0
    return im0, left, upper, right, lower, n0, l0, u0


def imcrop_points(left, upper, right, lower, l0, u0, points, k):
    nf = 0
    x = np.array(points[::2], dtype=float)
    y = np.array(points[1::2], dtype=float)
    #print([[left, upper], [right, upper], [right, lower], [left, lower]])
    pim = Polygon(np.array([[left, upper], [right, upper], [right, lower], [left, lower]]))
    ppo = Polygon(np.array([x, y]).transpose()).buffer(0)
    if pim.intersects(ppo) is True:
        inter = pim.intersection(ppo)
        if inter.area < ppo.area * 0.2:
            coords = []
            bbox = []
            area = []
            k = k
        else:
            nf = 1 if inter.area < ppo.area * 0.999 else 0
            out = get_poly_coords(inter)
            x = out[0, :] - left + l0
            y = out[1, :] - upper + u0
            area, coords, bbox = area_seg_bbox(x, y)
            k = k + 1
    else:
        coords = []
        bbox = []
        area = []
        k = k
    return coords, bbox, area, k, nf


def imres(img, img_size, res_size, n):
    n = n + 1
    im1 = cv2.resize(img, (res_size[0], res_size[1]))
    """if img_size[0] > img_size[1]:
        im1 = cv2.resize(img, (res_size[1], res_size[0]))
    else:
        im1 = cv2.resize(img, (res_size[0], res_size[1]))"""
    return im1, n


def imres_points(img_size, res_size, points, k):
    #print(len(points))
    x = np.array(points[::2], dtype=float)
    y = np.array(points[1::2], dtype=float)
    c_x = res_size[0] / img_size[1]
    c_y = res_size[1] / img_size[0]
    x = x * c_x
    y = y * c_y
    area, coords, bbox = area_seg_bbox(x, y)
    k = k + 1
    return coords, bbox, area, k


def imrot(img, img_size, ang, n):
    n = n + 1
    if ang == 90:
        im3 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img_size1 = [img_size[1], img_size[0]]
    elif ang == 180:
        im3 = cv2.rotate(img, cv2.ROTATE_180)
        img_size1 = img_size
    elif ang == 270:
        im3 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_size1 = [img_size[0], img_size[1]]
    else:
        rm = cv2.getRotationMatrix2D((img_size[1] / 2, img_size[0] / 2), ang, 1)
        im3 = cv2.warpAffine(img, rm, (img_size[1], img_size[0]))
        img_size1 = img_size
    return im3, n, img_size1


def imrot_points(img_size, ang, points, k):
    w, h = img_size[1], img_size[0]
    x = np.array(points[::2], dtype=float)
    y = np.array(points[1::2], dtype=float)
    nf = 0
    if ang == 90:
        x1 = h - y
        y1 = x
        area, coords, bbox = area_seg_bbox(x1, y1)
        k = k + 1
    elif ang == 180:
        x1 = w - x
        y1 = h - y
        area, coords, bbox = area_seg_bbox(x1, y1)
        k = k + 1
    elif ang == 270:
        x1 = y
        y1 = w - x
        area, coords, bbox = area_seg_bbox(x1, y1)
        k = k + 1
    else:
        x0 = w / 2
        y0 = h / 2
        ang = np.radians(ang)
        x1 = (x - x0) * np.cos(ang) + (y - y0) * np.sin(ang) + x0
        y1 = -(x - x0) * np.sin(ang) + (y - y0) * np.cos(ang) + y0
        pim = Polygon(np.array([[0, 0], [w, 0], [w, h], [0, h]]))
        ppo = Polygon(np.array([x1, y1]).transpose()).buffer(0)
        ppo.buffer(0.01)
        if pim.intersects(ppo) is True:
            try:
                inter = pim.intersection(ppo)
                if inter.area < ppo.area * 0.2:
                    coords = []
                    bbox = []
                    area = []
                    k = k
                else:
                    nf = 1 if inter.area < ppo.area * 0.999 else 0
                    out = get_poly_coords(inter)
                    x = out[0, :]
                    y = out[1, :]
                    area, coords, bbox = area_seg_bbox(x, y)
                    k = k + 1
            except:
                coords = []
                bbox = []
                area = []
                k = k
        else:
            coords = []
            bbox = []
            area = []
            k = k
    return coords, bbox, area, k, nf


def imflipver(img, n):
    n = n + 1
    im4 = cv2.flip(img, 0)
    return im4, n


def imflipver_points(img_size, points, k):
    x = np.array(points[::2], dtype=float)
    y = np.array(points[1::2], dtype=float)
    y = img_size[0] - y
    area, coords, bbox = area_seg_bbox(x, y)
    k = k + 1
    return coords, bbox, area, k


def imfliphor(img, n):
    n = n + 1
    im5 = cv2.flip(img, 1)
    return im5, n


def imfliphor_points(img_size, points, k):
    x = np.array(points[::2], dtype=float)
    y = np.array(points[1::2], dtype=float)
    x = img_size[1] - x
    area, coords, bbox = area_seg_bbox(x, y)
    k = k + 1
    return coords, bbox, area, k


def imwarp(img, img_size, n):
    n = n + 1
    sour_o = np.float32([[0, 0], [img_size[1], 0], [0, img_size[0]], [img_size[1], img_size[0]]])
    lh = [random.randint(1, img_size[1] // 4), random.randint(1, img_size[0] // 4)]
    ph = [img_size[1] // (4 / 3) + random.randint(1, img_size[1] // 4), random.randint(1, img_size[0] // 4)]
    ld = [random.randint(1, img_size[1] // 4), img_size[0] // (4 / 3) + random.randint(1, img_size[0] // 4)]
    pd = [img_size[1] // (4 / 3) + random.randint(1, img_size[1] // 4),
          img_size[0] // (4 / 3) + random.randint(1, img_size[0] // 4)]
    sour_n = np.float32([lh, ph, ld, pd])
    mat = cv2.getPerspectiveTransform(sour_o, sour_n)
    im6 = cv2.warpPerspective(img, mat, (img_size[1], img_size[0]))
    return im6, mat, n


def imwarp_points(mat, points, k):
    x = np.array(points[::2], dtype=float)
    y = np.array(points[1::2], dtype=float)
    pts = np.array([x, y], dtype=float).transpose().reshape(1, len(x), 2)
    pts_w = cv2.perspectiveTransform(pts, mat)
    pts_w = pts_w.transpose().reshape(2, len(x))
    x = pts_w[0, :]
    y = pts_w[1, :]
    if len(x) > 0:
        area, coords, bbox = area_seg_bbox(x, y)
        k = k + 1
    else:
        coords = []
        bbox = []
        area = []
        k = k
    return coords, bbox, area, k


def imblur(img, r, type_blur, n):  # type: #'gauss' or 'median'
    n = n + 1
    if type_blur == 'gauss':
        im7 = cv2.GaussianBlur(img, (r, r), 1)
    elif type_blur == 'median':
        im7 = cv2.medianBlur(img, r)
    else:
        im7 = []
    return im7, n


def imhist(img, type_h, n):  # type: #'eq_hist' or 'clahe'
    n = n + 1
    if type_h == 'eq_hist':
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
        im8 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    elif type_h == 'clahe':
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 2] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_hsv[:, :, 2])
        im8 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    else:
        im8 = []
    return im8, n


def imcontrast(img, con, n):  # con: recommended interval (0.2;2)
    n = n + 1
    im9 = np.array(img, dtype=float)
    im9 = con * (im9 - 128) + 128
    im9[im9 > 255] = 255
    im9[im9 < 0] = 0
    im9 = np.array(im9, dtype=np.uint8)
    return im9, n


def imbrightness(img, br, n):  # br: recommended interval (-0.5;0.5)
    n = n + 1
    im10 = np.array(img, dtype=float) + br * 255
    im10[im10 > 255] = 255
    im10[im10 < 0] = 0
    im10 = np.array(im10, dtype=np.uint8)
    return im10, n


def imnoise(img, s, n):  # s: interval (0:1), 0.7 recommended
    n = n + 1
    nn = np.random.normal(0, s, img.shape).astype('uint8')
    im11 = img + img * nn
    return im11, n


def imrgbshift(img, pr, n):  # pr: number of color shift of the hue channel in HSV image (5:70 recommended)
    n = n + 1
    im12 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    p = random.randint(-pr, pr)
    im12[:, :, 0] = im12[:, :, 0] + p
    im12 = cv2.cvtColor(im12, cv2.COLOR_HSV2RGB)
    return im12, n


"""--------------------------------------------------------"""
"""-------------coco-json-function-------------------------"""


def anotdict(k, n, c, coords, area, bbox):
    anot = {
        "id": k,
        "image_id": n,
        "category_id": c,
        "segmentation": coords,
        "area": area,
        "bbox": bbox,
        "bbox_mode": 1,
        "iscrowd": 0,
        "attributes": {
            "occluded": 'false'
        }
    }
    return anot


def anot2dict(lc, k, n, c, coords, area, bbox, anot_e):
    lc = lc + 1
    anotd = anotdict(k, n, c, coords, area, bbox)
    anot_e.append(anotd)
    return lc


def imdict(n, w, h, i):
    im = {
        "id": n,
        "width": w,
        "height": h,
        "file_name": i,
        "license": 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": 0
    }
    return im


def im2dict0(n, out_im_dir, img_p, im_e, tar_size, img_size, n_i):
    cv2.imwrite(out_im_dir + '/' + str(n).zfill(n_i) + '.jpg', img_p)
    if len(tar_size) > 1:
        imd = imdict(n, tar_size[0], tar_size[1], str(n).zfill(n_i) + '.jpg')
    else:
        imd = imdict(n, img_size[1], img_size[0], str(n).zfill(n_i) + '.jpg')
    im_e.append(imd)


def im2dict(lc, n, out_im_dir, img_p, im_e, tar_size, img_size, n_i):
    if lc > 0:
        cv2.imwrite(out_im_dir + '/' + str(n).zfill(n_i) + '.jpg', img_p)
        if len(tar_size) > 1:
            imd = imdict(n, tar_size[0], tar_size[1], str(n).zfill(n_i) + '.jpg')
        else:
            imd = imdict(n, img_size[1], img_size[0], str(n).zfill(n_i) + '.jpg')
        im_e.append(imd)
        nf = n
    else:
        nf = n - 1
    return nf


def make_json(lic, inf, cat, im_e, anot_e, name_file):
    x = {
        "licenses": lic,
        "info": inf,
        "categories": cat,
        "images": im_e,
        "annotations": anot_e
    }
    y = json.dumps(x)
    json_name = name_file
    with open(json_name, "w") as outfile:
        outfile.write(y)


def load_data(dir, jfile):
    f = dir + '/' + jfile
    f = open(f)
    data = json.load(f)
    return data


def get_im_anot(data):
    im = data['images']
    anot = data['annotations']
    im_list = []
    im1 = []
    anot_list = []
    anot1 = []
    for pre in anot:
        anot1.append(pre)
        anot_list.append(str(pre['image_id']))
    for abc in im:
        if str(abc['id']) in anot_list:
            p = anot_list.index(str(abc['id']))
            if anot[p]['iscrowd'] == 0:
                im_list.append(abc['file_name'])
                im1.append(abc)
    return im1, im_list, anot1, anot_list


def find_im(i, im1, im_list, n, im_e):  # anot, anot_list,
    por_im = im_list.index(i)
    im_id = im1[por_im]['id']
    return im_id

def merge_mat(mat):
    mat0 = []
    if len(mat[0]) > 1:
        for i in range(len(mat)):
            if len(mat[i]) > 1:
                for j in range(len(mat[i])):
                    if len(mat[i][j]) > 1:
                        for k in range(mat[i][j]):
                            mat0.append(mat[i][j][k])
                    else:
                        mat0.append(mat[i][j])
            else:
                mat0.append(mat[i])
    else:
        mat0 = mat
    return mat0

def find_anot(im_id, anot1, anot_list):
    anot_seg = []
    anot_cat = []
    anot_area = []
    anot_bbox = []
    coords = []
    li = [h for h, por in enumerate(anot_list) if por == str(im_id)]
    for j in range(0, len(li)):
        bbox = anot1[li[j]]['bbox']
        cbb = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0], bbox[1] + bbox[3]]
        coords = anot1[li[j]]['segmentation'] if len(anot1[li[j]]['segmentation']) > 0 else [cbb]
        #coords = merge_mat(coord0)
        area = anot1[li[j]]['area']

        c = anot1[li[j]]['category_id']
        anot_seg.append(coords)
        anot_cat.append(c)
        anot_area.append(area)
        anot_bbox.append(bbox)
    return anot_seg, anot_cat, anot_area, anot_bbox

def rename(im_e, anot_e, out_im_dir):
    im_list1, anot_list1 = [], []
    for image in im_e:
        im_list1.append(image['file_name'])
    for anot in anot_e:
        anot_list1.append(str(anot['image_id']))
    p = len(im_list1)  # počet souborů v adresáři
    r = random.sample(range(1, p + 1), p)
    n_i = len(str(p))
    id1 = 0
    an_id1 = 0
    im_df = []
    anot_df = []
    name_o = []
    name_n = []
    for i in r:
        if im_list1[i - 1] in os.listdir(out_im_dir):
            id1 += 1
            print(id1)
            name = im_list1[i - 1]
            name_o.append(name)
            name1 = str(id1).zfill(n_i) + '_1.jpg'
            name_n.append(name1)
            im_d1 = im_e[i - 1]
            im_d1['id'] = id1
            im_d1['file_name'] = name1
            im_df.append(im_d1)
            li = [j for j, por in enumerate(anot_list1) if por == str(i)]
            for j in range(0, len(li)):
                an_id1 += 1
                anot_d1 = anot_e[li[j]]
                anot_d1['image_id'] = id1
                anot_d1['id'] = an_id1
                anot_df.append(anot_d1)
    for i in name_o:
        naz0 = out_im_dir + "/" + i  # původní název souboru
        naz1 = out_im_dir + "/r_" + i  # nový název souboru
        os.rename(naz0, naz1)
    for i, j in enumerate(name_o):
        naz0 = out_im_dir + "/r_" + j  # původní název souboru
        naz1 = out_im_dir + "/" + name_n[i]  # nový název souboru
        os.rename(naz0, naz1)
    return im_df, anot_df


"""--------------------------------------------------------"""
"""--------------------------------------------------------"""

data = load_data(dir, jfile)
im1, im_list, anot1, anot_list = get_im_anot(data)
n_i = len(str(len(im_list) * 19))
for i in os.listdir(in_im_dir):
    if i[-3:] == 'jpg' or i[-3:] == 'png' or i[-4:] == 'jpeg' or i[-4:] == 'JPEG':
        if i in im_list:
            print(i)
            img = imread(in_im_dir + '/' + i, imread_type)
            img_size = img.shape
            im_id = find_im(i, im1, im_list, n, im_e)
            #img_size = [im1[im_id-1]['height'], im1[im_id-1]['width']]
            anot_seg, anot_cat, anot_area, anot_bbox = find_anot(im_id, anot1, anot_list)
            #print(anot_seg)
            anot_seg0, anot_cat0, anot_area0, anot_bbox0 = [], [], [], []
            if (img_size[0] > img_size[1] and res_size[0] > res_size[1]) or (
                    img_size[0] < img_size[1] and res_size[0] < res_size[1]):
                res_size0 = [res_size[1], res_size[0]]
                tar_size0 = [tar_size[1], tar_size[0]]
            else:
                res_size0 = [res_size[0], res_size[1]]
                tar_size0 = [tar_size[0], tar_size[1]]
            if 0 in aug:
                for m in range(0, 3):
                    img0, left, upper, right, lower, n, l0, u0 = imcrop(img, img_size, tar_size0, res_size0, n)
                    lc0 = 0
                    for r in range(0, len(anot_seg)):
                        [points, c] = [np.array(anot_seg[r]).flatten(), anot_cat[r]]
                        coords, bbox, area, k, nf = imcrop_points(left, upper, right, lower, l0, u0, points, k)
                        if nonfull_cat == True:
                            c = nfc[1] if (nf == 1 and c == nfc[0]) else c
                        if len(coords) > 0:
                            lc0 = anot2dict(lc0, k, n, c, coords, area, bbox, anot_e)
                    n = im2dict(lc0, n, out_im_dir, img0, im_e, tar_size0, [], n_i)
                if rot_crop == 'yes':
                    for m in range(0, 3):
                        tar_size_rot = [tar_size0[1], tar_size0[0]]
                        img0, left, upper, right, lower, n, l0, u0 = imcrop(img, img_size, tar_size_rot, res_size0, n)
                        lc0 = 0
                        for r in range(0, len(anot_seg)):
                            [points, c] = [np.array(anot_seg[r]).flatten(), anot_cat[r]]
                            coords, bbox, area, k, nf = imcrop_points(left, upper, right, lower, l0, u0, points, k)
                            if nonfull_cat == True:
                                c = nfc if nf == 1 else c
                            if len(coords) > 0:
                                lc0 = anot2dict(lc0, k, n, c, coords, area, bbox, anot_e)
                        n = im2dict(lc0, n, out_im_dir, img0, im_e, tar_size_rot, [], n_i)
            if 1 in aug:
                img1, n = imres(img, img_size, res_size0, n)
                im2dict0(n, out_im_dir, img1, im_e, res_size0, [], n_i)
                for r in range(0, len(anot_seg)):
                    [points, c] = [np.array(anot_seg[r]).flatten(), anot_cat[r]]
                    coords, bbox, area, k = imres_points(img_size, res_size0, points, k)
                    anotd = anotdict(k, n, c, coords, area, bbox)
                    anot_e.append(anotd), anot_seg0.append(coords), anot_cat0.append(c), anot_area0.append(
                        area), anot_bbox0.append(bbox)
                [img, anot_seg, anot_cat, anot_area, anot_bbox] = [img1.copy(), anot_seg0.copy(), anot_cat0.copy(),
                                                                   anot_area0.copy(), anot_bbox0.copy()]
                img_size = [res_size0[1], res_size0[0]]
            if 2 in aug:
                for m in range(0, 3):
                    if ang == 90:
                        ang1 = int(ang * (m + 1))
                    else:
                        ang1 = int(ang * random.uniform(0.5 * (m + 1), 0.8 * (m + 1)))
                    img2, n, img_size1 = imrot(img, img_size, ang1, n)
                    lc2 = 0
                    for r in range(0, len(anot_seg)):
                        [points, c] = [np.array(anot_seg[r]).flatten(), anot_cat[r]]
                        coords, bbox, area, k, nf = imrot_points(img_size, ang1, points, k)
                        if nonfull_cat == True:
                            c = nfc[1] if (nf == 1 and c == nfc[0]) else c
                        if len(coords) > 0:
                            lc2 = anot2dict(lc2, k, n, c, coords, area, bbox, anot_e)
                    n = im2dict(lc2, n, out_im_dir, img2, im_e, [], img_size1, n_i)
            if 3 in aug:
                img3, n = imflipver(img, n)
                im2dict0(n, out_im_dir, img3, im_e, [], img_size, n_i)
                for r in range(0, len(anot_seg)):
                    [points, c] = [np.array(anot_seg[r]).flatten(), anot_cat[r]]
                    coords, bbox, area, k = imflipver_points(img_size, points, k)
                    anotd = anotdict(k, n, c, coords, area, bbox)
                    anot_e.append(anotd)
            if 4 in aug:
                img4, n = imfliphor(img, n)
                im2dict0(n, out_im_dir, img4, im_e, [], img_size, n_i)
                for r in range(0, len(anot_seg)):
                    [points, c] = [np.array(anot_seg[r]).flatten(), anot_cat[r]]
                    coords, bbox, area, k = imfliphor_points(img_size, points, k)
                    anotd = anotdict(k, n, c, coords, area, bbox)
                    anot_e.append(anotd)
            if 5 in aug:
                for m in range(0, 3):
                    img5, mat, n = imwarp(img, img_size, n)
                    im2dict0(n, out_im_dir, img5, im_e, [], img_size, n_i)
                    for r in range(0, len(anot_seg)):
                        [points, c] = [np.array(anot_seg[r]).flatten(), anot_cat[r]]
                        coords, bbox, area, k = imwarp_points(mat, points, k)
                        anotd = anotdict(k, n, c, coords, area, bbox)
                        anot_e.append(anotd)
            if 6 in aug:
                img6, n = imblur(img, bl, type_blur, n)
                im2dict0(n, out_im_dir, img6, im_e, [], img_size, n_i)
                for r in range(0, len(anot_seg)):
                    [coords, c, bbox, area, k] = [anot_seg[r], anot_cat[r], anot_bbox[r], anot_area[r], k + 1]
                    anotd = anotdict(k, n, c, coords, area, bbox)
                    anot_e.append(anotd)
            if 7 in aug:
                img7, n = imhist(img, type_h, n)
                im2dict0(n, out_im_dir, img7, im_e, [], img_size, n_i)
                for r in range(0, len(anot_seg)):
                    [coords, c, bbox, area, k] = [anot_seg[r], anot_cat[r], anot_bbox[r], anot_area[r], k + 1]
                    anotd = anotdict(k, n, c, coords, area, bbox)
                    anot_e.append(anotd)
            if 8 in aug:
                img8, n = imcontrast(img, con, n)
                im2dict0(n, out_im_dir, img8, im_e, [], img_size, n_i)
                for r in range(0, len(anot_seg)):
                    [coords, c, bbox, area, k] = [anot_seg[r], anot_cat[r], anot_bbox[r], anot_area[r], k + 1]
                    anotd = anotdict(k, n, c, coords, area, bbox)
                    anot_e.append(anotd)
            if 9 in aug:
                for br in np.arange(-0.15, 0.3, 0.1):
                    img9, n = imbrightness(img, br, n)
                    im2dict0(n, out_im_dir, img9, im_e, [], img_size, n_i)
                    for r in range(0, len(anot_seg)):
                        [coords, c, bbox, area, k] = [anot_seg[r], anot_cat[r], anot_bbox[r], anot_area[r], k + 1]
                        anotd = anotdict(k, n, c, coords, area, bbox)
                        anot_e.append(anotd)
            if 10 in aug:
                img10, n = imnoise(img, s, n)
                im2dict0(n, out_im_dir, img10, im_e, [], img_size, n_i)
                for r in range(0, len(anot_seg)):
                    [coords, c, bbox, area, k] = [anot_seg[r], anot_cat[r], anot_bbox[r], anot_area[r], k + 1]
                    anotd = anotdict(k, n, c, coords, area, bbox)
                    anot_e.append(anotd)
            if 11 in aug:
                img11, n = imrgbshift(img, pr, n)
                im2dict0(n, out_im_dir, img11, im_e, [], img_size, n_i)
                for r in range(0, len(anot_seg)):
                    [coords, c, bbox, area, k] = [anot_seg[r], anot_cat[r], anot_bbox[r], anot_area[r], k + 1]
                    anotd = anotdict(k, n, c, coords, area, bbox)
                    anot_e.append(anotd)
im_e, anot_e = rename(im_e, anot_e, out_im_dir)
lic = [{"name": '', "id": 0, "url": ''}]
inf = {"contributor": '', "date_created": '', "description": '', "url": '', "version": '', "year": ''}
cat = data['categories']
name_file = out_im_dir + '/' + jfile_out
make_json(lic, inf, cat, im_e, anot_e, name_file)