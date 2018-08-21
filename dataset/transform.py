from common import *
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from random import randint

## for debug
def dummy_transform(image):
    print ('\tdummy_transform')
    return image

# kaggle science bowl-2 : ################################################################

def resize_to_factor2(image, mask, factor=16):

    H,W = image.shape[:2]
    h = (H//factor)*factor
    w = (W //factor)*factor
    return fix_resize_transform2(image, mask, w, h)



def fix_resize_transform2(image, mask, w, h):
    H,W = image.shape[:2]
    if (H,W) != (h,w):
        image = cv2.resize(image,(w,h))

        mask = mask.astype(np.float32)
        mask = cv2.resize(mask,(w,h),cv2.INTER_NEAREST)
        mask = mask.astype(np.int32)
    return image, mask




def fix_crop_transform2(image, mask, x,y,w,h):

    H,W = image.shape[:2]
    assert(H>=h)
    assert(W >=w)

    if (x==-1 & y==-1):
        x=(W-w)//2
        y=(H-h)//2

    if (x,y,w,h) != (0,0,W,H):
        image = image[y:y+h, x:x+w]
        mask = mask[y:y+h, x:x+w]

    return image, mask

def random_one_crop_transform(image, bboxes, mask, w, h, rand=True):
    H, W = image.shape[:2]
    n_bbox = len(bboxes)

    if rand:
        i = np.random.choice(n_bbox)
    else:
        i = 0
    x0, y0, x1, y1 = bboxes[i]
    c_x = (x0 + x1) / 2
    c_y = (y0 + y1) / 2
    x = max(0, int(c_x - w / 2))
    y = max(0, int(c_y - h / 2))

    if x + w > W:
        diff = x + w - W
        x = max(0, x - diff)

    if y + h > H:
        diff = y + h - H
        y = max(0, y - diff)

    # print('image (%d, %d), bbox (%d, %d)(%d, %d)' % (H, W, y0, x0, y1, x1))
    # print((y, x))

    return fix_crop_transform2(image, mask, x,y,w,h)

def random_crop_transform2(image, mask, w,h, u=0.5):
    x,y = -1,-1
    if random.random() < u:

        H,W = image.shape[:2]
        if H!=h:
            y = np.random.choice(H-h)
        else:
            y=0

        if W!=w:
            x = np.random.choice(W-w)
        else:
            x=0

    return fix_crop_transform2(image, mask, x,y,w,h)



def random_horizontal_flip_transform2(image, mask, u=0.5):
    if random.random() < u:
        image = cv2.flip(image,1)  #np.fliplr(img) ##left-right
        mask  = cv2.flip(mask,1)
    return image, mask

def random_vertical_flip_transform2(image, mask, u=0.5):
    if random.random() < u:
        image = cv2.flip(image,0)
        mask  = cv2.flip(mask,0)
    return image, mask

def random_rotate90_transform2(image, mask, u=0.5):
    if random.random() < u:

        angle=random.randint(1,3)*90
        if angle == 90:
            image = image.transpose(1,0,2)  #cv2.transpose(img)
            image = cv2.flip(image,1)
            mask = mask.transpose(1,0)
            mask = cv2.flip(mask,1)

        elif angle == 180:
            image = cv2.flip(image,-1)
            mask = cv2.flip(mask,-1)

        elif angle == 270:
            image = image.transpose(1,0,2)  #cv2.transpose(img)
            image = cv2.flip(image,0)
            mask = mask.transpose(1,0)
            mask = cv2.flip(mask,0)
    return image, mask


def relabel_multi_mask(multi_mask):
    data = multi_mask
    data = data[:,:,np.newaxis]
    unique_color = set( tuple(v) for m in data for v in m )
    #print(len(unique_color))


    H,W  = data.shape[:2]
    multi_mask = np.zeros((H,W),np.int32)
    for color in unique_color:
        #print(color)
        if color == (0,): continue

        mask = (data==color).all(axis=2)
        label  = skimage.morphology.label(mask)

        index = [label!=0]
        multi_mask[index] = label[index]+multi_mask.max()

    return multi_mask


def random_shift_scale_rotate_transform2( image, mask,
                        shift_limit=[-0.0625,0.0625], scale_limit=[1/1.2,1.2],
                        rotate_limit=[-15,15], borderMode=cv2.BORDER_REFLECT_101 , u=0.5):

    #cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    if random.random() < u:
        height, width, channel = image.shape

        angle  = random.uniform(rotate_limit[0],rotate_limit[1])  #degree
        scale  = random.uniform(scale_limit[0],scale_limit[1])
        sx    = scale
        sy    = scale
        dx    = round(random.uniform(shift_limit[0],shift_limit[1])*width )
        dy    = round(random.uniform(shift_limit[0],shift_limit[1])*height)

        cc = math.cos(angle/180*math.pi)*(sx)
        ss = math.sin(angle/180*math.pi)*(sy)
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)

        image = cv2.warpPerspective(image, mat, (width,height),flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

        mask = mask.astype(np.float32)
        mask = cv2.warpPerspective(mask, mat, (width,height),flags=cv2.INTER_NEAREST,#cv2.INTER_LINEAR
                                    borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
        mask = mask.astype(np.int32)
        mask = relabel_multi_mask(mask)

    return image, mask




# single image ########################################################

#agumentation (photometric) ----------------------
def random_brightness_shift_transform(image, limit=[16,64], u=0.5):
    if np.random.random() < u:
        alpha = np.random.uniform(limit[0], limit[1])
        image = image + alpha*255
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def random_brightness_transform(image, limit=[0.5,1.5], u=0.5):
    if np.random.random() < u:
        alpha = np.random.uniform(limit[0], limit[1])
        image = alpha*image
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def random_contrast_transform(image, limit=[0.5,1.5], u=0.5):
    if np.random.random() < u:
        alpha = np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587,  0.299]]]) #rgb to gray (YCbCr)
        gray = image * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        image = alpha*image  + gray
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def random_saturation_transform(image, limit=[0.5,1.5], u=0.5):
    if np.random.random() < u:
        alpha = np.random.uniform(limit[0], limit[1])
        coef  = np.array([[[0.114, 0.587,  0.299]]])
        gray  = image * coef
        gray  = np.sum(gray,axis=2, keepdims=True)
        image = alpha*image  + (1.0 - alpha)*gray
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image

# https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/transforms.py
# https://github.com/fchollet/keras/pull/4806/files
# https://zhuanlan.zhihu.com/p/24425116
# http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
def random_hue_transform(image, limit=[-0.1,0.1], u=0.5):
    if random.random() < u:
        h = int(np.random.uniform(limit[0], limit[1])*180)
        #print(h)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image


def random_noise_transform(image, limit=[0, 0.5], u=0.5):
    if random.random() < u:
        H,W = image.shape[:2]
        noise = np.random.uniform(limit[0],limit[1],size=(H,W))*255

        image = image + noise[:,:,np.newaxis]*np.array([1,1,1])
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


# geometric ---
def resize_to_factor(image, factor=16):
    height,width = image.shape[:2]
    h = (height//factor)*factor
    w = (width //factor)*factor
    return fix_resize_transform(image, w, h)


def fix_resize_transform(image, w, h):
    height,width = image.shape[:2]
    if (height,width) != (h,w):
        image = cv2.resize(image,(w,h))
    return image



def pad_to_factor(image, factor=16):
    height,width = image.shape[:2]
    h = math.ceil(height/factor)*factor
    w = math.ceil(width/factor)*factor

    image = cv2.copyMakeBorder(image, top=0, bottom=h-height, left=0, right=w-width,
                               borderType= cv2.BORDER_REFLECT101, value=[0,0,0] )

    return image

def scale_image_canals(image):
    for i in range(image.shape[2]):
        canal = image[:,:,i]
        canal = canal - canal.min()
        canalmax = canal.max()
        if canalmax > 0:
            factor = 255/canalmax
            canal = (canal * factor).astype(int)
        image[:,:,i] = canal
    return image

def elastic_transform(image, sigma=12, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """
    # assert len(image.shape)==2
    
    alpha = np.random.randint(50, 100)
    if random_state is None:
            random_state = np.random.RandomState(None)
    
    shape = image.shape[0:2]
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    if len(image.shape) > 2:
        for c in range(image.shape[2]):
            image[:,:,c] = map_coordinates(image[:,:,c], indices, order=1).reshape(shape)
    else:
        image = map_coordinates(image, indices, order=1).reshape(shape)
    return image


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    print('\nsucess!')