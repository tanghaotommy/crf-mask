import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import skimage.segmentation
import pandas as pd
import os
from train_0 import *

def score_single(labels, y_pred):
    # Compute number of objects
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    print("Number of true objects:", true_objects)
    print("Number of predicted objects:", pred_objects)

    # Compute intersection between all objects
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn)
        print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return prec
                         
                         
                         
def rleToMask(rleString,height,width):
    rows,cols = height,width
    rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
    rlePairs = np.array(rleNumbers).reshape(-1,2)
    img = np.zeros(rows*cols,dtype=np.uint8)
    for index,length in rlePairs:
        index -= 1
        img[index:index+length] = 255
    img = img.reshape(cols,rows)
    img = img.T
    return img
                 
def main(annos = '/home/htang6/remote/gru/workspace/data/dsb2018/stage1_label/stage1_solution.csv',
    # preds = '/home/htang6/workspace/mask-rcnn-resnet50-ver-01.a/results/mask-rcnn-0423/submit/balance.csv'):  
    #preds = '/home/htang6/workspace/mask-rcnn-ver-12-gray-4-fix-zero-detection/results/mask-rcnn-0423/submit/balance.csv'): 
    preds = '/home/htang6/remote/gru/workspace/mask-rcnn-ver-12-gray-4-fix-zero-detection/results/mask-rcnn-transfer/submit/filenames.csv'):
    preds = pd.read_csv(preds).dropna()
    annos = pd.read_csv(annos)
    filenames = np.unique(annos['ImageId'])

    data_dir = '/home/htang6/workspace/data/dsb2018/stage1_test/'
    all_prec = []
    for id in filenames:
        p = preds[preds['ImageId'] == id]
        gt = annos[annos['ImageId'] == id]
        file = os.path.join(data_dir, "{}/images/{}.png".format(id,id))
        image = skimage.io.imread(file)
        height, width, _ = image.shape
        
        pred_mask = np.zeros((height, width), np.uint16)
        count = 1
        for i, row in p.iterrows():
            rle = row['EncodedPixels']
            m = rleToMask(rle, height, width)
            pred_mask[m > 0] = count
            count += 1
            
        count = 1
        gt_mask = np.zeros((height, width), np.uint16)
        for i, row in gt.iterrows():
            rle = row['EncodedPixels']
            m = rleToMask(rle, height, width)
            gt_mask[m > 0] = count
            count += 1

        prec = score_single(gt_mask, pred_mask)
        all_prec.append(prec)

    all_prec = np.array(all_prec)
    mean = np.mean(all_prec,0)
    print("Thresh\tAvgPrec.")
    thr = np.arange(0.5, 1.0, 0.05)

    for i in range(len(mean)):
        print("{:1.3f}\t{:1.3f}".format(thr[i], mean[i]))
        
    print('Total test data: ', all_prec.shape[0])
    print('Score: ', np.mean(mean))

if __name__ == '__main__':
    main()
