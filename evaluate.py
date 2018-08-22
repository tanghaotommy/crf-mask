import os, sys
import cv2
import numpy as np
sys.path.append(os.path.dirname(__file__))

from train_0 import *

##--------------------------------------------------------------

def watershed(im):
    """
    Watershed algorithm for seperating each nuclei
    im: probability of the pixel being a nuclei
    return: each pixel labeled with integer indicating individual nucleis
    """
    # Watershed
    im = (im * 255).astype(np.uint8)

    ret, thresh = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)


    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    sure_bg = sure_bg

    # Finding sure foreground area
    # dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    # ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    eroded_img = cv2.erode(im, kernel, iterations=3)
    sure_fg = ((eroded_img > 200) * 255).astype(np.uint8)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers+1
    markers[unknown==255] = 0

    im = im[:,:,np.newaxis]
    im = np.repeat(im, 3, axis=-1)
    im = im.astype(np.uint8)

    markers = cv2.watershed(im,markers)
    im[markers == -1] = [255,0,0]
    
    return markers

## overwrite functions ###
def revert(net, images):

    def torch_clip_proposals (proposals, index, width, height):
        boxes = torch.stack((
             proposals[index,0],
             proposals[index,1].clamp(0, width  - 1),
             proposals[index,2].clamp(0, height - 1),
             proposals[index,3].clamp(0, width  - 1),
             proposals[index,4].clamp(0, height - 1),
             proposals[index,5],
             proposals[index,6],
        ), 1)
        return proposals

    # ----

    batch_size = len(images)
    for b in range(batch_size):
        image  = images[b]
        height,width  = image.shape[:2]


        # net.rpn_logits_flat  <todo>
        # net.rpn_deltas_flat  <todo>
        # net.rpn_window       <todo>
        # net.rpn_proposals    <todo>

        # net.rcnn_logits
        # net.rcnn_deltas
        # net.rcnn_proposals <todo>

        # mask --
        # net.mask_logits
        index = (net.detections[:,0]==b).nonzero().view(-1)
        net.detections   = torch_clip_proposals (net.detections, index, width, height)

        net.masks[b] = net.masks[b][:height,:width]

    return net, image



def eval_augment(image, multi_mask, meta, index):
    # image = normalize(image)
    pad_image = pad_to_factor(image, factor=16)
    input = torch.from_numpy(pad_image.transpose((2,0,1))).float()
    box, label, instance  = multi_mask_to_annotation(multi_mask)

    return input, box, label, instance, meta, image, index


def eval_collate(batch):

    batch_size = len(batch)
    #for b in range(batch_size): print (batch[b][0].size())
    inputs    = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    boxes     =             [batch[b][1]for b in range(batch_size)]
    labels    =             [batch[b][2]for b in range(batch_size)]
    instances =             [batch[b][3]for b in range(batch_size)]
    metas     =             [batch[b][4]for b in range(batch_size)]
    images    =             [batch[b][5]for b in range(batch_size)]
    indices   =             [batch[b][6]for b in range(batch_size)]

    return [inputs, boxes, labels, instances, metas, images, indices]






#--------------------------------------------------------------
def run_evaluate():

    # out_dir  = RESULTS_DIR + '/mask-rcnn-0313'
    # initial_checkpoint = \
    #     RESULTS_DIR + '/mask-rcnn-50-rcnn-02/checkpoint/00003500_model.pth'
        ##

    ## setup  ---------------------------
    os.makedirs(out_dir +'/evaluate/overlays', exist_ok=True)
    os.makedirs(out_dir +'/evaluate/npys', exist_ok=True)
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)
    os.makedirs(out_dir +'/backup', exist_ok=True)
    #backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.evaluate.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## net ------------------------------
    cfg = Configuration()
    # cfg.rpn_train_nms_pre_score_threshold = 0.8 #0.885#0.5
    # cfg.rpn_test_nms_pre_score_threshold  = 0.8 #0.885#0.5

    net = MaskRcnnNet(cfg).cuda()
    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))


    log.write('%s\n\n'%(type(net)))
    log.write('\n')



    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    set_name = 'filenames/histological_val.csv'
    test_dataset = ScienceDataset(DATA_DIR, set_name, mode='train', transform = eval_augment)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler = SequentialSampler(test_dataset),
                        batch_size  = 1,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = eval_collate)


    log.write('\ttest_dataset.set_name = %s\n'%(test_dataset.set_name))
    log.write('\tlen(test_dataset)  = %d\n'%(len(test_dataset)))
    log.write('\n')





    ## start evaluation here! ##############################################
    log.write('** start evaluation here! **\n')
    mask_average_precisions = []
    box_precisions_50  = []

    test_num  = 0
    test_loss = np.zeros(5,np.float32)
    test_acc  = 0
    for i, (inputs, truth_boxes, truth_labels, truth_instances, metas, images, indices) in enumerate(test_loader, 0):
        if all((truth_label>0).sum()==0 for truth_label in truth_labels): continue

        net.set_mode('test')
        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            net(inputs, truth_boxes,  truth_labels, truth_instances )
            #loss = net.loss(inputs, truth_boxes,  truth_labels, truth_instances)


        ##save results ---------------------------------------
        revert(net, images)


        batch_size = len(indices)
        assert(batch_size==1)  #note current version support batch_size==1 for variable size input
                               #to use batch_size>1, need to fix code for net.windows, etc

        batch_size,C,H,W = inputs.size()
        inputs = inputs.data.cpu().numpy()
        #
        # window          = net.rpn_window
        # rpn_logits_flat = net.rpn_logits_flat.data.cpu().numpy()
        # rpn_deltas_flat = net.rpn_deltas_flat.data.cpu().numpy()
        # proposals  = net.rpn_proposals
        masks      = net.masks
        detections = net.detections.cpu().numpy()


        for b in range(batch_size):
            #image0 = (inputs[b].transpose((1,2,0))*255).astype(np.uint8)
            image  = images[b]
            height,width  = image.shape[:2]
            mask = masks[b]

            index = np.where(detections[:,0]==b)[0]
            detection = detections[index]
            box = detection[:,1:5]


            truth_mask = instance_to_multi_mask(truth_instances[b])
            truth_box  = truth_boxes[b]
            truth_label= truth_labels[b]
            truth_instance= truth_instances[b]


            mask_average_precision, mask_precision =\
                compute_average_precision_for_mask(mask, truth_mask, t_range=np.arange(0.5, 1.0, 0.05))

            box_precision, box_recall, box_result, truth_box_result = \
                compute_precision_for_box(box, truth_box, truth_label, threshold=[0.5])
            box_precision = box_precision[0]

            mask_average_precisions.append(mask_average_precision)
            box_precisions_50.append(box_precision)




            # --------------------------------------------
            name = test_dataset.filenames[indices[b]]
            print('%d\t%s\t%0.5f  (%0.5f)'%(i,name,mask_average_precision, box_precision))

            #----
            contour_overlay  = multi_mask_to_contour_overlay(mask, image, color=[0,255,0])
            color_overlay    = multi_mask_to_color_overlay(mask, color='summer')
            color1_overlay   = multi_mask_to_contour_overlay(mask, color_overlay, color=[255,255,255])
            all1 = np.hstack((image,contour_overlay,color1_overlay))

            all6 = draw_multi_proposal_metric(cfg, image, detection, truth_box, truth_label,[0,255,255],[255,0,255],[255,255,0])
            all7 = draw_mask_metric(cfg, image, mask, truth_box, truth_label, truth_instance)

            os.makedirs(out_dir +'/evaluate/overlays/%s'%name, exist_ok=True)
            cv2.imwrite(out_dir +'/evaluate/overlays/%s/%s_contour.png'%(name,name),all1)
            cv2.imwrite(out_dir +'/evaluate/overlays/%s/%s_proposal.png'%(name,name),all6)
            cv2.imwrite(out_dir +'/evaluate/overlays/%s/%s_mask.png'%(name,name),all7)



            #image_show('overlay_mask',overlay_mask)
            #image_show('overlay_truth',overlay_truth)
            #image_show('overlay_error',overlay_error)
            #image_show('all1',all1)
            #image_show('all6',all6)
            #image_show('all7',all7)
            #cv2.waitKey(0)

        # print statistics  ------------
        test_acc  += 0 #batch_size*acc[0][0]
        # test_loss += batch_size*np.array((
        #                    loss.cpu().data.numpy(),
        #                    net.rpn_cls_loss.cpu().data.numpy(),
        #                    net.rpn_reg_loss.cpu().data.numpy(),
        #                     0,0,
        #                  ))
        test_num  += batch_size

    #assert(test_num == len(test_loader.sampler))
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num

    log.write('initial_checkpoint  = %s\n'%(initial_checkpoint))
    log.write('test_acc  = %0.5f\n'%(test_acc))
    log.write('test_loss = %0.5f\n'%(test_loss[0]))
    log.write('test_num  = %d\n'%(test_num))
    log.write('\n')

    mask_average_precisions = np.array(mask_average_precisions)
    box_precisions_50 = np.array(box_precisions_50)
    log.write('-------------\n')
    log.write('mask_average_precision = %0.5f\n'%mask_average_precisions.mean())
    log.write('box_precision@0.5 = %0.5f\n'%box_precisions_50.mean())
    log.write('\n')




## evaluate post process here ####-------------------------------------
# def run_evaluate_map():
#
#     out_dir = RESULTS_DIR + '/mask-rcnn-gray-011b-drop1'
#     split   = 'valid1_ids_gray_only_43'
#
#     #------------------------------------------------------------------
#     log = Logger()
#     log.open(out_dir+'/log.evaluate.txt',mode='a')
#
#     #os.makedirs(out_dir +'/eval/'+split+'/label', exist_ok=True)
#     #os.makedirs(out_dir +'/eval/'+split+'/final', exist_ok=True)
#
#
#     image_files = glob.glob(out_dir + '/submit/npys/*.png')
#     image_files.sort()
#
#     average_precisions = []
#     for image_file in image_files:
#         #image_file = image_dir + '/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732.png'
#
#         name  = image_file.split('/')[-1].replace('.png','')
#
#         image   = cv2.imread(DATA_DIR + '/image/stage1_train/' + name + '/images/' + name +'.png')
#         truth   = np.load(DATA_DIR    + '/image/stage1_train/' + name + '/multi_mask.npy').astype(np.int32)
#         predict = np.load(out_dir     + '/submit/npys/' + name + '.npy').astype(np.int32)
#         assert(predict.shape == truth.shape)
#         assert(predict.shape[:2] == image.shape[:2])
#
#
#         #image_show('image',image)
#         #image_show('mask',mask)
#         #cv2.waitKey(0)
#
#
#         #baseline labeling  -------------------------
#
#
#         # fill hole, file small, etc ...
#         # label = filter_small(label, threshold=15)
#
#
#         average_precision, precision = compute_average_precision(predict, truth)
#         average_precisions.append(average_precision)
#
#         #save and show  -------------------------
#         print(average_precision)
#
#         # overlay = (skimage.color.label2rgb(label, bg_label=0, bg_color=(0, 0, 0))*255).astype(np.uint8)
#         # cv2.imwrite(out_dir +'/eval/'+split+'/label/' + name + '.png',overlay)
#         # np.save    (out_dir +'/eval/'+split+'/label/' + name + '.npy',label)
#
#
#         # overlay1 = draw_label_contour (image, label )
#         # mask  = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
#         # final = np.hstack((image, overlay1, overlay, mask))
#         # final = final.astype(np.uint8)
#         # cv2.imwrite(out_dir +'/eval/'+split+'/final/' + name + '.png',final)
#         #
#         #
#         # image_show('image',image)
#         # image_show('mask',mask)
#         # image_show('overlay',overlay)
#         # cv2.waitKey(1)
#
#     ##----------------------------------------------
#     average_precisions = np.array(average_precisions)
#     log.write('-------------\n')
#     log.write('average_precision = %0.5f\n'%average_precisions.mean())
#     log.write('\n')
#








# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))


    run_evaluate()



    print('\nsucess!')
