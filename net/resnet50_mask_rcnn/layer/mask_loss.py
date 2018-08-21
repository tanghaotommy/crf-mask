from common import *


def weighted_binary_cross_entropy_with_logits(logits, labels, weights):

    loss = logits.clamp(min=0) - logits*labels + torch.log(1 + torch.exp(-logits.abs())) 
    loss = (weights*loss).sum()/(weights.sum()+1e-12)

    return loss
	
	
def binary_cross_entropy_with_logits(logits, labels):

    loss = logits.clamp(min=0) - logits*labels + torch.log(1 + torch.exp(-logits.abs()))
    loss = loss.sum()/len(loss)

    return loss


def mask_loss(logits, labels, instances ):

    batch_size, num_classes = logits.size(0), logits.size(1)

    logits_flat = logits.view (batch_size,num_classes, -1)
    dim =  logits_flat.size(2)

    # one hot encode
    select = Variable(torch.zeros((batch_size,num_classes))).cuda()
    select.scatter_(1, labels.view(-1,1), 1)
    select[:,0] = 0
    select = select.view(batch_size,num_classes,1).expand((batch_size,num_classes,dim)).contiguous().byte()

    # print(select.size())
    # print(logits_flat[select].shape)

    logits_flat = logits_flat[select].view(-1)
    labels_flat = instances.view(-1)
 
    loss = binary_cross_entropy_with_logits(logits_flat, labels_flat)
    return loss

# def mask_loss(logits, labels, instances):
#     batch_size, num_classes, mask_num_classes = logits.size(0), logits.size(1), logits.size(2)

#     logits_flat = logits.view (batch_size,num_classes, mask_num_classes,-1)
#     dim =  logits_flat.size(-1)

#     # one hot encode
#     select = Variable(torch.zeros((batch_size,num_classes))).cuda()
#     select.scatter_(1, labels.view(-1,1), 1)
#     select[:,0] = 0
#     select = select.view(batch_size,num_classes,1,1).expand((batch_size,num_classes,mask_num_classes,dim)).contiguous().byte()

#     # print('select size: ', select.size())
#     # print('logit size: ', logits_flat.size())
#     # print('label size: ', labels.size())
#     # print('logits_flat: ', logits_flat[select].size())
#     # logits_flat = logits_flat[:,labels,:,:]

#     # test = logits_flat[:,1,:,:]
#     # print('test ', test.size())

#     logits_flat = logits_flat[select].view(batch_size, mask_num_classes, dim)
#     # print(torch.equal(test, logits_flat))

#     logits_flat = logits_flat.permute([0, 2, 1]).contiguous().view(-1, mask_num_classes)
#     labels_flat = instances.view(-1)
#     labels_flat = labels_flat.type(torch.LongTensor).cuda()
 
#     # loss = binary_cross_entropy_with_logits(logits_flat, labels_flat)
#     num_background = np.sum(labels_flat.cpu().numpy() == 0)
#     num_object = np.sum(labels_flat.cpu().numpy() == 1)
#     num_boundary = np.sum(labels_flat.cpu().numpy() == 2)
#     # print(labels_flat)
#     # print(logits.size())
#     # print(instances.size())
#     # print('num_boundary: ', num_boundary)
#     # print('num_object: ', num_object)
#     # print('num_background: ', num_background)
#     # print('total: ', labels_flat.size())
#     # print('labels_flat: ', torch.max(labels_flat), ' ', torch.min(labels_flat))
#     weights = Variable(torch.FloatTensor([1, 1, 3])).cuda()
#     loss = F.cross_entropy(logits_flat, labels_flat, size_average=True, weight=weights)
#     return loss


# #-----------------------------------------------------------------------------
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

