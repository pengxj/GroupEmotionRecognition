import os, sys
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import myResnet as models
from PIL import Image

def image_loader_gray(path):
    return Image.open(path).convert('L')


def test_model(model, data_dir, transform, use_gpu, save_file):   
    def get_topn(root, imfiles, n = 3):
        if n==0: return imfiles
        imsizes = []
        for f in imfiles:
            imfile = os.path.join(root, f)
            im = image_loader(imfile)
            imsizes.append(im.width * im.height)
        np_sizes = np.array(imsizes)
        sorted_inds = np.argsort(-np_sizes)
        if len(imsizes)>n:
            return [imfiles[i] for i in sorted_inds[:n]]
        else:
            return imfiles
    def rm_small(root, imfiles, minsize = 24):
        if minsize==0: return imfiles
        val_files = []
        for f in imfiles:
            imfile = os.path.join(root, f)
            im = image_loader(imfile)
            if min(im.width, im.height)>=minsize:
                val_files.append(f)
        return val_files

    model.eval()
    if use_gpu: model.cuda()
    obj = open(save_file, 'w')
    for root, _, fnames in sorted(os.walk(data_dir)):
        if len(fnames)>0:
            tensors = []
            imfiles = get_topn(root, fnames, 3)
            if len(imfiles)>0:
                for fname in imfiles:
                    imfile = os.path.join(root, fname) 
                    im_tensor = transform(image_loader_gray(imfile))
                    tensors.append(im_tensor)
                inputs = torch.stack(tensors,dim=0)
                if use_gpu: inputs = inputs.cuda()
                inputs = torch.autograd.Variable(inputs, volatile=True) 
                outputs = model(inputs)
                out_cpu=outputs.data.cpu().numpy()
                score = out_cpu.mean(axis=0) #[vidx, :]

                line = '{} {} {} {}\n'.format(root.replace(data_dir,''), score[0],score[1],score[2])
                obj.write(line)

    obj.close()

if __name__ == '__main__':
	# https://github.com/Microsoft/FERPlus
	# all the path is fixed
    model_path = 'models/ferplus_resnet34_ft_trval.pth'
    test_img_faces_path = '../TEST_nonaligned_faces/'
    save_score_file = 'score_test_face_FERPlus_res34_trval_top3.txt'

    use_gpu = torch.cuda.is_available()    
    model = models.resnet34(False, 3, 1, 48, 0) #-------
    cp = torch.load(model_path)
    model.load_state_dict(cp)
    
    transform = transforms.Compose([
        transforms.Scale((48,48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    
    test_model(model, test_img_faces_path, transform, use_gpu, save_score_file)
    