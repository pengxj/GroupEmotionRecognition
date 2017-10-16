
import os, sys
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import myResnet as models
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def image_loader_rgb(path):
    return Image.open(path).convert('RGB')


def test_model(model, data_dir, transform,use_gpu, save_file):    
    model.eval()
    if use_gpu: model.cuda()
    obj = open(save_file, 'w')
    for root, _, fnames in sorted(os.walk(data_dir)):
        print len(fnames)
        if len(fnames)>0:
            for fname in fnames:
                if is_image_file(fname) and fname[0]!='.':
                    print fname
                    tensors = []
                    imfile = os.path.join(root, fname) # 
                    img = image_loader_rgb(imfile)
                    im_tensor = transform(img)
                    tensors.append(im_tensor)
                    flip_im_tensor = transform(img.transpose(Image.FLIP_LEFT_RIGHT))
                    tensors.append(flip_im_tensor)
                    inputs = torch.stack(tensors,dim=0)
                    if use_gpu: inputs = inputs.cuda()
                    inputs = torch.autograd.Variable(inputs, volatile=True) 
                    outputs = model(inputs)
                    out_cpu=outputs.data.cpu().numpy()
                    score = out_cpu.mean(axis=0) 

                    line = '{} {} {} {}\n'.format(imfile.replace(data_dir,''), score[0],score[1],score[2])
                    obj.write(line)

    obj.close()

if __name__ == '__main__':
    # all the path is fixed
    model_path = 'models/resnet101_extra_trval.pth'
    test_img_path = '../TEST/'
    save_score_file = 'score_test_global_resnet101_trvalextra.txt'

    use_gpu = torch.cuda.is_available()   

    # validation 
    model = models.resnet101(False,num_classes=3, input_c=3, input_size=224,drop=0)
    transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cp = torch.load(model_path)
    model.load_state_dict(cp)
    test_model(model, test_img_path, transform, use_gpu, save_score_file)
