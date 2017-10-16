clear;
%setting for GAF
remove_small_faces=false;
remove_small_faces_size=24;
%train.caffemodel is trained by trainset
%all.caffemodel is trained by trainset and valset
gaf_model='train.caffemodel';
%path of toolbox
caffe_path='/home/kwang/LargeMargin_Softmax_Loss-master';
pdollar_toolbox_path='toolbox-master'
caffe_model_path='./model/'
addpath(genpath(caffe_path));
addpath(genpath(pdollar_toolbox_path));

%use cpu
%caffe.set_mode_cpu();
gpu_id=0;
caffe.set_mode_gpu();	
caffe.set_device(gpu_id);

%three steps's threshold
threshold=[0.6 0.7 0.9];

%scale factor
factor=0.709;

%load caffe models
prototxt_dir =strcat(caffe_model_path,'det1.prototxt');
model_dir = strcat(caffe_model_path,'det1.caffemodel');
PNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'det2.prototxt');
model_dir = strcat(caffe_model_path,'det2.caffemodel');
RNet=caffe.Net(prototxt_dir,model_dir,'test');	
prototxt_dir = strcat(caffe_model_path,'det3.prototxt');
model_dir = strcat(caffe_model_path,'det3.caffemodel');
ONet=caffe.Net(prototxt_dir,model_dir,'test');
faces=cell(0);	
prototxt_dir = strcat(caffe_model_path,'deploy_enet.prototxt');
model_dir = strcat(caffe_model_path,gaf_model);
ENet=caffe.Net(prototxt_dir,model_dir,'test');
load Coord5points2
imglist=dir('./TEST/');
imglist=struct2cell(imglist);
imglist=imglist(1,3:end);
for i=686:length(imglist)
    i
	img=imread(['./TEST/' imglist{i}]);
    if ndims(img)==2
        img = img(:,:,[1,1,1]);
    end
	%we recommend you to set minsize as x * short side
	minl=min([size(img,1) size(img,2)]);
	minsize=fix(minl*0.05);
    [boudingboxes points]=detect_face(img,minsize,PNet,RNet,ONet,threshold,false,factor);
    boudingboxes = uint16(boudingboxes);
    
    boudingboxes(boudingboxes>size(img,2)) = size(img,2);
    boudingboxes(boudingboxes>size(img,1)) = size(img,1);
    % save the non-aligned faces
    save_dir = ['TEST_nonaligned_faces/' imglist{i}];
    mkdir(save_dir);

	numbox=size(boudingboxes,1);
    score=zeros(1,3);
    count=0;
    boudingboxes(boudingboxes==0) = 1;
	for j=1:numbox
		if boudingboxes(j,4)>size(img,2)
			boudingboxes(j,4)=size(img,2)
		end
		if boudingboxes(j,3)>size(img,1)
			boudingboxes(j,3)=size(img,1)
		end 
        non_aligned_face = img( boudingboxes(j,2):boudingboxes(j,4),boudingboxes(j,1):boudingboxes(j,3), :);
        imwrite(non_aligned_face, fullfile(save_dir, [num2str(j), '.png']));
        
    end
end