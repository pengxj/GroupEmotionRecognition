clear;
%setting for GAF
remove_small_faces=false;
remove_small_faces_size=24;
%train.caffemodel is trained by trainset
%all.caffemodel is trained by trainset and valset
gaf_model='train.caffemodel';
%path of toolbox
caffe_path='D:\iccv\caffe-windows-master\matlab';
pdollar_toolbox_path='D:\mtcnnv2\toolbox-master'
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
f=fopen('result.txt','w');
imglist=dir('../TEST/');
imglist=struct2cell(imglist);
imglist=imglist(1,3:end);
for i=1:length(imglist)
    i
	img=imread(['../TEST/' imglist{i}]);
	%we recommend you to set minsize as x * short side
	minl=min([size(img,1) size(img,2)]);
	minsize=fix(minl*0.05);
    [boudingboxes points]=detect_face(img,minsize,PNet,RNet,ONet,threshold,false,factor);
    % save the non-aligned faces
    S = regexp(imglist{i}, '.', 'split');
    save_dir = ['../TEST_nonaligned_faces/' S(1)];

	numbox=size(boudingboxes,1);
    score=zeros(1,3);
    count=0;
	for j=1:numbox
        non_aligned_face = img(boudingboxes(j,1):boudingboxes(j,3), boudingboxes(j,2):boudingboxes(j,4), :);
        imwrite(non_aligned_face, [save_dir, num2str(j), '.png'])
        if remove_small_faces && (boudingboxes(j,3)-boudingboxes(j,1)<remove_small_faces_size || boudingboxes(j,4)-boudingboxes(j,2)<remove_small_faces_size)
            continue;
        end
        Tfm =  cp2tform(double([points(1:5,j) points(6:10,j)]), Coord5points', 'similarity');
        cropImg = imtransform(img, Tfm,'XData',[1 96],'YData',[1 112],'Size',[112 96]);
        if size(cropImg, 3) < 3
            cropImg(:,:,2) = cropImg(:,:,1);
            cropImg(:,:,3) = cropImg(:,:,1);
        end
        cropImg = single(cropImg);
        cropImg = (cropImg - 127.5)/128;
        cropImg = permute(cropImg, [2,1,3]);
        cropImg = cropImg(:,:,[3,2,1]);

        cropImg_(:,:,1) = flipud(cropImg(:,:,1));
        cropImg_(:,:,2) = flipud(cropImg(:,:,2));
        cropImg_(:,:,3) = flipud(cropImg(:,:,3));
        score1 = ENet.forward({cropImg});
        score2 = ENet.forward({cropImg_});
        count=count+1;
        score = score+(score1{1}'+score2{1}')/2;
    end
	if count>0
		score=score/count;
	end
    fprintf(f,'%s %d %d %d\n',imglist{i},score(2),score(3),score(1));
end