caffe.set_mode_cpu();
model_file = 'base.prototxt';
net = caffe.Net(model_file, 'test');

fprintf('ello');
