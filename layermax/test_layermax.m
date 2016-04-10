caffe.set_mode_cpu();
model_file = 'test_layermax.prototxt';

net = caffe.Net(model_file, 'test');

% NOTE: THERE IS SOME DISCREPANCY IN THE SHAPE OF INPUT
% ERROR IN 6,7
input_arr_1 = [ 10, 1, 222, 33, -11, -11111, 0];
input_arr_1 = reshape(input_arr_1, [1,1,1,7]);

results = net.forward({input_arr_1})
results{1}
