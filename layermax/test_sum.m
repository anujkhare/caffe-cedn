caffe.set_mode_cpu();
model_file = 'test_sum.prototxt';
net = caffe.Net(model_file, 'test');

% input_arr_1 = [ 1, 1,  1, 1, 1, 1, 1, 1, 1, 1];
% input_arr_1 = reshape(input_arr_1, [1,1,1,10]);
% input_arr_2 = [ 1, 1,  1, 1, 1, 1, 1, 1, 1, 1];
% input_arr_2 = reshape(input_arr_2, [1,1,1,10]);
input_arr_1 = [ 10, 0];
input_arr_1 = reshape(input_arr_1, [1,1,1,2]);
input_arr_2 = [ 10, 20];
input_arr_2 = reshape(input_arr_2, [1,1,1,2]);

results = net.forward({input_arr_1; input_arr_2})
results{1}
