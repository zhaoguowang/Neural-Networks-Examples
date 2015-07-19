% This function trains a neural network language model.
function [model] = train(epochs)
% Inputs:
%   epochs: Number of epochs to run.
% Output:
%   model: A struct containing the learned weights and biases and vocabulary.

if size(ver('Octave'),1)
  OctaveMode = 1;
  warning('error', 'Octave:broadcast');
  start_time = time;
else
  OctaveMode = 0;
  start_time = clock;
end

% SET HYPERPARAMETERS HERE.
batchsize = 100;  % Mini-batch size.
learning_rate = 0.1;  % Learning rate; default = 0.1.
momentum = 0.9;  % Momentum; default = 0.9.
numhid1 = 50;  % Dimensionality of embedding space; default = 50.
numhid2 = 200;  % Number of units in hidden layer; default = 200.
init_wt = 0.01;  % Standard deviation of the normal distribution
                 % which is sampled to get the initial weights; default = 0.01

% VARIABLES FOR TRACKING TRAINING PROGRESS.
show_training_CE_after = 100;
show_validation_CE_after = 1000;

% LOAD DATA.
[train_input, train_target, valid_input, valid_target, ...
  test_input, test_target, vocab] = load_data(batchsize);
[numwords, batchsize, numbatches] = size(train_input); 

vocab_size = size(vocab, 2);

%numwords = 3
%numbatches = samples/batchsize
%train_input: numwords X batchsize X numbatches
%train_target: numwords X batchsize X numbatches


% INITIALIZE WEIGHTS AND BIASES.

%vocab_size X numhid1 = 250 X 50
word_embedding_weights = init_wt * randn(vocab_size, numhid1);
%word_embedding_weights = init_wt * ones(vocab_size, numhid1);

%(numwords * numhid1) X numhid2 = 150 X 200
embed_to_hid_weights = init_wt * randn(numwords * numhid1, numhid2);
%embed_to_hid_weights = init_wt * ones(numwords * numhid1, numhid2);

%numhid2 X vocab_size = 200 X 250
hid_to_output_weights = init_wt * randn(numhid2, vocab_size);
%hid_to_output_weights = init_wt * ones(numhid2, vocab_size);

% numhid2 X 1 = 200 X 1
hid_bias = zeros(numhid2, 1);

%vocab X 1 = 250 X 1
output_bias = zeros(vocab_size, 1);

%vocab_size X numhid1 = 250 X 50
word_embedding_weights_delta = zeros(vocab_size, numhid1);

%vocab_size X numhid1 = 250 X 50
word_embedding_weights_gradient = zeros(vocab_size, numhid1);

%(numwords * numhid1) X numhid2 = 150 X 200
embed_to_hid_weights_delta = zeros(numwords * numhid1, numhid2);

%numhid2 X vocab_size = 200 X 250
hid_to_output_weights_delta = zeros(numhid2, vocab_size);

% numhid2 X 1 = 200 X 1
hid_bias_delta = zeros(numhid2, 1);

%vocab X 1 = 250 X 1
output_bias_delta = zeros(vocab_size, 1);

%vocab X vocab = 250 X 250
expansion_matrix = eye(vocab_size);


count = 0;
tiny = exp(-30);

% TRAIN.
for epoch = 1:epochs
  fprintf(1, 'Epoch %d\n', epoch);
  this_chunk_CE = 0;
  trainset_CE = 0;
  % LOOP OVER MINI-BATCHES.
  for m = 1:numbatches
    %input_batch: numwords X batchsize = 3 X 100
    input_batch = train_input(:, :, m);

    %target_batch: numwords X batchsize = 1 X 100
    target_batch = train_target(:, :, m);

    % FORWARD PROPAGATE.
    % Compute the state of each layer in the network given the input batch
    % and all weights and biases
    [embedding_layer_state, hidden_layer_state, output_layer_state] = ...
      fprop(input_batch, ...
            word_embedding_weights, embed_to_hid_weights, ...
            hid_to_output_weights, hid_bias, output_bias);

    % COMPUTE DERIVATIVE.
    %% Expand the target to a sparse 1-of-K vector.
    % vocab_size X batchsize = 250 X 100
    expanded_target_batch = expansion_matrix(:, target_batch);

    %% Compute derivative of cross-entropy loss function.
    % vocab_size X batchsize = 250 X 100
    error_deriv = output_layer_state - expanded_target_batch;

    % MEASURE LOSS FUNCTION.
    CE = -sum(sum(...
      expanded_target_batch .* log(output_layer_state + tiny))) / batchsize;
    count =  count + 1;
    this_chunk_CE = this_chunk_CE + (CE - this_chunk_CE) / count;
    trainset_CE = trainset_CE + (CE - trainset_CE) / m;
    fprintf(1, '\rBatch %d Train CE %.3f', m, this_chunk_CE);
    if mod(m, show_training_CE_after) == 0
      fprintf(1, '\n');
      count = 0;
      this_chunk_CE = 0;
    end
    if OctaveMode
      fflush(1);
    end

    % BACK PROPAGATE.
    %% OUTPUT LAYER.
    % 200 X 250
    hid_to_output_weights_gradient =  hidden_layer_state * error_deriv';
    output_bias_gradient = sum(error_deriv, 2);

    % if e == 100
    	save -append  output_bias_gradient.mat  output_bias_gradient  
    % end

    % 200 X 100 
    back_propagated_deriv_1 = (hid_to_output_weights * error_deriv) ...
      .* hidden_layer_state .* (1 - hidden_layer_state);

    % if e == 100
    	save -append   back_propagated_deriv_1.mat   back_propagated_deriv_1 
    % end

    %% HIDDEN LAYER.
    % FILL IN CODE. Replace the line below by one of the options.
    %embed_to_hid_weights_gradient = zeros(numhid1 * numwords, numhid2);

    % 150 X 200
    embed_to_hid_weights_gradient = embedding_layer_state * back_propagated_deriv_1';
    
    % if e == 100
    	save -append  embed_to_hid_weights_gradient.mat  embed_to_hid_weights_gradient 
    % end

    % Options:
    % (a) embed_to_hid_weights_gradient = back_propagated_deriv_1' * embedding_layer_state;
    % (b) embed_to_hid_weights_gradient = embedding_layer_state * back_propagated_deriv_1';
    % (c) embed_to_hid_weights_gradient = back_propagated_deriv_1;
    % (d) embed_to_hid_weights_gradient = embedding_layer_state;

    % FILL IN CODE. Replace the line below by one of the options.
    %hid_bias_gradient = zeros(numhid2, 1);
    hid_bias_gradient = sum(back_propagated_deriv_1, 2);

    %save -append hid_bias_gradient.mat hid_bias_gradient

    % Options
    % (a) hid_bias_gradient = sum(back_propagated_deriv_1, 2);
    % (b) hid_bias_gradient = sum(back_propagated_deriv_1, 1);
    % (c) hid_bias_gradient = back_propagated_deriv_1;
    % (d) hid_bias_gradient = back_propagated_deriv_1';

    % FILL IN CODE. Replace the line below by one of the options.
    %back_propagated_deriv_2 = zeros(numhid2, batchsize);

    % (numwords * numhid1) X batchsize = 150 X 100
    back_propagated_deriv_2 = embed_to_hid_weights * back_propagated_deriv_1;

    % if e == 100
    	save -append back_propagated_deriv_2.mat back_propagated_deriv_2
    % end

    % Options
    % (a) back_propagated_deriv_2 = embed_to_hid_weights * back_propagated_deriv_1;
    % (b) back_propagated_deriv_2 = back_propagated_deriv_1 * embed_to_hid_weights;
    % (c) back_propagated_deriv_2 = back_propagated_deriv_1' * embed_to_hid_weights;
    % (d) back_propagated_deriv_2 = back_propagated_deriv_1 * embed_to_hid_weights';

    word_embedding_weights_gradient(:) = 0;
    %% EMBEDDING LAYER.
    for w = 1:numwords
       word_embedding_weights_gradient = word_embedding_weights_gradient + ...
         expansion_matrix(:, input_batch(w, :)) * ...
         (back_propagated_deriv_2(1 + (w - 1) * numhid1 : w * numhid1, :)');
    end

    % if e == 100
    	save -append  word_embedding_weights_gradient.mat  word_embedding_weights_gradient
    % end
   
    % UPDATE WEIGHTS AND BIASES.
    word_embedding_weights_delta = ...
      momentum .* word_embedding_weights_delta + ...
      word_embedding_weights_gradient ./ batchsize;
    word_embedding_weights = word_embedding_weights...
      - learning_rate * word_embedding_weights_delta;

    embed_to_hid_weights_delta = ...
      momentum .* embed_to_hid_weights_delta + ...
      embed_to_hid_weights_gradient ./ batchsize;
    embed_to_hid_weights = embed_to_hid_weights...
      - learning_rate * embed_to_hid_weights_delta;

    hid_to_output_weights_delta = ...
      momentum .* hid_to_output_weights_delta + ...
      hid_to_output_weights_gradient ./ batchsize;
    hid_to_output_weights = hid_to_output_weights...
      - learning_rate * hid_to_output_weights_delta;

    hid_bias_delta = momentum .* hid_bias_delta + ...
      hid_bias_gradient ./ batchsize;
    hid_bias = hid_bias - learning_rate * hid_bias_delta;

    output_bias_delta = momentum .* output_bias_delta + ...
      output_bias_gradient ./ batchsize;
    output_bias = output_bias - learning_rate * output_bias_delta;

    % VALIDATE.
    if mod(m, show_validation_CE_after) == 0
      fprintf(1, '\rRunning validation ...');
      if OctaveMode
        fflush(1);
      end
      [embedding_layer_state, hidden_layer_state, output_layer_state] = ...
        fprop(valid_input, word_embedding_weights, embed_to_hid_weights,...
              hid_to_output_weights, hid_bias, output_bias);
      datasetsize = size(valid_input, 2);
      expanded_valid_target = expansion_matrix(:, valid_target);
      CE = -sum(sum(...
        expanded_valid_target .* log(output_layer_state + tiny))) /datasetsize;
      fprintf(1, ' Validation CE %.3f\n', CE);
      if OctaveMode
        fflush(1);
      end
    end
  end
  fprintf(1, '\rAverage Training CE %.3f\n', trainset_CE);
end
fprintf(1, 'Finished Training.\n');
if OctaveMode
  fflush(1);
end
fprintf(1, 'Final Training CE %.3f\n', trainset_CE);

% EVALUATE ON VALIDATION SET.
fprintf(1, '\rRunning validation ...');
if OctaveMode
  fflush(1);
end
[embedding_layer_state, hidden_layer_state, output_layer_state] = ...
  fprop(valid_input, word_embedding_weights, embed_to_hid_weights,...
        hid_to_output_weights, hid_bias, output_bias);
datasetsize = size(valid_input, 2);
expanded_valid_target = expansion_matrix(:, valid_target);
CE = -sum(sum(...
  expanded_valid_target .* log(output_layer_state + tiny))) / datasetsize;
fprintf(1, '\rFinal Validation CE %.3f\n', CE);
if OctaveMode
  fflush(1);
end

% EVALUATE ON TEST SET.
fprintf(1, '\rRunning test ...');
if OctaveMode
  fflush(1);
end
[embedding_layer_state, hidden_layer_state, output_layer_state] = ...
  fprop(test_input, word_embedding_weights, embed_to_hid_weights,...
        hid_to_output_weights, hid_bias, output_bias);
datasetsize = size(test_input, 2);
expanded_test_target = expansion_matrix(:, test_target);
CE = -sum(sum(...
  expanded_test_target .* log(output_layer_state + tiny))) / datasetsize;
fprintf(1, '\rFinal Test CE %.3f\n', CE);
if OctaveMode
  fflush(1);
end

model.word_embedding_weights = word_embedding_weights;
model.embed_to_hid_weights = embed_to_hid_weights;
model.hid_to_output_weights = hid_to_output_weights;
model.hid_bias = hid_bias;
model.output_bias = output_bias;
model.vocab = vocab;

% In MATLAB replace line below with 'end_time = clock;'
if OctaveMode
  end_time = time;
  diff = end_time - start_time;
else
  end_time = clock;
  diff = etime(end_time, start_time);
end
fprintf(1, 'Training took %.2f seconds\n', diff);
end
