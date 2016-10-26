require 'torch'
local csv2tensor = require 'csv2tensor'
require 'nn'

-- Load our training and testing data.
train_data_raw, train_cols = csv2tensor.load("model_data.train.csv")
test_data_raw, test_cols = csv2tensor.load("model_data.test.csv")
classes = { '0 - 5', '5 - 10', '10 - 15', '15 - 20', '20+' }

-- print('Raw Test Data:')
-- print(test_data_raw)

train_attr_count = test_data_raw[1]:size(1)
test_attr_count = test_data_raw[1]:size(1)

if (train_attr_count ~= test_attr_count) then
    print('Training data set columns:')
    print(train_cols)
    print('Testing data set columns:')
    print(test_cols)
    error('Mismatch between training data set and testing data set structure.')
end

-- Extract the last column as the label value for each row.

function normalizeDataset (raw_data)
    attr_count = raw_data[1]:size(1)

    classification_matrix = raw_data[{ {}, {attr_count} }]
    -- classification = classification_matrix:transpose(1,2)[1]:byte()
    -- Start at 2 to skip the index column.
    model_inputs = torch.totable(raw_data[{ {}, {2,(attr_count - 1)} }])

    dataset = {}
    dataset.data = torch.Tensor(model_inputs)
    dataset.label = classification_matrix

    setmetatable(dataset,
        {__index = function(t, i)
                        return {t.data[i], t.label[i]}
                    end}
    );

    function dataset:size()
        return self.data:size(1)
    end

    return dataset
end

train_data = normalizeDataset( train_data_raw )
print('Training Set Size: ' .. train_data:size())

test_data = normalizeDataset( test_data_raw )
print('Testing Set Size: ' .. test_data:size())

-- Normalize the data.
attr_count = train_data.data[1]:size(1)
mean = {}
stdv = {}
for i=1,attr_count do
    local train_attr_data = train_data.data[{ {}, {i} }]
    local test_attr_data = test_data.data[{ {}, {i} }]

    mean[i] = train_attr_data:mean()
    print('Column ' .. i .. ' (' .. train_cols[i] .. '), Mean: ' .. mean[i])
    train_attr_data:add(-mean[i])
    test_attr_data:add(-mean[i])

    stdv[i] = train_attr_data:std()
    print('Column ' .. i .. ' (' .. train_cols[i] .. '), Standard Deviation: ' .. stdv[i])
    train_attr_data:div(stdv[i])
    test_attr_data:div(stdv[i])
end

-- print('Normalized Test Data:')
-- print(test_data)


-- Construct the model
net = nn.Sequential()
inputs = attr_count
outputs = 1

net:add(nn.Linear(inputs, 1))
--net:add(nn.Linear(10, 20))
--net:add(nn.Linear(20, 40))
--net:add(nn.Tanh())
--net:add(nn.Linear(40, 20))
--net:add(nn.Linear(20, 10))
net:add(nn.Linear(1, outputs))


-- Train the model
criterion = nn.MSECriterion()
learningRate = 0.01

for i = 1,train_data:size() do
    local input = train_data.data[i]
    local output = train_data.label[i]

    -- print('Training with input', input,' and output', output)

    criterion:forward(net:forward(input), output)

    net:zeroGradParameters()
    net:backward(input, criterion:backward(net.output, output))
    net:updateParameters(learningRate)
end


-- Test the model.
correct = 0
for i=1,test_data:size() do
    local input = test_data.data[i]
    local expected_class = test_data.label[i][1]

    -- print('Testing with input', input,' and expected output', expected_class)

    local prediction = net:forward(input)
    local predicted_class = math.floor(prediction[1] + 0.5)
    -- print(prediction[1])

    if expected_class == predicted_class then
        correct = correct + 1
    else
        print('Incorrect.', 'Expected:', expected_class, 'Predicted:', predicted_class)
    end
end

print(correct, 100 * correct / test_data:size() .. ' % ')
