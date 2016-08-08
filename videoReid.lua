-- Copyright (c) 2016 Niall McLaughlin, CSIT, Queen's University Belfast, UK
-- Contact: nmclaughlin02@qub.ac.uk
-- If you use this code please cite:
-- "Recurrent Convolutional Network for Video-based Person Re-Identification",
-- N McLaughlin, J Martinez Del Rincon, P Miller, 
-- IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
-- 
-- This software is licensed for research and non-commercial use only.
-- 
-- The above copyright notice and this permission notice shall be included in
-- all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
-- THE SOFTWARE.

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'cunn'
require 'cutorch'
require 'image'
require 'paths'
require 'rnn'

require 'buildModel'
require 'train'
require 'test'

local datasetUtils = require 'datasetUtils'
local prepDataset = require 'prepareDataset'

-- set the GPU
cutorch.setDevice(1)

cmd = torch.CmdLine()
cmd:option('-nEpochs',500,'number of training epochs')
cmd:option('-dataset',1,'1 -  ilids, 2 - prid')
cmd:option('-sampleSeqLength',16,'length of sequence to train network')
cmd:option('-gradClip',5,'magnitude of clip on the RNN gradient')
cmd:option('-saveFileName','basicnet','name to save dataset file')
cmd:option('-usePredefinedSplit',false,'Use predefined test/training split loaded from a file')
cmd:option('-dropoutFrac',0.6,'fraction of dropout to use between layers')
cmd:option('-dropoutFracRNN',0.6,'fraction of dropout to use between RNN layers')
cmd:option('-samplingEpochs',100,'how often to compute the CMC curve - dont compute too much - its slow!')
cmd:option('-disableOpticalFlow',false,'use optical flow features or not')
cmd:option('-seed',1,'random seed')
cmd:option('-learningRate',1e-3)
cmd:option('-momentum',0.9)
cmd:option('-nConvFilters',32)
cmd:option('-embeddingSize',128)
cmd:option('-hingeMargin',2)

opt = cmd:parse(arg)
print(opt)

function isnan(z)
    return z ~= z
end

torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

-- change these paths to point to the place where you store i-lids or prid datasets
homeDir = paths.home
if opt.dataset == 1 then
    seqRootRGB = homeDir .. '/Documents/i-LIDS-VID/sequences/'
    seqRootOF = homeDir .. '/Documents/i-LIDS-VID-OF-HVP/sequences/'
else
    seqRootRGB = homeDir .. '/Documents/PRID2011/multi_shot/'
    seqRootOF = homeDir .. '/Documents/PRID2011-OF-HVP/multi_shot/'
end

print('loading Dataset - ',seqRootRGB,seqRootOF)
dataset = prepDataset.prepareDataset(seqRootRGB,seqRootOF,'.png')
print('dataset loaded',#dataset,seqRootRGB,seqRootOF)

if opt.usePredefinedSplit then
    -- useful for debugging to run with exactly the same test/train split
    print('loading predefined test/training split')
    local datasetSplit
    if opt.dataset == 1 then
        datasetSplit = torch.load('./data/dataSplit.th7')
    else
        datasetSplit = torch.load('./data/dataSplit_PRID2011.th7')
    end    
    testInds = datasetSplit.testInds
    trainInds = datasetSplit.trainInds
else
    print('randomizing test/training split')
    trainInds,testInds = datasetUtils.partitionDataset(#dataset,0.5)
end

-- build the model
fullModel,criterion,Combined_CNN_RNN,baseCNN = buildModel_MeanPool_RNN(16,opt.nConvFilters,opt.nConvFilters,trainInds:size(1))

-- train the model
trainedModel,trainedConvnet,trainedBaseNet = trainSequence(fullModel,Combined_CNN_RNN,baseCNN,criterion,dataset,nSamplesPerPerson,trainInds,testInds,nEpochs)

-- save the Model and Convnet (which is part of the model) to a file
saveFileNameModel = './trainedNets/fullModel_' .. opt.saveFileName .. '.dat'
torch.save(saveFileNameModel,trainedModel)
saveFileNameConvnet = './trainedNets/convNet_' .. opt.saveFileName .. '.dat'
torch.save(saveFileNameConvnet,trainedConvnet)
saveFileNameBasenet = './trainedNets/baseNet_' .. opt.saveFileName .. '.dat'
torch.save(saveFileNameBasenet,trainedBaseNet)

------------------------------------------------------------------------------------------------------------------------------------
-- Evaluation
------------------------------------------------------------------------------------------------------------------------------------

trainedConvnet:evaluate()
nTestImages = {1,2,4,8,16,32,64,128}

for n = 1,#nTestImages do
    print('test multiple images '..nTestImages[n])
    -- default method of computing CMC curve
    computeCMC_MeanPool_RNN(dataset,testInds,trainedConvnet,128,nTestImages[n])
end