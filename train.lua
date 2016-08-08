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

local datasetUtils = require 'datasetUtils'
local prepDataset = require 'prepareDataset'

-- train the model on the given dataset
function trainSequence(model,Combined_CNN_RNN,baseCNN,criterion,personImages,samplesPerPerson,trainInds,testInds)

	local parameters,gradParameters = model:getParameters()
	print('Number of parameters',parameters:size(1))

	local batchErr  = 0
	local nTrainPersons = trainInds:size(1)

	local optim_state = {
		learningRate = opt.learningRate,
		momentum = opt.momentum,
		}		

    local x = {}
    local y = {}
    for t = 1,opt.sampleSeqLength do
    	table.insert(x,torch.zeros(5,56,40):cuda())
    	table.insert(y,torch.zeros(5,56,40):cuda())
    end
	local netInput = {x,y}

	for eph = 1,opt.nEpochs do

    	local timer = torch.Timer()
    	collectgarbage()

		local order = torch.randperm(nTrainPersons)

		for i = 1,(nTrainPersons*2) do

			-- choose the mode / similar - diff
			local pushPull
			local netInputA
			local netInputB
			local classLabel1
			local classLabel2
			local netTarget

	        if i % 2 == 0 then
	            -- choose a positive pair, both sequences show the same person
	            local pushPull = 1
	            local camA = 1
	            local camB = 2
	            local startA, startB
	            startA,startB,seq_length = datasetUtils.getPosSample(personImages,trainInds,order[i/2],opt.sampleSeqLength)
	            netInputA = personImages[trainInds[order[i/2]]][camA][{{startA,startA + seq_length - 1},{},{}}]:squeeze()
	            netInputB = personImages[trainInds[order[i/2]]][camB][{{startB,startB + seq_length - 1},{},{}}]:squeeze()
	            netTarget = {1,(order[i/2]),(order[i/2])}

	        else
	            -- choose a negative pair, both sequences show different persons
	            local pushPull = -1
	            local seqA,seqB,camA,camB,startA,startB
	            seqA,seqB,camA,camB,startA,startB,seq_length = datasetUtils.getNegSample(personImages,trainInds,opt.sampleSeqLength)
	            netInputA = personImages[trainInds[seqA]][camA][{{startA,startA + seq_length - 1},{},{},{}}]:squeeze()
	            netInputB = personImages[trainInds[seqB]][camB][{{startB,startB + seq_length - 1},{},{},{}}]:squeeze()
	            netTarget = {-1,seqA,seqB}
	        end

	        -- set the parameters for data augmentation. Note that we apply the same augmentation
	        -- to the whole sequence, rather than individual images
	        local crpxA = torch.floor(torch.rand(1):squeeze() * 8) + 1
	        local crpyA = torch.floor(torch.rand(1):squeeze() * 8) + 1
	        local crpxB = torch.floor(torch.rand(1):squeeze() * 8) + 1
	        local crpyB = torch.floor(torch.rand(1):squeeze() * 8) + 1
	        local flipA = torch.floor(torch.rand(1):squeeze() * 2) + 1
	        local flipB = torch.floor(torch.rand(1):squeeze() * 2) + 1

	        -- deal with the case where we have only a single image i.e. meanpool size == 1
	        if netInputA:dim() == 3 then
	        	netInputA:resize(1,netInputA:size(1),netInputA:size(2),netInputA:size(3))
	        	netInputB:resize(1,netInputB:size(1),netInputB:size(2),netInputB:size(3))
	        end

	        -- we can't (easily) deal with sequenes that are too short (complicates the code) - just skip them for now...
	        -- will try to deal with this later...
	        if netInputA:size(1) ~= opt.sampleSeqLength or netInputB:size(1) ~= opt.sampleSeqLength then
	        	goto continue -- yuck!
	        end

	        netInputA = doDataAug(netInputA,crpxA,crpyA,flipA)
	        netInputB = doDataAug(netInputB,crpxB,crpyB,flipB)

			for t = 1,opt.sampleSeqLength do
				netInput[1][t]:copy(netInputA[{{t},{},{},{}}]:squeeze())
				netInput[2][t]:copy(netInputB[{{t},{},{},{}}]:squeeze())
			end			

			-- note that due to a problem with SuperCriterion we must cast
			-- from CUDA to double and back before passing data to/from the
			-- criteiron layer - may be fixed in a future update of Torch...
			-- ... or maybe I'm just not using it right!
			local feval = function(x)

				local batchError = 0
				if x ~= parameters then
					parameters:copy(x)
				end
				gradParameters:zero()
					
				--forward
				local output = model:forward(netInput)
				for p = 1,#output do
					output[p] = output[p]:double()
				end
				local netError = criterion:forward(output,netTarget)

				--backward
				local gradCriterion = criterion:backward(output,netTarget)
				for c = 1,#gradCriterion do
					gradCriterion[c] = gradCriterion[c]:cuda()
				end
				model:backward(netInput,gradCriterion)

				batchErr = batchErr + netError			
				gradParameters:clamp(-opt.gradClip, opt.gradClip)

				return batchError,gradParameters
			end
         	optim.sgd(feval, parameters, optim_state)

			::continue::
		end

		if eph % 1 == 0 then
			local time = timer:time().real
			print(eph .. ' Batch Error ' .. batchErr .. ' ' .. time)
			batchErr = 0
		end

		if (eph % opt.samplingEpochs == 0) then

			model:evaluate()
			Combined_CNN_RNN:evaluate()

			local cmcTest,cmcTrain,simMatTest,simMatTrain

			cmcTest,simMatTest = computeCMC_MeanPool_RNN(personImages,testInds,Combined_CNN_RNN,opt.embeddingSize,opt.sampleSeqLength)
			cmcTrain,simMatTrain = computeCMC_MeanPool_RNN(personImages,trainInds,Combined_CNN_RNN,opt.embeddingSize,opt.sampleSeqLength)

			local outStringTest =  'Test  '
			local outStringTrain = 'Train '
			local printInds = {1,2,3,4,5,6,7,8,9,10}
			for c = 1,#printInds do
				if c < nTrainPersons then
					outStringTest = outStringTest .. torch.floor(cmcTest[printInds[c]]) .. ' '
					outStringTrain = outStringTrain .. torch.floor(cmcTrain[printInds[c]]) .. ' '
				end
			end
			print(outStringTest)
			print(outStringTrain)

			model:training()
			Combined_CNN_RNN:training()
		end
	end

	return model,Combined_CNN_RNN,baseCNN
end

-- perform data augmentation to a sequence of images stored in a torch tensor
function doDataAug(seq,cropx,cropy,flip)
    seqLen = seq:size(1)
    seqChnls = seq:size(2)
    seqDim1 = seq:size(3)
    seqDim2 = seq:size(4)

    -- print(seqLen,seqChnls,seqDim1,seqDim2,cropx,cropy)

    local daData = torch.zeros(seqLen,seqChnls,seqDim1-8,seqDim2-8)
    for t = 1,seqLen do
        -- do the data augmentation here
        thisFrame = seq[{{t}, {},{},{}}]:squeeze():clone()
        if flip == 1 then
            thisFrame = image.hflip(thisFrame)
        end

        thisFrame = image.crop(thisFrame,cropx,cropy,40 + cropx,56 + cropy)
        thisFrame = thisFrame - torch.mean(thisFrame)

        daData[{{t},{},{},{}}] = thisFrame
    end
    return daData
end