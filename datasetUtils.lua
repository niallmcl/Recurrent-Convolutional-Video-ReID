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

local dataset_utils = {}

-- given the dataset, which consists of a table where t[x] contains the images for person x
-- split the dataset into testing and training parts
function dataset_utils.partitionDataset(nTotalPersons,testTrainSplit)
    local splitPoint = torch.floor(nTotalPersons * testTrainSplit)
    local inds = torch.randperm(nTotalPersons)

    -- save the inds to a mat file
    --mattorch.save('rnnInds.mat',inds)

    trainInds = inds[{{1,splitPoint}}]
    testInds = inds[{{splitPoint+1,nTotalPersons}}]

    print('N train = ' .. trainInds:size(1))
    print('N test  = ' .. testInds:size(1))

    -- save the split to a file for later use
    -- datasetSplit = {
    --     trainInds = trainInds,
    --     testInds = testInds,
    -- }
    -- torch.save('./trainedNets/dataSplit_PRID2011.th7',datasetSplit)

    return trainInds,testInds
end

-- the dataset format is dataset[person][camera][nSeq][nCrop][FeatureVec]
-- choose a pair of sequences from the same person
function dataset_utils.getPosSample(dataset,trainInds,person,sampleSeqLen)

    -- choose the camera, ilids video only has two, but change this for other datasets
    local camA = 1
    local camB = 2

    local actualSampleSeqLen = sampleSeqLen
    local nSeqA = dataset[trainInds[person]][camA]:size(1)
    local nSeqB = dataset[trainInds[person]][camB]:size(1)

    -- what to do if the sequence is shorter than the sampleSeqLen 
    if nSeqA <= sampleSeqLen or nSeqB <= sampleSeqLen then
        if nSeqA < nSeqB then
            actualSampleSeqLen = nSeqA
        else
            actualSampleSeqLen = nSeqB
        end
    end

    local startA = torch.floor(torch.rand(1)[1] * ((nSeqA - actualSampleSeqLen) + 1)) + 1    
    local startB = torch.floor(torch.rand(1)[1] * ((nSeqB - actualSampleSeqLen) + 1)) + 1

    return startA,startB,actualSampleSeqLen
end

-- the dataset format is dataset[person][camera][nSeq][nCrop][FeatureVec]
-- choose a pair of sequences from different people
function dataset_utils.getNegSample(dataset,trainInds,sampleSeqLen)

    local permAllPersons = torch.randperm(trainInds:size(1))
    local personA = permAllPersons[1]--torch.floor(torch.rand(1)[1] * 2) + 1
    local personB = permAllPersons[2]--torch.floor(torch.rand(1)[1] * 2) + 1

    -- choose the camera, ilids video only has two, but change this for other datasets
    local camA = torch.floor(torch.rand(1)[1] * 2) + 1
    local camB = torch.floor(torch.rand(1)[1] * 2) + 1

    local actualSampleSeqLen = sampleSeqLen
    local nSeqA = dataset[trainInds[personA]][camA]:size(1)
    local nSeqB = dataset[trainInds[personB]][camB]:size(1)

    -- what to do if the sequence is shorter than the sampleSeqLen 
    if nSeqA <= sampleSeqLen or nSeqB <= sampleSeqLen then
        if nSeqA < nSeqB then
            actualSampleSeqLen = nSeqA
        else
            actualSampleSeqLen = nSeqB
        end
    end

    local startA = torch.floor(torch.rand(1)[1] * ((nSeqA - actualSampleSeqLen) + 1)) + 1  
    local startB = torch.floor(torch.rand(1)[1] * ((nSeqB - actualSampleSeqLen) + 1)) + 1

    return personA,personB,camA,camB,startA,startB,actualSampleSeqLen
end

return dataset_utils