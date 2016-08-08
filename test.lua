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

-- standard method of computing the CMC curve using sequences
function computeCMC_MeanPool_RNN(personImgs,cmcTestInds,net,outputSize,sampleSeqLength)

	net:evaluate()

    local nPersons = cmcTestInds:size(1)

    local avgSame = 0
    local avgDiff = 0
    local avgSameCount = 0
    local avgDiffCount = 0

    local simMat = torch.zeros(nPersons,nPersons)

    for shiftx = 1,8 do
        for doflip = 1,2 do
        local shifty = shiftx
            local feats_cam_a = torch.DoubleTensor(nPersons,outputSize)            
            for i = 1,nPersons do
                local actualSampleLen = 0
                local seqLen = personImgs[cmcTestInds[i]][1]:size(1)
                if seqLen > sampleSeqLength then
                    actualSampleLen = sampleSeqLength
                else
                    actualSampleLen = seqLen
                end
                seq_length = actualSampleLen
                local seq = personImgs[cmcTestInds[i]][1][{{1,1 + (actualSampleLen - 1)},{},{}}]:squeeze():clone()
                if seq:dim() == 3 then
	        		seq:resize(1,seq:size(1),seq:size(2),seq:size(3))
	        	end
                -- augment each of the images in the sequence
                local augSeq = {}
                local feats_cam_a_mp = {}
                for k = 1,actualSampleLen do
                    local u = seq[{{k},{},{},{}}]:squeeze():clone()
                    if doflip == 1 then
                        u = image.hflip(u)
                    end
                    u = image.crop(u,shiftx,shifty,40+shiftx,56+shifty)
                    u = u - torch.mean(u)
                    augSeq[k] = u:cuda():clone()
                end
                --feats_cam_a[{i,{}}] = net:forward(augSeq):double()
                feats_cam_a[{i,{}}] = net:forward(augSeq):double()
            end

            local feats_cam_b = torch.DoubleTensor(nPersons,outputSize)            
            for i = 1,nPersons do
                local actualSampleLen = 0
                local seqOffset = 0
                local seqLen = personImgs[cmcTestInds[i]][2]:size(1)                
                if seqLen > sampleSeqLength then
                    actualSampleLen = sampleSeqLength
                    seqOffset = (seqLen - sampleSeqLength) + 1
                else
                    actualSampleLen = seqLen
                    seqOffset = 1
                end
                seq_length = actualSampleLen
                local seq = personImgs[cmcTestInds[i]][2][{{seqOffset,seqOffset + (actualSampleLen - 1)},{},{}}]:squeeze():clone()
                if seq:dim() == 3 then
	        		seq:resize(1,seq:size(1),seq:size(2),seq:size(3))
	        	end
                -- augment each of the images in the sequence
                local augSeq = {}
                local feats_cam_b_mp = torch.DoubleTensor(actualSampleLen,outputSize)
                for k = 1,actualSampleLen do
                    local u = seq[{{k},{},{},{}}]:squeeze():clone()
                    if doflip == 1 then
                        u = image.hflip(u)
                    end
                    u = image.crop(u,shiftx,shifty,40+shiftx,56+shifty)
                    u = u - torch.mean(u)
                    augSeq[k] = u:cuda():clone()
                end
                --feats_cam_b[{i,{}}] = net:forward(augSeq):double()
                feats_cam_b[{i,{}}] = net:forward(augSeq):double()
            end
            
            for i = 1,nPersons do
                for j = 1, nPersons do
                    local fa = feats_cam_a[{{i},{}}]
                    local fb = feats_cam_b[{{j},{}}]
                    local dst = torch.sqrt(torch.sum(torch.pow(fa - fb,2)))
                    simMat[i][j] = simMat[i][j] + dst
                    if i == j then
                        avgSame = avgSame  + dst
                        avgSameCount = avgSameCount + 1
                    else
                        avgDiff = avgDiff + dst
                        avgDiffCount = avgDiffCount + 1
                    end
                end
            end
        end
    end

    avgSame = avgSame / avgSameCount
    avgDiff = avgDiff / avgDiffCount

    local cmcInds = torch.DoubleTensor(nPersons)
    local cmc = torch.zeros(nPersons)
    local samplingOrder = torch.zeros(nPersons,nPersons)
    for i = 1,nPersons do

        cmcInds[i] = i

        local tmp = simMat[{i,{}}]
        local y,o = torch.sort(tmp)

        --find the element we want
        local indx = 0
        local tmpIdx = 1
        for j = 1,nPersons do
            if o[j] == i then
                indx = j
            end

            -- build the sampling order for the next epoch
            -- we want to sample close images i.e. ones confused with this person
            if o[j] ~= i then
                samplingOrder[i][tmpIdx] = o[j]
                tmpIdx = tmpIdx + 1
            end
        end

        for j = indx,nPersons do
            cmc[j] = cmc[j] + 1
        end
    end
    cmc = (cmc / nPersons) * 100
    cmcString = ''
    for c = 1,50 do
    	if c <= nPersons then
        	cmcString = cmcString .. ' ' .. torch.floor(cmc[c])
    	end
    end
    print(cmcString)

    return cmc,simMat,samplingOrder,avgSame,avgDiff
end