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
require 'paths'
require 'image'

prepareDataset = {}

--load all images into a flat list
local function loadSequenceImages(cameraDir,opticalflowDir,filesList)

    local nImgs = #filesList
    local imagePixelData
    for i,file in ipairs(filesList) do  

        local filename = paths.concat(cameraDir,file)
        local filenameOF = paths.concat(opticalflowDir,file)

        local img = image.load(filename,3)
        img = image.scale(img,48,64)

        local imgof = image.load(filenameOF,3)
        imgof = image.scale(imgof,48,64)

        -- --allocate storage
        if i == 1 then
            local s = #img
            imagePixelData = torch.DoubleTensor(nImgs,5,s[2],s[3])
        end

        --img = image.rgb2lab(img):type('torch.DoubleTensor')
        img = image.rgb2yuv(img):type('torch.DoubleTensor')

        for c = 1,3 do
            local v = torch.sqrt(torch.var(img[c]))
            local m = torch.mean(img[c])
            img[c] = img[c] - m
            img[c] = img[c] / torch.sqrt(v)
            imagePixelData[{ {i}, {c}, {}, {}}] = img[c]
        end 
        for c = 1,2 do
            local v = torch.sqrt(torch.var(imgof[c]))
            local m = torch.mean(imgof[c])
            imgof[c] = imgof[c] - m
            imgof[c] = imgof[c] / torch.sqrt(v)
            imagePixelData[{ {i}, {c+3}, {}, {}}] = imgof[c]

            if opt.disableOpticalFlow then
                imagePixelData[{ {i}, {c+3}, {}, {}}]:mul(0)
            end
        end 
    end
    return imagePixelData
end

-- given a directory containing all images in a sequence get all the image filenames in order
local function getSequenceImageFiles(seqRoot,filesExt)

    local seqFiles = {}
    -- Go over all files in directory. We use an iterator, paths.files().
    for file in paths.files(seqRoot) do
       -- We only load files that match the extension
       if file:find(filesExt .. '$')then
          -- and insert the ones we care about in our table
          table.insert(seqFiles, file)
       end
    end

    -- Check files exist
    if #seqFiles == 0 then
       error('given directory doesnt contain any files' .. seqRoot)
    end

    -- function used to sort the filenames
    local function numOrd(a,b)
        local k = string.find(a,"%.")
        local in1 = 0
        if opt.dataset == 1 then
            in1 = tonumber(a:sub(k-4,k-1))
        else
            in1 = tonumber(a:sub(k-5,k-1))
        end     

        j = string.find(b,"_")
        k = string.find(b,"%.")
        local in2 = 0
        if opt.dataset == 1 then
            in2 = tonumber(b:sub(k-4,k-1))
        else
            in2 = tonumber(b:sub(k-5,k-1))
        end     

        if in1 < in2 then
            return true
        else
            return false
        end     
    end
    table.sort(seqFiles, numOrd)

    return seqFiles
end

-- get a sorted list of directories for all the persons in the dataset
local function getPersonDirsList(seqRootDir)

    local firstCameraDirName
    if opt.dataset == 1 then
        firstCameraDirName = 'cam1'
    else
        firstCameraDirName = 'cam_a'
    end
    local tmpSeqCam = paths.concat(seqRootDir,firstCameraDirName)

    local personDirs = {}
    -- Go over all files in directory. We use an iterator, paths.files().
    for file in paths.files(tmpSeqCam) do
       -- We only load files that match the extension
       if #file > 2 then
          -- and insert the ones we care about in our table
          table.insert(personDirs, file)
       end
    end

    -- Check files exist
    if #personDirs == 0 then
       error(seqRootDir .. ' directory does not contain any image files')
    end

    local function orderDirs(a,b)
        local strLen = #a
        local delimiter
        if opt.dataset == 1 then
            delimiter = "n"
        else
            delimiter = "_"
        end
        local j = string.find(a,delimiter)
        local pn1 = tonumber(a:sub(j+1,j+4))

        strLen = #b
        j = string.find(b,delimiter)

        local pn2 = tonumber(b:sub(j+1,j+4))

        if pn1 < pn2 then
            return true
        else
            return false
        end     
    end
    table.sort(personDirs, orderDirs)

    return personDirs
end

-- return all images - we can later split this into the training / validation sets
function prepareDataset.prepareDataset(datasetRootDir,datasetRootDirOF,fileExt)
    local dataset = {}
    local personDirs = getPersonDirsList(datasetRootDir)
    local nPersons = #personDirs
    local letter = {'a','b'}
    for i,pdir in ipairs(personDirs) do
        dataset[i] = {}
        for cam = 1,2 do
            local cameraDirName
            if opt.dataset == 1 then
                cameraDirName = 'cam'.. cam             
            else
                cameraDirName = 'cam_'..letter[cam]
            end
            local seqRoot = paths.concat(datasetRootDir,cameraDirName,pdir)
            local seqRootOF = paths.concat(datasetRootDirOF,cameraDirName,pdir)
            local seqImgs = getSequenceImageFiles(seqRoot,fileExt)          
            dataset[i][cam] = loadSequenceImages(seqRoot,seqRootOF,seqImgs)         
        end
        
        -- -- for faster debugging
        -- if i == 10 then
        --  return dataset
        -- end

        -- only use first 200 persons who appear in both cameras for PRID 2011
        if opt.dataset == 2 and i == 200 then
            return dataset
        end
    end
    return dataset
end

return prepareDataset