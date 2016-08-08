% Copyright (c) 2016 Niall McLaughlin, CSIT, Queen's University Belfast, UK
% Contact: nmclaughlin02@qub.ac.uk
% If you use this code please cite:
% "Recurrent Convolutional Network for Video-based Person Re-Identification",
% N McLaughlin, J Martinez Del Rincon, P Miller, 
% IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
% 
% This software is licensed for research and non-commercial use only.
% 
% The above copyright notice and this permission notice shall be included in
% all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THE SOFTWARE.

%read an image sequence in the ilids video / PRID dataset
%compute optical flow

rootDir = fullfile('C:','Users','3047122','Documents','Test Data');

for person = 1:319
    disp(person)
    for cam = 1:2
        
        camName = {'cam_a','cam_b'};
        
        dataDir = fullfile(rootDir,'i-LIDS-VID','sequences',['cam',num2str(cam)],['person',sprintf('%03i',person)]);
        %dataDir = fullfile(rootDir,'PRID2011','multi_shot',camName{cam},['person_',sprintf('%04i',person)]);
        files = dir(dataDir);
        
        if exist(dataDir)

            saveDir = fullfile(rootDir,'i-LIDS-VID-OF-HVP','sequences',['cam',num2str(cam)],['person',sprintf('%03i',person)]);
            %saveDir = fullfile(rootDir,'PRID2011-OF-HVP','multi_shot',camName{cam},['person_',sprintf('%04i',person)]);
            if ~exist(saveDir)
                mkdir(saveDir);
            end

            seqFiles = {};
            for f = 1:length(files)
                if length(files(f).name) > 4 && ~isempty(findstr(files(f).name,'.png'))
                    seqFiles = [seqFiles files(f).name];
                end
            end

            optical = vision.OpticalFlow('Method','Lucas-Kanade','OutputValue', 'Horizontal and vertical components in complex form');

            for f = 1:length(seqFiles)
                seqImg = imread(fullfile(dataDir,seqFiles{f}));
                optFlow = step(optical,double(rgb2gray(seqImg)));
                
                %separate optFlow into mag and phase components
                R = abs(optFlow);
                theta = angle(optFlow);                
                
                %threshold to remove pixels with large magnitude values
                ofThreshold = 50;
                R = min(R,ofThreshold);
                R = max(R,-1*ofThreshold);                
                
                %convert back to complex form
                Z = R.*exp(1i*theta);                

                H = imag(optFlow);
                V = real(optFlow);
                M = abs(optFlow);
                
                H = H + 127;
                V = V + 127;
                M = M + 127;
                P = theta + 127;

                imgDims = size(seqImg);
                tmpImg = zeros(imgDims);
                tmpImg(:,:,1) = H;
                tmpImg(:,:,2) = V;
                tmpImg(:,:,3) = 0;
                
                tmpImg(tmpImg < 0) = 0;
                tmpImg(tmpImg > 255) = 255;

                tmpImg = tmpImg ./ 255;            

                %save optical flow image to file
                saveFile = fullfile(saveDir,seqFiles{f});
                imwrite(tmpImg,saveFile);
            end
        end
    end
end