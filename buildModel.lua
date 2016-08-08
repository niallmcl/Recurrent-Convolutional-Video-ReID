function buildModel_MeanPool_RNN(nFltrs1,nFltrs2,nFltrs3,nPersonsTrain)

	local nFilters = {nFltrs1,nFltrs2,nFltrs3}

	local filtsize = {5,5,5}
	local poolsize = {2,2,2}
	local stepSize = {2,2,2}

	-- remember this adds padding to ALL SIDES of the image
	local padDim = 4

	local cnn = nn.Sequential()

	local ninputChannels = 5
	cnn:add(nn.SpatialZeroPadding(padDim, padDim, padDim, padDim))
	cnn:add(nn.SpatialConvolutionMM(ninputChannels, nFilters[1], filtsize[1], filtsize[1], 1, 1))
	cnn:add(nn.Tanh())
	cnn:add(nn.SpatialMaxPooling(poolsize[1],poolsize[1],stepSize[1],stepSize[1]))

	ninputChannels = nFilters[1]
	cnn:add(nn.SpatialZeroPadding(padDim, padDim, padDim, padDim))
	cnn:add(nn.SpatialConvolutionMM(ninputChannels, nFilters[2], filtsize[2], filtsize[2], 1, 1))
	cnn:add(nn.Tanh())
	cnn:add(nn.SpatialMaxPooling(poolsize[2],poolsize[2],stepSize[2],stepSize[2]))

	ninputChannels = nFilters[2]
	cnn:add(nn.SpatialZeroPadding(padDim, padDim, padDim, padDim))
	cnn:add(nn.SpatialConvolutionMM(ninputChannels, nFilters[3], filtsize[3], filtsize[3], 1, 1))
	cnn:add(nn.Tanh())
	cnn:add(nn.SpatialMaxPooling(poolsize[3],poolsize[3],stepSize[3],stepSize[3]))

	local nFullyConnected = nFilters[3]*10*8	

	cnn:add(nn.Reshape(1,nFullyConnected))
	cnn:add(nn.Dropout(opt.dropoutFrac))	
	cnn:add(nn.Linear(nFullyConnected,opt.embeddingSize))
	cnn:cuda()

	local h2h = nn.Sequential()
	h2h:add(nn.Tanh())
	h2h:add(nn.Dropout(opt.dropoutFracRNN))
	h2h:add(nn.Linear(opt.embeddingSize,opt.embeddingSize))
	h2h:cuda()

	local r1 = nn.Recurrent(
		opt.embeddingSize,
		cnn,
		h2h,
		nn.Identity(),
		opt.sampleSeqLength)

	local rnn1 = nn.Sequencer(
		nn.Sequential()
		:add(r1)
		)

	Combined_CNN_RNN_1 = nn.Sequential()
	Combined_CNN_RNN_1:add(rnn1)
	Combined_CNN_RNN_1:add(nn.JoinTable(1))
	Combined_CNN_RNN_1:add(nn.Mean(1))

	local r2 = nn.Recurrent(
		opt.embeddingSize,
		cnn:clone('weight','bias','gradWeight','gradBias'),
		h2h:clone('weight','bias','gradWeight','gradBias'),
		nn.Identity(),
		opt.sampleSeqLength)

	local rnn2 = nn.Sequencer(
		nn.Sequential()
		:add(r2)
		)

	Combined_CNN_RNN_2 = nn.Sequential()
	Combined_CNN_RNN_2:add(rnn2)
	Combined_CNN_RNN_2:add(nn.JoinTable(1))
	Combined_CNN_RNN_2:add(nn.Mean(1))

	-- Combined_CNN_RNN_2 = Combined_CNN_RNN_1:clone('weight','bias','gradWeight','gradBias')

	local mlp2 = nn.ParallelTable()
	mlp2:add(Combined_CNN_RNN_1)
	mlp2:add(Combined_CNN_RNN_2)
	mlp2:cuda()

	local mlp3 = nn.ConcatTable()
	mlp3:add(nn.Identity())
	mlp3:add(nn.Identity())
	mlp3:add(nn.Identity())
	mlp3:cuda()

	local mlp4 = nn.ParallelTable()
	mlp4:add(nn.Identity())
	mlp4:add(nn.SelectTable(1))
	mlp4:add(nn.SelectTable(2))
	mlp4:cuda()

	-- used to predict the identity of each person
	local classifierLayer = nn.Linear(opt.embeddingSize,nPersonsTrain)

	-- identification
	local mlp6 = nn.Sequential()
	mlp6:add(classifierLayer)
	mlp6:add(nn.LogSoftMax())
	mlp6:cuda()
	
	local mlp7 = nn.Sequential()
	mlp7:add(classifierLayer:clone('weight','bias','gradWeight','gradBias'))
	mlp7:add(nn.LogSoftMax())
	mlp7:cuda()

	local mlp5 = nn.ParallelTable()
	mlp5:add(nn.PairwiseDistance(2))
	mlp5:add(mlp6)
	mlp5:add(mlp7)
	mlp5:cuda()

	local fullModel = nn.Sequential()
	fullModel:add(mlp2)
	fullModel:add(mlp3)
	fullModel:add(mlp4)
	fullModel:add(mlp5)
	fullModel:cuda()

	local crit = nn.SuperCriterion()
	crit:add(nn.HingeEmbeddingCriterion(2),1)
	crit:add(nn.ClassNLLCriterion(),1)
	crit:add(nn.ClassNLLCriterion(),1)

	return fullModel, crit, Combined_CNN_RNN_1, cnn
end