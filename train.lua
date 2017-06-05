--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState, checkpoint)
   print('init trainer')
	self.model = model
   self.criterion = criterion
	--local learningRates, weightDecays = model:getOptimConfig(opt.LR, opt.weightDecay)
   self.optimState = optimState or {
	--	learningRates = learningRates,
	--	weightDecays = weightDecays,
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
	self.iter = 1
	if checkpoint then
		self.iter = checkpoint.iter
	end
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
	self.finish = false
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   print('training')

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
		if self.iter>=self.opt.maxIter then
			self.finish = true
			break
		end
		self.optimState.learningRate = self:learningRate(epoch)
	--	self.optimState.learningRates, self.optimState.weightDecays = self.model:getOptimConfig(self.optimState.learningRate, self.optimState.weightDecay)
		local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)
		self.input = self.input:cuda()
		self.target = self.target:cuda()
      local output = self.model:forward(self.input):float()
      --print(output:size())
		local batchSize = output:size(1)
		--print(self.target:size())
      local loss = self.criterion:forward(self.model.output, self.target)
      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)
      optim.sgd(feval, self.params, self.optimState)
		
      lossSum = lossSum + loss*batchSize
      N = N + batchSize
      print((' | Epoch: [%d][%d/%d][%d]    Time %.3f  Data %.3f  LR %.5f  Err %1.4f'):format(
         epoch, n, trainSize, self.iter, timer:time().real, dataTime, self.optimState.learningRate, loss))

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
		self.iter = self.iter + 1
   end

   return lossSum / N, self.finish
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local IOUSum = 0.0
   local lossSum = 0.0
   local N = 0

   self.model:evaluate()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      self.input = self.input:cuda()
      self.target = self.target:cuda()
      local output = self.model:forward(self.input):float()
      local accuracy, avgRecall, avgIOU = self:computeAccuracy(output, self.target:float())
      local batchSize = output:size(1) / nCrops
      local loss = self.criterion:forward(self.model.output, self.target)

      lossSum = lossSum + loss*batchSize
      IOUSum = IOUSum + avgIOU*batchSize
      N = N + batchSize

      print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  Accuracy %.3f  AvgRecall %.3f  AvgIOU %.3f (%.3f)'):format(
         epoch, n, size, timer:time().real, dataTime, loss, accuracy, avgRecall, avgIOU, IOUSum / N))

      timer:reset()
      dataTimer:reset()
   end
   self.model:training()

   return lossSum / N, IOUSum / N
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or (torch.CudaLongTensor and torch.CudaLongTensor()or torch.CudaTensor())
   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if self.opt.dataset == 'seg' then
      decay = 1 - self.iter/self.opt.maxIter
   elseif self.opt.dataset == 'cifar10' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   elseif self.opt.dataset == 'cifar100' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
   end
   return self.opt.LR * math.pow(decay, 0.9)
end

function Trainer:computeAccuracy( output, target )
   local batchSize = output:size(1)
   local classNum = output:size(2)
   local h = output:size(3)
   local w = output:size(4)
   local accuracy, avgRecall, avgIOU = 0.0, 0.0, 0.0
   for i = 1, batchSize do
      local _, maxMap = torch.max(output[{i,{},{},{}}], 1)
      local target_i = target[{i,{},{}}]:long()
      -- accuracy
      accuracy = accuracy + torch.sum(torch.eq(maxMap, target_i)) / (h*w)
      -- recall, IOU
      local recall = 0.0
      local IOU = 0.0
      local numClass = 0
      for c = 1, classNum do
         local num_c = torch.sum(torch.eq(target_i, c))
         local num_c_pred = torch.sum(torch.eq(maxMap, c))
         local numTrue = torch.sum(torch.cmul(torch.eq(maxMap, c), torch.eq(target_i, c)))
         local unionSize = num_c + num_c_pred - numTrue
         if numTrue == unionSize then
            -- avoid divede by zero
            IOU = IOU + 1.0
         else
            IOU = IOU + numTrue / unionSize
         end
         if num_c > 0 then
            recall = recall + numTrue / num_c
            numClass = numClass + 1
         end
      end
      recall = recall / numClass
      avgRecall = avgRecall + recall
      IOU = IOU / classNum
      avgIOU = avgIOU + IOU
   end
   accuracy = accuracy / batchSize
   avgRecall = avgRecall / batchSize
   avgIOU = avgIOU / batchSize
   return accuracy * 100, avgRecall * 100, avgIOU * 100
end

return M.Trainer
