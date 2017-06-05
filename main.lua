--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState, checkpoint)

if opt.testOnly then
   local Err, IOU = trainer:test(0, valLoader)
   print(string.format(' * Results loss: %.3f  IOU: %.3f', Err, IOU))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestLoss = math.huge
if checkpoint then
   bestLoss = checkpoint.bestLoss
end
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainLoss, finish = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local valLoss = trainer:test(epoch, valLoader)

   local bestModel = false
   if valLoss < bestLoss then
      bestModel = true
      bestLoss = valLoss
      print(' * Best model ', valLoss)
	else
		print(' * valLoss ', valLoss)
   end

   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt, trainer.iter, bestLoss)
	if finish then
		break
	end
end

print(string.format(' * Finished Err: %6.3f', bestLoss))
