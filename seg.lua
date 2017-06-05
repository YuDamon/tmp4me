local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local LaneDataset = torch.class('resnet.LaneDataset', M)

function LaneDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = opt.data
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function LaneDataset:get(i)
   local imgpath = ffi.string(self.imageInfo.imagePath[i]:data())
   local lbpath = ffi.string(self.imageInfo.labelPath[i]:data())
   local image = self:_loadImage(self.dir .. imgpath, 3, 'float')
   local label = self:_loadImage(self.dir .. lbpath, 1, 'byte')
--torch.zeros(1,512,1024)--跑自己的数据库做的更改
	label:add(1)
   return {
      input = image,
      target = label,
      imgpath = self.imageInfo.imagePath[i],
   }
end

function LaneDataset:get_seq(i, frame)
   local imgpath = ffi.string(self.imageInfo.imagePath[i+frame-1]:data())
   local lbpath = ffi.string(self.imageInfo.labelPath[i+frame-1]:data())
   local image = self:_loadImage(self.dir .. imgpath, 3, 'float')
   local label = self:_loadImage(self.dir .. lbpath, 1, 'byte')
	label:add(1)  
   return {
      input = image,
      target = label,
      imgpath = self.imageInfo.imagePath[i+frame-1],
   }
end

function LaneDataset:_loadImage(path, channel, ttype)
	local ok, input = pcall(function()
		return image.load(path, channel, ttype)
	end)

	if not ok then
		print("load image failed!")
		print(path)
		return -1
	end
	return input
end

function LaneDataset:size()
	return self.imageInfo.imagePath:size(1)
end

local meanstd = {
	mean = { 0.3598, 0.3653, 0.3662 },
	std = { 0.2573, 0.2663, 0.2756 },
}

function LaneDataset:preprocess()           -- Don't use data augmentation for training RNN
	if self.split == 'train' then
	return t.Compose{
			t.ScaleWH(512,256),    --WH都得是8的倍数 --cityscape 原图2048*1024
			t.ColorNormalize(meanstd),
		}
	elseif self.split == 'val' then
		return t.Compose{
			t.ScaleWH(512,256),  --WH都得是8的倍数
			t.ColorNormalize(meanstd),
		}
	else
		error('invalid split: ' .. self.split)
	end
end

function LaneDataset:preprocess_aug()
	if self.split == 'train' then
	return t.Compose{
			t.RandomScaleRatio(460,563,230,282),
			t.Rotation(5),
			t.RandomCrop(456, 224),
			t.ColorNormalize(meanstd),
		}
	elseif self.split == 'val' then
		return t.Compose{
			t.ScaleWH(512,256),
			t.ColorNormalize(meanstd),
		}
	else
		error('invalid split: ' .. self.split)
	end
end

return M.LaneDataset
