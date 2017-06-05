require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'lfs'
require 'paths'
ffi = require 'ffi'
image = require 'image'
local models = require 'models/init'
local opts = require 'opts'
local DataLoader = require 'dataloader'
local checkpoints = require 'checkpoints'

opt = opts.parse(arg)

local classNum = opt.classNum
local show = false
local color = --x/255--torch.rand(classNum, 3)
torch.Tensor({{0,  0,  0},{0,  0,  0},{0,  0,  0},{0,  0,  0},{0,  0,  0},{111, 74,  0},{ 81,  0, 81},{128, 64,128},{244, 35,232},{250,170,160},{230,150,140},{70, 70, 70},{102,102,156},{190,153,153},{180,165,180},{150,100,100},{150,120, 90},{153,153,153},{153,153,153},{250,170, 30},{220,220,  0},{107,142, 35},{152,251,152},{70,130,180},{220, 20, 60},{255,  0,  0},{ 0,  0,142},{ 0,  0, 70},{0, 60,100},{ 0,  0, 90},{  0,  0,110},{ 0, 80,100},{  0,  0,230},{119, 11, 32},{0,  0,142}})/255 
 

if show == true then
	opt.batchSize = 1
end

checkpoint, optimState = checkpoints.latest(opt)
model = models.setup(opt, checkpoint)
print(model)
local trainLoader, valLoader = DataLoader.create(opt)
print('data loaded')
input = torch.CudaTensor()
target = torch.CudaTensor()
function copyInputs(sample)
	input = input or (opt.nGPU == 1
		and torch.CudaTensor()
		or cutorch.createCudaHostTensor())
	target = target or (torch.CudaLongTensor and torch.CudaLongTensor()or torch.CudaTensor())
	input:resize(sample.input:size()):copy(sample.input)
	target:resize(sample.target:size()):copy(sample.target)
	return input, target
end

function sleep(n)
	os.execute("sleep " .. tonumber(n))
end

model:evaluate()
for n, sample in valLoader:run() do
	print(n)
	input, target = copyInputs(sample)
	local imgpath = sample.imgpath
	local img = image.load(opt.data .. ffi.string(imgpath[1]:data()), 3, 'float')
	output = model:forward(input):float()
	local outputn
	for b = 1, input:size(1) do
		print(ffi.string(imgpath[b]:data()))
		outputn = output[{b,{},{},{}}]
		local h = outputn:size(2)
		local w = outputn:size(3)
		results = torch.FloatTensor(3,h,w):fill(0)
		local max, maxMap = torch.max(outputn, 1)
		for i = 1,h do
			for j = 1,w do
				for channel = 1,3 do
					results[{channel,i,j}] = color[maxMap[1][i][j]][channel]
				end
			end
		end
		-- for class = 1, classNum do
		-- 	for channel = 1, 3 do
		-- 	results[{channel,{},{}}] = results[{channel,{},{}}] + outputn[{class,{},{}}]*color[class][channel]
		-- 	end
		-- end
		if show==false then
		local savePath, resPath
		local subPath = '.'
		savePath = opt.save .. string.sub(ffi.string(imgpath[b]:data()), 1, -5) .. '.png'
		resPath = savePath
			j = string.find(resPath, '/')
			while j do
				subPath = subPath .. '/' .. string.sub(resPath, 1, j-1)
				if not paths.dirp(subPath) then
					lfs.mkdir(subPath)
				end
				resPath = string.sub(resPath, j+1, -1)
				j = string.find(resPath, '/')
			end
		image.save(savePath, results)
		end
	end
	if show==true then
		image.display(results)
		image.display(img)
		sleep(3)
	end
end

