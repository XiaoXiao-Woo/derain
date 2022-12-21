-- usage example: DATA_ROOT=/path/to/data/ which_direction=BtoA name=expt1 th train.lua 
--
-- code derived from https://github.com/phillipi/pix2pix   and https://github.com/jcjohnson/fast-neural-style
--

require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
require 'image'
require 'models2'  -- load your own models
require 'torch'
require 'optim'
require 'image'

require 'fast_neural_style.DataLoader'
require 'fast_neural_style.PerceptualCriterion'
local utils = require 'fast_neural_style.utils'
local preprocess = require 'fast_neural_style.preprocess'
local models = require 'fast_neural_style.models'

local cmd = torch.CmdLine()


-- Generic options for Perceptual Loss

-- Generic loss function options
cmd:option('-pixel_loss_type', 'L2', 'L2|L1|SmoothL1')
cmd:option('-pixel_loss_weight', 0.0)
cmd:option('-percep_loss_weight', 1.0)
cmd:option('-tv_strength', 1e-6)
-- Options for feature reconstructin loss

cmd:option('-content_weights', '1.0')
cmd:option('-content_layers', '9')
cmd:option('-loss_network', 'per_loss/models/vgg16.t7')

-- Options for style reconstruction loss (Just forget this part, we set the style loss weights to be 0)
cmd:option('-style_image', 'images/styles/candy.jpg')
cmd:option('-style_image_size', 256)
cmd:option('-style_weights', '0.0')
cmd:option('-style_layers', '4,9,16,23')
cmd:option('-style_target_type', 'gram', 'gram|mean')


-- Backend options
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
cmd:option('-backend', 'cuda', 'cuda|opencl')




local opt = cmd:parse(arg)

  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)

 -- Parse layer strings and weights
  opt.content_layers, opt.content_weights =
    utils.parse_layers(opt.content_layers, opt.content_weights)
  opt.style_layers, opt.style_weights =
    utils.parse_layers(opt.style_layers, opt.style_weights)


  -- Set up the perceptual loss function
  local percep_crit
  if opt.percep_loss_weight > 0 then
    local loss_net = torch.load(opt.loss_network)
    local crit_args = {
      cnn = loss_net,
      style_layers = opt.style_layers,
      style_weights = opt.style_weights,
      content_layers = opt.content_layers,
      content_weights = opt.content_weights,
      agg_type = opt.style_target_type,
    }
    percep_crit = nn.PerceptualCriterion(crit_args):type(dtype)
end

-- Parameter for training ID-CGAN
opt = {
   DATA_ROOT = '',         -- path to images (should have subfolders 'train', 'val', etc)
   batchSize = 7,          -- # images in batch
   loadSize = 286,         -- scale images to this size
   fineSize = 256,         --  then crop to this size
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 48,               -- #  of discrim filters in first conv layer
   input_nc = 3,           -- #  of input image channels
   output_nc = 3,          -- #  of output image channels
   niter = 2000,            -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   flip = 1,               -- if flip the images for data argumentation
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'rain',              -- name of the experiment, should generally be passed on the command line
   which_direction = 'AtoB',    -- AtoB or BtoA
   phase = 'training',             -- train, val, test, etc
   preprocess = 'regular',      -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
   nThreads = 2,                -- # threads for loading data
   save_epoch_freq = 50,        -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
   save_latest_freq = 5000,     -- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
   print_freq = 50,             -- print the debug information every print_freq iterations
   display_freq = 100,          -- display the current results every display_freq iterations
   save_display_freq = 5000,    -- save the current display of results every save_display_freq_iterations
   continue_train=0,            -- if continue training, load the latest model: 1: true, 0: false
   serial_batches = 0,          -- if 1, takes images in order to make batches, otherwise takes them randomly
   serial_batch_iter = 1,       -- iter into serial image list
   checkpoints_dir = './checkpoints/model_name', -- models are saved here
   cudnn = 1,                         -- set to 0 to not use cudnn (untested)
   condition_GAN = 1,                 -- set to 0 to use unconditional discriminator
   use_GAN = 1,                       -- set to 0 to turn off GAN term
   use_L2 = 1,                        -- set to 0 to turn off L2 term
   use_Lp = 1,                        -- set to 0 to turn off Perceptual term
   which_model_netD = 'basic', -- selects model to use for netD
   which_model_netG = 'our_net',  -- selects model to use for netG
   lambda = 150,               -- weight on Perceptual loss term in objective
   lambda1 = 150,               -- weight on MSE term in objective
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local input_nc = opt.input_nc
local output_nc = opt.output_nc
-- translation direction
local idx_A = nil
local idx_B = nil

if opt.which_direction=='AtoB' then
    idx_A = {1, input_nc}
    idx_B = {input_nc+1, input_nc+output_nc}
elseif opt.which_direction=='BtoA' then
    idx_A = {input_nc+1, input_nc+output_nc}
    idx_B = {1, input_nc}
else
    error(string.format('bad direction %s',opt.which_direction))
end

if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local data_loader = paths.dofile('data/data.lua')
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())
tmp_d, tmp_paths = data:getBatch()

----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

function defineG(input_nc, output_nc, ngf, nz)
   
    if opt.which_model_netG == "our_net" then netG = defineGour_net(input_nc, output_nc, ngf)
    else error("unsupported netG model")
    end
   
   netG:apply(weights_init)
   
   return netG
end

function defineD(input_nc, output_nc, ndf)
    
    local netD = nil
    if opt.condition_GAN==1 then
        input_nc_tmp = input_nc
    else
        input_nc_tmp = 0 -- only penalizes structure in output channels
    end
    
    if     opt.which_model_netD == "basic" then netD = defineD_n_layers(input_nc, output_nc, ndf, 3)
    else error("unsupported netD model")
    end
    
    netD:apply(weights_init)
    
    return netD
end


-- load saved models and finetune
if opt.continue_train == 1 then
   print('loading previously trained netG...')
   netG = util.load(paths.concat(opt.checkpoints_dir, opt.name, '1900_net_G.t7'), opt)
   print('loading previously trained netD...')
   netD = util.load(paths.concat(opt.checkpoints_dir, opt.name, '1900_net_D.t7'), opt)
else
  print('define model netG...')
  netG = defineG(input_nc, output_nc, ngf, nz)
  print('define model netD...')
  netD = defineD(input_nc, output_nc, ndf)
end

print(netG)
print(netD)


local criterion = nn.BCECriterion()
local criterionAE1 = nn.MSECriterion()
local criterionAE = percep_crit



---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)
local real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local errD, errG, errL1 = 0, 0, 0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------

if opt.gpu > 0 then
   print('transferring to gpu...')
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   real_A = real_A:cuda();
   real_B = real_B:cuda(); fake_B = fake_B:cuda();
   real_AB = real_AB:cuda(); fake_AB = fake_AB:cuda();
   if opt.cudnn==1 then
      netG = util.cudnn(netG); netD = util.cudnn(netD);
   end
   netD:cuda(); netG:cuda(); criterion:cuda(); criterionAE:cuda(); criterionAE1:cuda();
   print('done')
end


local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()


function createRealFake()
    -- load real
    data_tm:reset(); data_tm:resume()
    local real_data, data_path = data:getBatch()
    data_tm:stop()
    
    real_A:copy(real_data[{ {}, idx_A, {}, {} }])
    real_B:copy(real_data[{ {}, idx_B, {}, {} }])
    
    if opt.condition_GAN==1 then
        real_AB = torch.cat(real_A,real_B,2)
    else
        real_AB = real_B -- unconditional GAN, only penalizes structure in B
    end
    
    -- create fake
    fake_B = netG:forward(real_A)
    
    if opt.condition_GAN==1 then
        fake_AB = torch.cat(real_A,fake_B,2)
    else
        fake_AB = fake_B -- unconditional GAN, only penalizes structure in B
    end
    local predict_real = netD:forward(real_AB)
    local predict_fake = netD:forward(fake_AB)
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersD:zero()
    
    -- Real
    local output = netD:forward(real_AB)
    local label = torch.FloatTensor(output:size()):fill(real_label):cuda()
    local errD_real = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(real_AB, df_do)
    
    -- Fake
    local output = netD:forward(fake_AB)
    label:fill(fake_label)
    local errD_fake = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(fake_AB, df_do)
    
    errD = (errD_real + errD_fake)/2
    
    return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersG:zero()
    
    -- GAN loss
    local df_dg = torch.zeros(fake_B:size()):cuda()
    if opt.use_GAN==1 then
       local output = netD.output -- netD:forward{input_A,input_B} was already executed in fDx, so save computation
       local label = torch.FloatTensor(output:size()):fill(real_label):cuda() -- fake labels are real for generator cost
       errG = criterion:forward(output, label)
       local df_do = criterion:backward(output, label)
       df_dg = netD:updateGradInput(fake_AB, df_do):narrow(2,fake_AB:size(2)-output_nc+1, output_nc)
    else
        errG = 0
    end
    
    -- Perceptual loss
    local df_do_AE = torch.zeros(fake_B:size()):cuda()
    if opt.use_Lp==1 then
       errLp = criterionAE:forward(fake_B, real_B)
       df_do_AE = criterionAE:backward(fake_B, real_B)
    else
        errLp = 0
    end

    -- Eucledean loss
    local df_do_AE1 = torch.zeros(fake_B:size()):cuda()
    if opt.use_L2==1 then
       errL2 = criterionAE1:forward(fake_B, real_B)
       df_do_AE1 = criterionAE1:backward(fake_B, real_B)
    else
        errL2 = 0
    end

    --netG:backward(x, grad_out)    
    netG:backward(real_A, df_dg + df_do_AE:mul(opt.lambda)+df_do_AE1:mul(opt.lambda1))
    
    return errG, gradParametersG
    --return loss, grad_params
end




-- train
local best_err = nil
paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.name)

-- save opt
file = torch.DiskFile(paths.concat(opt.checkpoints_dir, opt.name, 'opt.txt'), 'w')
file:writeObject(opt)
file:close()

local counter = 0
for epoch = 1, opt.niter do
    epoch_tm:reset()
    for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
        tm:reset()
        
        -- load a batch and run G on that batch
        createRealFake()
        
        -- (1) Update D network:
        if opt.use_GAN==1 then optim.adam(fDx, parametersD, optimStateD) end
        
        -- (2) Update G network:
        optim.adam(fGx, parametersG, optimStateG)
        
        -- display
        counter = counter + 1
        if counter % opt.display_freq == 0 and opt.display then
            createRealFake()
            if opt.preprocess == 'colorization' then 
                local real_A_s = util.scaleBatch(real_A:float(),100,100)
                local fake_B_s = util.scaleBatch(fake_B:float(),100,100)
                local real_B_s = util.scaleBatch(real_B:float(),100,100)
                disp.image(util.deprocessL_batch(real_A_s), {win=opt.display_id, title=opt.name .. ' input'})
                disp.image(util.deprocessLAB_batch(real_A_s, fake_B_s), {win=opt.display_id+1, title=opt.name .. ' output'})
                disp.image(util.deprocessLAB_batch(real_A_s, real_B_s), {win=opt.display_id+2, title=opt.name .. ' target'})
            else
                disp.image(util.deprocess_batch(util.scaleBatch(real_A:float(),100,100)), {win=opt.display_id, title=opt.name .. ' input'})
                disp.image(util.deprocess_batch(util.scaleBatch(fake_B:float(),100,100)), {win=opt.display_id+1, title=opt.name .. ' output'})
                disp.image(util.deprocess_batch(util.scaleBatch(real_B:float(),100,100)), {win=opt.display_id+2, title=opt.name .. ' target'})
            end
        end
      
        -- write display visualization to disk
        --  runs on the first batchSize images in the opt.phase set
        if counter % opt.save_display_freq == 0 and opt.display then
            local serial_batches=opt.serial_batches
            opt.serial_batches=1
            opt.serial_batch_iter=1
            
            local image_out = nil
            local N_save_display = 10
            for i3=1, torch.floor(N_save_display/opt.batchSize) do
            
                createRealFake()
                print('save to the disk')
                if opt.preprocess == 'colorization' then 
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then image_out = torch.cat(util.deprocessL(real_A[i2]:float()),util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()),3)/255.0
                        else image_out = torch.cat(image_out, torch.cat(util.deprocessL(real_A[i2]:float()),util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()),3)/255.0, 2) end
                    end
                else
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then image_out = torch.cat(util.deprocess(real_B[i2]:float()),util.deprocess(fake_B[i2]:float()),3)
                        else image_out = torch.cat(image_out, torch.cat(util.deprocess(real_B[i2]:float()),util.deprocess(fake_B[i2]:float()),3), 2) end
                    end
                end
            end
            image.save(paths.concat(opt.checkpoints_dir,  opt.name , counter .. '_train_res.png'), image_out)
            
            opt.serial_batches=serial_batches
        end
        
        -- logging
        if counter % opt.print_freq == 0 then
            print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                    .. '  Err_G: %.4f  Err_D: %.4f  ErrLp: %.4f ErrMSE: %.4f'):format(
                     epoch, ((i-1) / opt.batchSize),
                     math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                     tm:time().real / opt.batchSize, data_tm:time().real / opt.batchSize,
                     errG and errG or -1, errD and errD or -1, errLp and errLp or -1,  errL2 and errL2 or -1))
        end
        
        -- save latest model
        if counter % opt.save_latest_freq == 0 then
            print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), netG:clearState())
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), netD:clearState())
        end
        
    end
    
    
    parametersD, gradParametersD = nil, nil 
    parametersG, gradParametersG = nil, nil
    
    if epoch % opt.save_epoch_freq == 0 then
        torch.save(paths.concat(opt.checkpoints_dir, opt.name,  epoch .. '_net_G.t7'), netG:clearState())
        torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D.t7'), netD:clearState())
    end
    
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
    parametersD, gradParametersD = netD:getParameters() 
    parametersG, gradParametersG = netG:getParameters()
end
