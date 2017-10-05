require '../../module/Linear_PN'
require '../../module/Linear_Oblique'
require '../../module/Linear_PN_EI'


function create_model(opt)
  ------------------------------------------------------------------------------

  ------------------------------------------------------------------------------

   

  local model=nn.Sequential()          
  --local cfg_hidden=torch.Tensor({256,256,256,256})--used for SVD debug
  --local cfg_hidden=torch.Tensor({opt.n_hidden_number,opt.n_hidden_number,opt.n_hidden_number,opt.n_hidden_number,opt.n_hidden_number})
  local cfg_hidden=torch.Tensor({1000,750,250,250,250}) --used for get the best performance
  local n=cfg_hidden:size(1)
  
  local nonlinear 
  if opt.mode_nonlinear==0 then  --sigmod
      nonlinear=nn.Sigmoid
  elseif opt.mode_nonlinear==1 then --tanh
      nonlinear=nn.Tanh
  elseif opt.mode_nonlinear==2 then --ReLU
     nonlinear=nn.ReLU
  elseif opt.mode_nonlinear==3 then --ReLU
     nonlinear=nn.ELU
  end 
  
  local linear=nn.Linear
  local module_BN=nn.BatchLinear_FIM
  local module_PN=nn.Linear_PN
  local module_Oblique=nn.Linear_Oblique
  local module_PN_EI=nn.Linear_PN_EI
 
  local function block_sgd(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(nn.Dropout(opt.dropout))
    s:add(linear(n_input,n_output,opt.orth_intial))
    return s
  end
 
  local function block_Oblique(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(module_Oblique(n_input,n_output,opt.learningRate,opt.orth_intial))
    return s
  end
  local function block_Oblique_batch(n_input, n_output)
    local s=nn.Sequential()
    s:add(module_BN(n_input,true))
    s:add(nonlinear())
    s:add(module_Oblique(n_input,n_output,opt.learningRate,opt.orth_intial))
    return s
  end
  local function block_PN_EI(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(module_PN_EI(n_input,n_output,opt.learningRate,opt.orth_intial))
    return s
  end
  local function block_PN_EI_batch(n_input, n_output)
    local s=nn.Sequential()
    s:add(module_BN(n_input,true))
    s:add(nonlinear())
    s:add(module_PN_EI(n_input,n_output,opt.learningRate,opt.orth_intial))
    return s
  end
  local function block_PN(n_input, n_output)
    local s=nn.Sequential()
    s:add(nonlinear())
    s:add(module_PN(n_input,n_output,opt.orth_intial))
    return s
  end



-----------------------------------------model configure-------------------
--   model:add(nn.WhiteNoise_local(0, opt.noise_level))

  if opt.model_method=='sgd' then 
       model:add(linear(opt.n_inputs,cfg_hidden[1]))
    for i=1,n do
       if i==n then
         model:add(block_sgd(cfg_hidden[i],opt.n_outputs)) 

       else
        model:add(block_sgd(cfg_hidden[i],cfg_hidden[i+1])) 

       end
     end 


  elseif opt.model_method=='Oblique' then
   model:add(nn.Linear_Oblique(opt.n_inputs,cfg_hidden[1],opt.learningRate,opt.orth_intial))
     for i=1,n do
       if i==n then
         model:add(block_Oblique(cfg_hidden[i],opt.n_outputs))
       else
         model:add(block_Oblique(cfg_hidden[i],cfg_hidden[i+1]))
       end
     end

  elseif opt.model_method=='PN_EI' then
   model:add(nn.Linear_PN_EI(opt.n_inputs,cfg_hidden[1],opt.learningRate,opt.orth_intial))
     for i=1,n do
       if i==n then
         model:add(block_PN_EI(cfg_hidden[i],opt.n_outputs))
       else
         model:add(block_PN_EI(cfg_hidden[i],cfg_hidden[i+1]))
       end
     end

  elseif opt.model_method=='PN' then
   model:add(nn.Linear_PN(opt.n_inputs,cfg_hidden[1],opt.orth_intial))
     for i=1,n do
       if i==n then
         model:add(block_PN(cfg_hidden[i],opt.n_outputs))
       else
         model:add(block_PN(cfg_hidden[i],cfg_hidden[i+1]))
       end
     end

    
  end
  
  
  model:add(nn.LogSoftMax()) 
 

  ------------------------------------------------------------------------------
  -- LOSS FUNCTION
  ------------------------------------------------------------------------------
  local  criterion = nn.ClassNLLCriterion()

  return model, criterion
end

