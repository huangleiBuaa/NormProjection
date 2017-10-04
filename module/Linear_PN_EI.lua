local Linear_PN_EI, parent = torch.class('nn.Linear_PN_EI', 'nn.Module')

function Linear_PN_EI:__init(inputSize, outputSize,lr, orth_flag)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)
   
   self.FIM=torch.Tensor()
   self.conditionNumber={}
   self.epcilo=10^-100
 
   self.updateFIM_flag=false
   
   self.debug=false
   self.printDetail=false
   self.testWeightNormalized=false

    if lr ~= nil then
       self.lr = lr
     else
       self.lr = 0.1
     end


  if orth_flag ~= nil then
      assert(type(orth_flag) == 'boolean', 'orth_flag has to be true/false')
      
      
    if orth_flag then
      self:reset_orthogonal()
    else
    self:reset()
    end
  else
    self:reset()
  
  end

end

function Linear_PN_EI:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-stdv, stdv)

      self.bias:uniform(-stdv, stdv)
   end

   return self
end

function Linear_PN_EI:reset_orthogonal()
    local initScale = 1.1 -- math.sqrt(2)

    local M1 = torch.randn(self.weight:size(1), self.weight:size(1))
    local M2 = torch.randn(self.weight:size(2), self.weight:size(2))

    local n_min = math.min(self.weight:size(1), self.weight:size(2))

    -- QR decomposition of random matrices ~ N(0, 1)
    local Q1, R1 = torch.qr(M1)
    local Q2, R2 = torch.qr(M2)

    self.weight:copy(Q1:narrow(2,1,n_min) * Q2:narrow(1,1,n_min)):mul(initScale)

    self.bias:zero()
end

function Linear_PN_EI:updateOutput(input)
   --self.bias:fill(0)
  
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.bias:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.addBuffer = self.addBuffer or input.new()
      if self.addBuffer:nElement() ~= nframe then
         self.addBuffer:resize(nframe):fill(1)
      end
     self.buffer=self.buffer or self.weight.new()
      --retractor the weight to the oblique manifold. use this idea to follow the optimization framework in torch to update        weight. we don't explicitly update weight
      if self.train then
      --  print('----------------train-------------')
       self.buffer=self.weight:norm(2,2)
       self.weight:cdiv(self.buffer:expandAs(self.weight))
       self.testWeightNormalized=false
      -- we only excute the last operattion to retract the weight to oblique manifold in training mode.
    elseif not self.testWeightNormalized then
      --   print('----------------test-------------')
       self.buffer=self.weight:norm(2,2)
       self.weight:cdiv(self.buffer:expandAs(self.weight))
       self.testWeightNormalized=true
    end

      
      
      self.output:addmm(0, self.output, 1, input, self.weight:t())
      self.output:addr(1, self.addBuffer, self.bias)
   else
      error('input must be vector or matrix')
   end
   return self.output
end

function Linear_PN_EI:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end
    
    if self.printDetail then
     print("Linear_PN_EI: gradOutput, number fo example=20")
     print(gradOutput[{{1,20},{}}]) 
    end
      
      --------------------------------------------------
      --calculate the FIM----------
      --------------------------------------------------
     --  self.counter=self.counter+1
      return self.gradInput
   end
end

function Linear_PN_EI:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
   end
   
--   self.weight:add(-self.lr*self.gradWeight)
   
--   self.buffer=self.buffer or self.weight.new()
--   local temp=self.weight:norm(2,2)
--   self.buffer:repeatTensor(temp,1,self.weight:size(2))
    -- print('--------updateWeight----------------')
--   self.weight:cdiv(self.buffer)

  -- self.gradWeight:fill(0) --in case of the optimization call weight=weight+lr*gradWeight such that change weigt

end



-- we do not need to accumulate parameters when sharing
Linear_PN_EI.sharedAccUpdateGradParameters = Linear_PN_EI.accUpdateGradParameters


function Linear_PN_EI:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
