local Linear_Oblique, parent = torch.class('nn.Linear_Oblique', 'nn.Module')

function Linear_Oblique:__init(inputSize, outputSize,lr, orth_flag)
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

function Linear_Oblique:reset(stdv)
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

function Linear_Oblique:reset_orthogonal()
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

function Linear_Oblique:updateOutput(input)
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
      self.output:addmm(0, self.output, 1, input, self.weight:t())
      self.output:addr(1, self.addBuffer, self.bias)
   else
      error('input must be vector or matrix')
   end
   if self.printDetail then
     print("Linear_Oblique: input, number fo example=20")
     print(input[{{1,20},{}}]) 
     
     print("Linear_Oblique: weight")
     print(self.weight) 
     
     print("Linear_Oblique: bias")
     print(self.bias) 
     
     print("Linear_Oblique: activation, number fo example=20")
     print(self.output[{{1,20},{}}])  
   end
   return self.output
end

function Linear_Oblique:updateGradInput(input, gradOutput)
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
     print("Linear_Oblique: gradOutput, number fo example=20")
     print(gradOutput[{{1,20},{}}]) 
    end
      
      --------------------------------------------------
      --calculate the FIM----------
      --------------------------------------------------
     --  self.counter=self.counter+1
      return self.gradInput
   end
end

function Linear_Oblique:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
   end
   
    local nOutput=self.weight:size(1)
   self.buffer=self.buffer or self.weight.new()
   self.buffer1=self.buffer1 or self.weight.new()
   self.buffer:resize(nOutput,nOutput) 
   self.buffer:mm(self.weight, self.gradWeight:t()) -- W*(DL/DW)^T
   self.buffer1=torch.eye(self.buffer:size(1),self.buffer:size(1))
   self.buffer1:cmul(self.buffer)   --get the daigonal matrix
  -- local a=torch.Tensor()
 --  a:diag(self.buffer1)
--   print(a:norm()) 

   
   --print(self.buffer1)
   --print(self.weight)
   self.buffer:resizeAs(self.weight):mm(self.buffer1,self.weight)
  -- print(self.buffer)
   
   self.gradWeight:add(-self.buffer)
   
   self.weight:add(-self.lr*self.gradWeight)
   self.buffer=self.weight:norm(2,2)
   self.weight:cdiv(self.buffer:expandAs(self.weight))


   self.gradWeight:fill(0) --in case of the optimization call weight=weight+lr*gradWeight such that change weigt

end



-- we do not need to accumulate parameters when sharing
Linear_Oblique.sharedAccUpdateGradParameters = Linear_Oblique.accUpdateGradParameters


function Linear_Oblique:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
