# -*- coding: utf-8 -*-
import torch


class Brain(torch.nn.Module):
      
      def __init__(self,inputs,action_space):
            super(Brain,self).__init__()
            