#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 16:36:27 2019

@author: Song Yunhua
"""

import theano
import numpy as np
import keras.backend as K
from keras.layers.recurrent import LSTM#,GRU,RNN,LSTMCell
from keras.layers import Layer

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints

class emb_init(initializers.Initializer):
    def __init__(self,emb,minval=-0.05, maxval=0.05, seed=None):
        self.emb=emb
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
    def __call__(self,shape,dtype=None):
        if shape==self.emb.shape:
            #print(shape,'match with',self.emb.shape)
            return self.emb
        else:
            print("Init error:")
            print(shape,'do not match with',self.emb.shape)
            return K.random_uniform(shape, self.minval, self.maxval,
                                dtype=dtype, seed=self.seed)


class GDCell(Layer):
    def __init__(self, h_units,o_units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        super(GDCell, self).__init__(**kwargs)
        self.hidden_dim = h_units
        self.output_dim= o_units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.state_size = (self.output_dim, self.hidden_dim)
        self.output_size = self.output_dim
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        #        input_shape = self.input_shape
        dim = input_shape[-1]
        self.input_dim = dim
        self.init=self.kernel_initializer
        self.inner_init=self.recurrent_initializer
        hdim = self.hidden_dim
        outdim = self.output_dim

        self.W = self.add_weight(shape=(outdim, hdim * 3),
                                      name='kernel_W',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.W_r =self.W[:, :hdim]
        self.W_z =self.W[:, hdim:2*hdim]
        self.W_s =self.W[:, 2*hdim:3*hdim]
        
        self.U = self.add_weight(
                                        shape=(hdim, hdim * 3),
                                        name='recurrent_kernel_U',
                                        initializer=self.recurrent_initializer,
                                        regularizer=self.recurrent_regularizer,
                                        constraint=self.recurrent_constraint)
        self.U_r =self.U[:, :hdim]
        self.U_z =self.U[:, hdim:2*hdim]
        self.U_s =self.U[:, 2*hdim:3*hdim]
        
        self.B = self.add_weight(shape=(3*hdim),
                                        name='bias_b',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        self.b_r =self.B[ :hdim]
        self.b_z =self.B[ hdim:2*hdim]
        self.b_s =self.B[2*hdim:3*hdim]
        
        self.W_x = self.add_weight(shape=(hdim, outdim),
                                      name='kernel_wx',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.b_x = self.add_weight(shape=(outdim),
                                        name='bias_bx',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        self.V_r = self.add_weight(shape=(dim, hdim),
                                      name='kernel_vr',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.V_z = self.add_weight(shape=(dim, hdim ),
                                      name='kernel_vz',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.V_s = self.add_weight(shape=(dim, hdim ),
                                      name='kernel_vs',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.built = True
    def call(self,inputs,states):
        x_tm1=states[0]
        s_tm1=states[1]

        xr_t=K.dot(x_tm1, self.W_r)+ self.b_r+ K.dot(inputs, self.V_r)
        xz_t = K.dot(x_tm1, self.W_z)  + self.b_z+ K.dot(inputs, self.V_z)
        r_t  = self.recurrent_activation(xr_t + K.dot(s_tm1, self.U_r))
        z_t  = self.recurrent_activation(xz_t + K.dot(s_tm1, self.U_z))
        xs_t = K.dot(x_tm1, self.W_s)  + self.b_s+ K.dot(inputs, self.V_s)
        s1_t = self.activation(xs_t + K.dot(r_t*s_tm1, self.U_s))
        s_t = (1-z_t) * s_tm1 + z_t * s1_t
        x_t = self.activation(K.dot(s_t, self.W_x) + self.b_x)
        return x_t, [x_t,s_t]
           
    def get_config(self):
        config = {'hidden_units': self.hidden_dim,
                  'output_units': self.output_dim,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation}
        base_config = super(GDCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    

def peaked_softmax():
    
    return 0

from theano import tensor as T
def top_k(variable,k):
    return T.sort(variable)[:k]

def BSloss():
    return 0
#def beam_search_decoder(seq_c, k):
#    sequences = [[list(), 1.0]]
#    for row in seq_c:
#        all_candidates = list()
#        for i in range(len(sequences)):
#            seq, score = sequences[i]
#            for j,contribution in enumerate(row):
#                candidate = [seq + [j], score_fun(score,contribution)]# 概率  score -log(contribution)]
#                all_candidates.append(candidate)
#        # 所有候选根据分值排序
#        ordered = sorted(all_candidates, key=lambda tup:tup[1])
#        # 选择前k个
#        sequences = ordered[:k]
#    return sequences

#def score_fun(score,c):
#    return score -log(c)

class BScell(Layer):
    def __init__(self, K,h_units,o_units,t_units,act="softmax",
                 **kwargs):
        super(BScell, self).__init__(**kwargs)
        self.cellD=GDCell(h_units,o_units,t_units,**kwargs)
        self.K=K
        self.activation = activations.get(act)

    def build(self, input_shape):
        #        input_shape = self.input_shape
        dim = input_shape[-1]
        self.input_dim = dim
        self.init=self.kernel_initializer
        self.inner_init=self.recurrent_initializer
        hdim = self.hidden_dim
        outdim = self.output_dim

        self.W = self.add_weight(shape=(outdim, hdim * 3),
                                      name='kernel_W',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        super(BScell, self).build(input_shape)
    def call(self,inputs,states):
        t_t,_=self.cellD.call(inputs,states)
        t_t = self.activation(t_t)
        tag=BSloss(t_t)
        return tag, states

    
        
    

class LSTM_Decoder(LSTM):
    input_ndim = 3
    def __init__(self, output_length,output_dim=None, hidden_dim=None,state_input=True, **kwargs):
        self.output_length = output_length
        self.hidden_dim = hidden_dim
        
        self.output_dim=output_dim
        self.state_outputs = []
        self.state_input = state_input
        self.return_sequences = True #Decoder always returns a sequence.
        self.updates = []
        super(LSTM_Decoder, self).__init__(**kwargs)
    def build(self):
        input_shape = self.input_shape
        dim = input_shape[-1]
        self.input_dim = dim
        self.input = K.placeholder(input_shape)
        hdim = self.hidden_dim
        outdim = self.output_dim
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            self.reset_states()
        else:
            self.states = [None, None]
        self.W_i = self.init((outdim, hdim))
        self.U_i = self.inner_init((hdim, hdim))
        self.b_i = K.zeros((hdim))
        self.W_f = self.init((outdim, hdim))
        self.U_f = self.inner_init((hdim, hdim))
        self.b_f = self.forget_bias_init((hdim))
        self.W_c = self.init((outdim, hdim))
        self.U_c = self.inner_init((hdim, hdim))
        self.b_c = K.zeros((hdim))
        self.W_o = self.init((outdim, hdim))
        self.U_o = self.inner_init((hdim, hdim))
        self.b_o = K.zeros((hdim))
        self.W_x = self.init((hdim, outdim))
        self.b_x = K.zeros((outdim))
        self.V_i = self.init((dim, hdim))
        self.V_f = self.init((dim, hdim))
        self.V_c = self.init((dim, hdim))
        self.V_o = self.init((dim, hdim))
        self.trainable_weights = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
            self.W_x,           self.b_x,
            self.V_i, self.V_c, self.V_f, self.V_o
        ]
        self.input_length = self.input_shape[-2]
        if not self.input_length:
            raise Exception ('AttentionDecoder requires input_length.')
        super(LSTM_Decoder, self).build(input_shape)
    def set_previous(self, layer, connection_map={}):
        self.previous = layer
        self.build()
    def get_initial_states(self, X):
        # build an all-zero tensor of shape (samples, hidden_dim)
        initial_state = K.zeros_like(X)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.hidden_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, hidden_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states
    def ssstep(self,
               h,
              x_tm1,
              h_tm1, c_tm1,
              u_i, u_f, u_o, u_c, w_i, w_f, w_c, w_o, w_x,  v_i, v_f, v_c, v_o, b_i, b_f, b_c, b_o, b_x):
        xi_t = K.dot(x_tm1, w_i)+ b_i+ K.dot(h, v_i)
        xf_t = K.dot(x_tm1, w_f)  + b_f+ K.dot(h, v_f)
        xc_t = K.dot(x_tm1, w_c) + b_c+ K.dot(h, v_c)
        xo_t = K.dot(x_tm1, w_o)  + b_o+ K.dot(h, v_o)
        i_t = self.inner_activation(xi_t + K.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + K.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + K.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + K.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        x_t =self.activation(K.dot(h_t, w_x) + b_x)
        return x_t, h_t, c_t
    def call(self,H):
        Hh = K.permute_dimensions(H, (1, 0, 2))
        def rstep(o,index,Hh):
            return Hh[index],index-1
        [RHh,index],update = theano.scan(
        rstep,
        n_steps=Hh.shape[0],
        non_sequences=[Hh],
        outputs_info= [Hh[-1]]+[-1])
        X = K.permute_dimensions(H, (1, 0, 2))[-1]
        outdim=self.output_dim
        X1=X[:,:outdim]+X[:,outdim:]
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)
        [outputs,hidden_states, cell_states], updates = theano.scan(
            self.ssstep,
            sequences=RHh,
            n_steps = self.output_length,
            outputs_info=[X1] + initial_states,
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c,
                          self.W_i, self.W_f, self.W_c, self.W_o,
                          self.W_x, self.V_i, self.V_f, self.V_c,
                          self.V_o, self.b_i, self.b_f, self.b_c,
                          self.b_o, self.b_x])
        states = [hidden_states[-1], cell_states[-1]]
        if self.stateful and not self.state_input:
            self.updates = []
            for i in range(2):
                self.updates.append((self.states[i], states[i]))
        for o in self.state_outputs:
            o.updates = []
            for i in range(2):
                o.updates.append((o.states[i], states[i]))
        return K.permute_dimensions(outputs, (1, 0, 2))
    def compute_output_shape(self,input_shape):
        return (input_shape[:-1],self.output_dim)
   
#    def get_config(self):
#        config = {'name': self.__class__.__name__}
#        base_config = super(LSTM_Decoder, self).get_config()
#        return dict(list(base_config.items()) + list(config.items()))
class MYGRUDecoder(Layer):
    input_ndim = 3
    def __init__(self, output_length,output_dim, hidden_dim=None,**kwargs):
        self.output_length = output_length
        self.output_dim=output_dim
        self.hidden_dim = hidden_dim
        self.return_sequences = True #Decoder always returns a sequence.
#        self.updates = []
        super(MYGRUDecoder, self).__init__(**kwargs)
    def build(self,input_shape):
#        input_shape = self.input_shape
        dim = input_shape[-1]
        self.input_dim = dim
        self.init=self.kernel_initializer
        self.inner_init=self.recurrent_initializer
        hdim = self.hidden_dim
        outdim = self.output_dim

        self.W = self.add_weight(shape=(outdim, hdim * 3),
                                      name='kernel_W',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.W_r =self.W[:, :hdim]
        self.W_z =self.W[:, hdim:2*hdim]
        self.W_s =self.W[:, 2*hdim:3*hdim]
        
        self.U = self.add_weight(
                                        shape=(hdim, hdim * 3),
                                        name='recurrent_kernel_U',
                                        initializer=self.recurrent_initializer,
                                        regularizer=self.recurrent_regularizer,
                                        constraint=self.recurrent_constraint)
        self.U_r =self.U[:, :hdim]
        self.U_z =self.U[:, hdim:2*hdim]
        self.U_s =self.U[:, 2*hdim:3*hdim]
        
        self.B = self.add_weight(shape=(3*hdim),
                                        name='bias_b',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        self.b_r =self.B[ :hdim]
        self.b_z =self.B[ hdim:2*hdim]
        self.b_s =self.B[2*hdim:3*hdim]
        
        self.W_x = self.add_weight(shape=(hdim, outdim * 3),
                                      name='kernel_wx',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.b_x = self.add_weight(shape=(outdim),
                                        name='bias_bx',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        self.V_r = self.add_weight(shape=(dim, hdim),
                                      name='kernel_vr',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.V_z = self.add_weight(shape=(dim, hdim ),
                                      name='kernel_vz',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.V_s = self.add_weight(shape=(dim, hdim ),
                                      name='kernel_vs',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
      
#        self.input_length = self.input_shape[-2]
#        if not self.input_length:
#            raise Exception ('AttentionDecoder requires input_length.')
        super(MYGRUDecoder, self).build(input_shape)
    def set_previous(self, layer, connection_map={}):
        self.previous = layer
        self.build()
    def get_initial_states(self, X):
        initial_state = K.zeros_like(X)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.hidden_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, hidden_dim)
        initial_states = [initial_state]# for _ in range(len(self.states))]
        return initial_states
    def ssstep(self,
               h,
              x_tm1,
              s_tm1,
              u_r, u_z, u_s, w_r, w_z, w_s, w_x,  v_r, v_z, v_s, b_r, b_z, b_s, b_x):
        xr_t = K.dot(x_tm1, w_r)+ b_r+ K.dot(h, v_r)
        xz_t = K.dot(x_tm1, w_z)  + b_z+ K.dot(h, v_z)
        r_t  = self.recurrent_activation(xr_t + K.dot(s_tm1, u_r))
        z_t  = self.recurrent_activation(xz_t + K.dot(s_tm1, u_z))
        xs_t = K.dot(x_tm1, w_s)  + b_s+ K.dot(h, v_s)
        s1_t = self.activation(xs_t + K.dot(r_t*s_tm1, u_s))
        s_t = (1-z_t) * s_tm1 + z_t * s1_t
        x_t = self.activation(K.dot(s_t, w_x) + b_x)
        return x_t, s_t
    def call(self,H):
        Hh = K.permute_dimensions(H, (1, 0, 2))
        def rstep(o,index,Hh):
            return Hh[index],index-1
        [RHh,index],update = theano.scan(
        rstep,
        n_steps=Hh.shape[0],
        non_sequences=[Hh],
        outputs_info= [Hh[-1]]+[-1])
        X = K.permute_dimensions(H, (1, 0, 2))[-1]
        outdim=self.output_dim
        X1=X[:,:outdim]+X[:,outdim:]
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)
        [outputs,hidden_states], updates = theano.scan(
            self.ssstep,
            sequences=RHh,
            n_steps = self.output_length,
            outputs_info=[X1] + initial_states,
            non_sequences=[self.U_r, self.U_z, self.U_s,
                          self.W_r, self.W_z, self.W_s, self.W_x,
                          self.V_r, self.V_z, self.V_s,
                          self.b_r, self.b_z, self.b_s, self.b_x])
        states = [hidden_states[-1]]
        if self.stateful and not self.state_input:
            self.updates = []
            for i in range(2):
                self.updates.append((self.states[i], states[i]))

        return K.permute_dimensions(outputs, (1, 0, 2))
    
    def compute_output_shape(self,input_shape):
        return (input_shape[:-1],self.output_dim)
    def get_config(self):
        config = {'name': self.__class__.__name__}
        base_config = super(MYGRUDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))   
        
    
class ReverseLayer(Layer):
    def __init__(self,axis,**kwargs):
        super(ReverseLayer, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True
        self._reshape_required = False
    def call(self,x):
        return K.reverse(x,self.axis)
    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        masks=K.reverse(mask,self.axis)
        return masks
    def compute_output_shape(self,input_shape):
        return input_shape
class reshapecell(Layer):
    def __init__(self,kks,odim,**kwargs):
        super(reshapecell, self).__init__(**kwargs)
        self.kks=kks
        self.odim=odim
        #self.w1=
    def call(self,x):
        p=[]
        for i in range(self.kks):
            p.append(x[:,:,self.odim*i:self.odim*i+self.odim])
        return K.concatenate(p,axis=1)
    def compute_output_shape(self,input_shape):

        return (input_shape[0],input_shape[1]*self.kks,self.odim)
    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        p=[]

        for i in range(self.kks):
            p.append(mask[:,:])
        return K.concatenate(p,axis=1)
    
    
    def get_config(self):
        config = {'kks':self.kks}
        base_config = super(reshapecell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Merge_1(Layer):
    def __init__(self, layers, mode='sum', concat_axis=-1, dot_axes=-1):
        if len(layers) < 2:
            raise Exception('Please specify two or more input layers '
                            '(or containers) to merge')

        if mode not in {'sum', 'mul', 'concat', 'ave', 'join', 'cos', 'dot'}:
            raise Exception('Invalid merge mode: ' + str(mode))

        if mode in {'sum', 'mul', 'ave', 'cos'}:
            input_shapes = set([l.output_shape for l in layers])
            if len(input_shapes) > 1:
                raise Exception('Only layers of same output shape can '
                                'be merged using ' + mode + ' mode. ' +
                                'Layer shapes: %s' % ([l.output_shape for l in layers]))
        if mode in {'cos', 'dot'}:
            if len(layers) > 2:
                raise Exception(mode + ' merge takes exactly 2 layers')
            shape1 = layers[0].output_shape
            shape2 = layers[1].output_shape
            n1 = len(shape1)
            n2 = len(shape2)
            if mode == 'dot':
                if type(dot_axes) == int:
                    if dot_axes < 0:
                        dot_axes = [range(dot_axes % n1, n1), range(dot_axes % n2, n2)]
                    else:
                        dot_axes = [range(n1 - dot_axes, n2), range(1, dot_axes + 1)]
                if type(dot_axes) not in [list, tuple]:
                    raise Exception('Invalid type for dot_axes - should be a list.')
                if len(dot_axes) != 2:
                    raise Exception('Invalid format for dot_axes - should contain two elements.')
                if type(dot_axes[0]) not in [list, tuple, range] or type(dot_axes[1]) not in [list, tuple, range]:
                    raise Exception('Invalid format for dot_axes - list elements should have type "list" or "tuple".')
                for i in range(len(dot_axes[0])):
                    if shape1[dot_axes[0][i]] != shape2[dot_axes[1][i]]:
                        raise Exception('Dimension incompatibility using dot mode: ' +
                                        '%s != %s. ' % (shape1[dot_axes[0][i]], shape2[dot_axes[1][i]]) +
                                        'Layer shapes: %s, %s' % (shape1, shape2))
        elif mode == 'concat':
            input_shapes = set()
            for l in layers:
                oshape = list(l.output_shape)
                oshape.pop(concat_axis)
                oshape = tuple(oshape)
                input_shapes.add(oshape)
            if len(input_shapes) > 1:
                raise Exception('"concat" mode can only merge layers with matching ' +
                                'output shapes except for the concat axis. ' +
                                'Layer shapes: %s' % ([l.output_shape for l in layers]))
        self.mode = mode
        self.concat_axis = concat_axis
        self.dot_axes = dot_axes
        self.layers = layers
        self.trainable_weights = []
        self.regularizers = []
        self.constraints = []
        self.updates = []
        for l in self.layers:
            params, regs, consts, updates = l.get_params()
            self.regularizers += regs
            self.updates += updates
            # params and constraints have the same size
            for p, c in zip(params, consts):
                if p not in self.trainable_weights:
                    self.trainable_weights.append(p)
                    self.constraints.append(c)
        super(Merge_1, self).__init__()

    @property
    def input_shape(self):
        return [layer.input_shape for layer in self.layers]

    @property
    def output_shape(self):
        input_shapes = [layer.output_shape for layer in self.layers]
        if self.mode in ['sum', 'mul', 'ave']:
            return input_shapes[0]
        elif self.mode == 'concat':
            output_shape = list(input_shapes[0])
            for shape in input_shapes[1:]:
                output_shape[self.concat_axis] += shape[self.concat_axis]
            return tuple(output_shape)
        elif self.mode == 'join':
            return None
        elif self.mode == 'dot':
            shape1 = list(input_shapes[0])
            shape2 = list(input_shapes[1])
            dot_axes = []
            for axes in self.dot_axes:
                dot_axes.append([index-1 for index in axes])
            tensordot_output = np.tensordot(np.zeros(tuple(shape1[1:])),
                                            np.zeros(tuple(shape2[1:])),
                                            axes=dot_axes)
            if len(tensordot_output.shape) == 0:
                shape = (1,)
            else:
                shape = tensordot_output.shape
            return (shape1[0],) + shape
        elif self.mode == 'cos':
            return (input_shapes[0][0], 1)

    def get_params(self):
        return self.trainable_weights, self.regularizers, self.constraints, self.updates

    def get_output(self, train=False):
        if self.mode == 'sum' or self.mode == 'ave':
            s = self.layers[0].get_output(train)
            for i in range(1, len(self.layers)):
                s += self.layers[i].get_output(train)
            if self.mode == 'ave':
                s /= len(self.layers)
            return s
        elif self.mode == 'concat':
            inputs = [self.layers[i].get_output(train) for i in range(len(self.layers))]
            return K.concatenate(inputs, axis=self.concat_axis)
        elif self.mode == 'mul':
            s = self.layers[0].get_output(train)
            for i in range(1, len(self.layers)):
                s *= self.layers[i].get_output(train)
            return s
        elif self.mode == 'dot':
            l1 = self.layers[0].get_output(train)
            l2 = self.layers[1].get_output(train)
            output = K.batch_dot(l1, l2, self.dot_axes)
            output_shape = list(self.output_shape)
            output_shape[0] = -1
            output = K.reshape(output, (tuple(output_shape)))
            return output
        elif self.mode == 'cos':
            l1 = self.layers[0].get_output(train)
            l2 = self.layers[1].get_output(train)
            output = K.batch_dot(l1, l2, self.dot_axes) / K.sqrt(
                K.batch_dot(l1, l1, self.dot_axes) * K.batch_dot(l2, l2, self.dot_axes))
            output = output.dimshuffle((0, 'x'))
            return output
        else:
            raise Exception('Unknown merge mode.')
    def get_input(self, train=False):
        res = []
        for i in range(len(self.layers)):
            o = self.layers[i].get_input(train)
            if not type(o) == list:
                o = [o]
            for output in o:
                if output not in res:
                    res.append(output)
        return res
    @property
    def input(self):
        return self.get_input()

    def supports_masked_input(self):
        return False

    def get_output_mask(self, train=None):
        return None

    def get_weights(self):
        weights = []
        for l in self.layers:
            weights += l.get_weights()
        return weights

    def set_weights(self, weights):
        for i in range(len(self.layers)):
            nb_param = len(self.layers[i].get_weights())
            self.layers[i].set_weights(weights[:nb_param])
            weights = weights[nb_param:]

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'layers': [l.get_config() for l in self.layers],
                  'mode': self.mode,
                  'concat_axis': self.concat_axis,
                  'dot_axes': self.dot_axes}
        base_config = super(Merge_1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

    
class DecodeCell(Layer):
    def __init__(self, h_units,o_units,t_units,
                 activation='tanh',
                 tag_activation='softmax',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        super(DecodeCell, self).__init__(**kwargs)
        self.hidden_dim = h_units
        self.output_dim= o_units
        self.tag_dim=t_units
        
        self.activation = activations.get(activation)
        self.tag_activation = activations.get(tag_activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.state_size = (self.output_dim, self.hidden_dim)
        self.output_size = self.tag_dim#self.output_dim
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        #        input_shape = self.input_shape
        dim = input_shape[-1]
        self.input_dim = dim
        self.init=self.kernel_initializer
        self.inner_init=self.recurrent_initializer
        hdim = self.hidden_dim
        outdim = self.output_dim
        tagdim=self.tag_dim

        self.W = self.add_weight(shape=(outdim, hdim * 3),
                                      name='kernel_W',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.W_r =self.W[:, :hdim]
        self.W_z =self.W[:, hdim:2*hdim]
        self.W_s =self.W[:, 2*hdim:3*hdim]
        
        self.U = self.add_weight(
                                        shape=(hdim, hdim * 3),
                                        name='recurrent_kernel_U',
                                        initializer=self.recurrent_initializer,
                                        regularizer=self.recurrent_regularizer,
                                        constraint=self.recurrent_constraint)
        self.U_r =self.U[:, :hdim]
        self.U_z =self.U[:, hdim:2*hdim]
        self.U_s =self.U[:, 2*hdim:3*hdim]
        
        self.B = self.add_weight(shape=(3*hdim),
                                        name='bias_b',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        self.b_r =self.B[ :hdim]
        self.b_z =self.B[ hdim:2*hdim]
        self.b_s =self.B[2*hdim:3*hdim]
        
        self.W_x = self.add_weight(shape=(hdim, outdim),
                                      name='kernel_wx',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.b_x = self.add_weight(shape=(outdim),
                                        name='bias_bx',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        self.V_r = self.add_weight(shape=(dim, hdim),
                                      name='kernel_vr',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.V_z = self.add_weight(shape=(dim, hdim ),
                                      name='kernel_vz',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.V_s = self.add_weight(shape=(dim, hdim ),
                                      name='kernel_vs',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
        self.W_y = self.add_weight(shape=(outdim, tagdim),
                                      name='kernel_wy',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.b_y = self.add_weight(shape=(tagdim),
                                        name='bias_by',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        
        self.built = True
    def call(self,inputs,states,training=None):
        x_tm1=states[0]
        s_tm1=states[1]
        #t_tm1=states[0]
        
        xr_t=K.dot(x_tm1, self.W_r)+ self.b_r+ K.dot(inputs, self.V_r)
        xz_t = K.dot(x_tm1, self.W_z)  + self.b_z+ K.dot(inputs, self.V_z)
        r_t  = self.recurrent_activation(xr_t + K.dot(s_tm1, self.U_r))
        z_t  = self.recurrent_activation(xz_t + K.dot(s_tm1, self.U_z))
        xs_t = K.dot(x_tm1, self.W_s)  + self.b_s+ K.dot(inputs, self.V_s)
        s1_t = self.activation(xs_t + K.dot(r_t*s_tm1, self.U_s))
        s_t = (1-z_t) * s_tm1 + z_t * s1_t
        x_t = self.tag_activation(K.dot(s_t, self.W_x) + self.b_x)
        
        t_t=self.tag_activation(K.dot(x_t, self.W_y) + self.b_y)
        
        return t_t, [x_t,s_t]
           
    def get_config(self):
        config = {'hidden_units': self.hidden_dim,
                  'output_units': self.output_dim,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation}
        base_config = super(DecodeCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))  
    
    
    

class BSCell(DecodeCell):
    def __init__(self, Dcell,h_units,o_units,t_units,
                 activation='tanh',
                 tag_activation='softmax',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        super(BSCell, self).__init__(**kwargs)
        self.cell=Dcell
        self.hidden_dim = h_units
        self.output_dim= o_units
        self.tag_dim=t_units
        
        self.activation = activations.get(activation)
        self.tag_activation = activations.get(tag_activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.state_size = (self.output_dim, self.hidden_dim)
        self.output_size = self.tag_dim#self.output_dim
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

#    def build(self, input_shape):
#        super(BSCell,self).build(input_shape)

    def call(self,inputs,states,training=None):
        
        x_tm1=states[0]
        s_tm1=states[1]
        #t_tm1=states[0]
        
        xr_t=K.dot(x_tm1, self.W_r)+ self.b_r+ K.dot(inputs, self.V_r)
        xz_t = K.dot(x_tm1, self.W_z)  + self.b_z+ K.dot(inputs, self.V_z)
        r_t  = self.recurrent_activation(xr_t + K.dot(s_tm1, self.U_r))
        z_t  = self.recurrent_activation(xz_t + K.dot(s_tm1, self.U_z))
        xs_t = K.dot(x_tm1, self.W_s)  + self.b_s+ K.dot(inputs, self.V_s)
        s1_t = self.activation(xs_t + K.dot(r_t*s_tm1, self.U_s))
        s_t = (1-z_t) * s_tm1 + z_t * s1_t
        x_t = self.tag_activation(K.dot(s_t, self.W_x) + self.b_x)
        
        t_t=self.tag_activation(K.dot(x_t, self.W_y) + self.b_y)
        
        
        Beam=K
        return Beam, [x_t,s_t,score]
           
    def get_config(self):
        config = {'hidden_units': self.hidden_dim,
                  'output_units': self.output_dim,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation}
        base_config = super(BSCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))  
 
#def Beam_cell(t_t,t_tm1,)


    