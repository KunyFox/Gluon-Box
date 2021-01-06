# --------------------------------------------------------------------------------------
# Copyright 2021 by Kun Yuan, CortexLabs AI Group.
# All rights reserved.
# This file is part of the GLuon-Box (https://github.com/KunyFox/Gluon-Box),
# and is released under the "GNU General Public License v2.0". Please see the LICENSE.
# File that should have been included as part of this package.
#
# GTrainer is a Trainer including optimizer and learning-rate scheduler.
# --------------------------------------------------------------------------------------

import mxnet as mx 
import mxnet.gluon as gluon 
import mxnet.optimizer as optimizer 


class GTrainer(gluon.Trainer):
    """A trainer like mxnet.gluon.Trainer.

    Parameters
    ----------
    params : ParameterDict
        The set of parameters to optimize.
    num_epoch : int 
        The number of training epoch.
    optimizer : dict
        The optimizer config to use. 'type' should be initialized in the dict.
    lr_scheduler : dict 
        The learning rate scheduler.
    kvstore : str or KVStore
        kvstore type for multi-gpu and distributed training.
    compression_params : dict
        Specifies type of gradient compression and additional arguments depending
        on the type of compression being used. For example, 2bit compression requires a threshold.
        Arguments would then be {'type':'2bit', 'threshold':0.5}
        See mxnet.KVStore.set_gradient_compression method for more details on gradient compression.
    update_on_kvstore : bool, default None
        Whether to perform parameter updates on kvstore. If None, then trainer will choose the more
        suitable option depending on the type of kvstore. If the `update_on_kvstore` argument is
        provided, environment variable `MXNET_UPDATE_ON_KVSTORE` will be ignored.
    """
    def __init__(self,
                 params,
                 num_epoch,
                 optimizer=dict(
                     type='SGD',
                     momentum=0.0,
                     wd=1e-5,
                     clip_gradient=None
                 ),
                 lr_scheduler=dict(
                     type='step', 
                     learning_rate=0.1,
                     decay_rate=0.1, 
                     steps=[8, 11],
                     warmup='line',
                     warmup_iters=500,
                     warmup_ratio=0.1),
                 kvstore='device',
                 compression_params=None, 
                 update_on_kvstore=None):
        super(GTrainer, self).__init__()
        param_list = []
        if isinstance(params, (dict, gluon.ParameterDict)):
            for key in sorted(list(params.keys())):
                param_list.append(params[key])
            params = param_list
        if not isinstance(params, (list, tuple)):
            raise ValueError(
                "First argument must be a list or dict of Parameters, " \
                "got %s."%(type(params)))
        self._params = []
        self._contains_sparse_weight = False
        self._contains_sparse_grad = False
        self._param2idx = {}
        for i, param in enumerate(params):
            if not isinstance(param, Parameter):
                raise ValueError(
                    "First argument must be a list or dict of Parameters, " \
                    "got list of %s."%(type(param)))
            self._param2idx[param.name] = i
            self._params.append(param)
            param._set_trainer(self)
            if param._stype != 'default':
                self._contains_sparse_weight = True
            if param._grad_stype != 'default':
                self._contains_sparse_grad = True

        self._compression_params = compression_params
        self._contexts = self._check_contexts()

        self._num_epoch = num_epoch 

        self._init_optimizer(optimizer)
        self._scale = self._optimizer.rescale_grad

        assert isinstance(lr_scheduler, dict) and 'type' in lr_scheduler  \
                and 'learning_rate' in lr_scheduler
        if lr_scheduler.get('warmup', None):
            assert lr_scheduler.get('warmup_iters') > 0
            assert 0 < lr_scheduler.get('warmup_ratio') < 1
        self.lr_scheduler = lr_scheduler

        self._kvstore_params = {'kvstore': kvstore, 'update_on_kvstore':  \
                                update_on_kvstore}
        self._kv_initialized = False
        self._kvstore = None
        self._update_on_kvstore = None
        self._distributed = None
        self._params_to_init = []
        self._reset_kvstore()

        self.epoch = 0
        self.iteration = 0


    def _init_optimizer(self, optimizer_cfg):
        param_dict = {i: param for i, param in enumerate(self._params)}

        assert isinstance(optimizer_cfg, dict) and 'type' in optimizer_cfg

        # initialize optimizer
        _cfg = optimizer_cfg.copy()
        optimizer_type = _cfg.pop('type')
        _cfg.update({'param_dict': param_dict})
        self._optimizer = getattr(optimizer, optimizer_type)(**_cfg)
        self._updaters = [optimizer.get_updater(self._optimizer) for _ in  \
                          self._contexts]

        # initialize learning rate 
        if self.lr_scheduler.get('warmup', None):
            self.warmup_finished = False 
            self._optimizer.set_learning_rate(self._epoch_lr *  \
                self.lr_scheduler.get('warmup_ratio'))
        else:
            self.warmup_finished = True 
            self._optimizer.set_learning_rate(self._epoch_lr)
    


    def reset(self):
        if self.epoch > 0:
            self._update_learning_rate()
        self.epoch += 1
        if self.epoch > self._num_epoch:
            return -1
        return self.epoch

        
    def _update_learning_rate(self, mode=None):
        if mode == 'warmup' and not self.warmup_finished:
            lr = self._get_warmup_lr()
        else:
            decay_type = self.lr_scheduler.get('type')
            decay_rate = self.lr_scheduler.get('decay_rate')
            if decay_type == 'step':
                lr = lr * decay_rate if epoch in \
                    self.lr_scheduler.get('steps') else lr
            elif  decay_type == 'poly':
                lr = lr * decay_rate
            self._epoch_lr = lr 

        self._optimizer.set_learning_rate(lr)


    def _get_warmup_lr(self):
        lr = self.learning_rate()
        mode = self.lr_scheduler.get('warmup')
        ratio = self.lr_scheduler.get('warmup_ratio')
        iters = self.lr_scheduler.get('warmup_iters')
        base_lr = self.lr_scheduler.get('learning_rate')
        if mode == 'line':
            k = (1 - self.iteration / iters) * (1 - ratio)
            lr = base_lr * (1 - k)
        elif mode == 'constant': 
            lr = base_lr * ratio 
        elif mode == 'exp':
            k = ratio**(1 - self.iteration / iters)
            lr = base_lr * k 
        
        if lr >= self._epoch_lr:
            lr = self._epoch_lr 
            self.warmup_finished = True 

        return lr 


    def step(self, batch_size, ignore_stale_grad=False):
        self.iteration += 1
        rescale_grad = self._scale / batch_size
        self._check_and_rescale_grad(rescale_grad)
        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

        self._allreduce_grads()
        self._update(ignore_stale_grad)

        if not self.warmup_finished:
            self._update_learning_rate(mode='warmup')


    def save_states(self, fname):
        assert self._optimizer is not None

        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

        trainer_states = self._updaters[0].get_states(dump_optimizer=True)

        states = dict{
            epoch = self.epoch, 
            learning_rate = self._epoch_lr,
            states = trainer_states
        }

        with open(fname, 'wb') as fout:
            fout.write(states)


    def load_states(self, fname, only_states=False):
        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

        with open(fname, 'rb') as f:
            states = f.read()
        
        if not only_states:
            self.epoch = states.get('epoch', self.epoch)
            self._epoch_lr = states.get('learning_rate', self._epoch_lr)
        
        trainer_states = states.get('states', None)
        if trainer_states:
            print("Loading trainer states from {}".format(fname))
            for updater in self._updaters:
                updater.set_states(trainer_states)
                updater.optimizer = self._updaters[0].optimizer
            self._optimizer = self._updaters[0].optimizer
        param_dict = {i: param for i, param in enumerate(self._params)}
        self._optimizer.param_dict = param_dict

    @property 
    def learning_rate(self):
        return self._optimizer.learning_rate

    @property 
    def num_epoch(self):
        return self._num_epoch 