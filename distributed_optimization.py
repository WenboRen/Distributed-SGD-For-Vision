# Version on June 8th 2020, written by Wenbo Ren

import torch
import torch.nn.parallel
import torch.distributed
import torch.multiprocessing
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist


# In principle, the only public interface
def get_distributed_optimizer(alg, local_optimizer, rank, world_size, group, arg_dict):
    """Public interface to get the distributed optimizer specified by alg

    Arguments:
        alg (str): specifies the distributed optimization algorithm.
            Current version supports
                -(Post-)local SGD. When initial_steps = 0, Post-local SGD
                    reduces to local SGD
            Raise error if alg is not specified or supported

        local_optimizer (torch.optim.optimizer): The optimizer used for the
            local optimization

        rank (int): the rank of this process

        world_size (int): number of processes in the group this process
            belongs to

        group (list of int): the list of process ids that belong to this process
            group

        arg_dict (dictionary <str key type>): specifies the arguments for
            the distributed optimizer.

    Returns an instace of the distrbuted optimizer for the specified algorithm

    Supported algorthms:
        1. (Post)-Local SGD.
            First perform global optimization for $initial_steps steps,
            then, average the weights after every $local_steps steps
            "Lin, T., Stich, S. U., Patel, K. K., & Jaggi, M. (2018).
            Don't Use Large Mini-Batches, Use Local SGD.
            arXiv preprint arXiv:1808.07217."
        Arguments:
            'initial_steps' (int): number of initial global steps
            'local_steps' (int): number of local steps betwen two weights
                average
            'init_step_method' (str): method for running initial steps
        Init_step_method:
            1. 'multiple_processes': Run initial steps on all processes and
                average models after each step.
            2. 'single_process': Run initial steps on one process and then
                copy the model to other processes.

    """
    if alg in {'local_sgd', 'post_local_sgd'}:
        return PostLocalSGD(local_optimizer, rank, world_size, group, arg_dict)
    raise ValueError('Algorithm {} not specified or not supported, \
        supported algorithms include \'local_sgd\' and \'post_local_sgd\''
        .format(alg))


# Virtual class, father of all optimization algorithms
# Should not declare an instance of it
class DistributedOptimizer():
    # Public, constructor
    def __init__(self, local_optimizer, rank, world_size, group, arg_dict):
        self._rank = rank
        self._world_size = world_size
        self._group = group
        self.set_process_group()
        self.local_optimizer = local_optimizer

    # Private, for setting the process group
    def set_process_group(self):
        self._process_group = dist.group.WORLD if self._group is None \
            else dist.new_group(ranks=self._group)

    # Public, virtual method, awaiting implementation by successors
    def step(self):
        with torch.no_grad():
            self.local_optimizer.step()

    # Private
    def state_dict(self):
        dict = self.local_optimizer.state_dict()
        dict['rank'] = self._rank
        dict['world_size'] = self._world_size
        dict['group'] = self._group
        return dict

    # Private
    def load_state_dict(self, dict):
        self.local_optimizer.load_state_dict(dict)
        self._rank = dict['rank']
        self._world_size = dict['world_size']
        self._group = dict['group']
        self.set_process_group()

    # Private
    def average_weights(self):
        for group in self.local_optimizer.param_groups:
            for param in group['params']:
                dist.all_reduce(
                    param.data,
                    op=dist.ReduceOp.SUM, 
                    group=self._process_group
                )
                param.data /= self._world_size

    # Private
    # Broadcast the weights from src to all other processes in self._process_group
    def broadcast_weights(self, src=None):
        if src is None:
            if self._process_group is None:
                src = 0
            elif self._leader is not None:
                src = self._leader
            else:
                raise ValueError('src {} unrecognized'.format(src))

        for group in self.local_optimizer.param_groups:
            for param in group['params']:
                dist.broadcast(param.data, src, self._process_group)


# Implement the (post-)local SGD algrithm
class PostLocalSGD(DistributedOptimizer):
    # Private constant
    # Set default values to minimize the chance of raising an error
    _default_local_steps = 8
    _default_initial_steps = 0
    _default_initial_step_method = 'single_process'

    # Public constructor
    def __init__(self, local_optimizer, rank, world_size, group, arg_dict):
        super().__init__(local_optimizer, rank, world_size, group, arg_dict)

        if 'local_steps' not in arg_dict:
            self._local_steps = self._default_local_steps
        else:
            self._local_steps = arg_dict['local_steps']

        if 'initial_steps' not in arg_dict:
            self._initial_steps = self._default_initial_steps
        else:
            self._initial_steps = arg_dict['initial_steps']

        if 'initial_step_method' not in arg_dict:
            self._initial_step_method = self._default_initial_step_method
        else:
            self._initial_step_method = arg_dict['initial_step_method']

        self._step_counter = 0
        self._leader = 0 if group is None else group[0]

    # Public
    def zero_grad(self):
        self.local_optimizer.zero_grad()

    # Override
    @torch.no_grad()
    def step(self):
        self._step_counter += 1

        if self._initial_step_method == 'multiple_processes':
            super().step()
            if self._step_counter <= self._initial_steps \
                    or self._step_counter % self._local_steps == 0:
                self.average_weights()

        elif self._initial_step_method == 'single_process':
            if self._step_counter <= self._initial_steps \
                    and self._rank != self._leader:
                return

            if self._step_counter == self._initial_steps + 1:
                self.broadcast_weights(self._leader)

            super().step()

            if self._step_counter > self._initial_steps \
                    and self._step_counter % self._local_steps == 0:
                self.average_weights()

        else:
            raise ValueError('initial step method {} not specified or supported, \
                supported methods include \'single_process\' and \'multiple_processes\''
                .format(self._initial_step_method))

    # Override
    def state_dict(self):
        dict = super().state_dict()
        dict['local_steps'] = self._local_steps
        dict['initial_steps'] = self._initial_steps
        dict['step_counter'] = self._step_counter
        dict['initial_step_method'] = self._initial_step_method
        return dict

    # Override
    def load_state_dict(self, dict):
        super().load_state_dict(dict)
        self._local_steps = dict['local_steps']
        self._initial_steps = dict['initial_steps']
        self._step_counter = dict['step_counter']
        self._initial_step_method = dict['initial_step_method']

