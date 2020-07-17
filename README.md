# Distributed SGD for Computer Vision

Implement local SGD and apply it to train imageneta and cifar.

     Introduction: First perform the initial step method for $initial_steps steps, and then perform the local SGD algorithm, i.e., each process steps on its own samples, and all processes average the models after every $local_steps steps. When $initial_steps = 0, (post)-local SGD algorithm reduces to local SGD. ["Lin, T., Stich, S. U., Patel, K. K., & Jaggi, M. (2018). Don't Use Large Mini-Batches, Use Local SGD. arXiv preprint arXiv:1808.07217."]

You will need at least two CUDA devices to make the programs perform as expected.

### Sample commands to run cifar.py

    path=~/datasets/cifar-10-batches-py
    local_steps=4
    initial_steps=100
    initial_step_method=single_process
    # initial_step_method=multiple_processes

    python cifar.py $path \
        --dist-url 'tcp://127.0.0.1:23453' --dist-backend 'nccl' \
        -p 10 --epochs 5 --batch-size 256 \
        --local-steps $local_steps \
        --initial-steps $initial_steps \
        --initial-step-method $initial_step_method \
        | tee cifar.log
        
### Sample commands to run image.py

    path=~/datasets/imagenet
    local_steps=4
    initial_steps=100
    initial_step_method=single_process
    # initial_step_method=multiple_processesrint("")

    python imtest.py $path -a resnet18 \
        --dist-url 'tcp://127.0.0.1:23451' --dist-backend 'nccl' \
        --multiprocessing-distributed \
        --world-size 1 --rank 0 --epochs 2 \
        --local-sgd --local-steps $local_steps \
        --initial_steps $initial_steps \
        --initial_step_method $initial_step_method \
        | tee imagenet.log

### How to use distributed_optimization.py:
First, add arguments t= arg_dic. Second, use a local_optimizer (type torch.optim.optimizer) and pass it to distributed.optimization.get_distributed_optimizer() to get a distributed_optimizer. Third, use distributed_optimizer just like torch.optim.optimizer. 

### Sample code: 
    
    local_optimizer = torch.optim.SGD(
        model.parameters(), 
        args.lr,
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    arg_dict = {
        'local_steps': args.local_steps,
        'initial_steps': args.initial_steps,
        'initial_step_method': args.initial_step_method,
    }
    optimizer = get_distributed_optimizer(
        'local_sgd', 
        local_optimizer, 
        args.rank,
        args.world_size, 
        args.group, 
        arg_dict
    )
        
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    for steps:
        output = model(images)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
### Document:

    def get_distributed_optimizer(alg, local_optimizer, rank, world_size, group, arg_dict):
        """
    
        Public interface to get the distributed optimizer specified by alg

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

        Note: If want to let some param_groups not processed by distributed
             optimizer, then add a pair ('allow_data_parallelism', False) in the
            corresponding param_group of the local_optimizer.

        Supported algorthms:
        1. (Post)-Local SGD.
            Introduction:
                First perform the initial step method for $initial_steps steps, and
                then perform the local SGD algorithm, i.e., each process steps on
                its own samples, and all processes average the models after every
                $local_steps steps. When $initial_steps = 0, (post)-local SGD
                algorithm reduces to local SGD.
                "Lin, T., Stich, S. U., Patel, K. K., & Jaggi, M. (2018).
                Don't Use Large Mini-Batches, Use Local SGD.
                arXiv preprint arXiv:1808.07217."
            Arguments:
                'initial_steps' (int): number of initial global steps (default 0)
                'local_steps' (int): number of local steps betwen two weights
                    average (default 4)
                'init_step_method' (str): method for running initial steps (default
                    'single_process')
            init_step_method:
                1. 'multiple_processes': Run initial steps on all processes and
                    average models after each step.
                2. 'single_process': Run initial steps on one process and then
                    copy the model to other processes.

        """
