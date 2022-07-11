"""
Distributed helpers functions.

Mostly copy-paste from torchvision references.
"""
import os

import torch
import torch.distributed as dist


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_local_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.no_distributed:
        print('Not using distributed mode')
        args.distributed = False
        return

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    # print('world_size', world_size)
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def fn_on_master_if_distributed(is_distributed, fn, *args, **kwargs):
    if is_distributed:
        if is_main_process():
            return fn(*args, **kwargs)
    else:
        return fn(*args, **kwargs)


def gather_object_in_order_backup(src_tensor, to_cpu=False):
    # use the process rank to ensure the order
    with torch.no_grad():
        tensor_list = [{x: torch.ones_like(src_tensor)} for x in range(get_world_size())]
        dist.all_gather_object(tensor_list, {get_rank(): src_tensor})
        tensor_dict = {}
        for item in tensor_list:
            tensor_dict.update(item)
        out_tensor = torch.cat([tensor_dict[x].cpu() if to_cpu else tensor_dict[x] for x in range(get_world_size())], dim=0)
    return out_tensor


def gather_object_in_order(src_tensor):
    tensor_list = [torch.ones_like(src_tensor) for _ in range(get_world_size())]
    dist.all_gather(tensor_list, src_tensor)
    out_tensor = torch.cat(tensor_list, dim=0)
    return out_tensor
