import os
import time

import hyperparameter as hp

import torch
import torch.distributed as dist


@hp.param("blueprinting.diag")
def prepare_cluster(backend="nccl", device="cuda", timeout=1, hostname=None):
    from datetime import timedelta

    dist.init_process_group(backend=backend, init_method="env://")
    if device == "cuda":
        torch.cuda.set_device(dist.get_rank())

    tcpstore = dist.TCPStore(
        os.environ["MASTER_ADDR"],
        int(os.environ["MASTER_PORT"]),
        dist.get_world_size(),
        timeout=timedelta(seconds=timeout),
    )
    tcpstore.set(f"rank_hostname[{dist.get_rank()}]", hostname or get_hostname())
    tcpstore.add(f"ready_ranks", 1)

    while True:
        ready_ranks = tcpstore.get("ready_ranks")
        if int(ready_ranks) < dist.get_world_size():
            time.sleep(0.1)
            continue
        else:
            break

    if dist.get_rank() == 0:
        for i in range(dist.get_world_size()):
            print(f"\trank {i}: {tcpstore.get(f'rank_hostname[{i}]').decode()}")

    if dist.get_rank() == 0:
        import threading

        thread = threading.Thread(
            target=poll_result,
            args=(tcpstore, list(range(dist.get_world_size()))),
        )
        thread.start()

    return tcpstore


@hp.param("blueprinting.diag")
def test_all_reduce(group, ranks, device="cuda"):
    try:
        tensor = torch.tensor([dist.get_rank()]).to(device)
        expected = sum(ranks)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)

        return tensor.item() == expected
    except Exception as e:
        print(f"[Rank {dist.get_rank()}] Error in all_reduce: {e}")
        return False


def test_group(tcpstore, ranks):
    my_rank = dist.get_rank()
    in_group = my_rank in ranks

    mid = len(ranks) // 2
    left_ranks, right_ranks = ranks[:mid], ranks[mid:]

    if len(left_ranks) > 1 and len(right_ranks) > 1:
        left_success = test_group(tcpstore, left_ranks)
        right_success = test_group(tcpstore, right_ranks)
        if not left_success or not right_success:
            return False

    group = dist.new_group(ranks=ranks)
    current_success = test_all_reduce(group, ranks) if in_group else True
    tcpstore.set(f"test_{ranks}_allreduce", "SUCC" if current_success else "FAIL")

    return current_success


def generate_rank_list(ranks):
    mid = len(ranks) // 2
    left_ranks, right_ranks = ranks[:mid], ranks[mid:]

    if len(left_ranks) > 1 and len(right_ranks) > 1:
        return (
            [ranks] + generate_rank_list(left_ranks) + generate_rank_list(right_ranks)
        )
    else:
        return [ranks]


def poll_result(tcpstore, ranks):
    rank_list = generate_rank_list(ranks)
    rank_list.sort(key=lambda x: len(x))

    print(f"Rank list: {rank_list}")

    completed = set()

    while len(completed) < len(rank_list):
        for rl in rank_list:
            key = f"test_{rl}_allreduce"
            if key in completed:
                continue
            try:
                ret = tcpstore.get(key).decode()
                completed.add(key)
                key = key.replace("test_", "").replace("_allreduce", "")
                print(f"{str(ret)}: {key}")
            except:
                pass

        time.sleep(0.1)


def get_hostname():
    import socket

    return socket.gethostname()


def main(args):
    tcpstore = prepare_cluster()
    success = test_group(tcpstore, list(range(dist.get_world_size())))

    # 输出最终结果
    if dist.get_rank() == 0:
        print("\nFinal test result:", "SUCCESS" if success else "FAILURE")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CCL Test")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for testing")
    parser.add_argument(
        "--hostname", type=str, default=None, help="Hostname for the cluster"
    )
    args = parser.parse_args()
    if os.getenv('RANK') == "0":
        print(f"Running with arguments: {args}")

    with hp.scope(
        **{
            "blueprinting.diag": {
                "backend": (
                    "nccl" if args.cuda and torch.cuda.is_available() else "gloo"
                ),
                "device": "cuda" if args.cuda and torch.cuda.is_available() else "cpu",
                "hostname": args.hostname,
                "timeout": 1,
            }
        }
    ):
        main(args)
