"""
REUSE-FIRST: torch ships its own environment-collection script
(torch.utils.collect_env) that already reports Python/torch/CUDA/OS
versions correctly across platforms. We just expose it under our CLI
rather than reimplementing environment introspection.
"""


def collect_env_info() -> str:
    from torch.utils.collect_env import get_pretty_env_info

    return get_pretty_env_info()


if __name__ == "__main__":
    print(collect_env_info())
