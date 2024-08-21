from typing import Optional

import torch


def assert_size_of_tensor(name: str, tensor: Optional[torch.Tensor], *, expected_size: tuple[int, ...], optional=True) -> None:
    if optional and tensor is None:
        return
    else:
        assert tensor.size() == torch.Size(expected_size), f"tensor {name} size is {tensor.size()}, expected {expected_size}"
