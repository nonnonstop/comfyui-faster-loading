import safetensors.torch

_load_file_org = safetensors.torch.load_file


def _load_file_for_wsl(filename, device="cpu", *args, **kwargs):
    try:
        if device == "cpu":
            with open(filename, "rb") as f:
                return safetensors.torch.load(f.read())
    except Exception:
        pass
    return _load_file_org(filename, device, *args, **kwargs)


safetensors.torch.load_file = _load_file_for_wsl
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
