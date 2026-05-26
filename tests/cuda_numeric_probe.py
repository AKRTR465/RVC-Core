import argparse
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
OLD_ROOT = ROOT / "Retrieval-based-Voice-Conversion-WebUI"
NEW_ROOT = ROOT / "RVC_rebuild"


def add_import_roots():
    for path in (str(NEW_ROOT), str(OLD_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def max_abs_diff(left, right):
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def tensor_report(left, right):
    return {
        "equal": bool(torch.equal(left, right)),
        "max_abs_diff": max_abs_diff(left, right),
        "shape": list(left.shape),
    }


def sine_wave(device, samples=64, freq=220.0, sr=16000):
    t = torch.arange(samples, device=device, dtype=torch.float32) / float(sr)
    return torch.sin(2.0 * math.pi * freq * t)


def compare_rand_slice(device, seed):
    from infer.lib.infer_pack import commons as old_commons
    from src.utils.infer_pack import commons as new_commons

    x = torch.arange(2 * 3 * 32, device=device, dtype=torch.float32).reshape(2, 3, 32)
    lengths = torch.tensor([32, 28], device=device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    old_slice, old_ids = old_commons.rand_slice_segments(x, lengths, 8)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    new_slice, new_ids = new_commons.rand_slice_segments(x, lengths, 8)
    return {
        "ids": tensor_report(old_ids, new_ids),
        "slice": tensor_report(old_slice, new_slice),
        "old_ids": old_ids.detach().cpu().tolist(),
        "new_ids": new_ids.detach().cpu().tolist(),
    }


def compare_wn_train_step(device, seed):
    from infer.lib.infer_pack.modules import WN as OldWN
    from src.utils.infer_pack.modules import WN as NewWN

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    old_model = OldWN(4, 3, 1, 2, gin_channels=0, p_dropout=0).to(device)
    new_model = NewWN(4, 3, 1, 2, gin_channels=0, p_dropout=0).to(device)
    new_model.load_state_dict(old_model.state_dict(), strict=True)

    base = sine_wave(device)
    x = torch.stack(
        [base, base * 0.5, -base * 0.25, torch.cos(base)], dim=0
    ).unsqueeze(0)
    x_mask = torch.ones_like(x)

    def step(model):
        optim = torch.optim.AdamW(model.parameters(), 1e-4, betas=(0.8, 0.99), eps=1e-9)
        optim.zero_grad(set_to_none=True)
        out = model(x.clone(), x_mask)
        loss = out.square().mean()
        loss.backward()
        optim.step()
        return (
            out.detach(),
            loss.detach(),
            {key: value.detach().clone() for key, value in model.state_dict().items()},
        )

    old_out, old_loss, old_state = step(old_model)
    new_out, new_loss, new_state = step(new_model)
    state_mismatches = []
    for key in old_state:
        if not torch.equal(old_state[key], new_state[key]):
            state_mismatches.append(
                {
                    "key": key,
                    "max_abs_diff": max_abs_diff(old_state[key], new_state[key]),
                }
            )
    return {
        "out": tensor_report(old_out, new_out),
        "loss": tensor_report(old_loss, new_loss),
        "loss_old": float(old_loss.item()),
        "loss_new": float(new_loss.item()),
        "state_equal": len(state_mismatches) == 0,
        "state_mismatch_count": len(state_mismatches),
        "first_state_mismatch": state_mismatches[:1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    add_import_roots()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")

    result = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "input_device": str(sine_wave(device).device),
        "torch": torch.__version__,
        "seed": args.seed,
        "rand_slice": compare_rand_slice(device, args.seed + 1),
        "wn_train_step": compare_wn_train_step(device, args.seed + 2),
        "allocated_bytes": torch.cuda.memory_allocated(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    result["all_equal"] = (
        result["rand_slice"]["ids"]["equal"]
        and result["rand_slice"]["slice"]["equal"]
        and result["wn_train_step"]["out"]["equal"]
        and result["wn_train_step"]["loss"]["equal"]
        and result["wn_train_step"]["state_equal"]
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
