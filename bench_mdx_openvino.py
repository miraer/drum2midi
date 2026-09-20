"""Can MDX23C run through OpenVINO, and is it faster than what we use now?

The NPU investigation left one gap and said so rather than guessing past it: the toy
convolution stack and the real ADTOF model were both measured through OpenVINO, and both
said the runtime overhead swallows any device gain. MDX23C -- the separator, which is by
far the most expensive stage -- was never tested, because a result on a toy says nothing
reliable about a 417 MB network.

This closes that gap or records exactly where it stops. Three steps, each of which can
fail informatively:

  1. build the model from its checkpoint and config and run one torch forward pass
  2. export it to ONNX, which is where models with STFT front-ends usually fail
  3. convert to OpenVINO IR and time it against torch on the same input

A failure at step 2 or 3 is a real answer: it means the path is not available without
rewriting the model, which is a different project.

Timings describe this machine -- Intel Arc integrated GPU, no NVIDIA -- and are not a
claim about the model.

    python bench_mdx_openvino.py
    python bench_mdx_openvino.py --runs 5 --seconds 6
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "uvr" / "models" / "MDX23C-DrumSep-aufr33-jarredou.ckpt"
CONFIG = ROOT / "uvr" / "models" / "config_drumsep_mdx23c.yaml"


def build_model():
    import torch
    import yaml
    from ml_collections import ConfigDict
    from audio_separator.separator.uvr_lib_v5.tfc_tdf_v3 import TFC_TDF_net

    cfg = ConfigDict(yaml.load(CONFIG.read_text(encoding="utf-8"), Loader=yaml.FullLoader))
    model = TFC_TDF_net(cfg, device="cpu")
    state = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


class SpectralCore:
    """The convolutional body, with the STFT and iSTFT left outside.

    ONNX has no complex tensor type, so the STFT front-end cannot cross the boundary --
    that is what blocks a whole-model export. It is also the cheap end. Everything
    between `cac2cws` and `cws2cac` is the 109 M parameters that actually cost time, and
    it is real-valued throughout, so it can be exported on its own.

    Built as a plain wrapper rather than by editing the vendored model, so the weights
    and the graph are exactly the ones the product runs.
    """

    def __new__(cls, model):
        import torch.nn as nn

        class _Core(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, spec):
                import torch
                m = self.m
                mix = x = spec
                first_conv_out = x = m.first_conv(x)
                x = x.transpose(-1, -2)
                skips = []
                for block in m.encoder_blocks:
                    x = block.tfc_tdf(x)
                    skips.append(x)
                    x = block.downscale(x)
                x = m.bottleneck_block(x)
                for block in m.decoder_blocks:
                    x = block.upscale(x)
                    x = torch.cat([x, skips.pop()], 1)
                    x = block.tfc_tdf(x)
                x = x.transpose(-1, -2)
                x = x * first_conv_out
                return m.final_conv(torch.cat([mix, x], 1))

        return _Core(model).eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--audio", type=Path, default=ROOT / "input" / "latin-groove.wav",
                    help="a real recording to take the test chunk from")
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "mdx_openvino")
    args = ap.parse_args()

    import numpy as np
    import torch

    torch_xpu = None
    for f in (CKPT, CONFIG):
        if not f.exists():
            print(f"missing: {f}")
            return 1

    print("step 1: build the model and run one torch forward pass")
    try:
        model, cfg = build_model()
    except Exception as exc:
        print(f"  FAILED to build: {type(exc).__name__}: {exc}")
        return 1
    n_param = sum(p.numel() for p in model.parameters())
    chunk = int(cfg.audio.chunk_size)
    print(f"  built: {n_param/1e6:.1f}M parameters, chunk {chunk} samples "
          f"({chunk/cfg.audio.sample_rate:.1f}s at {cfg.audio.sample_rate} Hz)")

    # Real audio, not noise: the accuracy comparison below is only meaningful on the
    # kind of spectrogram the model actually sees, and a fixed slice keeps it
    # reproducible between runs.
    x = None
    if args.audio and args.audio.exists():
        import soundfile as sf
        data, sr = sf.read(str(args.audio), always_2d=True, dtype="float32")
        if data.shape[0] >= chunk:
            seg = data[sr * 2: sr * 2 + chunk]
            if seg.shape[0] == chunk:
                if seg.shape[1] == 1:
                    seg = np.repeat(seg, 2, axis=1)
                x = torch.from_numpy(seg[:, :2].T.copy()).unsqueeze(0)
                print(f"  input: {args.audio.name}, 2 s in, {chunk} samples")
    if x is None:
        torch.manual_seed(0)
        x = torch.randn(1, 2, chunk)
        print("  input: seeded noise (no audio file given)")
    with torch.no_grad():
        t0 = time.perf_counter()
        y = model(x)
        first = time.perf_counter() - t0
    print(f"  forward ok: {tuple(y.shape)}, {first:.2f}s for the first pass")

    times = []
    with torch.no_grad():
        for _ in range(args.runs):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    torch_cpu = float(np.median(times))
    print(f"  torch cpu: {torch_cpu:.2f}s median of {args.runs}")

    print("\nstep 2: export to ONNX")
    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "mdx23c.onnx"
    whole_model_failed = None
    try:
        torch.onnx.export(
            model, x, str(onnx_path),
            input_names=["mix"], output_names=["stems"],
            opset_version=18, do_constant_folding=True)
        print(f"  whole model exported: {onnx_path.stat().st_size/1e6:.0f} MB")
        export_input, exported = x, onnx_path
    except Exception as exc:
        whole_model_failed = f"{type(exc).__name__}"
        print(f"  whole model FAILED: {whole_model_failed}")
        print("  cause: the STFT front-end works on complex tensors and ONNX has no")
        print("  complex type, so the boundary cannot be crossed intact.")
        print("\n  retrying with the STFT left outside, which is the cheap end anyway")
        core = SpectralCore(model)
        with torch.no_grad():
            spec = model.cac2cws(model.stft(x))
        print(f"  spectrogram input {tuple(spec.shape)}")
        onnx_path = args.out / "mdx23c_core.onnx"
        try:
            torch.onnx.export(
                core, spec, str(onnx_path),
                input_names=["spec"], output_names=["mask"],
                opset_version=18, do_constant_folding=True)
            print(f"  core exported: {onnx_path.stat().st_size/1e6:.0f} MB")
            export_input, exported = spec, onnx_path
        except Exception as exc2:
            print(f"  core FAILED too: {type(exc2).__name__}: {str(exc2)[:220]}")
            print("\nThat is the answer: no ONNX export, so no OpenVINO path short of")
            print("reimplementing the model, which is a different project.")
            return 0

        # Time the torch side of the same subgraph, so the comparison is like for like.
        with torch.no_grad():
            core(spec)
            ts = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                core(spec)
                ts.append(time.perf_counter() - t0)
        torch_cpu = float(np.median(ts))
        print(f"  torch cpu on the same subgraph: {torch_cpu:.2f}s")

        # The pipeline already runs this model on the Arc GPU through torch XPU, so
        # torch CPU is not the baseline any comparison should be made against. Without
        # this row, OpenVINO on a GPU would be credited with beating a CPU.
        torch_xpu = None
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            try:
                core_x = core.to("xpu")
                spec_x = spec.to("xpu")
                with torch.no_grad():
                    core_x(spec_x)
                    torch.xpu.synchronize()
                    ts = []
                    for _ in range(args.runs):
                        t0 = time.perf_counter()
                        core_x(spec_x)
                        torch.xpu.synchronize()
                        ts.append(time.perf_counter() - t0)
                torch_xpu = float(np.median(ts))
                print(f"  torch xpu (what the pipeline uses): {torch_xpu:.2f}s, "
                      f"{torch_cpu/torch_xpu:.2f}x over torch cpu")
                core = core.to("cpu")
            except Exception as exc:
                print(f"  torch xpu failed: {type(exc).__name__}: {str(exc)[:120]}")

    print("\nstep 3: convert to OpenVINO and time it")
    try:
        import openvino as ov
        core_ov = ov.Core()
        devices = core_ov.available_devices
        print(f"  devices: {devices}")
        ov_model = ov.convert_model(str(exported))
    except Exception as exc:
        print(f"  FAILED to convert: {type(exc).__name__}: {str(exc)[:300]}")
        return 0

    xn = export_input.detach().numpy()
    with torch.no_grad():
        ref = (core(export_input) if whole_model_failed else model(export_input)).numpy()
    print(f"  {'device':<8}{'time':>8}{'vs cpu':>9}{'vs xpu':>9}{'max rel err':>13}")
    for dev in [d for d in devices if d.split(".")[0] in ("CPU", "GPU", "NPU")]:
        try:
            compiled = core_ov.compile_model(ov_model, dev)
            got = list(compiled(xn).values())[0]
            compiled(xn)                      # warm up, excluded from the median
            ts = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                compiled(xn)
                ts.append(time.perf_counter() - t0)
            med = float(np.median(ts))
            # A speedup on a graph that computes something else is not a speedup.
            rel = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9))
            flag = "" if rel < 0.001 else ("  lossy" if rel < 0.02 else "  DIFFERS")
            vs_xpu = f"{torch_xpu/med:8.2f}x" if torch_xpu else f"{'-':>9}"
            print(f"  {dev:<8}{med:7.2f}s{torch_cpu/med:8.2f}x{vs_xpu}"
                  f"{rel:12.4%}{flag}")
        except Exception as exc:
            print(f"  {dev:<8} FAILED: {type(exc).__name__}: {str(exc)[:160]}")

    if whole_model_failed:
        print("\nNote: this compares the convolutional body only. The STFT and iSTFT")
        print("stay in torch either way, so a whole-pipeline speedup would be smaller")
        print("than these numbers suggest.")
    print("Any device whose output differs from torch is trading accuracy for speed,")
    print("and what that costs in onset F1 is not measured here.")
    print(f"\ntorch cpu baseline {torch_cpu:.2f}s per "
          f"{chunk/cfg.audio.sample_rate:.1f}s chunk. Timings describe this machine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
