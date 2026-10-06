"""ConvSegment: FlatMLP plus one Conv1d stage that learns across L and channels
within each segment, before falling back to FlatMLP's own flatten -> Linear tail.

MOTIVATION: FlatMLP (no spatial structure at all) already works well. This
backbone asks for the smallest addition on top of it -- some layer-level and
channel-level learning -- while keeping the part after that addition IDENTICAL
to FlatMLP, so any gain over FlatMLP is attributable to that one new stage.

SEGMENT-MAJOR RESHAPE: the field arrives as (N, in_ch, L, M) -- L rows, M
columns (see QKVFieldDataset). This backbone's Conv1d treats "one segment's L
values" as the unit a single conv window should be able to span, so the (L, M)
plane is first transposed and flattened to (N, in_ch, M*L) ordered as M BLOCKS
of L -- i.e. position [0:L) is segment 0's L layer-values, [L:2L) is segment
1's, etc. (plain `.flatten()` of the untransposed (L, M) tensor would instead
interleave an M-stride pattern, putting all L *layers'* values for a fixed
column together across segments, not within one). With KERNEL_SIZE = STRIDE =
L (the default below), each conv step reads exactly one whole segment and
produces one output position per segment -- i.e. a learned per-segment
summary across L and across channels (Conv1d mixes all in_ch input channels
into every output channel, which a plain reshape never would).

KERNEL_SIZE / STRIDE are hardcoded constants below, not config fields --
change them directly in this file. Changing them changes what each conv
window spans (e.g. KERNEL_SIZE=2*L, STRIDE=L makes each window straddle two
adjacent segments) and, when STRIDE != KERNEL_SIZE, how many output positions
the conv produces relative to M. `_checked_padding` below always pads so the
conv's own output covers the input length with no leftover tail (a real
acceptability check, done once at import time against the actual (L, M) this
run uses); whenever OUT_CHANNELS != in_ch or the output length comes out
!= M*L (which happens whenever STRIDE > 1, since no padding can undo a
stride's downsampling), `nn.Upsample` restretches the sequence back to
exactly M*L before the channel-projection step below -- so the rest of this
backbone (the part shared with FlatMLP) always sees the same (in_ch, L, M)
shape it would without this stage at all, regardless of KERNEL_SIZE/STRIDE.

CHANNEL PROJECTION BACK TO in_ch BEFORE FLATTEN: the Conv1d's own OUT_CHANNELS
(set below, can exceed in_ch to give the conv itself more capacity) is
projected back down to in_ch with a final 1x1 Conv1d before flattening --
flattening at OUT_CHANNELS directly would multiply proj's fan-in by
OUT_CHANNELS/in_ch for no benefit, since nothing downstream needs the extra
channels once the conv stage is done.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: Each conv window spans exactly one segment's L layer-values (kernel) and
#: steps by a full segment at a time (stride) -- the "look at 1 segment of L
#: numbers per step" setting from this backbone's own motivation. Edit these
#: directly to try e.g. a window that straddles 2 segments (KERNEL_SIZE =
#: 2 * L, STRIDE = L) or a finer stride (STRIDE = L // 2) -- `_checked_padding`
#: below will raise a clear error if a choice can't be tiled cleanly.
KERNEL_SIZE: int | None = 1   # None -> resolved to n_rows (= L) at construction
STRIDE: int | None = None        # None -> resolved to KERNEL_SIZE (one segment/step)
CONV_OUT_CHANNELS = 3


def _checked_padding(seq_len: int, kernel: int, stride: int) -> int:
    """The minimal symmetric-ish (left-padding only, via nn.Conv1d's single
    `padding` value applied to both sides) padding such that `kernel` windows
    stepping by `stride` tile `seq_len` with no leftover: i.e. the last window
    ends exactly at the (padded) sequence's end. Raises if `kernel` itself
    can't fit even once -- the one way this family of knobs is a real,
    checkable mistake rather than just "a different valid shape".
    """
    if kernel > seq_len + 2 * (kernel - 1):
        raise ValueError(
            f"conv_segment: kernel_size={kernel} cannot fit inside the "
            f"segment-flattened sequence of length {seq_len} even with padding."
        )
    # Smallest padding p (applied to both ends, so + 2p) such that
    # (seq_len + 2p - kernel) is an exact multiple of stride -- i.e. no
    # partial/leftover window at the tail.
    remainder = (seq_len - kernel) % stride
    if remainder == 0:
        return 0
    pad_total = stride - remainder
    if pad_total % 2 != 0:
        pad_total += 1  # nn.Conv1d's `padding` pads both sides equally
    return pad_total // 2


class ConvSegment(nn.Module):
    """FlatMLP + one per-segment Conv1d stage. See module docstring.

    Built eagerly at construction (needs `n_rows`, `n_segments` up front) for
    the same reason FlatMLP is: `proj`'s shape depends on them, and
    `test.py`/`cam.py`/`forecasting.py` call `load_state_dict` with no forward
    pass first -- see flat_mlp.py's docstring for the full story.
    """

    def __init__(
        self,
        n_rows: int,
        n_segments: int,
        embed_dim: int = 128,
        dropout: float = 0.0,
        in_ch: int = 3,
    ):
        super().__init__()
        self.n_rows = n_rows
        self.n_segments = n_segments
        self.in_ch = in_ch
        seq_len = n_rows * n_segments

        kernel = KERNEL_SIZE if KERNEL_SIZE is not None else n_rows
        stride = STRIDE if STRIDE is not None else kernel
        padding = _checked_padding(seq_len, kernel, stride)
        self.conv = nn.Conv1d(
            in_ch, CONV_OUT_CHANNELS, kernel_size=kernel, stride=stride, padding=padding
        )
        self.act = nn.GELU()
        # Restore the exact seq_len the rest of this backbone (shared with
        # FlatMLP) expects, regardless of what (kernel, stride) did to it --
        # a no-op shape-wise whenever stride == 1 already lands on seq_len.
        conv_out_len = (seq_len + 2 * padding - kernel) // stride + 1
        self.resize = (
            nn.Identity()
            if conv_out_len == seq_len
            else nn.Upsample(size=seq_len, mode="linear", align_corners=False)
        )
        # Back to in_ch before flattening -- see module docstring.
        self.channel_proj = nn.Conv1d(CONV_OUT_CHANNELS, in_ch, kernel_size=1)

        # From here on, identical to FlatMLP's tail.
        self.proj = nn.Linear(in_ch * n_rows * n_segments, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (N, in_ch, L, M)
        n = x.shape[0]
        # (N, in_ch, L, M) -> (N, in_ch, M, L) -> (N, in_ch, M*L), segment-major.
        seq = x.transpose(-1, -2).reshape(n, self.in_ch, self.n_segments * self.n_rows)
        seq = self.act(self.conv(seq))                     # (N, CONV_OUT_CHANNELS, ~M*L)
        seq = self.resize(seq)                              # (N, CONV_OUT_CHANNELS, M*L)
        seq = self.channel_proj(seq)                        # (N, in_ch, M*L)
        # Undo the segment-major reshape so flatten sees the same (in_ch, L, M)
        # layout FlatMLP's input always has.
        img = seq.reshape(n, self.in_ch, self.n_segments, self.n_rows).transpose(-1, -2)
        flat = img.flatten(1)                                # (N, in_ch*L*M)
        return self.proj(self.drop(flat))                    # (N, E)
