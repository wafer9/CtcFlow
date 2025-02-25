import re
from jiwer.process import _word2char, AlignmentChunk
import rapidfuzz


def split_en(s):
    return s.strip().split()


def split_zh(s):
    pattern = r"[\u4e00-\u9fff]|[a-z\-\.'@0-9%$¢€£]+|\S"
    return re.findall(pattern, s)


def compute_wer(ref, hyp, lan='en'):
    if lan in ['zh', 'en']:
        transform = {'zh': split_zh, 'en': split_en,}[lan]
        ref_units = transform(ref)
        hyp_units = transform(hyp)
    else:
        ref_units = ref.strip().split() # transform(ref)
        hyp_units = hyp.strip().split() # transform(hyp)

    if len(ref_units) == 0:
        H, D, S, I, wer = 0, 0, 0, len(hyp_units), float('inf')
        alignments = [
            AlignmentChunk(
                type='insert',
                ref_start_idx=-1,
                ref_end_idx=-1,
                hyp_start_idx=0,
                hyp_end_idx=len(hyp_units),
            )
        ]
    else:
        ref_as_chars, hyp_as_chars = _word2char(ref_units, hyp_units)
        edit_ops = rapidfuzz.distance.Levenshtein.editops(
            ref_as_chars, hyp_as_chars)
        S = sum(1 if op.tag == "replace" else 0 for op in edit_ops)
        D = sum(1 if op.tag == "delete" else 0 for op in edit_ops)
        I = sum(1 if op.tag == "insert" else 0 for op in edit_ops)
        H = len(ref_units) - (S + D)
        wer = float(S + D + I) / float(H + S + D)
        alignments = [
            AlignmentChunk(
                type=op.tag,
                ref_start_idx=op.src_start,
                ref_end_idx=op.src_end,
                hyp_start_idx=op.dest_start,
                hyp_end_idx=op.dest_end,
            ) for op in rapidfuzz.distance.Opcodes.from_editops(edit_ops)
        ]

    return {
        'references': ref_units,
        'hypotheses': hyp_units,
        'hits': H,
        'deletions': D,
        'substitutions': S,
        'insertions': I,
        'wer': wer,
        'alignments': alignments,
    }
