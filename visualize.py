import sys
import torch

ORANGE = "\033[38;2;255;100;50m"
BLUE   = "\033[38;2;50;150;255m"
GREEN  = "\033[38;2;50;200;50m"
RED    = "\033[38;2;220;50;50m"
RESET = "\033[0m"

def show_binary_rationale(ids, z, idx2word, tofile=False):
    """
    Visualize rationale with torch support (works in VSCode terminal).
    Inputs:
        ids -- torch.Tensor, numpy array, or list of token ids (sequence_length,)
        z -- torch.Tensor, numpy array, or list of binary rationale (sequence_length,)
        idx2word -- dict or list mapping id -> word
        tofile -- if True, return string instead of printing
    """
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().tolist()
    idx2word[0] = "UNK"
    text = [idx2word[idx] for idx in ids]
    output = ""
    for i, word in enumerate(text):
        if z[i] == 1:
            output += word + " "
        else:
            output += "O" + " "

    if tofile:
        return output
    else:
        try:
            print(output)
        except Exception as e:
            print(e)

        sys.stdout.flush()
