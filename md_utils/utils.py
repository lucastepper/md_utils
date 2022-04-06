import os


def pad_with_spaces(item, length):
    """ Cast item to string and pad it with spaces to given length, adding to the left. """
    item = str(item)
    n_to_pad = length - len(item)
    return n_to_pad * " " + item


def save_ndx(save_dir, name, idxs, len_line=80):
    """ Save a list of indexes (1-based) to file in gmx ndx format. """
    idxs = list(idxs)
    assert all([isinstance(idx, int) for idx in idxs]), "Indexes need to be integers"
    save_name = os.path.join(save_dir, name + ".ndx")
    len_per_idx = max([len(str(idx)) for idx in idxs])
    lines = [f"[ {name} ]\n"]
    line = ""
    while len(idxs) > 0:
        item_to_add = pad_with_spaces(idxs.pop(0), len_per_idx) + " "
        if len(line + item_to_add) < len_line:
            line += item_to_add
        else:
            lines.append(line.rstrip(" ") + "\n")
            line = item_to_add
    lines.append(line.rstrip(" ") + "\n")
    with open(save_name, "w") as fh:
        fh.writelines(lines)