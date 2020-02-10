import typing as t


def read_codebook(path: str) -> t.Mapping[str, str]:
    """Codebook file is a tab splited file,
    record the mapping from barcode to genes:
        <code>  <gene name/id>
    """
    code2gene = {}
    with open(path) as f:
        for line in f:
            items = line.strip().split()
            code2gene[items[0]] = items[1]
    return code2gene


def get_chars2chidx(channels: str,
                    k: int) -> t.Mapping[str, int]:
    assert 0 < k < len(channels)
    if k == 1:
        chars2chidx = {}
    else:
        from twintail.utils.spots.channel import channel_combinations
        combs = channel_combinations(list(range(len(channels))), k)
        chars2chidx = {
            "".join(sorted([channels[i] for i in oidxs])): idx
            for idx, oidxs in enumerate(combs)
        }
    return chars2chidx


def get_code2chidxs(codes: t.Iterable[str],
                    channels: str,
                    k: int,
                    ) -> t.Mapping[str, t.List[int]]:
    chars2chidx = get_chars2chidx(channels, k)
    code2chidxs = {}
    for code in codes:
        chidxs = []
        for i in range(len(code) // k):
            chars = code[i*k:(i+1)*k]
            chars = "".join(sorted(chars))
            chidx = chars2chidx[chars]
            chidxs.append(chidx)
        code2chidxs[code] = chidxs
    return code2chidxs
