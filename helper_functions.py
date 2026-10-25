import numpy as np

def jukes_cantor(reference_sequence: str, distant_sequence: str) -> float:
    """The Jukes-Cantor correction for estimating genetic distances
    calculated with Hamming distance.
    Should return genetic distance with the same unit as if not corrected.

    Parameters
    ----------
    reference_sequence: str
        A string of nucleotides in a sequence used as a reference
        in an alignment with other (e.g. AGGT-GA)
    distant_sequence: str
        A string of nucleotides in a sequence after the alignment
        with a reference (e.g. AGC-AGA)

    Returns
    -------
    float
        The Jukes-Cantor corrected genetic distance using Hamming distance.
        For example 1.163.

    """
    ref = np.array(list(reference_sequence.upper()))
    dist = np.array(list(distant_sequence.upper()))

    # Ignore positions where sequence has an indel ('-' or '.')
    valid = (ref != '-') & (dist != '-') & (ref != '.') & (dist != '.')

    ref_valid = ref[valid]
    dist_valid = dist[valid]

    L = len(ref_valid)
    if L == 0:
        return 0.0
    
    mismatch = np.sum(ref_valid != dist_valid)
    p = mismatch / L

    # prevent log domain errors if p approaches or exceeds 3/4

    if p >= 3.0 / 4.0: 
        p = 3.0 / 4.0 - 1e-6
    
    d_site = - (3.0 / 4.0) * np.log(1.0 - (4.0 / 3.0) * p)

    return float(L * d_site)


def kimura_two_parameter(reference_sequence: str, distant_sequence: str) -> float:
    """The Kimura Two Parameter correction for estimating genetic distances
    calculated with Hamming distance.
    Should return genetic distance with the same unit as if not corrected.

    Parameters
    ----------
    reference_sequence: str
        A string of nucleotides in a sequence used as a reference
        in an alignment with other (e.g. AGGT-GA)
    distant_sequence: str
        A string of nucleotides in a sequence after the alignment
        with a reference (e.g. AGC-AGA)

    Returns
    -------
    float
        The Kimura corrected genetic distance using Hamming distance.
        For example 1.196.

    """
    raise NotImplementedError()
