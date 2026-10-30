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
    valid = (ref != '-') & (dist != '-') & (ref != '.') & (dist != '.') & (ref != 'N') & (dist != 'N')

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

    # return total corrected genetic distance
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

    ref = reference_sequence.upper()
    dist = distant_sequence.upper()

    transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}

    valid_comparisons = 0
    p_count = 0
    q_count = 0

    for r_char, d_char in zip(ref, dist):
        # Ignore gaps and ambigoys characters
        if r_char in '-NX' or d_char in '-NX':
            continue
        if r_char == '.' or d_char == '.':
            continue

        valid_comparisons += 1
        if r_char != d_char: 
            if (r_char, d_char) in transitions:
                p_count += 1
            else:
                q_count += 1

    if valid_comparisons == 0:
        return 0.0
    
    P = p_count / valid_comparisons
    Q = q_count / valid_comparisons

    term1 = 1.0 - 2.0 * P - Q
    term2 = 1.0 - 2.0 * Q

    if term1 <= 0 or term2 <= 0: 
        # Fallback for highly saturated sequences
        return float('inf')
    
    k2p_dist = -0.5 * np.log(term1) - 0.25 * np.log(term2)
    return k2p_dist * valid_comparisons 
