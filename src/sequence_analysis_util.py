import pandas as pd

def extract_aa_at_positions(sequence, positions):
    """
    Extract amino acids from a sequence at the given positions.
    
    Parameters:
    - sequence (str): The amino acid sequence.
    - positions (dict): Dictionary where keys are reference positions 
                        and values are the corresponding positions in 
                        the MSA sequence.

    Returns:
    - str: A string of amino acids extracted from the specified positions.
    """
    extracted_aas = ''.join([sequence[pos] if pos < len(sequence) else '-' for pos in positions.values()])
    return extracted_aas

corresponding_positions = {0: 8,
 32: 40,
 34: 42,
 36: 44,
 78: 86,
 101: 109,
 107: 115,
 138: 146,
 139: 147,
 145: 153,
 154: 162,
 157: 165,
 159: 167,
 162: 170,
 166: 174}

def get_AA_at_15_loci(mutation, positions, wt):
    # Create a copy of the wild type sequence
    sequence = wt.copy()
    
    # Extract the position and mutated amino acid from the mutation string
    original_aa = mutation[0]
    mutated_aa = mutation[-1]
    mutation_position = mutation[1:-1]
    
    # If the mutation position is in the list of positions, replace the corresponding amino acid
    if mutation_position in positions:
        index = positions.index(mutation_position)
        sequence[index] = mutated_aa
    
    return ''.join(sequence)

def hamming_distance(seq1, seq2):
    """Compute the Hamming distance between two sequences."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length")
    return sum(ch1 != ch2 for ch1, ch2 in zip(seq1, seq2))


