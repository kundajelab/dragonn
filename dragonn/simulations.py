from __future__ import absolute_import, division, print_function
import numpy as np
from dragonn.synthetic.synthetic import (
    RepeatedEmbedder, SubstringEmbedder, ReverseComplementWrapper,
    UniformPositionGenerator, InsideCentralBp,
    LoadedEncodeMotifs, PwmSamplerFromLoadedMotifs,
    UniformIntegerGenerator, ZeroOrderBackgroundGenerator,
    EmbedInABackground, GenerateSequenceNTimes,
    RandomSubsetOfEmbedders, IsInTraceLabelGenerator,
    EmbeddableEmbedder, PairEmbeddableGenerator,
)
from dragonn.synthetic.util import DiscreteDistribution
from pkg_resources import resource_filename

ENCODE_MOTIFS_PATH = resource_filename('dragonn.synthetic', 'motifs.txt.gz')
loaded_motifs = LoadedEncodeMotifs(ENCODE_MOTIFS_PATH, pseudocountProb=0.001)


def get_distribution(GC_fraction):
    return DiscreteDistribution({
        'A': (1 - GC_fraction) / 2, 'C': GC_fraction / 2,
        'G': GC_fraction / 2, 'T': (1 - GC_fraction) / 2})


def simple_motif_embedding(motif_name, seq_length, num_seqs, GC_fraction):
    """
    returns sequence array
    """
    if motif_name is None:
        embedders = []
    else:
        substring_generator = PwmSamplerFromLoadedMotifs(
            loaded_motifs, motif_name)
        embedders = [SubstringEmbedder(
            ReverseComplementWrapper(substring_generator))]
    embed_in_background = EmbedInABackground(
        ZeroOrderBackgroundGenerator(
            seq_length, discreteDistribution=get_distribution(GC_fraction)),
        embedders)
    generated_sequences = GenerateSequenceNTimes(
        embed_in_background, num_seqs).generateSequences()
    sequence_arr = np.asarray(
        [generated_seq.seq for generated_seq in generated_sequences])

    return sequence_arr


def motif_density(motif_name, seq_length, num_seqs,
                  min_counts, max_counts, GC_fraction,
                  central_bp=None):
    """
    returns sequences with motif density.
    """
    substring_generator = PwmSamplerFromLoadedMotifs(loaded_motifs, motif_name)
    if central_bp is not None:
        position_generator = InsideCentralBp(central_bp)
    else:
        position_generator = UniformPositionGenerator()
    quantity_generator = UniformIntegerGenerator(min_counts, max_counts)
    embedders = [
        RepeatedEmbedder(
            SubstringEmbedder(
                ReverseComplementWrapper(
                    substring_generator), position_generator),
            quantity_generator)]
    embed_in_background = EmbedInABackground(
        ZeroOrderBackgroundGenerator(
            seq_length, discreteDistribution=get_distribution(GC_fraction)),
        embedders)
    generated_sequences = GenerateSequenceNTimes(
        embed_in_background, num_seqs).generateSequences()
    sequence_arr = np.asarray(
        [generated_seq.seq for generated_seq in generated_sequences])

    return sequence_arr


def simulate_single_motif_detection(motif_name, seq_length,
                                    num_pos, num_neg, GC_fraction):
    """
    Simulates two classes of seqeuences:
        - Positive class sequence with a motif
          embedded anywhere in the sequence
        - Negative class sequence without the motif

    Parameters
    ----------
    motif_name : str
        encode motif name
    seq_length : int
        length of sequence
    num_pos : int
        number of positive class sequences
    num_neg : int
        number of negative class sequences
    GC_fraction : float
        GC fraction in background sequence

    Returns
    -------
    sequence_arr : 1darray
        Array with sequence strings.
    y : 1darray
        Array with positive/negative class labels.
    """
    motif_sequence_arr = simple_motif_embedding(
        motif_name, seq_length, num_pos, GC_fraction)
    random_sequence_arr = simple_motif_embedding(
        None, seq_length, num_neg, GC_fraction)
    sequence_arr = np.concatenate((motif_sequence_arr, random_sequence_arr))
    y = np.array([[True]] * num_pos + [[False]] * num_neg)

    return sequence_arr, y


def simulate_motif_counting(motif_name, seq_length, pos_counts, neg_counts,
                            num_pos, num_neg, GC_fraction):
    """
    Generates data for motif counting task.
    Parameters
    ----------
    motif_name : str
    seq_length : int
    pos_counts : list
        (min_counts, max_counts) for positive set.
    neg_counts : list
        (min_counts, max_counts) for negative set.
    num_pos : int
    num_neg : int
    GC_fraction : float
    Returns
    -------
    sequence_arr : 1darray
        Contains sequence strings.
    y : 1darray
        Contains labels.
    """
    pos_count_sequence_array = motif_density(
        motif_name, seq_length, num_pos,
        pos_counts[0], pos_counts[1], GC_fraction)
    neg_count_sequence_array = motif_density(
        motif_name, seq_length, num_pos,
        neg_counts[0], neg_counts[1], GC_fraction)
    sequence_arr = np.concatenate(
        (pos_count_sequence_array, neg_count_sequence_array))
    y = np.array([[True]] * num_pos + [[False]] * num_neg)

    return sequence_arr, y


def simulate_motif_density_localization(
        motif_name, seq_length, center_size, min_motif_counts,
        max_motif_counts, num_pos, num_neg, GC_fraction):
    """
    Simulates two classes of seqeuences:
        - Positive class sequences with multiple motif instances
          in center of the sequence.
        - Negative class sequences with multiple motif instances
          anywhere in the sequence.
    The number of motif instances is uniformly sampled
    between minimum and maximum motif counts.

    Parameters
    ----------
    motif_name : str
        encode motif name
    seq_length : int
        length of sequence
    center_size : int
        length of central part of the sequence where motifs can be positioned
    min_motif_counts : int
        minimum number of motif instances
    max_motif_counts : int
        maximum number of motif instances
    num_pos : int
        number of positive class sequences
    num_neg : int
        number of negative class sequences
    GC_fraction : float
        GC fraction in background sequence

    Returns
    -------
    sequence_arr : 1darray
        Contains sequence strings.
    y : 1darray
        Contains labels.
    """
    localized_density_sequence_array = motif_density(
        motif_name, seq_length, num_pos,
        min_motif_counts, max_motif_counts, GC_fraction, center_size)
    unlocalized_density_sequence_array = motif_density(
        motif_name, seq_length, num_neg,
        min_motif_counts, max_motif_counts, GC_fraction)
    sequence_arr = np.concatenate(
        (localized_density_sequence_array, unlocalized_density_sequence_array))
    y = np.array([[True]] * num_pos + [[False]] * num_neg)

    return sequence_arr, y


def simulate_multi_motif_embedding(motif_names, seq_length, min_num_motifs,
                                   max_num_motifs, num_seqs, GC_fraction):
    """
    Generates data for multi motif recognition task.
    Parameters
    ----------
    motif_names : list
        List of strings.
    seq_length : int
    min_num_motifs : int
    max_num_motifs : int
    num_seqs : int
    GC_fraction : float
    Returns
    -------
    sequence_arr : 1darray
        Contains sequence strings.
    y : ndarray
        Contains labels for each motif.
    """

    def get_embedder(motif_name):
        substring_generator = PwmSamplerFromLoadedMotifs(
            loaded_motifs, motif_name)
        return SubstringEmbedder(
                ReverseComplementWrapper(substring_generator),
                name=motif_name)

    embedders = [get_embedder(motif_name) for motif_name in motif_names]
    quantity_generator = UniformIntegerGenerator(
        min_num_motifs, max_num_motifs)
    combined_embedder = [RandomSubsetOfEmbedders(
        quantity_generator, embedders)]
    embed_in_background = EmbedInABackground(
        ZeroOrderBackgroundGenerator(
            seq_length, discreteDistribution=get_distribution(GC_fraction)),
        combined_embedder)
    generated_sequences = GenerateSequenceNTimes(
        embed_in_background, num_seqs).generateSequences()
    label_generator = IsInTraceLabelGenerator(np.asarray(motif_names))
    data_arr = np.asarray(
        [[generated_seq.seq] + label_generator.generateLabels(generated_seq)
         for generated_seq in generated_sequences])
    sequence_arr = data_arr[:, 0]
    y = data_arr[:, 1:].astype(bool)

    return sequence_arr, y


def simulate_differential_accessibility(
        pos_motif_names, neg_motif_names, seq_length,
        min_num_motifs, max_num_motifs, num_pos, num_neg, GC_fraction):
    """
    Generates data for differential accessibility task.

    Parameters
    ----------
    pos_motif_names : list
        List of strings.
    neg_motif_names : list
        List of strings.
    seq_length : int
    min_num_motifs : int
    max_num_motifs : int
    num_pos : int
    num_neg : int
    GC_fraction : float

    Returns
    -------
    sequence_arr : 1darray
        Contains sequence strings.
    y : 1darray
        Contains labels.
    """
    pos_motif_sequence_arr, _ = simulate_multi_motif_embedding(
        pos_motif_names, seq_length,
        min_num_motifs, max_num_motifs, num_pos, GC_fraction)
    neg_motif_sequence_arr, _ = simulate_multi_motif_embedding(
        neg_motif_names, seq_length,
        min_num_motifs, max_num_motifs, num_neg, GC_fraction)
    sequence_arr = np.concatenate(
        (pos_motif_sequence_arr, neg_motif_sequence_arr))
    y = np.array([[True]] * num_pos + [[False]] * num_neg)

    return sequence_arr, y


def simulate_heterodimer_grammar(
        motif1, motif2, seq_length,
        min_spacing, max_spacing, num_pos, num_neg, GC_fraction):
    """
    Simulates two classes of sequences with motif1 and motif2:
        - Positive class sequences with motif1 and motif2 positioned
          min_spacing and max_spacing
        - Negative class sequences with independent motif1 and motif2 positioned
        anywhere in the sequence, not as a heterodimer grammar

    Parameters
    ----------
    seq_length : int, length of sequence
    GC_fraction : float, GC fraction in background sequence
    num_pos : int, number of positive class sequences
    num_neg : int, number of negatice class sequences
    motif1 : str, encode motif name
    motif2 : str, encode motif name
    min_spacing : int, minimum inter motif spacing
    max_spacing : int, maximum inter motif spacing

    Returns
    -------
    sequence_arr : 1darray
        Array with sequence strings.
    y : 1darray
        Array with positive/negative class labels.
    """

    motif1_generator = PwmSamplerFromLoadedMotifs(loaded_motifs, motif1)
    motif2_generator = PwmSamplerFromLoadedMotifs(loaded_motifs, motif2)
    separation_generator = UniformIntegerGenerator(min_spacing, max_spacing)
    embedder = EmbeddableEmbedder(PairEmbeddableGenerator(
        motif1_generator, motif2_generator, separation_generator))
    embed_in_background = EmbedInABackground(ZeroOrderBackgroundGenerator(
        seq_length, discreteDistribution=get_distribution(GC_fraction)), [embedder])
    generated_sequences = GenerateSequenceNTimes(
        embed_in_background, num_pos).generateSequences()
    grammar_sequence_arr = np.asarray(
        [generated_seq.seq for generated_seq in generated_sequences])
    nongrammar_sequence_arr, _ = simulate_multi_motif_embedding(
        [motif1, motif2], seq_length, 2, 2, num_neg, GC_fraction)
    sequence_arr = np.concatenate(
        (grammar_sequence_arr, nongrammar_sequence_arr))
    y = np.array([[True]] * num_pos + [[False]] * num_neg)

    return sequence_arr, y
