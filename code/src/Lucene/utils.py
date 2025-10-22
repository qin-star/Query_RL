import numpy as np


# def get_predicted_ranking(predicted_list, target_list, true_scores):
#     target_score_dict = {item: score for item, score in zip(target_list, true_scores)}
#     return [target_score_dict.get(item, 0) for item in predicted_list]


def dcg_at_k(retrieved, target, k, rel_scores=None):
    """
    Compute DCG@k (Discounted Cumulative Gain).
    Default target is an ordered list of relevant documents, from highest to lowest relevance.
    """
    retrieved = retrieved[:k]
    if rel_scores is None:
        gains = np.array(retrieved) == target
    else:
        assert len(target) == len(rel_scores)
        rel_scores_dict = {item: rel_scores[i] for i, item in enumerate(target)}
        gains = np.array([rel_scores_dict.get(doc, 0) for doc in retrieved])
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return np.sum(gains / discounts)

def ndcg_at_k(retrieved, target, k, rel_scores=None):
    """
    Compute NDCG@k.
    """
    dcg = dcg_at_k(retrieved, target, k, rel_scores)
    if isinstance(target, list):
        ideal_dcg = dcg_at_k(target, target, k, rel_scores)
    else:
        ideal_dcg = dcg_at_k([target], target, k, rel_scores)  # Ideal DCG: only the target at top
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def dcg_at_k_rank(rank, k):
    """
    Compute DCG@k (Discounted Cumulative Gain) for a single relevant document at a specific rank.
    
    Args:
        rank (int): The rank of the first occurrence of the answer (1-indexed)
        k (int): Number of documents to consider
        
    Returns:
        float: DCG score
    """
    if rank > k:
        return 0.0
    
    # Create a relevance array with a single 1 at the position of the first answer
    relevance = np.zeros(k)
    if rank <= k:
        relevance[rank-1] = 1.0
    
    # Calculate discounted gains using log2(i+1) for positions i (0-indexed)
    discounts = np.log2(np.arange(2, k + 2))
    return np.sum(relevance / discounts)


def ideal_dcg_at_k(k):
    """
    Compute ideal DCG@k where the answer is at rank 1.
    
    Args:
        k (int): Number of documents to consider
        
    Returns:
        float: Ideal DCG score
    """
    # Ideal case: answer is at rank 1
    return dcg_at_k_rank(1, k)


def ndcg_for_rank(rank, k):
    """
    Compute NDCG@k for a document at a specific rank.
    
    Args:
        rank (int): The rank of the first occurrence of the answer (1-indexed)
        k (int): Number of documents to consider
        
    Returns:
        float: NDCG score between 0 and 1
    """
    dcg = dcg_at_k_rank(rank, k)
    ideal_dcg = ideal_dcg_at_k(k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0