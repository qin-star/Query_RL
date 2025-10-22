def parse_qrel(qrel):
    """
    Parse a TSV file and return a dictionary where:
    - Keys are query IDs.
    - Values are dictionaries containing 'targets' (list of corpus IDs) and 'scores' (list of scores).
    """
    query_dict = {}

    # Skip the header
    for line in qrel:
        query_id, corpus_id, score = line

        if query_id not in query_dict:
            query_dict[query_id] = {"targets": [], "scores": []}

        query_dict[query_id]["targets"].append(corpus_id)
        query_dict[query_id]["scores"].append(int(score))

    return query_dict