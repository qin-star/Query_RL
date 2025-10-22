import json
import requests
import traceback
import time
import pandas as pd

CTGOV_BASE_URL = "https://clinicaltrials.gov/api/v2/studies?query.term="
CTGOV_NCTID_BASE_URL = "https://clinicaltrials.gov/api/v2/studies?&query.id="


class CTGovAPI:
    """A wrapper class for the CT.gov API with optimized performance.
    """
    def __init__(self, retry=1, request_delay=0.2):
        self.retry = retry
        self.request_delay = request_delay  # Delay between requests to avoid rate limiting

    def search(self, query, topk=-1):
        """Get the response of nctids."""
        search_results = self.search_with_query(query, topk=topk)
        return search_results

    def search_with_query(self, query, topk=-1):
        """Search with query input to get the response of nctids. Optimized for speed."""
        err_msg = ""
        nctid_list = []

        # Ensure we have a reasonable page size
        if "pageSize=" not in query:
            # Adjust page size based on topk - small pageSize for small topk to improve performance
            if topk > 0 and topk <= 100:
                query += f"&pageSize={topk}"  # Just fetch what we need
            else:
                query += "&pageSize=1000"  # Max allowed by API
        
        for attempt in range(self.retry + 1):
            if attempt > 0:
                print(f"Retry {attempt} times")
                time.sleep(self.request_delay * 2)  # Longer delay on retries
            
            try:
                # Use iteration instead of recursion for better control and less overhead
                next_page_token = None
                page_num = 1
                
                while True:
                    # Construct query for current page
                    current_query = query
                    if next_page_token:
                        current_query += f"&pageToken={next_page_token}"
                    
                    # Execute request
                    response = requests.get(current_query)
                    time.sleep(self.request_delay)  # Small delay to avoid rate limiting
                    
                    if response.status_code != 200:
                        raise ConnectionError(f"CTGov connection error occurred - {response.status_code}: {response.text}")
                    
                    # Parse response
                    results = json.loads(response.text)
                    total_count = results.get("totalCount", 0)
                    study_results = results.get("studies", [])
                    next_page_token = results.get("nextPageToken", "")
                    
                    # Process current page of results
                    new_nctids = []
                    for study in study_results:
                        protocol_section = study.get("protocolSection", {})
                        identification = protocol_section.get("identificationModule", {})
                        nct_id = identification.get("nctId")
                        if nct_id:
                            new_nctids.append(nct_id)
                    
                    # Add to our results list
                    nctid_list.extend(new_nctids)
                    
                    # Check if we have enough results or need to continue
                    if topk > 0 and len(nctid_list) >= topk:
                        nctid_list = nctid_list[:topk]  # Trim to exact count
                        break
                    
                    # Check if we have more pages
                    if not next_page_token:
                        break
                    
                    # If we didn't get any new results, stop to prevent infinite loop
                    if not new_nctids:
                        break
                    
                    page_num += 1
                
                # Success - break retry loop
                break
                
            except Exception as e:
                err_msg = traceback.format_exc()
                print(f"Error in CTGov API search: {e}")
                print(err_msg)

        if err_msg != "" and not nctid_list:
            raise RuntimeError("A CTGOV API error occurred")

        return nctid_list

    def search_with_keywords(self, keywords: str, topk=-1):
        """Search with keywords input to get the response of nctids."""
        query = CTGOV_BASE_URL + keywords
        search_results = self.search_with_query(query, topk=topk)
        return search_results
    
    def get_trials_by_nctids(self, nctid_list):
        """Search nctids to get the summary of trials."""
        if not nctid_list:
            return pd.DataFrame()
            
        err_msg = ""
        trials = None

        for attempt in range(self.retry + 1):
            if attempt > 0:
                print(f"Retry {attempt} times")
                time.sleep(self.request_delay * 2)  # Longer delay on retries
                
            try:
                # Process in batches to avoid URL length limitations
                batch_size = 100  # Keep batch size reasonable
                all_trials = []
                
                for i in range(0, len(nctid_list), batch_size):
                    batch_nctids = nctid_list[i:i+batch_size]
                    
                    # Use OR with parentheses to optimize query
                    nctid_query_str = " OR ".join(batch_nctids)
                    query = CTGOV_NCTID_BASE_URL + f"({nctid_query_str})&pageSize=1000"
                    
                    # Iteratively fetch pages if needed
                    next_page_token = None
                    batch_trials = []
                    
                    while True:
                        current_query = query
                        if next_page_token:
                            current_query += f"&pageToken={next_page_token}"
                        
                        response = requests.get(current_query)
                        time.sleep(self.request_delay)  # Small delay to avoid rate limiting
                        
                        if response.status_code != 200:
                            raise ConnectionError(f"CTGOV connection error occurred - {response.status_code}: {response.text}")
                        
                        results = json.loads(response.text)
                        study_results = results.get("studies", [])
                        next_page_token = results.get("nextPageToken", "")
                        
                        # Process results in bulk instead of individual parsing
                        if study_results:
                            data = []
                            for res in study_results:
                                protocol_section = res.get("protocolSection", {})
                                identification = protocol_section.get("identificationModule", {})
                                description = protocol_section.get("descriptionModule", {})
                                
                                data.append({
                                    'NCT Number': identification.get('nctId'),
                                    'Study Title': identification.get('briefTitle'),
                                    'Brief Summary': description.get('briefSummary')
                                })
                            
                            # Create a single DataFrame for this page
                            if data:
                                batch_trials.append(pd.DataFrame(data))
                        
                        # Check if we have more pages
                        if not next_page_token:
                            break
                    
                    # Combine all pages for this batch
                    if batch_trials:
                        all_trials.extend(batch_trials)
                
                # Combine all batches
                if all_trials:
                    trials = pd.concat(all_trials, axis=0).reset_index(drop=True)
                else:
                    trials = pd.DataFrame()
                
                break
                
            except Exception as e:
                err_msg = traceback.format_exc()
                print(f"Error in get_trials_by_nctids: {e}")
                print(err_msg)

        if err_msg != "" and trials is None:
            raise RuntimeError("A CTGOV API error occurred")

        return trials