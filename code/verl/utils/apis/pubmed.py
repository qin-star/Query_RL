import json
import requests
import traceback
import time

import pandas as pd
import xml.etree.ElementTree as ET

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term="
PUBMED_SUMMARY_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id="
PUBMED_EFETCH_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id="


class PubmedAPI:
    """A wrapper class for the Pubmed API with optimized performance.
    """
    def __init__(self, retry=1, api_key=None, request_delay=0.2):
        self.retry = retry
        self.api_key = api_key
        self.request_delay = request_delay  # Delay between requests to avoid rate limiting

    def search(self, query, topk=-1):
        """Get the response of pmids."""
        search_results = self.search_with_query(query, topk=topk)
        return search_results

    def search_with_query(self, query, topk=-1):
        """Search with query input to get the response of pmids. Optimized for speed."""
        err_msg = ""
        if self.api_key:
            query += f"&api_key={self.api_key}"
        if '&retmode=json' not in query:
            query += '&retmode=json'

        # Optimize retmax based on topk - no need to fetch large batches if topk is small
        # For unlimited retrieval or large topk values, use larger batch sizes
        retstart = 0
        
        if topk > 0 and topk <= 1000:
            retmax = topk  # If we need few results, just fetch them in one go
        else:
            retmax = 5000  # Higher value for better throughput, up to API limits
        
        original_query = query  # Save the original query without retmax/retstart parameters
        pmid_list = []

        for i in range(self.retry + 1):
            if i > 0:
                print(f"Retry {i} times")
                time.sleep(self.request_delay * 2)  # Longer delay on retries
            try:
                # First get the count of results
                count_query = original_query + f'&retmax=0'
                response = requests.get(count_query)
                
                if response.status_code != 200:
                    raise ConnectionError(f"Pubmed connection error occurred - {response.text}")
                
                response_dict = json.loads(response.text)
                total_count = int(response_dict['esearchresult']['count'])
                
                # If topk specified, limit to that number
                count_to_fetch = min(total_count, topk) if topk > 0 else total_count
                
                # No need to continue if no results or topk=0
                if count_to_fetch <= 0:
                    return []
                
                # Calculate how many batches we need
                batch_limit = min(10000, count_to_fetch)  # PubMed has a hard limit around 10,000
                
                # Fetch results in batches
                while retstart < batch_limit and len(pmid_list) < count_to_fetch:
                    # Adjust retmax for final batch if needed
                    current_retmax = min(retmax, count_to_fetch - len(pmid_list))
                    current_query = original_query + f'&retmax={current_retmax}&retstart={retstart}'
                    
                    response = requests.get(current_query)
                    time.sleep(self.request_delay)  # Small delay to avoid rate limiting
                    
                    if response.status_code != 200:
                        raise ConnectionError(f"Pubmed connection error occurred - {response.text}")
                    
                    response_dict = json.loads(response.text)
                    batch_results = response_dict['esearchresult']['idlist']
                    
                    # If no results returned in this batch, break the loop
                    if not batch_results:
                        break
                    
                    pmid_list.extend(batch_results)
                    retstart += current_retmax
                    
                    # If we have enough results, stop fetching
                    if topk > 0 and len(pmid_list) >= topk:
                        pmid_list = pmid_list[:topk]
                        break
                
                # We only need to remove duplicates when fetching unlimited results
                if topk <= 0:
                    pmid_list = list(dict.fromkeys(pmid_list))  # Remove duplicates while preserving order
                
                break
            except Exception as e:
                err_msg = traceback.format_exc()
                print(f"Error in PubMed API: {e}")
                print(err_msg)

        if err_msg != "":
            raise RuntimeError("A Pubmed API error occurred")

        return pmid_list
    
    def search_with_keywords(self, keywords: str, topk=-1):
        """Search with keywords input to get the response of pmids."""
        query = PUBMED_BASE_URL + keywords
        search_results = self.search_with_query(query, topk=topk)
        return search_results
    
    def get_papers_by_pmids(self, pmid_list):
        """Search pmids to get the summary of paper."""
        if not pmid_list:
            return pd.DataFrame()

        err_msg = ""
        
        for i in range(self.retry + 1):
            if i > 0:
                print(f"Retry {i} times")
                time.sleep(self.request_delay * 2)  # Longer delay on retries
            try:
                # Process in batches to avoid URL length limitations
                batch_size = 200  # Keep batch size reasonable to avoid HTTP 414 errors
                all_papers = []
                
                for i in range(0, len(pmid_list), batch_size):
                    batch_pmids = pmid_list[i:i+batch_size]
                    
                    # Build the query to get the summary of the articles
                    pmid_list_str = ','.join(batch_pmids)
                    summary_query = PUBMED_SUMMARY_BASE_URL + pmid_list_str + "&retmode=json"
                    if self.api_key:
                        summary_query += f"&api_key={self.api_key}"
                    
                    response = requests.get(summary_query)
                    time.sleep(self.request_delay)  # Small delay to avoid rate limiting

                    if response.status_code != 200:
                        if response.status_code == 414:
                            raise ConnectionError(f"Pubmed query too long!")
                        else:
                            raise ConnectionError(f"Pubmed connection error occurred: {response.text}")

                    response = json.loads(response.text)
                    results = response.get("result", {})
                    uids = results.get("uids", [])

                    if len(uids) == 0:
                        continue
                    
                    # Build dataframe for this batch
                    batch_papers = []
                    for idx in range(len(uids)):
                        try:
                            cur_res = results[uids[idx]]
                            parse_res = self._parse_json_summary_response(cur_res)
                            batch_papers.append(parse_res)
                        except Exception as e:
                            print(f"Error parsing paper {uids[idx]}: {e}")
                    
                    if batch_papers:
                        batch_df = pd.concat(batch_papers, axis=0).reset_index(drop=True)
                        
                        # Retrieve abstract and mesh terms for this batch
                        abstracts, mesh_terms = self._retrieve_abstract_and_mesh_term_by_efetch(uids)
                        batch_df['Abstract'] = abstracts
                        batch_df['Mesh Term'] = mesh_terms
                        
                        all_papers.append(batch_df)
                
                # Combine all batches
                if all_papers:
                    papers = pd.concat(all_papers, axis=0).reset_index(drop=True)
                else:
                    papers = pd.DataFrame()
                
                break
            except Exception as e:
                err_msg = traceback.format_exc()
                print(f"Error in get_papers_by_pmids: {e}")
                print(err_msg)
        
        if err_msg != "":
            raise RuntimeError("A Pubmed API error occurred")

        return papers
    
    def _parse_json_summary_response(self, res):
        """Parse the summary json response to pd dataframe from the API call."""
        pmid = res.get("uid", None)
        pub_date = res.get("pubdate", None)
        journal = res.get("fulljournalname", None)
        if journal is None:
            journal = res.get("source", None)
        title = res.get("title", None)
        authors = res.get("authors", [])
        authors = [author.get("name", None) for author in authors]
        authors = "; ".join(authors)
        volume = res.get("volume", None)
        issue = res.get("issue", None)
        pages = res.get("pages", None)
        pubtypes = res.get("pubtype", [])
        url = 'https://pubmed.ncbi.nlm.nih.gov/' + pmid

        df = pd.DataFrame({
            "PMID": [pmid],
            "Publication Date": [pub_date],
            "Title": [title],
            "Authors": [authors],
            "Journal": [journal],
            "Volume": [volume],
            "Issue": [issue],
            "Pages": [pages],
            "Pubtypes": [pubtypes],
            "URL": [url]
        })
        return df
    
    def _retrieve_abstract_and_mesh_term_by_efetch(self, pmid_list):
        '''Retrieve abstract based on pmids from efetch API'''
        # Process in batches to avoid URL length limitations
        batch_size = 200
        all_abstracts = {}
        all_mesh_terms = {}
        
        for i in range(0, len(pmid_list), batch_size):
            batch_pmids = pmid_list[i:i+batch_size]
            pmid_list_str = ','.join(batch_pmids)
            
            query = PUBMED_EFETCH_BASE_URL + pmid_list_str + "&retmode=xml"
            if self.api_key:
                query += "&api_key=" + self.api_key

            response = requests.get(query)
            time.sleep(self.request_delay)  # Small delay to avoid rate limiting
            
            if response.status_code != 200:
                print(f"Error fetching abstracts for batch {i}-{i+batch_size}: {response.status_code}")
                continue
            
            try:
                response_text = response.text
                tree = ET.ElementTree(ET.fromstring(response_text))
                articles = tree.findall(".//PubmedArticle")
                
                for article in articles:
                    abstract = article.find(".//AbstractText")
                    if abstract is not None:
                        abstract_text = abstract.text
                    else:
                        abstract_text = ""

                    # Get pmid
                    article_ids = [a for a in article.findall(".//ArticleId") if a.get("IdType").lower() == "pubmed"]
                    if len(article_ids) > 0:
                        pmid = article_ids[0].text
                        all_abstracts[pmid] = abstract_text
                    else:
                        continue
                
                    # Get MeSH terms and Qualifiers
                    mesh_terms = []
                    mesh_headings = article.findall(".//MeshHeading")
                    for mesh_heading in mesh_headings:
                        descriptor = mesh_heading.find("DescriptorName")
                        if descriptor is not None:
                            qualifiers = mesh_heading.findall("QualifierName")
                            if qualifiers:
                                for qualifier in qualifiers:
                                    mesh_terms.append(f"{descriptor.text} ({qualifier.text})")
                            else:
                                mesh_terms.append(descriptor.text)

                    all_mesh_terms[pmid] = mesh_terms
            except Exception as e:
                print(f"Error parsing XML for batch {i}-{i+batch_size}: {e}")
        
        # Make sure the input and output are in the same order and length
        output_abstracts = []
        output_mesh_terms = []
        for pmid in pmid_list:
            output_abstracts.append(all_abstracts.get(pmid, ""))
            output_mesh_terms.append(all_mesh_terms.get(pmid, []))

        return output_abstracts, output_mesh_terms