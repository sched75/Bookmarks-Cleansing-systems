import json
import requests
from bs4 import BeautifulSoup, Tag
import time
import argparse
from urllib.parse import urlparse
from typing import List, Dict, Optional, Tuple
import openai
import concurrent.futures
import threading
from tqdm import tqdm
import logging
import datetime
import os
import signal
import sys
from html import escape # Import escape for HTML safety

# Suppress only the InsecureRequestWarning from urllib3 needed for verify=False fallback
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure a logger for the application
logger = logging.getLogger("BookmarkSummarizer")
logger.propagate = False

# Flag to check if lxml is available, determined in main()
_lxml_available = False

class BookmarkSummarizer:
    def __init__(self, deepseek_api_key: str, input_file: str, output_file: str, max_workers: int = 5, fetch_timeout: float = 10.0):
        self.input_file = input_file
        self.output_file = output_file # Path for JSON output
        self.max_workers = max_workers if max_workers > 0 else 1
        self.fetch_timeout = fetch_timeout

        self.client = openai.OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com/v1",
            timeout=60 # API call timeout (seconds) - Can be adjusted
        )

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        self.failed_urls = []
        self._failed_urls_lock = threading.Lock()

    def validate_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def load_bookmarks(self) -> List[Dict]:
        """Load bookmarks from JSON file with validation"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                bookmarks_data = json.load(f)

            if not isinstance(bookmarks_data, list):
                raise ValueError("JSON root should be an array of bookmarks")

            valid_bookmarks = []
            for i, bookmark in enumerate(bookmarks_data):
                if not isinstance(bookmark, dict):
                    logger.warning(f"Skipping item at index {i} - not a dictionary.")
                    continue
                if 'uri' not in bookmark:
                    logger.warning(f"Skipping item at index {i} - missing 'uri'. Title: {bookmark.get('title', 'N/A')}")
                    continue
                if 'title' not in bookmark:
                     bookmark['title'] = "No Title"
                if not self.validate_url(bookmark['uri']):
                    logger.warning(f"Skipping item at index {i} - invalid URL: {bookmark['uri']}")
                    continue
                valid_bookmarks.append(bookmark)

            return valid_bookmarks
        except FileNotFoundError:
            logger.critical(f"Input file not found: {self.input_file}")
            raise Exception(f"Input file not found: {self.input_file}")
        except json.JSONDecodeError:
            logger.critical(f"Invalid JSON in file: {self.input_file}")
            raise Exception(f"Invalid JSON in file: {self.input_file}")
        except Exception as e:
            logger.critical(f"Error loading bookmarks: {str(e)}")
            raise Exception(f"Error loading bookmarks: {str(e)}")

    def save_json_bookmarks(self, bookmarks: List[Dict]) -> None:
        """Save bookmarks to JSON file with error handling"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(bookmarks, f, indent=2, ensure_ascii=False)
            # print(f"JSON output saved to: {self.output_file}")
        except IOError as e:
            logger.error(f"Failed to write JSON output file {self.output_file}: {str(e)}")
            raise Exception(f"Failed to write JSON output file: {str(e)}")
        except Exception as e:
            logger.error(f"Error saving JSON bookmarks to {self.output_file}: {str(e)}")
            raise Exception(f"Error saving JSON bookmarks: {str(e)}")

    def fetch_webpage(self, url: str) -> Optional[str]:
        """Fetch webpage content with robust error handling"""
        try:
            response = self.session.get(
                url,
                timeout=self.fetch_timeout,
                allow_redirects=True,
                verify=True
            )
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type and not any(ct in content_type for ct in ['text/plain', 'application/json', 'application/xml']):
               logger.warning(f"Unsupported content type '{content_type}' for {url}. Attempting parse anyway.")

            return response.text

        except requests.exceptions.SSLError:
            logger.warning(f"SSL Error for {url}. Retrying without verification.")
            try:
                response = self.session.get(url, timeout=self.fetch_timeout, verify=False)
                response.raise_for_status()
                return response.text
            except Exception as e_fallback:
                logger.error(f"Fallback fetch (verify=False) failed for {url}: {type(e_fallback).__name__} - {str(e_fallback)}")
                return None
        except requests.exceptions.Timeout:
             logger.error(f"Fetch timeout after {self.fetch_timeout} seconds for {url}.")
             return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {type(e).__name__} - {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during fetch for {url}: {type(e).__name__} - {str(e)}")
            return None

    def extract_main_content(self, html: str, url: str) -> Optional[str]:
        """Extract main content from HTML with improved parsing"""
        if not html:
            return None

        parser = 'lxml' if _lxml_available else 'html.parser'
        soup = None

        try:
            soup = BeautifulSoup(html, parser)
        except Exception as e:
            if parser == 'lxml':
                logger.warning(f"lxml parsing failed for {url}: {type(e).__name__} - {str(e)}. Falling back to html.parser for this URL.")
                try:
                    soup = BeautifulSoup(html, 'html.parser')
                except Exception as e_fallback:
                    logger.error(f"HTML parsing also failed for {url}: {type(e_fallback).__name__} - {str(e_fallback)}")
                    return None
            else:
                 logger.error(f"HTML parsing failed for {url}: {type(e).__name__} - {str(e)}")
                 return None

        if soup:
            for element in soup(['script', 'style', 'nav', 'footer', 'aside',
                               'iframe', 'noscript', 'svg', 'img', 'video', 'audio',
                               'button', 'form', 'comment', 'header', 'link', 'meta']):
                element.decompose()

            selectors = [
                'main', 'article', '.entry-content', '.post-content',
                '.td-post-content', '.content', '#content', '#main',
                '.main-content', '.post', '.entry', 'div[role="main"]'
            ]

            main_content_element = None
            for selector in selectors:
                main_content_element = soup.select_one(selector)
                if main_content_element:
                    break

            if main_content_element:
                text = main_content_element.get_text(separator=' ', strip=True)
            else:
                body = soup.find('body')
                if body:
                    text = body.get_text(separator=' ', strip=True)
                else:
                    text = soup.get_text(separator=' ', strip=True)
                if text.strip():
                     logger.warning(f"Could not find specific content container for {url}, extracted from body/full page.")
                else:
                     logger.warning(f"Could not find specific content container and extracted text is empty for {url}.")


            text = ' '.join(text.split())
            return text[:15000]

        return None

    def generate_summary(self, text: Optional[str], title: str, url: str) -> str:
        """Generate summary using DeepSeek API"""
        if not text:
            logger.warning(f"No content extracted for {url}. Summarizing based on title only.")
            prompt = f"""The webpage content could not be retrieved or parsed.
Please provide a *very brief* (1-2 sentence) educated guess about the page's purpose based *only* on its title and URL. State that the summary is based on metadata.

Page Title: {title}
URL: {url}

Respond with just the summary guess. Start with "Based on the title/URL...". """
            max_tokens_val = 100
        else:
            prompt = f"""Please provide a concise 2-3 sentence summary of the following webpage.
Focus on identifying the main purpose, content, and any key features or services offered.

Page Title: {title}
URL: {url}

Relevant Content Excerpt (up to 10k chars):
{text[:10000]}

Please respond with just the summary, no additional commentary or labels. The summary should be complete sentences that could stand alone."""
            max_tokens_val = 200

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an expert assistant skilled at summarizing web pages concisely."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens_val,
                temperature=0.6,
                # API timeout is set in the client init
            )
            summary = response.choices[0].message.content.strip()
            if summary.lower().startswith("sure, here") or summary.lower().startswith("okay, here"):
                 summary = summary.split('\n', 1)[-1].strip()
            return summary if summary else "Summary generation failed (empty response)."

        except openai.RateLimitError:
            logger.warning(f"Rate limit hit for {url}. Marking as failed.")
            return "Summary unavailable (Rate Limit)"
        except openai.APITimeoutError:
            logger.error(f"API Timeout for {url}. Marking as failed.")
            return "Summary unavailable (API Timeout)"
        except Exception as e:
            logger.error(f"Error generating summary for {url}: {type(e).__name__} - {str(e)}")
            return f"Summary unavailable (API Error: {type(e).__name__})"

    def generate_category(self, summary: str, title: str, url: str) -> str:
        """Generate category path using DeepSeek API based on summary/title/url."""
        if not summary or "Summary unavailable" in summary or "Failed to fetch" in summary:
            # Cannot categorize if summary failed
            logger.warning(f"Cannot categorize URL {url}: Summary unavailable. Assigning 'Uncategorized'.")
            return "Uncategorized"

        # Create a prompt for categorization
        prompt = f"""Categorize the following bookmark based on its summary, title, and URL.
Provide a hierarchical category path using a forward slash (/) as a separator.
For example: "Technology/AI", "News/Politics", "Programming/Python/LibraryName".
Keep category names concise (e.g., "AI" instead of "Artificial Intelligence").
If a bookmark fits multiple categories, pick the most relevant one.
If categorization is unclear or the content is generic (like a homepage with many links), provide a general category or 'Uncategorized'.
Respond with *only* the category path string.

Page Title: {title}
URL: {url}
Summary: {summary}

Category Path:"""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat", # Or a more advanced model if needed
                messages=[
                    {"role": "system", "content": "You are an expert at classifying web pages into concise hierarchical categories."},
                    {"role_profile": "user", "content": prompt}
                ],
                max_tokens=50, # Category path should be short
                temperature=0.2 # Lower temperature for more consistent output
                # API timeout set in client init
            )
            category_path = response.choices[0].message.content.strip()

            # Basic validation: Remove leading/trailing slashes, replace multiple slashes
            category_path = category_path.strip('/').replace('//', '/')
            # Replace potentially invalid characters with underscores (depends on target system, / is common)
            # Simple approach: only allow alphanumeric, spaces, hyphens, slashes, underscores
            # category_path = ''.join(c if c.isalnum() or c in ' /_-' else '_' for c in category_path)
            # Replace spaces with underscores for file system compatibility if needed, but keep spaces for readability
            # category_path = category_path.replace(' ', '_')

            if not category_path:
                 logger.warning(f"Category API returned empty string for {url}. Assigning 'Uncategorized'.")
                 return "Uncategorized"

            return category_path

        except openai.RateLimitError:
            logger.warning(f"Category API Rate limit hit for {url}. Assigning 'Uncategorized'.")
            return "Uncategorized (Rate Limit)"
        except openai.APITimeoutError:
            logger.error(f"Category API Timeout for {url}. Assigning 'Uncategorized'.")
            return "Uncategorized (API Timeout)"
        except Exception as e:
            logger.error(f"Error generating category for {url}: {type(e).__name__} - {str(e)}. Assigning 'Uncategorized'.")
            return f"Uncategorized (API Error: {type(e).__name__})"


    def _process_single_bookmark(self, bookmark: Dict) -> Dict:
        """Processes a single bookmark: fetch, extract, summarize, categorize."""
        url = bookmark['uri']
        title = bookmark.get('title', 'No Title')

        # Initialize placeholder summary and category
        bookmark['summary'] = "Error during processing"
        bookmark['category_path'] = "Uncategorized (Processing Error)" # Initialize category placeholder

        try:
            # Step 1: Fetch Webpage
            html = self.fetch_webpage(url)
            if html is None:
                with self._failed_urls_lock:
                    if url not in self.failed_urls:
                        self.failed_urls.append(url)
                bookmark['summary'] = "Failed to fetch or parse webpage content"
                # category_path remains initialized value
                return bookmark # Return early on fetch failure

            # Step 2: Extract Content
            content = self.extract_main_content(html, url)
            # content can be None

            # Step 3: Generate Summary
            summary = self.generate_summary(content, title, url)
            bookmark['summary'] = summary # Update summary

            # Step 4: Generate Category
            # Use the generated summary, title, and URL for categorization
            category_path = self.generate_category(summary, title, url)
            bookmark['category_path'] = category_path # Update category

            # Check if summary or category indicates failure and mark URL in failed_urls if so
            if "Failed" in summary or "unavailable" in summary or "Error" in summary or \
               "Uncategorized" in category_path: # Also consider uncategorized API errors as failures
                 with self._failed_urls_lock:
                    if url not in self.failed_urls:
                        self.failed_urls.append(url)

        except Exception as e:
            # This catches exceptions not explicitly handled in fetch/extract/summarize/categorize
            if isinstance(e, KeyboardInterrupt):
                 logger.warning(f"Processing interrupted by user for URL {url}")
                 bookmark['summary'] = "Processing interrupted by user"
                 bookmark['category_path'] = "Uncategorized (Interrupted)"
            else:
                 logger.exception(f"Unexpected exception processing URL {url}")
                 bookmark['summary'] = f"Unexpected Error: {type(e).__name__}"
                 bookmark['category_path'] = f"Uncategorized (Error: {type(e).__name__})"

            with self._failed_urls_lock:
                 if url not in self.failed_urls:
                    self.failed_urls.append(url)

        return bookmark


    def process_bookmarks(self) -> List[Dict]:
        """
        Main processing method using ThreadPoolExecutor for concurrency.
        Returns the list of all bookmarks (skipped + processed),
        now including 'summary' and 'category_path'.
        """
        try:
            all_bookmarks = self.load_bookmarks()
        except Exception: # load_bookmarks already logged the critical error
            return [] # Return empty list if loading fails

        # Determine which bookmarks need processing (don't have summary or category_path)
        # Or you might want to re-process if category_path is a failure message?
        # Let's re-process if summary or category_path is missing.
        bookmarks_to_process = [b for b in all_bookmarks if 'summary' not in b or 'category_path' not in b]
        bookmarks_skipped = [b for b in all_bookmarks if 'summary' in b and 'category_path' in b]

        total_to_process = len(bookmarks_to_process)
        if total_to_process == 0:
            print("No new bookmarks to process (all have summary and category).")
            # If output file is different, save the original/skipped ones
            if self.output_file != self.input_file:
                 print(f"Saving original/skipped bookmarks ({len(all_bookmarks)} items) to {self.output_file}")
                 try:
                     self.save_json_bookmarks(all_bookmarks)
                 except Exception: # save_bookmarks already logged the error
                     pass
            return all_bookmarks # Return original list if nothing to process

        print(f"Starting processing for {total_to_process} bookmarks using up to {self.max_workers} workers...")
        print(f"({len(bookmarks_skipped)} bookmarks already have summary and category and will be skipped)")
        print(f"Webpage fetch timeout set to {self.fetch_timeout} seconds.")
        print("Press Ctrl+C to initiate graceful shutdown.")


        executor = None
        future_to_bookmark = {}
        final_processed_results = []
        processing_interrupted = False

        try:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
            future_to_bookmark = {executor.submit(self._process_single_bookmark, bookmark): bookmark for bookmark in bookmarks_to_process}

            # Iterate through completed futures (for progress bar and potential intermediate results)
            for future in tqdm(concurrent.futures.as_completed(future_to_bookmark), total=total_to_process, desc="Processing"):
                 pass # Results collected in finally

        except KeyboardInterrupt:
            processing_interrupted = True
            print("\nShutdown requested. Cancelling remaining tasks...")
            for future in future_to_bookmark:
                future.cancel()
            if executor:
                 executor.shutdown(wait=False) # Don't wait, go to finally

        except Exception as e:
            logger.critical(f"Fatal error during processing loop: {str(e)}")
            if executor:
                executor.shutdown(wait=True)
            raise

        finally:
            # Collect results from all futures
            for future, original_bookmark in future_to_bookmark.items():
                if future.done():
                    try:
                        if not future.cancelled():
                           result = future.result() # Get the result (updated bookmark dict)
                           final_processed_results.append(result)
                        else:
                           url = original_bookmark.get('uri', 'unknown URL')
                           logger.warning(f"Task for URL {url} was cancelled.")
                           # _process_single_bookmark should have set a summary/category,
                           # but ensure 'Processing cancelled' is set if not.
                           if original_bookmark.get('summary') == "Error during processing":
                                original_bookmark['summary'] = "Processing cancelled"
                           if original_bookmark.get('category_path') == "Uncategorized (Processing Error)":
                                original_bookmark['category_path'] = "Uncategorized (Cancelled)"
                           final_processed_results.append(original_bookmark) # Add the marked original bookmark
                    except concurrent.futures.CancelledError:
                        pass # Handled by the 'if not future.cancelled()' branch
                    except Exception as exc:
                        # Any other error getting result from a *done* task (less common)
                        url = original_bookmark.get('uri', 'unknown URL')
                        logger.error(f"Error retrieving final result for {url}: {type(exc).__name__} - {str(exc)}")
                        original_bookmark['summary'] = f"Error retrieving final result: {type(exc).__name__}"
                        original_bookmark['category_path'] = f"Uncategorized (Error: {type(exc).__name__})"
                        final_processed_results.append(original_bookmark)
                else:
                    url = original_bookmark.get('uri', 'unknown URL')
                    # Task was not done (was pending or running when interrupt occurred)
                    original_bookmark['summary'] = "Processing interrupted"
                    original_bookmark['category_path'] = "Uncategorized (Interrupted)"
                    logger.warning(f"Task for URL {url} was left pending/interrupted.")
                    final_processed_results.append(original_bookmark)


            # Combine skipped bookmarks with the collected results (final_processed_results)
            # Rebuild the final list ensuring the original order from all_bookmarks
            processed_map_by_id = {id(b): b for b in final_processed_results}
            final_bookmarks_list = []
            for original_b in all_bookmarks:
                 if id(original_b) in processed_map_by_id:
                     final_bookmarks_list.append(processed_map_by_id[id(original_b)])
                 else:
                     # This should be the skipped bookmarks
                     final_bookmarks_list.append(original_b)


            # --- Final Summary ---
            print("\nProcessing finished (possibly interrupted).")
            total_attempted = len(bookmarks_to_process)
            attempted_in_final = [b for b in final_bookmarks_list if id(b) in {id(bm) for bm in bookmarks_to_process}]

            # Count based on summary and category_path status
            successfully_processed_count = sum(1 for b in attempted_in_final
                                               if 'summary' in b and not any(msg in b['summary'] for msg in ["Failed", "unavailable", "Error", "Unexpected Error", "cancelled", "interrupted", "unfinished"])
                                               and 'category_path' in b and not any(msg in b['category_path'] for msg in ["Uncategorized", "Error", "cancelled", "interrupted"])
                                               ) # More strict definition of success

            failed_count = sum(1 for b in attempted_in_final
                               if 'summary' in b and any(msg in b['summary'] for msg in ["Failed", "unavailable", "Error", "Unexpected Error"]) # Fetch/Extract/Summary failed
                               or ('category_path' in b and any(msg in b['category_path'] for msg in ["Uncategorized (Rate Limit)", "Uncategorized (API Timeout)", "Uncategorized (Error"])) # Category API failed
                              )

            cancelled_or_interrupted_count = sum(1 for b in attempted_in_final
                                                if 'summary' in b and any(msg in b['summary'] for msg in ["Processing cancelled", "Processing interrupted"])
                                                or ('category_path' in b and any(msg in b['category_path'] for msg in ["Uncategorized (Cancelled)", "Uncategorized (Interrupted)"]))
                                               )

            # Bookmarks that were attempted but didn't fall into success, failed, or cancelled/interrupted
            # This might catch bookmarks where only category failed but summary succeeded, etc.
            # Let's count them based on the final list state more directly.
            total_in_final = len(final_bookmarks_list)
            total_skipped = len(bookmarks_skipped)
            total_attempted = total_in_final - total_skipped # Should match len(bookmarks_to_process)

            successful = 0
            failed = 0 # Includes fetch, extract, summary, category API errors
            interrupted = 0 # Includes cancelled and interrupted

            for b in attempted_in_final:
                 summary_status = b.get('summary', '')
                 category_status = b.get('category_path', '')

                 if "interrupted" in summary_status or "cancelled" in summary_status or \
                    "Interrupted" in category_status or "Cancelled" in category_status:
                     interrupted += 1
                 elif "Failed" in summary_status or "unavailable" in summary_status or "Error" in summary_status or \
                      "Uncategorized (Rate Limit)" in category_status or "Uncategorized (API Timeout)" in category_status or "Uncategorized (Error" in category_status:
                     failed += 1
                 elif 'summary' in b and 'category_path' in b and "Uncategorized" not in category_status:
                     # A successful run means summary is not a failure message, and category is not an error/default
                     successful += 1
                 else:
                     # Bookmarks where summary is okay but category is 'Uncategorized' (not an error)
                     # Treat these as partially successful or needing review, count them with failed for simplicity in this summary.
                     # Or we could create a new category like "Needs Manual Categorization".
                     failed += 1 # Count these as failed categorization for this summary


            print(f"Total bookmarks in input: {len(all_bookmarks)}")
            print(f"Skipped (already had summary/category): {total_skipped}")
            print(f"Attempted processing: {total_attempted}")
            print(f"  Successfully categorized & summarized: {successful}")
            print(f"  Failed processing (API, Fetch, Parse, etc.): {failed}")
            print(f"  Cancelled or Interrupted: {interrupted}")


            unique_failed_urls = sorted(list(set(self.failed_urls)))
            if unique_failed_urls:
                 print(f"\nFailed URLs ({len(unique_failed_urls)}): (See log file for details)")
                 for i, url in enumerate(unique_failed_urls[:10]):
                     print(f"- {url}")
                 if len(unique_failed_urls) > 10:
                     print(f"... {len(unique_failed_urls) - 10} more failed URLs.")
            else:
                 print("\nNo explicitly failed URLs recorded.")


            try:
                 self.save_json_bookmarks(final_bookmarks_list)
                 print(f"\nUpdated JSON bookmarks saved to: {self.output_file}")
            except Exception:
                 pass # Error logged inside save_json_bookmarks

            # Ensure executor is properly shut down
            if executor and not getattr(executor, '_shutdown', False):
                 logger.debug("Executor not already shut down, calling shutdown(wait=True)")
                 executor.shutdown(wait=True)

            # Return the final list of all bookmarks for subsequent HTML saving
            return final_bookmarks_list


# --- Functions outside the class for building tree and generating HTML ---

def build_bookmark_tree(bookmarks: List[Dict], uncategorized_folder_name: str = "Uncategorized") -> Dict:
    """
    Builds a nested dictionary tree structure from a list of bookmarks
    based on their 'category_path'.
    """
    # Root of the tree will be a dictionary. Values are either other dictionaries (folders)
    # or lists of bookmark dictionaries (bookmarks in a folder).
    root = {}
    bookmark_count = 0

    for bookmark in tqdm(bookmarks, desc="Building HTML tree"):
        # Get the category path, default to uncategorized if missing or error-related
        category_path_str = bookmark.get('category_path')

        # Assign uncategorized path if it's missing, None, or indicates an error state
        if not category_path_str or any(msg in category_path_str for msg in ["Uncategorized", "Error", "Cancelled", "Interrupted", "Task unfinished"]):
            category_path_str = uncategorized_folder_name

        # Split the path into parts
        path_parts = category_path_str.split('/')
        current_level = root

        # Traverse or create the nested dictionary structure
        for i, part in enumerate(path_parts):
            part = part.strip() # Clean up whitespace
            if not part: # Skip empty parts from e.g., "//" or "/folder"
                 continue

            if part not in current_level:
                # If it's the last part of the path, the value is a list for bookmarks
                if i == len(path_parts) - 1 or all(not p.strip() for p in path_parts[i+1:]):
                    current_level[part] = []
                else:
                    # Otherwise, it's a folder, the value is another dictionary
                    current_level[part] = {}

            # Move down the tree. If the current part leads to a list,
            # we can only append bookmarks at this level.
            if isinstance(current_level[part], list):
                 # This part should be the final level for bookmarks
                 # Append the bookmark here and stop traversing this path
                 current_level[part].append(bookmark)
                 bookmark_count += 1
                 break # Path processed for this bookmark
            else:
                # Move to the next level (sub-dictionary)
                current_level = current_level[part]

    print(f"Organized {bookmark_count} bookmarks into the tree structure.")
    return root

def write_html_node(soup: BeautifulSoup, parent_tag: Tag, node_name: str, content: any):
    """
    Recursively writes HTML for a node (folder or list of bookmarks) in the tree.
    content is either a dict (folder) or a list (bookmarks).
    """
    # Create a DT tag for the item (either folder or bookmark link)
    dt_tag = soup.new_tag("dt")
    parent_tag.append(dt_tag) # Append DT to the parent DL

    if isinstance(content, dict):
        # This is a folder node
        h3_tag = soup.new_tag("h3")
        h3_tag.string = escape(node_name) # Escape folder name
        dt_tag.append(h3_tag)

        # Create a new DL for the contents of this folder
        dl_tag = soup.new_tag("dl")
        dt_tag.insert_after(dl_tag) # Insert the DL after the DT (containing the H3)
        # Optional: Add <p> after DL for Netscape format compatibility
        p_tag = soup.new_tag("p")
        dl_tag.insert_after(p_tag)


        # Sort folder contents alphabetically by name/title before recursing
        # Separate sub-folders (dict) and bookmarks (list within dict)
        folder_items = sorted([(name, item) for name, item in content.items()])

        for name, item in folder_items:
             if isinstance(item, dict) or (isinstance(item, list) and all(isinstance(b, dict) for b in item)):
                 # Recurse for sub-folders or lists of bookmarks
                 write_html_node(soup, dl_tag, name, item)
             else:
                 # This shouldn't typically happen if build_bookmark_tree works correctly,
                 # but handle unexpected items at this level.
                 logger.warning(f"Unexpected item '{name}' in tree structure under folder '{node_name}'. Skipping.")


    elif isinstance(content, list) and all(isinstance(b, dict) for b in content):
        # This is a list of bookmarks under a folder name (node_name is the folder name)
        # Or this could be a list of bookmarks directly under the root if node_name is the key at root level

        # If node_name is not the default uncategorized name,
        # create a folder for this list of bookmarks if they were grouped implicitly.
        # However, build_bookmark_tree already handles creating the list *at* the final level,
        # so node_name here is typically the final folder name.
        # Bookmarks are directly appended to the list associated with the final folder key.

        # Let's restructure write_html_node slightly.
        # Call it passing the parent tag (the DL), and the content (dict or list).
        # If content is a dict, create H3 and DL.
        # If content is a list, iterate through bookmarks and create DT/A/DD.

        # This case (content is list) needs to be handled *within* the recursion
        # when a key's value is found to be a list. Let's adjust the main recursive function.
        # Redefine write_html_content to take (soup, parent_dl_tag, tree_node_content)

        # --- Refined Approach for HTML Writing ---
        # The recursive function should take the current 'parent_dl_tag' and the 'current_node_content' (dict or list)
        # If current_node_content is a dict (folder):
        #   Iterate through items (folder_name, folder_content).
        #   Create DT/H3 for folder_name.
        #   Create new DL after DT.
        #   Call recursive function on the new DL and folder_content.
        # If current_node_content is a list (bookmarks):
        #   Iterate through bookmarks in the list.
        #   Create DT/A/DD for each bookmark and append to the parent_dl_tag.

        # Let's abandon the old write_html_node signature and create a new recursive helper.
        pass # This branch will be replaced by the helper


def write_folder_content(soup: BeautifulSoup, parent_dl_tag: Tag, folder_content: Dict):
    """
    Recursive helper to write the content of a folder (sub-folders and bookmarks)
    to the given parent_dl_tag.
    """
    # Sort the items within the current folder level
    # Directories (dict values) first, then bookmarks (list values implicitly handled as leaf)
    # Sorting by key (folder name)
    sorted_items = sorted(folder_content.items())

    for name, content in sorted_items:
        if isinstance(content, dict):
            # This item is a sub-folder
            dt_tag = soup.new_tag("dt")
            parent_dl_tag.append(dt_tag)

            h3_tag = soup.new_tag("h3")
            h3_tag.string = escape(name)
            dt_tag.append(h3_tag)

            # Create a new DL for the contents of this sub-folder
            sub_dl_tag = soup.new_tag("dl")
            dt_tag.insert_after(sub_dl_tag)
            # Optional: Add <p> after DL
            p_tag = soup.new_tag("p")
            sub_dl_tag.insert_after(p_tag)

            # Recurse into the sub-folder
            write_folder_content(soup, sub_dl_tag, content)

        elif isinstance(content, list) and all(isinstance(b, dict) for b in content):
            # This item is a list of bookmarks (at a leaf node of the folder structure)
            # Iterate through bookmarks and add their HTML
            # Sort bookmarks by title within the folder
            sorted_bookmarks = sorted(content, key=lambda b: b.get('title', '').lower())

            for bookmark in sorted_bookmarks:
                dt_tag = soup.new_tag("dt")
                parent_dl_tag.append(dt_tag)

                a_tag = soup.new_tag("a", href=bookmark.get('uri', '#'))
                # Add common Netscape attributes (optional but good practice)
                a_tag['add_date'] = int(time.time()) # Or use actual add date if available
                a_tag.string = escape(bookmark.get('title', 'No Title')) # Link text is the title
                dt_tag.append(a_tag)

                # Add the summary as a DD tag if it exists and is not an error/status message
                summary = bookmark.get('summary')
                if summary and not any(status_msg in summary for status_msg in ["Failed to fetch", "Summary unavailable", "Error during processing", "Unexpected Error", "Processing cancelled", "Processing interrupted", "Task unfinished"]):
                     dd_tag = soup.new_tag("dd")
                     dd_tag.string = escape(summary)
                     dt_tag.append(dd_tag) # Append DD tag inside DT tag, after the A tag


        else:
             # Should not happen with correctly built tree
             logger.error(f"Unexpected content type ({type(content)}) in tree structure for key '{name}'. Skipping.")


def generate_netscape_html_tree(bookmark_tree: Dict, html_output_path: str, title: str = "Bookmarks with Summaries"):
    """
    Generates a Netscape HTML bookmarks file from the in-memory tree structure.
    """
    if not html_output_path:
        logger.info("HTML output path not provided. Skipping HTML generation.")
        return

    print(f"\nGenerating Netscape HTML bookmark file: {html_output_path}")

    try:
        # Create the basic HTML structure
        soup = BeautifulSoup("", 'html.parser', features="xml") # Start with an empty document
        html_tag = soup.new_tag("html")
        soup.append(html_tag)

        head_tag = soup.new_tag("head")
        html_tag.append(head_tag)

        meta_charset = soup.new_tag("meta", charset="UTF-8")
        head_tag.append(meta_charset)

        title_tag = soup.new_tag("title")
        title_tag.string = title
        head_tag.append(title_tag)

        body_tag = soup.new_tag("body")
        html_tag.append(body_tag)

        h1_tag = soup.new_tag("h1")
        h1_tag.string = title
        body_tag.append(h1_tag)

        # The main DL that contains the entire tree
        main_dl_tag = soup.new_tag("dl")
        main_dl_tag[' عاي'] = "0" # Common attribute in Netscape format root DL
        body_tag.append(main_dl_tag)

        # Start the recursive writing process from the root of the tree
        # The root dictionary's keys are the top-level folders/items
        write_folder_content(soup, main_dl_tag, bookmark_tree)


        # Save the modified HTML
        with open(html_output_path, 'w', encoding='utf-8') as f:
            # Use formatter="html" and prettify for standard, readable output
            f.write(soup.prettify(formatter="html"))

        print(f"Netscape HTML tree output saved to: {html_output_path}")

    except Exception as e:
        logger.critical(f"Error generating Netscape HTML tree file {html_output_path}: {str(e)}")
        print(f"Critical Error generating Netscape HTML tree file: {str(e)}")


def main():
    global _lxml_available

    parser = argparse.ArgumentParser(
        description="Bookmark Summarizer - Add AI summaries and categories to bookmarks and generate Netscape HTML tree output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--api-key',
        required=True,
        help='DeepSeek API key'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input JSON file path with bookmarks (array of objects with "uri" and "title")'
    )
    parser.add_argument(
        '--output-json', '-oj',
        help='Optional output JSON file path for bookmarks with summaries and categories (if not provided, JSON is not saved)'
    )
    parser.add_argument(
        '--html-output', '-ho',
        required=True, # HTML output is now the primary structured output
        help='Output Netscape HTML file path for bookmarks organized into a directory tree by AI categories'
    )
    parser.add_argument(
        '--max-workers', '-w',
        type=int,
        default=5,
        help='Maximum number of concurrent worker threads for summarizing and categorizing'
    )
    parser.add_argument(
        '--log-dir', '-l',
        default='logs',
        help='Directory to save error log files'
    )
    parser.add_argument(
        '--timeout', '-t',
        type=float,
        default=10.0,
        help='Timeout in seconds for fetching each webpage (HTTP request)'
    )
    parser.add_argument(
        '--uncategorized-folder',
        default='Uncategorized',
        help='Name of the folder for bookmarks that could not be categorized'
    )


    args = parser.parse_args()

    # --- Logging Setup ---
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"{timestamp}_bookmark_summarizer_errors.log")
    logger.setLevel(logging.WARNING)
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    if not logger.handlers:
       logger.addHandler(file_handler)

    print(f"Error and warning messages will be logged to: {log_file_path}")
    # --- End Logging Setup ---

    # --- Check for lxml availability ---
    try:
        import lxml
        _lxml_available = True
    except ImportError:
        _lxml_available = False
        warning_msg = "lxml not found. Falling back to html.parser for webpage content extraction. Install lxml (`pip install lxml`) for potentially faster parsing."
        logger.warning(warning_msg)
        print(f"Warning: {warning_msg}")
    except Exception as e:
         _lxml_available = False
         warning_msg = f"Error importing lxml: {str(e)}. Falling back to html.parser."
         logger.warning(warning_msg)
         print(f"Warning: {warning_msg}")

    # --- End lxml Check ---


    start_time = time.time()
    processed_bookmarks_list = []

    try:
        summarizer = BookmarkSummarizer(
            deepseek_api_key=args.api_key,
            input_file=args.input,
            output_file=args.output_json, # Pass the JSON output path (optional)
            max_workers=args.max_workers,
            fetch_timeout=args.timeout
        )

        # Step 1: Process bookmarks (fetch, extract, summarize, categorize)
        processed_bookmarks_list = summarizer.process_bookmarks()

        # Step 2: Build the tree structure from processed bookmarks
        if processed_bookmarks_list:
            bookmark_tree = build_bookmark_tree(processed_bookmarks_list, args.uncategorized_folder)

            # Step 3: Generate the Netscape HTML file from the tree
            generate_netscape_html_tree(bookmark_tree, args.html_output)
        else:
            print("\nNo bookmarks processed successfully. Skipping HTML generation.")


    except Exception as e:
        logger.critical(f"Program execution failed: {str(e)}")
        print(f"\nCritical Error: {str(e)}")
        sys.exit(1)

    finally:
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()