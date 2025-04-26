# Bookmark Summarizer and Organizer

This Python script fetches the content of bookmarks from a JSON input file, generates concise summaries and hierarchical categories for each using the DeepSeek API, saves the updated data back to a JSON file (optional), and generates a structured Netscape HTML bookmark file organized by the AI-generated categories. It uses multitasking (threading) to speed up the process of fetching webpages and calling the API.

## Features

* Fetch webpage content with robust error handling (timeouts, SSL fallback).
* Extract main textual content from HTML using BeautifulSoup (prefers lxml for speed).
* Generate 2-3 sentence summaries of bookmarks using the DeepSeek Chat API.
* Generate hierarchical category paths (e.g., "Technology/AI", "News/Politics") using the DeepSeek Chat API based on summaries, titles, and URLs.
* Process bookmarks concurrently using a thread pool.
* Save the processed bookmarks (including summaries and categories) to a JSON file.
* Generate a Netscape HTML bookmark file with bookmarks organized into folders based on the AI-generated category paths.
* Log warnings and errors to a timestamped file for debugging.
* Graceful shutdown on `Ctrl+C`, saving progress made so far.

## Prerequisites

* **Python 3.7+**: You need a modern version of Python installed.
* **DeepSeek API Key**: Obtain an API key from the DeepSeek website ([https://www.deepseek.com/](https://www.deepseek.com/)). This script will incur costs based on your API usage.
* **Bookmarks in Flat JSON Format**: The script expects an input JSON file containing a *flat array* of bookmark objects. Each object **must** have a `uri` key (the URL) and **should** have a `title` key. Example:

    ```json
    [
      {
        "title": "Example Domain",
        "uri": "https://example.com"
      },
      {
        "title": "DeepSeek AI",
        "uri": "https://www.deepseek.com/"
      },
      {
        "title": "Python Programming Language",
        "uri": "https://www.python.org/"
      }
      // ... more bookmarks
    ]
    ```

    *Note*: Most browsers export bookmarks in Netscape HTML format, not JSON. You will likely need an intermediate step to convert your browser's HTML export into this flat JSON format. Tools or simple scripts can achieve this. This script *does not* perform the HTML-to-JSON conversion.

## Installation

1. **Save the script:** Save the Python code provided previously as a file (e.g., `summarize_bookmarks.py`).
2. **Install dependencies:** Open your terminal or command prompt and run the following command:

    ```bash
    pip install requests beautifulsoup4 openai tqdm lxml
    ```

    * `requests`: For fetching web pages.
    * `beautifulsoup4`: For parsing HTML.
    * `openai`: The DeepSeek API uses the OpenAI library with a custom `base_url`.
    * `tqdm`: For displaying a progress bar.
    * `lxml`: (Optional but Recommended) A faster HTML parser. The script will fall back to `html.parser` if `lxml` is not installed, but parsing might be slower. The script will warn you if lxml is not found.

## Exporting Bookmarks from Browsers (Getting the Source Data)

The script requires a flat JSON file as input. Your browsers typically export bookmarks as Netscape HTML files. You'll need to export the HTML and then use another method (a custom script, an online converter tool, etc.) to convert that HTML into the required flat JSON format.

Here's how to export bookmarks to HTML from common browsers:

### Google Chrome

1. Open Chrome.
2. Click the three vertical dots in the top right corner (`⋮`).
3. Hover over "Bookmarks".
4. Click "Bookmark manager" (or press `Ctrl+Shift+O`).
5. In the Bookmark manager, click the three vertical dots next to the search bar (`⋮`).
6. Click "Export bookmarks".
7. Choose a location and save the file (e.g., `chrome_bookmarks.html`).

### Mozilla Firefox

1. Open Firefox.
2. Click the Library icon (looks like stacked books).
3. Click "Bookmarks".
4. Click "Manage Bookmarks" (or press `Ctrl+Shift+B`).
5. In the Library window, click "Import and Backup".
6. Click "Export Bookmarks to HTML...".
7. Choose a location and save the file (e.g., `firefox_bookmarks.html`).

### Microsoft Edge

1. Open Edge.
2. Click the three horizontal dots in the top right corner (`...`).
3. Hover over "Favorites".
4. Click "Manage favorites".
5. Click the three horizontal dots below "Favorites" (`...`).
6. Click "Export favorites".
7. Choose a location and save the file (e.g., `edge_bookmarks.html`).

After exporting to HTML, you will need to convert this HTML file into the flat JSON format expected by the script.

## Usage

Run the script from your terminal or command prompt.

```bash
python summarize_bookmarks.py --api-key YOUR_DEEPSEEK_API_KEY --input path/to/your/bookmarks.json --html-output path/to/save/output_bookmarks.html 
```

Use code with caution.

Arguments:

--api-key (required): Your DeepSeek API key.

--input, -i (required): Path to your input JSON file containing the bookmarks (flat array of objects with uri and title).

--html-output, -ho (required): Path where the output Netscape HTML file with summarized and categorized bookmarks will be saved.

--output-json, -oj (optional): Path where the processed JSON file (including summaries and categories) will be saved. If not provided, the JSON data is not saved to a file.

--max-workers, -w (optional, default: 5): Maximum number of concurrent threads to use for fetching pages and calling the API. Adjust this based on your network speed and API limits.

--log-dir, -l (optional, default: logs): Directory where timestamped error log files will be saved.

--timeout, -t (optional, default: 10.0): Timeout in seconds for fetching each webpage.

--uncategorized-folder (optional, default: Uncategorized): The name of the folder in the output HTML file where bookmarks that could not be categorized by the AI will be placed.
Example Command:

```bash
python summarize_bookmarks.py --api-key sk-YOUR_API_KEY --input my_bookmarks.json --output-json processed_bookmarks.json --html-output categorized_bookmarks.html -w 10 -t 15 --log-dir my_logs
```

Use code with caution.

This command will:

*Load bookmarks from my_bookmarks.json.

* Use your DeepSeek API key.

* Process up to 10 bookmarks concurrently, with a 15-second timeout for each web request.

* Generate summaries and categories using the API.

* Save the updated bookmark data to processed_bookmarks.json.

* Build a directory tree based on AI categories and save it as a Netscape HTML file to categorized_bookmarks.html.

* Log errors and warnings to files in the my_logs directory.

## Output Files

JSON Output (--output-json): If specified, this file will contain the original bookmarks plus the new summary and category_path keys added to each bookmark object. This can be useful for inspecting the results or for future runs, as the script will skip bookmarks that already have these keys.
Netscape HTML Output (--html-output): This is the primary output file containing your bookmarks. It will have a standard Netscape HTML bookmark structure (`<H1>`, `<DL>`, `<DT>`, `<H3>` for folders, `<A>` for links). Bookmarks will be placed into a hierarchy of folders based on the category_path assigned by the AI. The AI-generated summary for each bookmark will be included as a `<DD>` tag immediately following the `<DT><A>` tag.

## Logging

The script logs warnings and errors to a timestamped file (e.g., logs/YYYYMMDD_HHMMSS_bookmark_summarizer_errors.log) in the directory specified by --log-dir. Check this file if bookmarks fail to process, fetching errors occur, or API calls fail.
Graceful Shutdown (Ctrl+C)
You can press Ctrl+C at any time during the processing. The script will catch the interrupt, attempt to finish tasks that are already close to completion, cancel pending tasks, and save the results obtained so far to the output files before exiting. Bookmarks that were in progress or pending when interrupted will be marked accordingly.

## Important Notes

DeepSeek API Costs: Each summary and category generation consumes API tokens. Be mindful of your usage and costs.
API Key Security: Keep your API key confidential and do not share it publicly.
Rate Limits: DeepSeek API might impose rate limits. Using too many workers or having very short timeouts might hit these limits, resulting in API errors logged by the script.
Categorization Accuracy: The quality of the AI-generated categories depends on the AI model and the prompt. Results may vary and might require manual adjustments to the output HTML file. Some bookmarks might be placed in the "Uncategorized" folder if the AI cannot determine a clear category.
Performance: Performance is heavily dependent on network speed and the responsiveness of the websites being scraped. The max_workers setting is crucial for balancing speed and resource usage. Installing lxml is recommended for faster HTML parsing.

## Contributing

Feel free to open issues or pull requests if you find bugs or have suggestions for improvements.
