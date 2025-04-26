from bs4 import BeautifulSoup
import json
import argparse

def extract_bookmarks_to_json(html_file_path, json_file_path):
    """
    Extracts bookmark titles and URIs from a Netscape bookmarks HTML file
    and saves them to a JSON file.

    Args:
        html_file_path (str): Path to the input HTML bookmarks file.
        json_file_path (str): Path to the output JSON file.
    """
    bookmarks = []
    try:
        with open(html_file_path, 'r', encoding='utf-8') as fp:
            html_content = fp.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        a_tags = soup.find_all('a')

        for a_tag in a_tags:
            uri = a_tag.get('href')
            title = a_tag.string
            if uri and title: # Ensure both uri and title exist
                bookmarks.append({'title': title.strip(), 'uri': uri.strip()})

        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(bookmarks, json_file, indent=4, ensure_ascii=False)

        print(f"Bookmarks extracted and saved to '{json_file_path}'")

    except FileNotFoundError:
        print(f"Error: HTML file not found at '{html_file_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description='Extract bookmarks from Netscape HTML bookmarks file to JSON.')
    parser.add_argument('-i', '--input', required=True, help='Path to the input HTML bookmarks file.')
    parser.add_argument('-o', '--output', required=True, help='Path to the output JSON file.')
    args = parser.parse_args()

    extract_bookmarks_to_json(args.input, args.output)

if __name__ == "__main__":
    main()