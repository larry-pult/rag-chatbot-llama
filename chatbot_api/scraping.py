from bs4 import BeautifulSoup
import requests
import re


def clean_document(text):
    # remove trailing whitespaces
    text = text.strip()

    # remove numerical reference marks
    text = re.sub(r'\[\d+\]', "", text)

    # remove letter reference marks
    text = re.sub(r'\[[a-z]\]', "", text)

    # remove multiple line breaks
    text = re.sub(r'\n\n+', "\n", text)
    
    return text


def scrap_and_save_page_into_txt(target_filename, url):
    """scraps a url, gets all text inside <p> tag, saves to ./documents/target_file_name.txt"""

    response = requests.get(url)

    soup = BeautifulSoup(response.content, "html.parser")

    paragraphs = soup.find_all("p")

    document = ""
    for p in paragraphs:
        document += p.get_text() + "\n"

    document = clean_document(document)

    with open(f"./documents/{target_filename}.txt", "w", errors="ignore") as file:
        file.write(document)


# example usage
def main():
    filenames_and_urls_example = (
        ("crowdstrike", "https://en.wikipedia.org/wiki/CrowdStrike"),
        ("genshin", "https://en.wikipedia.org/wiki/Genshin_Impact"),
        ("langchain", "https://python.langchain.com/v0.2/docs/concepts/")
    )

    # scrap all pages in the variable FILENAMES_AND_URLS
    for filename, url in filenames_and_urls_example:
        scrap_and_save_page_into_txt(
            target_filename=filename, 
            url=url
        )
    
    # bee movie script
    # adding a big pile of trash on purpose to see how the retriever performs against thousands of irrelevant documents
    with open(f"./documents/bee_movie.txt", "w", errors="ignore") as file:
        response = requests.get("https://courses.cs.washington.edu/courses/cse163/20wi/files/lectures/L04/bee-movie.txt")
        soup = BeautifulSoup(response.content, "html.parser")
        file.write(soup.prettify())


if __name__ == "__main__":
    main()