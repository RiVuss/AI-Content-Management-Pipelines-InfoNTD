import requests, csv, os, time
from typing import List, Dict

FIELDNAMES = [
    "doi", "title", "abstract", "authors", "language",
    "pub_year", "pub_month", "type", "publisher",
    "funders", "subjects", "url"
]

API_URL = "https://api.crossref.org/works"


def fetch_crossref(start_date: str, end_date: str, output_csv: str = "temp/crossref_raw.csv") -> str:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    date_filter = f"from-pub-date:{start_date},until-pub-date:{end_date}"
    
    resp = requests.get(API_URL, params={"filter": date_filter, "rows": 0})
    resp.raise_for_status()
    total = resp.json()["message"]["total-results"]
    print(f"Total CrossRef works: {total}")

    cursor = "*"
    rows = 1000
    fetched = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, FIELDNAMES)
        writer.writeheader()

        while fetched < total:
            params = {
                "filter": date_filter,
                "rows": rows,
                "cursor": cursor,
            }
            r = requests.get(API_URL, params=params)
            r.raise_for_status()
            msg = r.json()["message"]

            items = msg.get("items", [])
            if not items:
                break

            for item in items:
                doi = item.get("DOI", "")
                title = " ".join(item.get("title", []))
                abstract = item.get("abstract", "").replace("<jats:p>", "").replace("</jats:p>", "")
                authors = "; ".join(
                    f"{a.get('given','')} {a.get('family','')}"
                    for a in item.get("author", []) if a.get("given") and a.get("family")
                )
                language = item.get("language", "")
                issued = item.get("issued", {}).get("date-parts", [])
                pub_year = issued[0][0] if issued and issued[0] else ""
                pub_month = issued[0][1] if issued and len(issued[0]) > 1 else ""
                typ = item.get("type", "")
                publisher = item.get("publisher", "")
                funders = "; ".join(f.get("name", "") for f in item.get("funder", []))
                subjects = "; ".join(item.get("subject", []))
                url = item.get("URL", "")

                writer.writerow({
                    "doi": doi, "title": title, "abstract": abstract,
                    "authors": authors, "language": language,
                    "pub_year": pub_year, "pub_month": pub_month,
                    "type": typ, "publisher": publisher,
                    "funders": funders, "subjects": subjects, "url": url
                })

            fetched += len(items)
            print(f"Fetched {fetched}/{total} works…")
            cursor = msg.get("next-cursor")
            if not cursor:
                break
            time.sleep(0.4)

    return output_csv
