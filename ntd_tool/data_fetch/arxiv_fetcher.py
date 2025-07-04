import requests, csv, datetime, time, os
import xml.etree.ElementTree as ET
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict

FIELDNAMES = [
    "arxiv_id", "title", "abstract", "authors",
    "pub_year", "pub_month", "doi",
    "primary_category", "categories", "pdf_url"
]

OAI_URL = "http://export.arxiv.org/oai2"
METADATA_PREF = "arXiv"
PAUSE_SECONDS = 5

ns = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/"
}

def fetch_arxiv(start_date: str, end_date: str, output_csv: str = "temp/arxiv_raw.csv") -> str:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))

    resumption_token = None
    fetched = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        while True:
            params = {"verb": "ListRecords"}
            if resumption_token:
                params["resumptionToken"] = resumption_token
            else:
                params.update({
                    "metadataPrefix": METADATA_PREF,
                    "from": start_date,
                    "until": end_date
                })

            resp = session.get(OAI_URL, params=params)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)

            for rec in root.findall(".//oai:record", ns):
                md = rec.find("oai:metadata/arxiv:arXiv", ns)
                if md is None:
                    continue

                created = md.findtext("arxiv:created", namespaces=ns)
                if not created:
                    continue
                cdate = datetime.datetime.strptime(created, "%Y-%m-%d").date()
                if cdate < start_dt or cdate > end_dt:
                    continue

                aid = md.findtext("arxiv:id", namespaces=ns).strip()
                title = md.findtext("arxiv:title", namespaces=ns).strip()
                abstr = md.findtext("arxiv:abstract", namespaces=ns).strip()

                authors = []
                for a in md.findall("arxiv:authors/arxiv:author", ns):
                    fore = a.findtext("arxiv:forenames", namespaces=ns) or ""
                    key = a.findtext("arxiv:keyname", namespaces=ns) or ""
                    full = " ".join(p for p in (fore, key) if p)
                    if full:
                        authors.append(full)
                authors_str = "; ".join(authors)

                py, pm = cdate.year, cdate.month
                doi = md.findtext("arxiv:doi", namespaces=ns) or ""

                cat_text = md.findtext("arxiv:categories", namespaces=ns) or ""
                categories = cat_text.split()
                primary = categories[0] if categories else ""

                pdf_url = f"http://arxiv.org/pdf/{aid}.pdf"

                writer.writerow({
                    "arxiv_id": aid,
                    "title": title,
                    "abstract": abstr,
                    "authors": authors_str,
                    "pub_year": py,
                    "pub_month": pm,
                    "doi": doi,
                    "primary_category": primary,
                    "categories": "; ".join(categories),
                    "pdf_url": pdf_url
                })
                fetched += 1

            token_el = root.find(".//oai:resumptionToken", ns)
            if token_el is None or not token_el.text:
                break
            resumption_token = token_el.text
            time.sleep(PAUSE_SECONDS)

    print(f"\n✅ ArXiv: harvested {fetched} records → {output_csv}")
    return output_csv
