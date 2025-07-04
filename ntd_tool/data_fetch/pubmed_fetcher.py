# data_fetch/pubmed_fetcher.py
import ftplib, requests, gzip, xml.etree.ElementTree as ET
import datetime, time, csv, os

FIELDNAMES = [
    "pmid", "title", "abstract", "authors", "language",
    "pub_year", "pub_month", "publication_type", "country",
    "agency", "mesh_terms", "doi"
]

FTP_HOST = "ftp.ncbi.nlm.nih.gov"
FTP_DIR = "pubmed/updatefiles"
BASE_URL = f"https://{FTP_HOST}/{FTP_DIR}/"

def daterange(start, end):
    for i in range((end - start).days + 1):
        yield start + datetime.timedelta(i)

def list_xml_gz_for_date(day):
    ftp = ftplib.FTP(FTP_HOST)
    ftp.login()
    ftp.cwd(FTP_DIR)
    files = ftp.nlst()
    matched = []
    for fn in files:
        if not fn.endswith(".xml.gz"):
            continue
        try:
            ts = ftp.sendcmd(f"MDTM {fn}").split()[1]
            if datetime.datetime.strptime(ts, "%Y%m%d%H%M%S").date() == day:
                matched.append(fn)
        except:
            pass
    ftp.quit()
    return matched

def parse_updatefile(url, writer, start_year, end_year):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with gzip.GzipFile(fileobj=r.raw) as gz:
            for _, elem in ET.iterparse(gz, events=("end",)):
                if elem.tag != "PubmedArticle":
                    continue
                mc = elem.find("MedlineCitation")
                ai = mc.find("Article") if mc is not None else None
                pd = elem.find("PubmedData")

                pub_year = pub_month = None
                if ai is not None:
                    pdj = ai.find("Journal/JournalIssue/PubDate")
                    if pdj is not None:
                        y = pdj.findtext("Year", "")
                        m = pdj.findtext("Month", "")
                        if y.isdigit():
                            pub_year = int(y)
                        if m.isdigit():
                            pub_month = int(m)
                        else:
                            MONTH_MAP = {
                                "Jan":1,"Feb":2,"Mar":3,"Apr":4,
                                "May":5,"Jun":6,"Jul":7,"Aug":8,
                                "Sep":9,"Oct":10,"Nov":11,"Dec":12
                            }
                            pub_month = MONTH_MAP.get(m, None)

                if pub_year is None or not (start_year <= pub_year <= end_year):
                    elem.clear()
                    continue

                pmid  = mc.findtext("PMID", "")
                title = ai.findtext("ArticleTitle", "") if ai is not None else ""
                abstract = ""
                if ai is not None:
                    ab = ai.find("Abstract")
                    if ab is not None:
                        abstract = "".join(ab.itertext())
                authors = ""
                if ai is not None:
                    authors = "; ".join(
                        f"{a.findtext('ForeName','')} {a.findtext('LastName','')}"
                        for a in ai.findall("AuthorList/Author")
                    )
                language = ai.findtext("Language", "") if ai is not None else ""
                mesh_terms = "; ".join(
                    mh.findtext("DescriptorName", "")
                    for mh in mc.findall("MeshHeadingList/MeshHeading")
                ) if mc is not None else ""
                publication_type = "; ".join(
                    pt.text for pt in ai.findall("PublicationTypeList/PublicationType")
                ) if ai is not None else ""
                country = mc.findtext("MedlineJournalInfo/Country", "")
                agency = "; ".join(
                    g.findtext("Agency", "") for g in ai.findall("GrantList/Grant")
                ) if ai is not None else ""
                doi = ""
                if pd is not None:
                    for aid in pd.findall("ArticleIdList/ArticleId"):
                        if aid.get("IdType") == "doi" and aid.text:
                            doi = aid.text
                            break

                writer.writerow({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "language": language,
                    "pub_year": pub_year,
                    "pub_month": pub_month,
                    "publication_type": publication_type,
                    "country": country,
                    "agency": agency,
                    "mesh_terms": mesh_terms,
                    "doi": doi
                })
                elem.clear()

def fetch_pubmed(start_date: str, end_date: str, output_csv: str = "temp/pubmed_raw.csv") -> str:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt   = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    start_year, end_year = start_dt.year, end_dt.year

    with open(output_csv, "w", newline="", encoding="utf-8") as outf:
        writer = csv.DictWriter(outf, FIELDNAMES)
        writer.writeheader()
        for day in daterange(start_dt, end_dt):
            print(f"Processing {day}…")
            for fn in list_xml_gz_for_date(day):
                url = BASE_URL + fn
                print(f"  ↳ {fn}")
                try:
                    parse_updatefile(url, writer, start_year, end_year)
                except Exception as e:
                    print(f"    ! parse error: {e}")
                time.sleep(0.1)
    return output_csv

