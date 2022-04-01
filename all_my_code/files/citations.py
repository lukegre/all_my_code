
def extract_citation_from_doi(doi):
    from pandas import to_datetime
    from crossref.restful import Works

    works = Works()

    out = works.doi(doi)
    if out is None:
        return ''
    
    refs = out.pop('reference')

    date = to_datetime(out['created']['date-time']).year
    doi = out['DOI']
    author = ', '.join([f"{a['family']}, {a['given']}" for a in out['author']])
    if len(out['author']) > 2:
        a = out['author'][0]
        author = f"{a['family']} et al."

    citation = f"{author} ({date})"

    return citation