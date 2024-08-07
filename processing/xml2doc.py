import zipfile
import requests
import io
from lxml import etree
from tqdm import tqdm
import unicodedata
import re


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def get_process_xml(process_uuid):
    version = "02.44.152"
    url = "https://data.probas.umweltbundesamt.de/resource/processes/{}?format=xml&version={}"\
        .format(process_uuid, version)

    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    return None

fields = {'Name': './process:processInformation/process:dataSetInformation/process:name/process:baseName',
          'Land': './process:processInformation/process:geography/process:locationOfOperationSupplyOrProduction/@location',
          'Jahr': './process:processInformation/process:time/common:referenceYear',
          'Kategorie': './process:processInformation/process:dataSetInformation/process:classificationInformation/common:classification/common:class',
          'Beschreibung': './process:processInformation/process:technology/process:technologyDescriptionAndIncludedProcesses'
          }

def xml2txt(xml_doc):
    nsmap = {'process': 'http://lca.jrc.it/ILCD/Process',
             'common': 'http://lca.jrc.it/ILCD/Common'}
    infodict = {}
    tree = etree.ElementTree(etree.fromstring(xml_doc, parser=etree.XMLParser(encoding='utf-8')))
    for field_name, field_xpath in fields.items():
        nodes = tree.xpath(field_xpath, namespaces=nsmap)
        node_text = ''
        if len(nodes) == 1:
            node_text = nodes[0] if isinstance(nodes[0], str) else nodes[0].text
        elif len(nodes) > 1:
            node_text = ' - '.join([node.text for node in nodes])
        infodict[field_name] = node_text
    fname = slugify('{}_{}_{}'.format(infodict['Name'], infodict['Land'], infodict['Jahr']))
    filename = '../dbtxt/{}.txt'.format(fname)
    docstring = '\n'.join("{}: {}".format(k, v) for k, v in infodict.items())
    with open(filename, 'w+') as f:
        f.write(docstring)

for i in range(1,6):
    tree = etree.parse('probas_full_{}.xml'.format(i))
    nsmap = tree.getroot().nsmap.copy()
    elements = tree.xpath('./p:process/sapi:uuid', namespaces=nsmap)

    for el in tqdm(elements):
        xml_doc = get_process_xml(el.text)
        if xml_doc is not None:
            xml2txt(xml_doc)
