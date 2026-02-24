import json
import urllib.request

url = "https://search.rcsb.org/rcsbsearch/v2/query"
query = {
    "query": {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute": "rcsb_entry_info.polymer_entity_count_protein",
            "operator": "equals",
            "value": 1
        }
    },
    "return_type": "entry",
    "request_options": {
        "paginate": {
            "start": 0,
            "rows": 5
        },
        "results_verbosity": "compact"
    }
}
data = json.dumps(query).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req) as response:
    result = json.loads(response.read().decode('utf-8'))
    print(json.dumps(result, indent=2))
